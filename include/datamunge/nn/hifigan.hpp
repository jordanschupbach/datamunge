#pragma once

/// @file hifigan.hpp
/// @brief Compact HiFi-GAN waveform generator and training objectives.

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "datamunge/nn/audio.hpp"
#include "datamunge/nn/wavenet_stack_models.hpp"

namespace datamunge::nn {

/// @brief Compute 8 kHz log-mel conditioning with a 64-sample frame and hop.
/// @param wave Floating waveform [batch, samples, 1] with at least 64 samples.
/// @return Log-mel tensor [batch, floor(samples/64), 8].
/// @throws std::invalid_argument If the waveform shape or type is invalid.
/// @par Example
/// @code
/// auto mel = dm::nn::hifigan_log_mel(dm::nn::zeros({1,128,1}));
/// @endcode
/// @see log_mel_spectrogram(), hifigan_generate()
inline Tensor hifigan_log_mel(const Tensor& wave) {
  if (!wave.defined() || wave.dim() != 3 || wave.size(0) < 1 || wave.size(1) < 64 ||
      wave.size(2) != 1 || (wave.dtype() != DType::Float32 && wave.dtype() != DType::Float64)) {
    throw std::invalid_argument("Invalid HiFi-GAN waveform");
  }
  const auto weights = linear_to_mel_weight_matrix(8,33,8000,0,4000,wave.dtype());
  return log_mel_spectrogram(power_spectrogram(stft(squeeze(wave,{2}),64,64,64)),weights);
}

/// @brief Upsample log-mel frames into a waveform using compact HiFi-GAN.
/// Three 4× nearest-neighbor stages each have two residual branches with
/// distinct kernel widths. Weights and biases initialize on first use.
/// @param generator Trainable parameter store matching condition dtype.
/// @param condition Log-mel frames [batch, frames, 8] with positive batch and frames.
/// @return Synthesized samples [batch, 64*frames, 1] in [-1,1].
/// @throws std::invalid_argument If the shape, dtype or stored parameters are invalid.
/// @par Example
/// @code
/// dm::nn::ParameterStore generator(5);
/// auto wave = dm::nn::hifigan_generate(generator, dm::nn::zeros({1,2,8}));
/// @endcode
/// @par References
/// - Kong et al. (2020), HiFi-GAN, https://arxiv.org/abs/2010.05646
/// @see hifigan_log_mel(), hifigan_objective()
inline Tensor hifigan_generate(ParameterStore& generator, const Tensor& condition) {
  if (!condition.defined() || condition.dim() != 3 || condition.size(0) < 1 ||
      condition.size(1) < 1 || condition.size(2) != 8 || condition.dtype() != generator.dtype()) {
    throw std::invalid_argument("Invalid HiFi-GAN condition");
  }
  constexpr int width = 12;
  auto state = conv1d(generator,"stem",condition*.2,width,3);
  for (int stage = 0; stage < 3; ++stage) {
    const auto name = "stage"+std::to_string(2 << (2*stage));
    const auto up = repeat(leaky_relu(state,.1),4,1);
    const auto mixed = conv1d(generator,name+"/up",up,width,5);
    auto combined = zeros_like(mixed);
    for (int kernel : {3,5}) {
      auto branch = mixed;
      for (int dilation : {1,2}) {
        const auto block = name+"/k"+std::to_string(kernel)+"/d"+std::to_string(dilation);
        const auto h = models::WaveStackConv(generator,block+"/dilated",leaky_relu(branch,.1),
                                            width,kernel,dilation);
        branch = branch + conv1d(generator,block+"/plain",leaky_relu(h,.1),width,kernel);
      }
      combined = combined + branch;
    }
    state = combined*.5;
  }
  return tanh(conv1d(generator,"wave",leaky_relu(state,.1),1,7));
}

/// @brief Scores and two feature maps from one waveform critic.
/// @par Example
/// @code
/// dm::nn::HiFiGanCritic critic;
/// @endcode
struct HiFiGanCritic {
  Tensor score; ///< Least-squares score at each output position.
  Tensor first; ///< First hidden activation.
  Tensor second; ///< Second hidden activation.
};

/// @brief Run two period critics and two scale critics on a waveform.
/// Periods 2 and 3 use time-axis convolutions on phase-folded, zero-padded
/// waveforms. Scale critics operate at full and half temporal resolution.
/// @param store Mutable discriminator parameter store, initialized on first use.
/// @param wave Waveform [batch, samples, 1] matching the store dtype, with at least two samples.
/// @return Critics ordered by period 2, period 3, full rate and half rate.
/// @throws std::invalid_argument If shape, dtype or stored parameters are incompatible.
/// @par Example
/// @code
/// dm::nn::ParameterStore critic(5);
/// auto scores = dm::nn::hifigan_critics(critic, dm::nn::zeros({1,128,1}));
/// @endcode
/// @par References
/// - Kong et al. (2020), HiFi-GAN, https://arxiv.org/abs/2010.05646
/// @see hifigan_objective()
inline std::array<HiFiGanCritic,4> hifigan_critics(ParameterStore& store, const Tensor& wave) {
  if (!wave.defined() || wave.dim() != 3 || wave.size(0) < 1 || wave.size(1) < 2 ||
      wave.size(2) != 1 || wave.dtype() != store.dtype()) {
    throw std::invalid_argument("Invalid HiFi-GAN critic input");
  }
  std::array<HiFiGanCritic,4> out;
  for (int period : {2,3}) {
    const auto name = "period"+std::to_string(period);
    const auto extra = (period-wave.size(1)%period)%period;
    const auto image = reshape(pad(wave,{{0,0},{0,extra},{0,0}}),{wave.size(0),-1,period,1});
    /// Applies an anisotropic SAME convolution with independent phase columns.
    const auto layer = [&](const char* key,const Tensor& x,int channels,int stride) {
      const auto prefix = name+"/"+key;
      const auto weight = store.get(prefix+"/W",{3,1,x.size(3),channels},init::normal(.12));
      const auto bias = store.get(prefix+"/b",{channels},init::zeros());
      return bias_add(conv2d(x,weight,{.strides={stride,1},.padding=Padding::Same}),bias);
    };
    const auto a = leaky_relu(layer("a",image,8,2),.1);
    const auto b = leaky_relu(layer("b",a,12,2),.1);
    out[period-2] = {layer("score",b,1,1),a,b};
  }
  const auto half = squeeze(avg_pool2d(expand_dims(wave,1),{1,2},{1,2}),{1});
  for (int scale : {1,2}) {
    const auto name = "scale"+std::to_string(scale);
    const auto& input = scale == 1 ? wave : half;
    const auto a = leaky_relu(conv1d(store,name+"/a",input,8,7,{.stride=2}),.1);
    const auto b = leaky_relu(conv1d(store,name+"/b",a,12,5,{.stride=2}),.1);
    out[scale+1] = {conv1d(store,name+"/score",b,1,3),a,b};
  }
  return out;
}

/// @brief Scalar HiFi-GAN objectives for alternating updates.
/// @par Example
/// @code
/// dm::nn::HiFiGanLoss loss;
/// @endcode
struct HiFiGanLoss {
  Tensor discriminator; ///< Sum of real and fake least-squares losses.
  Tensor generator; ///< Adversarial loss plus weighted feature and mel losses.
  Tensor mel; ///< Mean absolute error in log-mel space.
};

/// @brief Evaluate adversarial, feature-matching and log-mel objectives.
/// The generator adds twice the feature loss and 45 times the mel loss.
/// Select the desired scalar for the optimizer owning that model's parameters.
/// @param real Real-waveform critic outputs from hifigan_critics().
/// @param fake Synthesized-waveform critic outputs in the same order and shapes.
/// @param wave Synthesized waveform [batch, 64*frames, 1].
/// @param condition Matching reference log-mel tensor [batch, frames, 8].
/// @return Differentiable discriminator, generator and mel scalars.
/// @throws std::invalid_argument If operands have invalid or mismatched shapes.
/// @par Example
/// @code
/// dm::nn::ParameterStore critic(5);
/// auto wave = dm::nn::zeros({1,128,1});
/// auto scores = dm::nn::hifigan_critics(critic,wave);
/// auto loss = dm::nn::hifigan_objective(scores,scores,wave,dm::nn::hifigan_log_mel(wave));
/// @endcode
/// @par References
/// - Kong et al. (2020), HiFi-GAN, https://arxiv.org/abs/2010.05646
/// @see hifigan_critics(), hifigan_generate()
inline HiFiGanLoss hifigan_objective(const std::array<HiFiGanCritic,4>& real,
                                     const std::array<HiFiGanCritic,4>& fake,
                                     const Tensor& wave, const Tensor& condition) {
  const auto mel = hifigan_log_mel(wave);
  if (!condition.defined() || mel.shape() != condition.shape() || mel.dtype() != condition.dtype()) {
    throw std::invalid_argument("HiFi-GAN requires matching log-mel tensors");
  }
  auto d = zeros({},wave.dtype()), g = d, fm = d;
  for (std::size_t i=0; i<real.size(); ++i) {
    if (real[i].score.shape() != fake[i].score.shape() ||
        real[i].first.shape() != fake[i].first.shape() ||
        real[i].second.shape() != fake[i].second.shape()) {
      throw std::invalid_argument("HiFi-GAN requires corresponding critic maps");
    }
    d = d + mean(square(real[i].score-1)) + mean(square(fake[i].score));
    g = g + mean(square(fake[i].score-1));
    fm = fm + mean(abs(real[i].first-fake[i].first)) + mean(abs(real[i].second-fake[i].second));
  }
  const auto error = mean(abs(mel-condition));
  return {d,g+2*fm+45*error,error};
}

}  // namespace datamunge::nn
