#pragma once

/// @file waveglow.hpp
/// @brief Compact conditional waveform flow with exact inverse and density.

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

#include "datamunge/nn/wavenet_stack_models.hpp"

namespace datamunge::nn {

/// @brief Sample-space or latent-space values and their change in log volume.
/// @par Example
/// @code
/// dm::nn::WaveGlowResult result;
/// @endcode
struct WaveGlowResult {
  Tensor values; ///< Transformed waveform or latent vector, shape [batch, samples].
  Tensor log_det; ///< Log absolute determinant per batch item, signed for the chosen direction.
};

/// @brief Transform paired audio samples with three log-mel conditioned flow steps.
///
/// Each step applies a learned triangular-factor channel mixer and an affine
/// coupling. A gated causal convolution predicts the coupling from its first
/// channel and the aligned log-mel features. The inverse reverses both the
/// layer order and the order of operations within each layer.
/// @param model Store whose parameters initialize on first use.
/// @param values Float tensor [batch, samples], with 64 samples per conditioning frame.
/// @param mel Log-mel tensor [batch, frames, 8] in the store dtype.
/// @param inverse Whether to map latents to audio instead of audio to latents.
/// @return Transformed values and one signed log determinant per clip.
/// @throws std::invalid_argument For incompatible shapes, dtypes or stored parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(3);
/// auto latent = dm::nn::waveglow_transform(model, dm::nn::zeros({1,128}), dm::nn::zeros({1,2,8}));
/// @endcode
/// @par References
/// - Prenger, Valle and Catanzaro (2019), WaveGlow. https://arxiv.org/abs/1811.00002
/// @see waveglow_nll()
inline WaveGlowResult waveglow_transform(ParameterStore& model, const Tensor& values,
                                         const Tensor& mel, bool inverse = false) {
  if (!values.defined() || !mel.defined() || values.dim()!=2 || mel.dim()!=3 ||
      values.size(0)<1 || values.size(0)!=mel.size(0) || values.size(1)<64 ||
      values.size(1)%64!=0 || values.size(1)/64!=mel.size(1) || mel.size(2)!=8 ||
      values.dtype()!=model.dtype() || mel.dtype()!=model.dtype()) {
    throw std::invalid_argument("Invalid WaveGlow values or conditioning");
  }
  const auto batch = values.size(0), steps = values.size(1)/2;
  auto state = reshape(values,{batch,steps,2});
  const auto condition = repeat(mel*.2,32,1);
  auto log_det = zeros({batch},values.dtype());
  for (int i=0; i<3; ++i) {
    const auto key = "flow"+std::to_string(inverse ? 2-i : i);
    /// Applies a two-channel triangular-factor mixer in the chosen direction.
    const auto mix = [&](const Tensor& input) {
      const auto log_a = model.get(key+"/log_a",{},init::zeros());
      const auto log_d = model.get(key+"/log_d",{},init::zeros());
      const auto b = model.get(key+"/upper",{},init::constant(.1));
      const auto c = model.get(key+"/lower",{},init::constant(-.1));
      const auto channels = split(input,2,2);
      log_det = log_det + (inverse ? -1. : 1.)*steps*(log_a+log_d);
      if (inverse) {
        const auto second = (channels[1]-c*channels[0])*exp(-log_d);
        return concat({(channels[0]-b*second)*exp(-log_a),second},2);
      }
      const auto first = exp(log_a)*channels[0]+b*channels[1];
      return concat({first,c*first+exp(log_d)*channels[1]},2);
    };
    /// Applies the conditioned affine coupling and accumulates its volume change.
    const auto couple = [&](const Tensor& input) {
      const auto channels = split(input,2,2);
      const auto signal = models::WaveStackConv(model,key+"/audio",channels[0],16,3,2);
      const auto context = models::WaveStackConv(model,key+"/mel",condition,16);
      const auto gates = split(signal+context,2,2);
      const auto hidden = tanh(gates[0])*sigmoid(gates[1]);
      const auto affine = split(models::WaveStackConv(model,key+"/affine",hidden,2),2,2);
      const auto log_scale = tanh(affine[0]);
      log_det = log_det + (inverse ? -1. : 1.)*sum(log_scale,{1,2});
      const auto second = inverse ? (channels[1]-affine[1])*exp(-log_scale)
                                  : channels[1]*exp(log_scale)+affine[1];
      return concat({channels[0],second},2);
    };
    state = inverse ? mix(couple(state)) : couple(mix(state));
  }
  return {reshape(state,values.shape()),log_det};
}

/// @brief Compute average negative log likelihood per sample under a standard Gaussian base.
/// @param encoded Forward transform with values [batch, samples] and log determinants [batch].
/// @return Scalar negative log likelihood in nats per sample.
/// @throws std::invalid_argument If the encoded shapes or dtypes are incompatible.
/// @par Example
/// @code
/// auto loss = dm::nn::waveglow_nll({dm::nn::zeros({1,128}),dm::nn::zeros({1})});
/// @endcode
/// @see waveglow_transform()
inline Tensor waveglow_nll(const WaveGlowResult& encoded) {
  const auto& z = encoded.values;
  const auto& det = encoded.log_det;
  if (!z.defined() || !det.defined() || z.dim()!=2 || z.size(0)<1 || z.size(1)<1 ||
      det.shape()!=Shape{z.size(0)} || det.dtype()!=z.dtype() ||
      (z.dtype()!=DType::Float32 && z.dtype()!=DType::Float64)) {
    throw std::invalid_argument("Invalid WaveGlow encoding");
  }
  return .5*std::log(2*std::numbers::pi)+.5*mean(square(z))-mean(det)/z.size(1);
}

}  // namespace datamunge::nn
