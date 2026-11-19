#pragma once

/// @file speech_denoising.hpp
/// @brief Residual waveform enhancement with dilated convolutions.

#include <stdexcept>
#include <string>
#include <utility>

#include "datamunge/nn/wavenet_stack_models.hpp"

namespace datamunge::nn {

/// @brief Estimate noise and subtract it from a batch of mono clips.
///
/// Four causal residual blocks refine features from a width-seven stem.
/// The noise projection initializes to zero, so the initial estimate
/// reproduces the input. Fit the enhanced waveform against clean clips.
/// @param model Mutable parameter store initialized on first use.
/// @param noisy Floating waveform [batch, samples, 1] matching the store dtype.
/// @return Pair of enhanced waveform and estimated noise, both [batch, samples].
/// @throws std::invalid_argument For incompatible input shape, dtype or stored parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(4);
/// auto [clean, noise] = dm::nn::denoise_speech(model, dm::nn::ones({2,64,1}));
/// @endcode
/// @see models::WaveStackConv()
inline std::pair<Tensor,Tensor> denoise_speech(ParameterStore& model, const Tensor& noisy) {
  if (!noisy.defined() || noisy.dim()!=3 || noisy.size(0)<1 || noisy.size(1)<1 ||
      noisy.size(2)!=1 || noisy.dtype()!=model.dtype()) {
    throw std::invalid_argument("Invalid speech denoising waveform");
  }
  constexpr int width=16;
  auto state=conv1d(model,"stem",noisy,width,7);
  for (int dilation : {1,2,4,8}) {
    const auto update=models::WaveStackConv(model,"block"+std::to_string(dilation),
                                           relu(state),width,3,dilation);
    state=state+.25*update;
  }
  const auto noise=squeeze(dense(model,"noise",relu(state),1,
                                {.bias=false,.init=init::zeros()}),{2});
  return {squeeze(noisy,{2})-noise,noise};
}

}  // namespace datamunge::nn
