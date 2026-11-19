#pragma once

/// @file audio_visual_speech.hpp
/// @brief Paired audio and video classification.

#include <stdexcept>
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Classify aligned audio and video clips using separate dense encoders.
/// The two encodings are concatenated, then projected to syllable logits.
/// @param model Parameter store initialized on first use.
/// @param audio Floating [batch,features] acoustic input.
/// @param video Floating [batch,features] visual input in the same store dtype.
/// @param classes Positive number of syllables, default four.
/// @return Classification logits [batch,classes].
/// @throws std::invalid_argument For incompatible shapes, dtypes or stored parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto logits=dm::nn::audio_visual_logits(model,dm::nn::zeros({2,32}),dm::nn::zeros({2,512}));
/// @endcode
/// @see dense(), sparse_softmax_cross_entropy()
inline Tensor audio_visual_logits(ParameterStore& model,const Tensor& audio,const Tensor& video,
                                  int classes=4) {
  if (!audio.defined() || !video.defined() || audio.dim()!=2 || video.dim()!=2 ||
      audio.size(0)<1 || audio.size(1)<1 || video.size(1)<1 || audio.size(0)!=video.size(0) ||
      audio.dtype()!=model.dtype() || video.dtype()!=model.dtype() || classes<1)
    throw std::invalid_argument("Invalid aligned audio and video inputs");
  const auto acoustic=dense(model,"acoustic",audio,20,Activation::Tanh);
  const auto visual=dense(model,"visual",video,20,Activation::Tanh);
  const auto fused=dense(model,"fusion",concat({acoustic,visual},1),24,Activation::Tanh);
  return dense(model,"head",fused,classes);
}

}  // namespace datamunge::nn
