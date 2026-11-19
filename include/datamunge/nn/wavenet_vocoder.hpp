#pragma once

/// @file wavenet_vocoder.hpp
/// @brief Locally conditioned WaveNet prediction and synthesis.

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/audio.hpp"
#include "datamunge/nn/wavenet_stack_models.hpp"

namespace datamunge::nn {

/// @brief Score next mu-law codes with the compact conditioned WaveNet.
/// @param model Mutable parameter store initialized on first use.
/// @param history Previous-sample integer codes [batch, samples] in [0,255].
/// @param condition Aligned log-mel values [batch, samples, bands] in store dtype.
/// @return Causal logits [batch, samples, 256].
/// @throws std::invalid_argument For invalid tensor shapes, dtypes, codes or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(4);
/// auto scores = dm::nn::wavenet_vocoder_logits(model,
///     dm::nn::full({1,64},128,dm::nn::DType::Int32),dm::nn::zeros({1,64,8}));
/// @endcode
/// @par References
/// - van den Oord et al. (2016), WaveNet. https://arxiv.org/abs/1609.03499
/// @see wavenet_vocoder_generate()
inline Tensor wavenet_vocoder_logits(ParameterStore& model, const Tensor& history,
                                     const Tensor& condition) {
  if (!history.defined() || history.dim()!=2 || history.size(0)<1 || history.size(1)<1 ||
      (history.dtype()!=DType::Int32 && history.dtype()!=DType::Int64) ||
      min(history).item()<0 || max(history).item()>255 || !condition.defined()) {
    throw std::invalid_argument("Invalid WaveNet vocoder inputs");
  }
  return models::WaveStack(model,one_hot(history,256,1,0,model.dtype()),{8,12,3,1,256},condition*.2).logits;
}

/// @brief Synthesize mu-law codes greedily from aligned log-mel features.
/// Starts from the silence code 128 and feeds each argmax back as history.
/// Only the nine-sample receptive field needs to be evaluated each step.
/// @param model Mutable store holding the trained vocoder weights.
/// @param condition Aligned log-mel tensor [batch, samples, bands].
/// @return Integer codes [batch, samples] suitable for mu_law_decode().
/// @throws std::invalid_argument For invalid conditioning shape, dtype or parameters.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(4);
/// auto codes = dm::nn::wavenet_vocoder_generate(model,dm::nn::zeros({1,64,8}));
/// @endcode
/// @see wavenet_vocoder_logits(), mu_law_decode()
inline Tensor wavenet_vocoder_generate(ParameterStore& model, const Tensor& condition) {
  if (!condition.defined() || condition.dim()!=3 || condition.size(0)<1 ||
      condition.size(1)<1 || condition.size(2)<1 || condition.dtype()!=model.dtype()) {
    throw std::invalid_argument("Invalid WaveNet vocoder condition");
  }
  NoGradGuard guard;
  const auto batch=condition.size(0), samples=condition.size(1);
  const auto field=models::WaveStackField({8,12,3,1,256});
  std::vector<std::int32_t> output(batch*samples);
  for (std::int64_t t=0;t<samples;++t) {
    const auto begin=std::max<std::int64_t>(0,t+1-field), width=t+1-begin;
    std::vector<std::int32_t> history(batch*width);
    for (std::int64_t row=0;row<batch;++row)
      for (auto i=begin;i<=t;++i) history[row*width+i-begin]=i ? output[row*samples+i-1] : 128;
    const auto scores=wavenet_vocoder_logits(model,from_vector(history,{batch,width}),
                                             slice_axis(condition,1,begin,t+1));
    const auto next=argmax(select(scores,1,width-1)).to_vector<std::int32_t>();
    for (std::int64_t row=0;row<batch;++row) output[row*samples+t]=next[row];
  }
  return from_vector(output,{batch,samples});
}

}  // namespace datamunge::nn
