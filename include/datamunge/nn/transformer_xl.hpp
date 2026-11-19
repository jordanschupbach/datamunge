#pragma once

/// @file transformer_xl.hpp
/// @brief Causal segments with detached memory and content-dependent relative positions.

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Logits, detached layer inputs and per-layer attention for a segment.
/// @par Example
/// @code
/// const dm::nn::TransformerXlSegment empty{};
/// @endcode
struct TransformerXlSegment {
  Tensor logits; ///< Character scores [batch,time,65].
  std::vector<Tensor> memory; ///< Two detached [batch,time,16] layer input states.
  std::vector<Tensor> attention; ///< Two [batch,2,time,history+time] probability tensors.
};

/// @brief Decode one Transformer-XL character segment with optional previous layer inputs.
/// Cached states enter attention as keys and values. Causal masking covers
/// the current segment, and differentiation stops at the cached inputs.
/// @param parameters Trainable CPU parameter store.
/// @param encoded Floating CPU one-hot characters [batch,time,65], matching the store dtype.
/// @param history Optional pair of previous layer inputs [batch,history,16].
/// @return Logits, detached layer inputs for the next segment, and attention.
/// @throws std::invalid_argument For incompatible segment or history geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore model(7);
/// const auto input = n::one_hot(n::zeros({2,4}, n::DType::Int32),65);
/// const auto first = n::transformer_xl_segment(model,input);
/// const auto second = n::transformer_xl_segment(model,input,first.memory);
/// @endcode
/// @par References
/// - Dai et al. (2019), Transformer-XL, https://aclanthology.org/P19-1285/.
/// @see sinusoidal_encoding()
inline TransformerXlSegment transformer_xl_segment(ParameterStore& parameters, const Tensor& encoded,
                                                   const std::vector<Tensor>& history = {}) {
  if (!encoded.defined() || encoded.dim() != 3 || encoded.numel() == 0 ||
      encoded.size(2) != 65 || encoded.dtype() != parameters.dtype() ||
      !is_floating(encoded.dtype()) || encoded.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("transformer_xl_segment: expected CPU [batch,time,65] input");
  const auto batch = encoded.size(0);
  const auto time = encoded.size(1);
  std::int64_t past = 0;
  if (!history.empty()) {
    if (history.size()!=2 || !history.front().defined() || history.front().dim()!=3)
      throw std::invalid_argument("transformer_xl_segment: expected two history layers");
    past = history.front().size(1);
    for (const auto& layer : history)
      if (!layer.defined() || layer.shape()!=Shape{batch,past,16} ||
          layer.dtype()!=encoded.dtype() || layer.device().backend!=compute::Backend::CPU)
        throw std::invalid_argument("transformer_xl_segment: incompatible history");
  }
  const auto span = past + time;
  const auto distance = arange(past, span, 1, DType::Int64).reshape({time,1}) -
                        arange(0, span, 1, DType::Int64);
  const auto allowed = greater_equal(distance, 0);
  const auto relative_index = where(allowed, distance, zeros_like(distance));
  const auto position = sinusoidal_encoding(span,16,10000.,encoded.dtype());
  const auto heads = [&](const Tensor& x, std::int64_t length) {
    return x.reshape({batch,length,2,8}).transpose({0,2,1,3});
  };
  auto state = matmul(encoded, parameters.get("token",{65,16},init::normal(.1)));
  TransformerXlSegment result;
  for (int layer=0; layer<2; ++layer) {
    const auto name = "layer" + std::to_string(layer);
    const auto matrix = [&](const std::string& suffix) {
      return parameters.get(name + "/" + suffix,{16,16},init::normal(.15));
    };
    result.memory.push_back(state.detach());
    const auto context = past ? concat({history[layer].detach(),state},1) : state;
    const auto query = heads(matmul(state,matrix("query")),time);
    const auto key = heads(matmul(context,matrix("key")),span);
    const auto value = heads(matmul(context,matrix("value")),span);
    const auto u = parameters.get(name+"/u",{2,1,8},init::zeros());
    const auto v = parameters.get(name+"/v",{2,1,8},init::zeros());
    const auto projected = matmul(position,matrix("relative"));
    const auto relative = gather(projected,relative_index).reshape({time,span,2,8})
                              .transpose({2,0,1,3});
    const auto content = matmul(query+u,key,false,true);
    const auto displacement = sum(expand_dims(query+v,3)*relative,{-1});
    const auto scores = (content+displacement)/std::sqrt(8.);
    const auto weight = softmax(where(allowed,scores,full(scores.shape(),-1e9,encoded.dtype())),-1);
    result.attention.push_back(weight);
    const auto mixed = matmul(weight,value).transpose({0,2,1,3}).reshape({batch,time,16});
    state = layer_norm(parameters,name+"/ln1",state+matmul(mixed,matrix("out")));
    const auto expanded = dense(parameters,name+"/ff1",state,32,Activation::Relu);
    state = layer_norm(parameters,name+"/ln2",state+dense(parameters,name+"/ff2",expanded,16));
  }
  result.logits = dense(parameters,"head",state,65);
  return result;
}

} // namespace datamunge::nn
