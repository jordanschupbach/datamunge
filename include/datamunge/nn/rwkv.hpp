/// @file
/// @brief Compact RWKV-4 time and channel mixing.

#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Apply one causal RWKV-4 block from zero history.
///
/// Time mixing interpolates the current and previous normalized inputs
/// before projecting keys, values and receptance. The weighted key-value
/// recurrence tracks the exponent maximum so it can rescale its numerator
/// and denominator. Channel mixing applies a gated squared-ReLU expansion.
/// Both sublayers add a residual connection. Retaining gradients through
/// the full sequence supports training.
/// @param store Trainable parameters, matching the input dtype.
/// @param name Parameter prefix, with an empty string using root names.
/// @param input Nonempty CPU Float32 or Float64 sequence [batch,time,width].
/// @return Block output and ungated weighted key-value averages, both [batch,time,width].
/// @throws std::invalid_argument Invalid geometry, dtype, device or parameter shape.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore weights(7);
/// const auto [out, history] = n::rwkv4_block(weights,"rwkv",n::zeros({1,3,2}));
/// std::print("{} {}\n", out.size(1), history.size(2)); // 3 2
/// @endcode
/// @par References
/// - Peng et al. (2023), RWKV: Reinventing RNNs for the Transformer Era.
///   https://arxiv.org/abs/2305.13048
/// @see recurrent(), layer_norm()
inline std::pair<Tensor,Tensor> rwkv4_block(ParameterStore& store,const std::string& name,
                                          const Tensor& input) {
  if (!input.defined() || input.dim()!=3 || input.numel()==0 ||
      !is_floating(input.dtype()) || input.dtype()!=store.dtype() ||
      input.device().backend!=compute::Backend::CPU) {
    throw std::invalid_argument("rwkv4_block: expected CPU [batch,time,width] input matching store");
  }
  const auto batch=input.size(0), time=input.size(1), width=input.size(2);
  const auto prefix=name.empty() ? name : name+"/";
  const auto mix=[&](const Tensor& x,const std::string& key,std::int64_t units) {
    const auto zero=zeros({batch,1,width},input.dtype());
    const auto previous=time==1 ? zero : concat({zero,slice(x,{0,0,0},{batch,time-1,width})},1);
    const auto amount=sigmoid(store.get(prefix+key+"/mix",{width},init::constant(.4)));
    return dense(store,prefix+key,x*amount+previous*(1.0-amount),units);
  };
  const auto normalized=layer_norm(store,prefix+"time/norm",input);
  const auto keys=time_steps(mix(normalized,"time/key",width));
  const auto values=time_steps(mix(normalized,"time/value",width));
  const auto gate=sigmoid(mix(normalized,"time/receptance",width));
  const auto decay=exp(store.get(prefix+"time/log_decay",{width},init::constant(-1.5)));
  const auto bonus=store.get(prefix+"time/bonus",{width},init::constant(.2));
  auto numerator=values.front(), denominator=ones_like(numerator), scale=keys.front();
  std::vector<Tensor> trace{numerator};
  for (std::size_t t=1;t<keys.size();++t) {
    const auto current=keys[t]+bonus;
    const auto out_scale=maximum(scale,current);
    const auto old_weight=exp(scale-out_scale), new_weight=exp(current-out_scale);
    trace.push_back((old_weight*numerator+new_weight*values[t]) /
                    (old_weight*denominator+new_weight));
    const auto aged=scale-decay;
    scale=maximum(aged,keys[t]);
    const auto keep=exp(aged-scale), insert=exp(keys[t]-scale);
    numerator=keep*numerator+insert*values[t];
    denominator=keep*denominator+insert;
  }
  const auto wkv=stack_time(trace);
  const auto residual=input+dense(store,prefix+"time/output",gate*wkv,width);
  const auto channel_input=layer_norm(store,prefix+"channel/norm",residual);
  const auto expanded=square(relu(mix(channel_input,"channel/key",2*width)));
  const auto channel=dense(store,prefix+"channel/value",expanded,width);
  const auto channel_gate=sigmoid(mix(channel_input,"channel/receptance",width));
  return {residual+channel_gate*channel,wkv};
}

} // namespace datamunge::nn
