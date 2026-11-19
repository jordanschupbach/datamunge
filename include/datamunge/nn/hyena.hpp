/// @file
/// @brief Two-stage Hyena sequence mixing with implicit causal filters.

#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Mix a sequence with two gated depthwise convolutions and implicit filters.
///
/// Project to two gates and a value branch, then apply a three-tap causal
/// convolution before splitting. A sinusoidal MLP generates two filters
/// from normalized lag and two Fourier bands. Learned positive decays
/// modulate their taps. Each stage multiplies by a gate, convolves over
/// the prefix, and adds a learned per-channel skip. Short sequences use
/// direct convolution. Apply residual connections and normalization outside
/// this mixer as needed.
/// @param store Parameters owned by the caller, with the same dtype as input.
/// @param name Parameter prefix. An empty string puts names at the store root.
/// @param input CPU Float32 or Float64 [batch, time, width], with at least two time steps.
/// @return Pair of mixed values [batch,time,width] and lag-ordered filters [time,2*width].
/// @throws std::invalid_argument Invalid input or incompatible parameter shape.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore weights(5);
/// const auto [out, taps] = n::hyena_mixer(weights, "mix", n::zeros({1,3,2}));
/// std::print("{} {}\n", out.size(2), taps.size(1)); // 2 4
/// @endcode
/// @par References
/// - Poli et al. (2023), Hyena Hierarchy. https://arxiv.org/abs/2302.10866
/// @see depthwise_conv2d(), dense()
inline std::pair<Tensor,Tensor> hyena_mixer(ParameterStore& store, const std::string& name,
                                          const Tensor& input) {
  if (!input.defined() || input.dim()!=3 || input.numel()==0 || input.size(1)<2 ||
      input.device().backend!=compute::Backend::CPU || !is_floating(input.dtype()) ||
      store.dtype()!=input.dtype()) {
    throw std::invalid_argument("hyena_mixer: expected CPU [batch,time,width] input matching the store");
  }
  const auto batch=input.size(0), time=input.size(1), width=input.size(2);
  const auto prefix=name.empty() ? name : name+"/";
  const auto convolve=[&](const Tensor& signal,const Tensor& taps) {
    const auto length=taps.size(0), channels=signal.size(2);
    Conv2DOptions options;
    options.padding=Padding::Explicit;
    options.explicit_paddings={length-1,0,0,0};
    return depthwise_conv2d(signal.reshape({batch,time,1,channels}),
        reverse(taps,{0}).reshape({length,1,channels,1}),options).reshape({batch,time,channels});
  };
  const auto projected=dense(store,prefix+"project",input,3*width);
  const auto short_kernel=store.get(prefix+"short/kernel",{3,3*width},init::uniform(.7));
  const auto short_bias=store.get(prefix+"short/bias",{3*width},init::zeros());
  const auto branches=split(convolve(projected,short_kernel)+short_bias,3,2);
  std::vector<double> lags, positions;
  for (std::int64_t i=0;i<time;++i) {
    const double lag=input.dtype()==DType::Float32
        ? static_cast<float>(i)/static_cast<float>(time-1)
        : static_cast<double>(i)/(time-1);
    lags.push_back(lag);
    positions.push_back(lag);
    for (int band=1;band<=2;++band) {
      const double phase=2*std::numbers::pi*band*lag;
      positions.push_back(std::cos(phase));
      positions.push_back(std::sin(phase));
    }
  }
  const auto features=from_vector(positions,{time,5}).to(input.dtype());
  const auto hidden=sin(dense(store,prefix+"filter/hidden",features,32));
  const auto decay=softplus(store.get(prefix+"filter/decay",{2*width},init::constant(.5)));
  const auto filters=dense(store,prefix+"filter/output",hidden,2*width) *
      exp(-from_vector(lags,{time,1}).to(input.dtype())*decay);
  const auto taps=split(filters,2,1);
  auto state=branches[2];
  for (int stage=0;stage<2;++stage) {
    const auto gated=state*branches[1-stage];
    const auto skip=store.get(prefix+"skip"+std::to_string(stage),{width},init::ones());
    state=convolve(gated,taps[stage])+gated*skip;
  }
  return {state,filters};
}

} // namespace datamunge::nn
