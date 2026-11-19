/// @file
/// @brief Selective diagonal state-space scan.

#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Scan a diagonal state-space model with input-dependent steps and projections.
///
/// Starting from zero memory, each token decays the state by
/// exp(-step * exp(log_rates)), adds step times the channel input times
/// its write projection, and reads the result with its read projection.
/// The direct skip is applied per channel. Gradients propagate through
/// all operations. This version unrolls the recurrence on CPU.
/// @param signal CPU floating-point input [batch,time,channels], with positive extents.
/// @param step Positive step sizes, same geometry as signal.
/// @param write Write projection at each token [batch,time,state].
/// @param read Read projection at each token, same geometry as write.
/// @param log_rates Log positive decay rates [channels,state].
/// @param skip Direct per-channel term [channels].
/// @return Sequence [batch,time,channels] and final memory [batch,channels,state].
/// @throws std::invalid_argument Incompatible shapes, dtypes, devices, or undefined inputs.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::ones({1,1,1});
/// const auto [y,h] = n::selective_scan(x,x,x,x,n::zeros({1,1}),n::ones({1}));
/// std::print("{:.0f}\n", y.item()); // 2
/// @endcode
/// @par References
/// - Gu and Dao (2023), Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
///   https://arxiv.org/abs/2312.00752
/// @see time_steps(), stack_time()
inline std::pair<Tensor,Tensor> selective_scan(
    const Tensor& signal,const Tensor& step,const Tensor& write,const Tensor& read,
    const Tensor& log_rates,const Tensor& skip) {
  for (const auto* tensor : {&signal,&step,&write,&read,&log_rates,&skip}) {
    if (!tensor->defined() || tensor->dtype()!=signal.dtype() ||
        tensor->device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("selective_scan: expected matching CPU tensors");
    }
  }
  if (!is_floating(signal.dtype()) || signal.dim()!=3 || signal.numel()==0 ||
      step.shape()!=signal.shape() || write.dim()!=3 || write.numel()==0 ||
      write.size(0)!=signal.size(0) || write.size(1)!=signal.size(1) ||
      read.shape()!=write.shape() || log_rates.shape()!=Shape{signal.size(2),write.size(2)} ||
      skip.shape()!=Shape{signal.size(2)}) {
    throw std::invalid_argument("selective_scan: incompatible shapes");
  }
  const auto batch=signal.size(0), channels=signal.size(2), states=write.size(2);
  const auto x=time_steps(signal), delta=time_steps(step);
  const auto b=time_steps(write), c=time_steps(read);
  const auto rates=-exp(log_rates);
  auto memory=zeros({batch,channels,states},signal.dtype());
  std::vector<Tensor> output;
  output.reserve(x.size());
  for (std::size_t t=0;t<x.size();++t) {
    const auto dt=delta[t].reshape({batch,channels,1});
    const auto drive=dt*x[t].reshape({batch,channels,1})*b[t].reshape({batch,1,states});
    memory=exp(dt*rates)*memory+drive;
    output.push_back(sum(memory*c[t].reshape({batch,1,states}),{2})+skip*x[t]);
  }
  return {stack_time(output),memory};
}

} // namespace datamunge::nn
