/// @file
/// @brief Normal-plus-low-rank HiPPO dynamics and scalar causal convolution.

#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Discretize a real S4 transition with a rank-one Woodbury correction.
///
/// Builds A = N - exp(log_strength) * p p^T where N is the fixed
/// normal part of HiPPO-LegS. Applies the bilinear transform at delta,
/// using a rank-one update to the inverse of I - delta*N/2.
/// @param input Floating CPU input projection B [state,1].
/// @param log_strength Scalar floating CPU log correction scale matching B's dtype.
/// @param delta Positive finite time step.
/// @return Continuous A, discrete A, and discrete B, differentiable in learned inputs.
/// @throws std::invalid_argument Invalid shapes, dtypes, devices, or time step.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto [a, ad, bd] = n::s4_discretize(n::ones({1,1}),n::from_scalar(0.0f),0.5);
/// std::print("{:.1f}\n", ad.item()); // 0.6
/// @endcode
/// @par References
/// - Gu, Goel and Ré (2022), Efficiently Modeling Long Sequences with Structured State Spaces.
///   https://arxiv.org/abs/2111.00396
/// @see s4_convolve(), solve()
inline std::array<Tensor,3> s4_discretize(const Tensor& input,const Tensor& log_strength,double delta) {
  if (!input.defined() || !log_strength.defined() || input.dim()!=2 || input.size(0)<=0 ||
      input.size(1)!=1 || log_strength.dim()!=0 || !is_floating(input.dtype()) ||
      log_strength.dtype()!=input.dtype() || !(delta>0 && std::isfinite(delta)) ||
      input.device().backend!=compute::Backend::CPU || log_strength.device().backend!=compute::Backend::CPU) {
    throw std::invalid_argument("s4_discretize: invalid inputs or step");
  }
  const auto dim=input.size(0);
  const auto type=input.dtype();
  const auto geometry=[&]<typename T>() {
    std::vector<T> p, normal;
    for (std::int64_t i=0;i<dim;++i) p.push_back(std::sqrt(T(i)+T(.5)));
    for (std::int64_t i=0;i<dim;++i) {
      for (std::int64_t j=0;j<dim;++j) {
        normal.push_back(i==j ? T(-.5) : (i>j ? -1 : 1)*p[i]*p[j]);
      }
    }
    return std::pair{from_vector(p,{dim,1}),from_vector(normal,{dim,dim})};
  };
  const auto [p,normal]=type==DType::Float32 ? geometry.template operator()<float>() :
      geometry.template operator()<double>();
  const auto identity=eye(dim,type);
  const auto scale=exp(log_strength);
  const auto a=normal-scale*matmul(p,p,false,true);
  const auto inverse=solve(identity-(.5*delta)*normal,identity);
  const auto left=matmul(inverse,p), right=matmul(p,inverse,true,false);
  const auto weight=(.5*delta)*scale;
  const auto updated=inverse-(weight/(1.0+weight*matmul(right,p)))*matmul(left,right);
  return {a,matmul(updated,identity+(.5*delta)*a),delta*matmul(updated,input)};
}

/// @brief Apply a zero-state scalar state-space filter as a causal convolution.
///
/// Derives the lag kernel C times powers of A times B up to the input
/// length. The direct skip D is added separately to each output position.
/// @param signal Nonempty CPU Float32 or Float64 signal [batch,time].
/// @param transition Discrete state matrix [state,state].
/// @param input Discrete input projection [state,1].
/// @param readout Output projection [1,state].
/// @param skip Scalar direct term. All parameters match signal's device and dtype.
/// @return Differentiable predictions [batch,time] and lag kernel [time].
/// @throws std::invalid_argument Undefined inputs or incompatible geometry or dtype.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto state = n::ones({1,1});
/// const auto [y,k] = n::s4_convolve(n::ones({1,2}),state,state,state,n::from_scalar(0.0f));
/// std::print("{:.0f}\n", y.to_vector<float>().back()); // 2
/// @endcode
/// @see s4_discretize(), conv2d()
inline std::pair<Tensor,Tensor> s4_convolve(const Tensor& signal,const Tensor& transition,
                                           const Tensor& input,const Tensor& readout,const Tensor& skip) {
  if (!signal.defined() || !transition.defined() || !input.defined() || !readout.defined() ||
      !skip.defined() || signal.dim()!=2 || signal.numel()==0 || !is_floating(signal.dtype()) ||
      transition.dim()!=2 || transition.size(0)<=0 || transition.size(0)!=transition.size(1) ||
      input.shape()!=Shape{transition.size(0),1} || readout.shape()!=Shape{1,transition.size(0)} ||
      skip.dim()!=0) throw std::invalid_argument("s4_convolve: invalid tensor geometry");
  for (const Tensor* tensor : {&signal,&transition,&input,&readout,&skip}) {
    if (tensor->dtype()!=signal.dtype() || tensor->device().backend!=compute::Backend::CPU)
      throw std::invalid_argument("s4_convolve: expected matching CPU floating tensors");
  }
  const auto batch=signal.size(0), time=signal.size(1);
  auto power=input;
  std::vector<Tensor> taps;
  for (std::int64_t t=0;t<time;++t) {
    taps.push_back(matmul(readout,power));
    if (t+1<time) power=matmul(transition,power);
  }
  const auto kernel=concat(taps,0).reshape({time});
  Conv2DOptions options;
  options.padding=Padding::Explicit;
  options.explicit_paddings={time-1,0,0,0};
  const auto filtered=conv2d(signal.reshape({batch,time,1,1}),
      reverse(kernel,{0}).reshape({time,1,1,1}),options);
  return {filtered.reshape({batch,time})+skip*signal,kernel};
}

} // namespace datamunge::nn
