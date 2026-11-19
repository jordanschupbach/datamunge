#pragma once

/// @file lagrangian_network.hpp
/// @brief Structured one-coordinate Lagrangian and its Euler–Lagrange field.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Dynamics and differential terms, each shaped [batch,1].
/// @par Example
/// @code
/// dm::nn::LagrangianNetworkOutput empty{};
/// @endcode
struct LagrangianNetworkOutput {
  Tensor acceleration; ///< Euler–Lagrange acceleration.
  Tensor lagrangian; ///< Kinetic energy minus potential.
  Tensor mass; ///< Positive inertia.
  Tensor partial_q; ///< Derivative of the Lagrangian with respect to position.
  Tensor mixed; ///< Mixed position and velocity derivative.
  Tensor potential; ///< Learned potential relative to its value at zero.
};

/// @brief Evaluate a learned position-dependent inertia and potential.
/// A shared tanh network predicts log inertia and potential. Subtracting
/// the hidden state at zero anchors both functions before the output projection.
/// @param model Store for temporal-independent hidden and output weights.
/// @param state Floating CPU rows [position,velocity] matching the store dtype.
/// @param width Positive hidden width, default 24.
/// @return Euler–Lagrange terms with live parameter gradients.
/// @throws std::invalid_argument For incompatible state geometry or width.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto out=dm::nn::lagrangian_network(model,dm::nn::zeros({2,2}));
/// @endcode
/// @par References
/// - Cranmer et al., Lagrangian Neural Networks, https://arxiv.org/abs/2003.04630.
inline LagrangianNetworkOutput lagrangian_network(ParameterStore& model,const Tensor& state,
                                                    std::int64_t width=24) {
  if (!state.defined() || state.dim()!=2 || !state.numel() || state.size(1)!=2 ||
      state.dtype()!=model.dtype() || state.device().backend!=compute::Backend::CPU || width<1)
    throw std::invalid_argument("lagrangian_network: expected CPU [batch,2] states");
  const auto q=slice_axis(state,1,0,1), velocity=slice_axis(state,1,1,2);
  const auto w=model.get("w",{1,width},init::uniform(.8));
  const auto b=model.get("b",{width},init::uniform(.3));
  const auto head=model.get("head",{width,2},init::uniform(.2));
  const auto h=tanh(matmul(q,w)+b);
  const auto values=matmul(h-tanh(b),head);
  const auto derivatives=matmul(w*(1.-square(h)),head);
  const auto mass=exp(slice_axis(values,1,0,1));
  const auto potential=slice_axis(values,1,1,2);
  const auto mass_q=mass*slice_axis(derivatives,1,0,1);
  const auto kinetic=.5*mass*square(velocity);
  const auto mixed=mass_q*velocity;
  const auto partial_q=.5*mass_q*square(velocity)-slice_axis(derivatives,1,1,2);
  return {(partial_q-mixed*velocity)/mass,kinetic-potential,mass,partial_q,mixed,potential};
}

} // namespace datamunge::nn
