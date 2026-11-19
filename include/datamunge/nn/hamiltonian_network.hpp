#pragma once

/// @file hamiltonian_network.hpp
/// @brief Canonical vector field from a scalar neural Hamiltonian.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Hamiltonian and corresponding canonical time derivatives.
/// @par Example
/// @code
/// dm::nn::HamiltonianNetworkOutput empty{};
/// @endcode
struct HamiltonianNetworkOutput {
  Tensor energy; ///< Scalar values [batch,1].
  Tensor derivative; ///< Canonical time derivatives [batch,2*dof].
};

/// @brief Evaluate a tanh Hamiltonian network and its analytic phase gradient.
/// @param model Parameters for the hidden affine layer and bias-free energy head.
/// @param state CPU matrix of ordered positions followed by momenta [batch,2*dof].
/// @param width Positive hidden width, default 32.
/// @return Energy and canonical time derivatives with live parameter gradients.
/// @throws std::invalid_argument For incompatible geometry, dtype or device.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto out=dm::nn::hamiltonian_network(model,dm::nn::zeros({2,2}));
/// @endcode
/// @par References
/// - Greydanus et al., Hamiltonian Neural Networks, https://arxiv.org/abs/1906.01563.
inline HamiltonianNetworkOutput hamiltonian_network(ParameterStore& model,const Tensor& state,
                                                     std::int64_t width=32) {
  if (!state.defined() || state.dim()!=2 || !state.numel() || state.size(1)%2 ||
      state.dtype()!=model.dtype() || state.device().backend!=compute::Backend::CPU || width<1)
    throw std::invalid_argument("hamiltonian_network: expected paired canonical coordinates");
  const auto dimension=state.size(1)/2;
  const auto hidden=dense(model,"hidden",state,width,Activation::Tanh);
  const auto head=model.get("energy/W",{width,1},init::uniform(.2));
  const auto derivative=matmul((1.-square(hidden))*head.transpose({1,0}),
                               model.at("hidden/W"),false,true);
  return {matmul(hidden,head),concat({slice_axis(derivative,1,dimension,2*dimension),
                                     -slice_axis(derivative,1,0,dimension)},1)};
}

} // namespace datamunge::nn
