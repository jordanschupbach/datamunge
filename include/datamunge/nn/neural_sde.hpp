#pragma once

/// @file neural_sde.hpp
/// @brief Scalar Euler transitions with learned drift and diffusion.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <numbers>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Conditional mean and standard deviation with shape [batch,1].
/// @par Example
/// @code
/// dm::nn::NeuralSdeTransition empty{};
/// @endcode
struct NeuralSdeTransition {
  Tensor mean; ///< Conditional Euler mean.
  Tensor sigma; ///< Conditional Euler standard deviation.
};

/// @brief Evaluate a scalar learned Itô SDE’s Gaussian Euler transition.
/// @param model Shared drift and diffusion model parameters.
/// @param state Floating CPU state column [batch,1] matching the model dtype.
/// @param dt Positive finite time increments of the same shape and dtype.
/// @param width Positive hidden width, default 16.
/// @return Conditional mean and standard deviation with live parameter gradients.
/// @throws std::invalid_argument For incompatible inputs or invalid time increments.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto path=dm::nn::neural_sde_transition(model,dm::nn::zeros({2,1}),dm::nn::full({2,1},.05));
/// @endcode
/// @see neural_sde_nll()
inline NeuralSdeTransition neural_sde_transition(ParameterStore& model,const Tensor& state,
                                                 const Tensor& dt,std::int64_t width=16) {
  if (!state.defined() || !dt.defined() || state.dim()!=2 || state.size(1)!=1 || !state.numel() ||
      dt.shape()!=state.shape() || state.dtype()!=model.dtype() || dt.dtype()!=state.dtype() ||
      state.device().backend!=compute::Backend::CPU || dt.device().backend!=compute::Backend::CPU ||
      width<1 || !(min(dt).item()>0 && std::isfinite(max(dt).item())))
    throw std::invalid_argument("neural_sde_transition: expected aligned states and positive steps");
  const auto h=dense(model,"hidden",state,width,Activation::Tanh);
  const auto drift=dense(model,"drift",h,1);
  const auto diffusion=softplus(dense(model,"diffusion",h,1))+.02;
  return {state+dt*drift,sqrt(dt)*diffusion};
}

/// @brief Evaluate mean Gaussian negative log likelihood of observed endpoints.
/// @param transition Aligned conditional means and positive standard deviations.
/// @param target Observed endpoints with the same nonempty shape.
/// @return Mean negative log likelihood including the Gaussian normalization constant.
/// @throws std::invalid_argument For undefined or incompatible tensors.
/// @par Example
/// @code
/// const auto x=dm::nn::zeros({2,1}), scale=dm::nn::ones({2,1});
/// const auto loss=dm::nn::neural_sde_nll({x,scale},x);
/// @endcode
/// @see neural_sde_transition()
inline Tensor neural_sde_nll(const NeuralSdeTransition& transition,const Tensor& target) {
  if (!target.defined() || !target.numel() || !transition.mean.defined() ||
      !transition.sigma.defined() || transition.mean.shape()!=target.shape() ||
      transition.sigma.shape()!=target.shape())
    throw std::invalid_argument("neural_sde_nll: expected aligned observations and transitions");
  return mean(.5*square((target-transition.mean)/transition.sigma)+
              log(transition.sigma)+.5*std::log(2*std::numbers::pi));
}

} // namespace datamunge::nn
