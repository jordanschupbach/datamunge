#pragma once

/// @file langevin.hpp
/// @brief Euler transitions with explicit Gaussian innovations for posterior sampling.

#include <datamunge/nn/ops.hpp>
#include <cmath>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Advance stochastic-gradient Langevin dynamics by one step.
/// The drift uses one half of the negative-log-density gradient and the
/// innovation has standard deviation sqrt(step). The caller controls the
/// minibatch gradient estimate, step schedule and random number stream.
/// @param position Current floating tensor of positions.
/// @param potential_gradient Gradient of the negative log density, same shape and dtype.
/// @param innovation Caller-supplied standard Gaussian values, same shape and dtype.
/// @param step Finite positive step size.
/// @return Updated positions with differentiable arithmetic.
/// @throws std::invalid_argument Invalid shapes, dtypes or step size.
/// @code
/// namespace n = dm::nn;
/// auto next = n::stochastic_gradient_langevin_step(
///     n::zeros({2, 3}), n::zeros({2, 3}), n::ones({2, 3}), .01);
/// @endcode
inline Tensor stochastic_gradient_langevin_step(
    const Tensor& position, const Tensor& potential_gradient, const Tensor& innovation,
    double step) {
  if (!position.defined() || !potential_gradient.defined() || !innovation.defined() ||
      position.shape() != potential_gradient.shape() || position.shape() != innovation.shape() ||
      position.dtype() != potential_gradient.dtype() || position.dtype() != innovation.dtype() ||
      (position.dtype() != DType::Float32 && position.dtype() != DType::Float64) ||
      !std::isfinite(step) || step <= 0.) {
    throw std::invalid_argument("invalid Langevin position, gradient, innovation or step");
  }
  return position - (.5 * step) * potential_gradient + std::sqrt(step) * innovation;
}

}  // namespace datamunge::nn
