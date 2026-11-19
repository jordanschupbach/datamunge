#pragma once

/// @file differentiable_kalman.hpp
/// @brief Batched differentiable scalar Kalman filtering and innovation likelihood.

#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Filtered states and innovation diagnostics from a scalar linear Gaussian model.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto result = n::differentiable_kalman(n::zeros({2, 3}), .85, n::zeros({}), n::zeros({}));
/// std::println("{}", result.filtered.size(1)); // 3
/// @endcode
/// @see differentiable_kalman()
struct DifferentiableKalmanResult {
  Tensor prior; ///< State means before each measurement [batch, time].
  Tensor filtered; ///< State means after each measurement [batch, time].
  Tensor variance; ///< Innovation variance shared across series [time].
  Tensor covariance; ///< Posterior state variance shared across series [time].
  Tensor objective; ///< Mean Gaussian negative log-likelihood per reading.
  Tensor process_variance; ///< Positive scalar process noise variance.
  Tensor measurement_variance; ///< Positive scalar measurement noise variance.
};

/// @brief Filter independent series and differentiate their innovation likelihood.
///
/// The state starts at zero with stationary prior variance. The observation
/// coefficient is one. Softplus plus 1e-5 maps both raw variances to positive
/// values. The scalar Joseph update preserves nonnegative posterior variance.
/// @param observed Float32 or Float64 CPU matrix [batch, time].
/// @param transition Fixed scalar state transition, finite and strictly between -1 and 1.
/// @param raw_q Same-dtype CPU scalar for unconstrained process variance.
/// @param raw_r Same-dtype CPU scalar for unconstrained measurement variance.
/// @return Filter history and scalar objective, connected to both raw variance tensors.
/// @throws std::invalid_argument Invalid dimensions, dtype, device, or transition.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto result = n::differentiable_kalman(n::zeros({2, 3}), .85, n::zeros({}), n::zeros({}));
/// std::println("{}", result.variance.size(0)); // 3
/// @endcode
/// @par References
/// - Kalman (1960), A New Approach to Linear Filtering and Prediction Problems, https://doi.org/10.1115/1.3662552
/// @see grad(), softplus()
[[nodiscard]] inline DifferentiableKalmanResult differentiable_kalman(
    const Tensor& observed, double transition, const Tensor& raw_q, const Tensor& raw_r) {
  /// @brief Check an unconstrained scalar variance parameter against the observation dtype.
  const auto scalar = [&](const Tensor& value) {
    return value.defined() && value.dim() == 0 && value.dtype() == observed.dtype() &&
           value.device().backend == compute::Backend::CPU;
  };
  if (!observed.defined() || observed.dim() != 2 || observed.size(0) < 1 || observed.size(1) < 1 ||
      observed.device().backend != compute::Backend::CPU ||
      (observed.dtype() != DType::Float32 && observed.dtype() != DType::Float64) ||
      !scalar(raw_q) || !scalar(raw_r) || !(std::abs(transition) < 1.0))
    throw std::invalid_argument("differentiable_kalman: invalid observations, variances, or transition");

  const auto q = softplus(raw_q) + 1e-5;
  const auto r = softplus(raw_r) + 1e-5;
  const auto squared = transition * transition;
  auto state = zeros({observed.size(0)}, observed.dtype());
  auto covariance = q / ((1.0 - transition) * (1.0 + transition));
  std::vector<Tensor> priors, filtered, variances, covariances, losses;
  for (const auto& reading : unstack(observed, 1)) {
    const auto prior = transition * state;
    const auto predicted_covariance = squared * covariance + q;
    const auto variance = predicted_covariance + r;
    const auto gain = predicted_covariance / variance;
    const auto innovation = reading - prior;
    state = prior + gain * innovation;
    covariance = square(1.0 - gain) * predicted_covariance + square(gain) * r;
    priors.push_back(prior);
    filtered.push_back(state);
    variances.push_back(variance);
    covariances.push_back(covariance);
    losses.push_back(.5 * (square(innovation) / variance + log(variance) +
                           std::log(2.0 * std::numbers::pi)));
  }
  return {stack(priors, 1), stack(filtered, 1), stack(variances, 0), stack(covariances, 0),
          mean(stack(losses, 1)), q, r};
}

}  // namespace datamunge::nn
