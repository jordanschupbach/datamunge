/// @file
/// @brief Case-weight derivatives of held-out loss for ridge regression.

#pragma once

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Weighted ridge fit and sensitivity of its held-out half squared loss.
///
/// @par Example
/// @code
/// const auto x = dm::nn::from_vector<double>({1, 1}, {2, 1});
/// const auto r = dm::nn::ridge_influence(x, dm::nn::from_vector<double>({0, 2}),
///     dm::nn::from_vector<double>({1, 1}), dm::nn::from_vector<double>({0}),
///     dm::nn::from_vector<double>({1}), 2.0);
/// std::print("{:.1f}\n", r.loss); // 0.5
/// @endcode
/// @see ridge_influence()
struct RidgeInfluenceResult {
  Tensor coefficients; ///< Fitted coefficients, shape [features].
  double loss; ///< Half squared prediction error on the held-out row.
  Tensor weight_derivatives; ///< Derivative of held-out loss per training weight, shape [rows].
};

/// @brief Fit weighted ridge regression and differentiate held-out loss with respect to row weights.
///
/// Minimizes half the sum of weighted squared training residuals plus half
/// the diagonal quadratic penalty. Weights are absolute, with no normalization.
/// Add a constant design column and use zero penalty for an unpenalized intercept.
/// The weighted normal matrix must be nonsingular. Computation is in Float64
/// on CPU. Returned tensors own their values and do not track gradients.
/// @param design Finite CPU design matrix [rows, features] with positive extents.
/// @param response Finite CPU training response [rows].
/// @param weights Finite nonnegative CPU training weights [rows].
/// @param penalty Finite nonnegative CPU diagonal penalty [features].
/// @param held_out Finite CPU held-out features [features].
/// @param target Finite held-out response.
/// @return Coefficients, held-out loss, and local weight derivatives at the supplied weights.
/// @throws std::invalid_argument Incorrect shapes, devices or values, or singular normal matrix.
///
/// @par Example
/// @code
/// const auto x = dm::nn::from_vector<double>({1, 1}, {2, 1});
/// const auto r = dm::nn::ridge_influence(x, dm::nn::from_vector<double>({0, 2}),
///     dm::nn::from_vector<double>({1, 1}), dm::nn::from_vector<double>({0}),
///     dm::nn::from_vector<double>({1}), 2.0);
/// std::print("{:.1f}\n", r.weight_derivatives.to_vector<double>()[0]); // 0.5
/// @endcode
/// @par References
/// - Koh and Liang (2017), Understanding Black-box Predictions via Influence Functions.
///   https://proceedings.mlr.press/v70/koh17a.html
/// @see RidgeInfluenceResult, solve()
inline RidgeInfluenceResult ridge_influence(
    const Tensor& design, const Tensor& response, const Tensor& weights,
    const Tensor& penalty, const Tensor& held_out, double target) {
  if (!design.defined() || !response.defined() || !weights.defined() ||
      !penalty.defined() || !held_out.defined() || design.shape().size() != 2 || design.size(0) == 0 || design.size(1) == 0 ||
      response.shape() != Shape{design.size(0)} || weights.shape() != response.shape() ||
      penalty.shape() != Shape{design.size(1)} || held_out.shape() != penalty.shape() ||
      !std::isfinite(target)) {
    throw std::invalid_argument("ridge_influence: incompatible shapes or nonfinite target");
  }
  const NoGradGuard no_grad;
  for (const Tensor* input : {&design, &response, &weights, &penalty, &held_out}) {
    if (input->device().backend != compute::Backend::CPU) {
      throw std::invalid_argument("ridge_influence: expected CPU tensors");
    }
    const auto values = input->to_vector<double>();
    for (const double value : values) {
      if (!std::isfinite(value) || ((input == &weights || input == &penalty) && value < 0)) {
        throw std::invalid_argument("ridge_influence: expected finite inputs and nonnegative weights and penalty");
      }
    }
  }
  const auto x = design.to(DType::Float64);
  const auto y = response.to(DType::Float64).reshape({-1, 1});
  const auto w = weights.to(DType::Float64).reshape({-1, 1});
  const auto test = held_out.to(DType::Float64).reshape({1, -1});
  const auto xt = x.transpose();
  const auto hessian = matmul(xt, w * x) + matrix_diag(penalty.to(DType::Float64));
  const auto coefficients = solve(hessian, matmul(xt, w * y));
  const double residual = matmul(test, coefficients).item() - target;
  const auto direction = solve(hessian, test.transpose() * residual);
  return {coefficients.reshape({-1}), 0.5 * residual * residual,
          (-(matmul(x, coefficients) - y) * matmul(x, direction)).reshape({-1})};
}

} // namespace datamunge::nn
