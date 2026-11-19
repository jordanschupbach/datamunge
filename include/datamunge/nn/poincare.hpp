#pragma once

/// @file poincare.hpp
/// @brief Row-wise Poincaré distance and projected Riemannian SGD for embeddings.

#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Computes distances between corresponding points in the open unit ball.
/// Uses the log1p form of acosh(1 + t) where t is twice squared Euclidean
/// separation divided by the two conformal denominators. Coincident points
/// return zero with a chosen zero subgradient.
/// @param left Nonempty CPU Float32 or Float64 point matrix [rows, dimension].
/// @param right Matching point matrix with each row strictly inside the unit ball.
/// @return Distances [rows, 1] in the input dtype.
/// @throws std::invalid_argument Invalid matrices or points outside the open unit ball.
/// @par Example
/// @code
/// const auto origin = dm::nn::zeros({1, 2});
/// const auto d = dm::nn::poincare_distance(origin, dm::nn::tensor({.5, 0}, {1, 2}));
/// std::print("{:.5f}\n", d.item()); // 1.09861
/// @endcode
/// @par References
/// - Nickel and Kiela (2017), https://arxiv.org/abs/1705.08039.
/// @see poincare_sgd_step()
inline Tensor poincare_distance(const Tensor& left, const Tensor& right) {
  if (!left.defined() || !right.defined() || left.dim() != 2 ||
      left.shape() != right.shape() || left.size(0) == 0 || left.size(1) == 0 ||
      (left.dtype() != DType::Float32 && left.dtype() != DType::Float64) ||
      left.dtype() != right.dtype() || left.device().backend != compute::Backend::CPU ||
      right.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("poincare_distance: expected matching nonempty CPU floating matrices");
  const auto left_norm = sum(square(left), {1}, true);
  const auto right_norm = sum(square(right), {1}, true);
  if (!all(logical_and(less(left_norm, 1.), less(right_norm, 1.))).item<bool>())
    throw std::invalid_argument("poincare_distance: all points must lie in the open unit ball");
  const auto t = 2. * sum(square(left - right), {1}, true) /
                 ((1. - left_norm) * (1. - right_norm));
  const auto nonzero = greater(t, 0.);
  const auto safe = where(nonzero, t, 1.);
  const auto distance = log1p(safe + sqrt(safe * (safe + 2.)));
  return where(nonzero, distance, 0.);
}

/// @brief Takes a rescaled gradient step and projects each point to a ball of radius less than one.
/// @param points Nonempty CPU Float32 or Float64 point matrix [rows, dimension] inside the open unit ball.
/// @param euclidean_gradient Finite gradient of the same shape, dtype and device type.
/// @param rate Finite nonnegative step size.
/// @param radius Finite projection radius strictly between zero and one.
/// @return Updated point matrix in the input dtype.
/// @throws std::invalid_argument Invalid operands or settings.
/// @throws std::overflow_error If numerical overflow or rounding prevents a finite interior result.
/// @par Example
/// @code
/// const auto points = dm::nn::zeros({1, 1});
/// const auto next = dm::nn::poincare_sgd_step(points, dm::nn::ones({1, 1}), .8, .96);
/// std::print("{:.1f}\n", next.item()); // -0.2
/// @endcode
/// @par References
/// - Nickel and Kiela (2017), https://arxiv.org/abs/1705.08039.
/// @see poincare_distance()
inline Tensor poincare_sgd_step(const Tensor& points, const Tensor& euclidean_gradient,
                                double rate, double radius) {
  if (!points.defined() || !euclidean_gradient.defined() || points.dim() != 2 ||
      points.shape() != euclidean_gradient.shape() || points.size(0) == 0 || points.size(1) == 0 ||
      (points.dtype() != DType::Float32 && points.dtype() != DType::Float64) ||
      points.dtype() != euclidean_gradient.dtype() || points.device().backend != compute::Backend::CPU ||
      euclidean_gradient.device().backend != compute::Backend::CPU || !(rate >= 0) ||
      !std::isfinite(rate) || !(radius > 0 && radius < 1))
    throw std::invalid_argument("poincare_sgd_step: invalid matrices, rate or radius");
  const double finite_limit = points.dtype() == DType::Float32 ? std::numeric_limits<float>::max()
                                                              : std::numeric_limits<double>::max();
  const auto gradient_abs = abs(euclidean_gradient);
  const auto norm = sum(square(points), {1}, true);
  if (!all(less(norm, 1.)).item<bool>() ||
      !all(less_equal(gradient_abs, finite_limit)).item<bool>())
    throw std::invalid_argument("poincare_sgd_step: requires interior points and finite gradients");
  const auto raw = points - rate * (.25 * square(1. - norm) * euclidean_gradient);
  const auto length = sqrt(sum(square(raw), {1}, true));
  if (!all(less_equal(length, finite_limit)).item<bool>())
    throw std::overflow_error("poincare_sgd_step: step overflowed");
  const auto next = raw * (radius / maximum(full({}, radius, points.dtype()), length));
  if (!all(less(sum(square(next), {1}, true), 1.)).item<bool>())
    throw std::overflow_error("poincare_sgd_step: projected point is outside the open unit ball");
  return next;
}

} // namespace datamunge::nn
