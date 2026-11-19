/// @file
/// @brief Regularized class prototypes for linear image scores.
#pragma once

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Solve a box-constrained prototype problem for each linear score.
///
/// Maximizes the score of the displacement from a uniform gray image,
/// minus half the penalty times the squared displacement norm. Each
/// coordinate has a known optimum before clipping, which can be used
/// to verify the iterative solution.
/// @param directions Signed Float32 or Float64 linear score rows [classes,pixels].
/// @param penalty Positive quadratic penalty.
/// @param steps Number of projected ascent steps from midgray.
/// @param rate Positive step size smaller than twice the inverse penalty.
/// @return Iterative prototypes in [0,1], shaped like directions.
/// @throws std::invalid_argument Invalid score geometry or optimization settings.
inline Tensor fit_linear_prototypes(const Tensor& directions, double penalty, int steps, double rate) {
  if (!directions.defined() || directions.dim() != 2 || directions.numel() == 0 ||
      (directions.dtype() != DType::Float32 && directions.dtype() != DType::Float64) ||
      !std::isfinite(penalty) || penalty <= 0 || steps < 0 ||
      !std::isfinite(rate) || rate <= 0 || rate * penalty >= 2) {
    throw std::invalid_argument("fit_linear_prototypes: invalid directions or optimizer settings");
  }
  auto images = full_like(directions, 0.5);
  for (int step = 0; step < steps; ++step) {
    images = clip(images + rate * (directions - penalty * (images - 0.5)), 0.0, 1.0);
  }
  return images;
}

} // namespace datamunge::nn
