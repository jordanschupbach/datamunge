/// @file
/// @brief Sparse updates for explicit-rating matrix factorization.

#pragma once

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/linalg.hpp"

namespace datamunge::nn {

/// @brief One centered rating between a row and a column of a sparse matrix.
///
/// @par Example
/// @code
/// const dm::nn::ExplicitRating rating{0, 1, -0.5};
/// std::print("{:.1f}\n", rating.value); // -0.5
/// @endcode
/// @see explicit_als_rows(), explicit_sgd_epoch()
struct ExplicitRating {
  std::int64_t row; ///< Zero-based row index.
  std::int64_t column; ///< Zero-based column index.
  double value; ///< Observed rating with any baseline already subtracted.
};

/// @brief Fit one side of a sparse explicit-rating factorization by ridge regression.
///
/// Solves a regularized least-squares problem for each row using only
/// supplied observations. Repeated observations contribute separately.
/// A row with no observations returns zeros.
/// @param fixed CPU Float64 matrix [columns, rank] with positive extents and finite factors.
/// @param ratings Sparse observations, indexed into the output rows and fixed columns.
/// @param rows Positive number of output rows.
/// @param penalty Positive finite diagonal regularization.
/// @return New CPU Float64 factors [rows, rank].
/// @throws std::invalid_argument Bad shape, device, penalty or observation, or a failed solve.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fixed = n::from_vector<double>({2}, {1,1});
/// const std::array ratings{n::ExplicitRating{0,0,5}};
/// std::print("{:.1f}\n", n::explicit_als_rows(fixed, ratings, 1, 1).item()); // 2.0
/// @endcode
/// @par References
/// - Koren, Bell and Volinsky (2009), Matrix Factorization Techniques for Recommender Systems.
///   https://doi.org/10.1109/MC.2009.263
/// @see explicit_sgd_epoch(), solve()
inline Tensor explicit_als_rows(const Tensor& fixed, std::span<const ExplicitRating> ratings,
                                std::int64_t rows, double penalty) {
  if (!fixed.defined() || fixed.dim() != 2 || fixed.size(0) <= 0 || fixed.size(1) <= 0 ||
      fixed.dtype() != DType::Float64 || fixed.device().backend != compute::Backend::CPU ||
      rows <= 0 || !(penalty > 0 && std::isfinite(penalty))) {
    throw std::invalid_argument("explicit_als_rows: invalid factors, row count or penalty");
  }
  const auto rank = fixed.size(1);
  const auto data = fixed.to_vector<double>();
  auto matrices = zeros({rows,rank,rank}, DType::Float64);
  auto targets = zeros({rows,rank,1}, DType::Float64);
  auto* lhs = matrices.data_ptr<double>();
  auto* rhs = targets.data_ptr<double>();
  for (std::int64_t row = 0; row < rows; ++row) {
    for (std::int64_t d = 0; d < rank; ++d) lhs[(row*rank+d)*rank+d] = penalty;
  }
  for (const auto& rating : ratings) {
    if (rating.row < 0 || rating.row >= rows || rating.column < 0 ||
        rating.column >= fixed.size(0) || !std::isfinite(rating.value)) {
      throw std::invalid_argument("explicit_als_rows: invalid observation");
    }
    const auto* factor = data.data() + rating.column * rank;
    for (std::int64_t d = 0; d < rank; ++d) {
      rhs[rating.row*rank+d] += rating.value * factor[d];
      for (std::int64_t j = 0; j < rank; ++j) {
        lhs[(rating.row*rank+d)*rank+j] += factor[d] * factor[j];
      }
    }
  }
  return solve(matrices, targets).reshape({rows,rank});
}

/// @brief Apply one ordered pass of regularized SGD to sparse rating observations.
///
/// Each observation updates both factor rows using their pre-update values.
/// The penalty is applied per observation. Inputs are preserved, and the
/// returned factors can be passed to the next epoch.
/// @param rows CPU Float64 row factors [rows, rank], with positive extents and finite values.
/// @param columns CPU Float64 column factors [columns, rank], with positive extents and finite values.
/// @param ratings Sparse centered observations in update order.
/// @param rate Positive finite learning rate.
/// @param penalty Nonnegative finite L2 penalty.
/// @return New CPU Float64 row and column factors in that order.
/// @throws std::invalid_argument Invalid shapes, devices, observation, rate or penalty.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto initial = n::from_vector<double>({1}, {1,1});
/// const std::array ratings{n::ExplicitRating{0,0,3}};
/// const auto [rows, cols] = n::explicit_sgd_epoch(initial, initial, ratings, 0.25, 0);
/// std::print("{:.1f} {:.1f}\n", rows.item(), cols.item()); // 1.5 1.5
/// @endcode
/// @par References
/// - Koren, Bell and Volinsky (2009), Matrix Factorization Techniques for Recommender Systems.
///   https://doi.org/10.1109/MC.2009.263
/// @see explicit_als_rows()
inline std::pair<Tensor,Tensor> explicit_sgd_epoch(
    const Tensor& rows, const Tensor& columns, std::span<const ExplicitRating> ratings,
    double rate, double penalty) {
  for (const Tensor* input : {&rows, &columns}) {
    if (!input->defined() || input->dim() != 2 || input->size(0) <= 0 || input->size(1) <= 0 ||
        input->dtype() != DType::Float64 || input->device().backend != compute::Backend::CPU) {
      throw std::invalid_argument("explicit_sgd_epoch: invalid factor matrix");
    }
  }
  if (rows.size(1) != columns.size(1) || !(rate > 0 && std::isfinite(rate)) ||
      !(penalty >= 0 && std::isfinite(penalty))) {
    throw std::invalid_argument("explicit_sgd_epoch: invalid rank, rate or penalty");
  }
  const auto rank = rows.size(1);
  auto u = rows.to_vector<double>(), v = columns.to_vector<double>();
  for (const auto& rating : ratings) {
    if (rating.row < 0 || rating.row >= rows.size(0) || rating.column < 0 ||
        rating.column >= columns.size(0) || !std::isfinite(rating.value)) {
      throw std::invalid_argument("explicit_sgd_epoch: invalid observation");
    }
    auto* left = u.data() + rating.row * rank;
    auto* right = v.data() + rating.column * rank;
    double prediction = 0;
    for (std::int64_t d = 0; d < rank; ++d) prediction += left[d] * right[d];
    const double residual = rating.value - prediction;
    for (std::int64_t d = 0; d < rank; ++d) {
      const double a = left[d], b = right[d];
      left[d] += rate * (residual * b - penalty * a);
      right[d] += rate * (residual * a - penalty * b);
    }
  }
  return {from_vector(u, rows.shape()), from_vector(v, columns.shape())};
}

} // namespace datamunge::nn
