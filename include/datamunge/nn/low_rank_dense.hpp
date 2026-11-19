#pragma once

/// @file low_rank_dense.hpp
/// @brief Truncated SVD factors for dense-layer inference.

#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/ops.hpp>

#include <stdexcept>
#include <utility>

namespace datamunge::nn {

/// @brief Truncated dense weight factors and their approximation diagnostics.
/// @par Example
/// @code
/// const auto fit = dm::nn::factor_dense_weights(dm::nn::eye(3), 2);
/// std::println("{}", fit.discarded_energy); // 1
/// @endcode
/// @see factor_dense_weights()
struct LowRankDense {
  Tensor left; ///< Input-to-rank weights U[:, :r] scaled by singular values.
  Tensor right; ///< Rank-to-output weights V[:, :r]^T.
  Tensor singular_values; ///< All singular values in descending order.
  double discarded_energy{}; ///< Sum of squared omitted singular values.

  /// @brief Apply the low-rank operator as two matrix products.
  /// @param features CPU tensor with rank at least two and matching last dimension and dtype.
  /// @return Tensor with the final dimension replaced by the number of outputs.
  /// @throws std::invalid_argument Incompatible shape, dtype, or device.
  /// @par Example
  /// @code
  /// const auto fit = dm::nn::factor_dense_weights(dm::nn::eye(3), 3);
  /// std::println("{}", fit.predict(dm::nn::ones({2, 3})).values_str());
  /// @endcode
  [[nodiscard]] Tensor predict(const Tensor& features) const {
    if (!features.defined() || features.dtype() != left.dtype() ||
        features.device().backend != compute::Backend::CPU)
      throw std::invalid_argument("LowRankDense: expected CPU features matching the weight dtype");
    return matmul(matmul(features, left), right);
  }
};

/// @brief Fold singular values into the left factor of a rank-r dense weight approximation.
///
/// The thin SVD supplies the Frobenius-optimal rank-constrained
/// approximation. Parameters are detached copies for inference. Storage
/// decreases when rank * (inputs + outputs) < inputs * outputs.
/// @param weights Nonempty finite Float32 or Float64 CPU matrix [inputs, outputs].
/// @param rank Number of singular directions in [1, min(inputs, outputs)].
/// @return Factor matrices and discarded squared Frobenius energy.
/// @throws std::invalid_argument Unsupported dtype, device, shape, rank, or nonfinite weights.
/// @par Example
/// @code
/// const auto fit = dm::nn::factor_dense_weights(dm::nn::eye(3), 2);
/// std::println("{}", fit.left.size(1)); // 2
/// @endcode
/// @par References
/// - Eckart and Young (1936), The approximation of one matrix by another of lower rank.
///   https://doi.org/10.1007/BF02288367
/// @see svd(), LowRankDense
[[nodiscard]] inline LowRankDense factor_dense_weights(const Tensor& weights, std::int64_t rank) {
  if (!weights.defined() || weights.dim() != 2 || weights.device().backend != compute::Backend::CPU ||
      (weights.dtype() != DType::Float32 && weights.dtype() != DType::Float64) ||
      rank < 1 || rank > weights.size(0) || rank > weights.size(1) ||
      !all(is_finite(weights)).item<bool>())
    throw std::invalid_argument("factor_dense_weights: expected finite CPU weights and a feasible rank");

  const auto decomposition = svd(weights.detach());
  const auto values = decomposition.s.to_vector<double>();
  double discarded = 0.;
  for (std::size_t i = static_cast<std::size_t>(rank); i < values.size(); ++i)
    discarded += values[i] * values[i];
  auto left = slice(decomposition.u, {0, 0}, {weights.size(0), rank}) *
              slice(decomposition.s, {0}, {rank});
  auto right = matrix_transpose(slice(decomposition.v, {0, 0}, {weights.size(1), rank})).contiguous();
  return {std::move(left), std::move(right), decomposition.s, discarded};
}

} // namespace datamunge::nn
