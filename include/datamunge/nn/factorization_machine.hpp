/// @file
/// @brief Differentiable second-order factorization-machine interactions.

#pragma once

#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Sum all distinct feature-pair interactions using shared low-rank factors.
///
/// The score is half the sum over latent coordinates of the squared projected
/// feature vector minus the projected squares. This excludes self-interactions
/// and costs O(rows * features * rank), without allocating the feature pairs.
/// Gradients propagate to both inputs through the tensor operations.
/// @param features Floating-point feature matrix [rows, features].
/// @param factors Floating-point factor matrix [features, rank] on the same
/// device and with the same dtype. Feature and rank extents must be positive.
/// @return Interaction scores [rows, 1], excluding any bias or linear term.
/// @throws std::invalid_argument Undefined inputs, invalid ranks or extents,
/// incompatible dtypes, or mismatched devices.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::from_vector<float>({2,3}, {1,2});
/// const auto v = n::from_vector<float>({4,5}, {2,1});
/// std::print("{:.0f}\n", n::factorized_interactions(x,v).item()); // 120
/// @endcode
/// @par References
/// - Rendle (2010), Factorization Machines. https://doi.org/10.1109/ICDM.2010.127
/// @see matmul(), square()
inline Tensor factorized_interactions(const Tensor& features, const Tensor& factors) {
  if (!features.defined() || !factors.defined() || features.dim() != 2 ||
      factors.dim() != 2 || features.size(1) <= 0 || factors.size(1) <= 0 ||
      features.size(1) != factors.size(0) || features.dtype() != factors.dtype() ||
      !is_floating(features.dtype())) {
    throw std::invalid_argument("factorized_interactions: incompatible feature and factor matrices");
  }
  return 0.5 * sum(square(matmul(features, factors)) -
                   matmul(square(features), square(factors)), {1}, true);
}

} // namespace datamunge::nn
