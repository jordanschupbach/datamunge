/// @file
/// @brief Exact nearest-neighbor classification of frozen descriptors.
#pragma once

#include <datamunge/nn/retrieval_index.hpp>

#include <numeric>

namespace datamunge::nn {

/// @brief Own detached predictions and the neighbors used for each query.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto result = n::descriptor_knn(n::zeros({2, 1}), n::arange(0, 2, 1, n::DType::Int64),
///                                       n::zeros({1, 1}), 2, 2);
/// @endcode
/// @see descriptor_knn()
struct DescriptorKNNResult {
  Tensor predicted; ///< Lowest class ID with the most votes per query [Q], Int64.
  Tensor neighbors; ///< Gallery row positions in rank order [Q,k], Int64.
  Tensor distances; ///< Corresponding squared Euclidean distances [Q,k], Float64.
};

/// @brief Classify frozen descriptors by unweighted exact neighbor voting.
///
/// Distances use the original feature coordinates promoted to Float64. Equal
/// distances prefer lower gallery rows. Equal vote counts prefer lower class
/// IDs. This evaluation detaches all tensors and leaves its inputs unchanged.
///
/// @param gallery Finite floating training descriptors [N,D], with D positive.
/// @param labels Integral class IDs [N] in [0,classes).
/// @param queries Finite floating query descriptors [Q,D], with Q positive.
/// @param classes Positive number of classes.
/// @param k Number of neighbors in [1,N], default 5.
/// @return Predictions, neighbor indices and squared distances.
/// @throws std::invalid_argument If an input violates its shape, type or value requirements.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto result = n::descriptor_knn(n::zeros({2, 1}), n::arange(0, 2, 1, n::DType::Int64),
///                                       n::zeros({1, 1}), 2, 2);
/// @endcode
/// @see models::TensorDescriptorIndex
inline DescriptorKNNResult descriptor_knn(const Tensor &gallery, const Tensor &labels,
                                          const Tensor &queries, int classes, int k = 5) {
  NoGradGuard guard;
  if (gallery.dim() != 2 || labels.dim() != 1 || labels.shape()[0] != gallery.shape()[0] ||
      !is_integral(labels.dtype()) || queries.dim() != 2 || queries.shape()[0] < 1 ||
      classes < 1 || k < 1 || k > gallery.shape()[0] ||
      min(labels).item<std::int64_t>() < 0 || max(labels).item<std::int64_t>() >= classes)
    throw std::invalid_argument("descriptor_knn: invalid gallery, labels, queries or counts");

  std::vector<std::int64_t> ids(gallery.shape()[0]);
  std::iota(ids.begin(), ids.end(), 0);
  const models::TensorDescriptorIndex index(gallery, std::move(ids));
  std::vector<Tensor> rows, distances;
  rows.reserve(queries.shape()[0]);
  distances.reserve(queries.shape()[0]);
  for (std::int64_t query = 0; query < queries.shape()[0]; ++query) {
    const auto rank = index.Rank(queries, query);
    rows.push_back(slice_axis(rank.rows, 0, 0, k));
    distances.push_back(slice_axis(rank.distances, 0, 0, k));
  }
  const auto neighbors = stack(rows, 0);
  const auto votes = one_hot(gather(labels, neighbors.reshape({-1})), classes, 1, 0, DType::Int64)
                         .reshape({queries.shape()[0], k, classes});
  return {argmax(sum(votes, {1}), 1), neighbors, stack(distances, 0)};
}

}  // namespace datamunge::nn
