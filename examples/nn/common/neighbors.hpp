// datamunge::nn port of tflow_examples src/common/ml/neighbors.h.
// Brute-force k-nearest neighbors: pairwise distances, top_k on negative
// distances, indexed labels and weighted voting/regression.
// For equal distances top_k prefers the lower training index; equal class votes
// choose the lowest class ID. In inverse-distance mode, exact matches alone
// receive weight if any are present among the neighbors. Otherwise use 1/d.
// SquaredDistances uses direct float32 coordinate differences.
// Caller supplies finite [N,D]/[Q,D] tensors and labels in [0,classes).
// Indices and predicted classes are int64 (TF int32).
#pragma once

#include <stdexcept>

#include "common/kernels.hpp"

namespace ml {

struct NearestNeighbors {
  nn::Tensor indices, squared_distances, weights;
  NearestNeighbors(const nn::Tensor& train, const nn::Tensor& query, int k, bool inverse_distance) {
    if (k <= 0 || k > train.size(0)) throw std::invalid_argument("k must be in [1, training samples]");
    auto top          = nn::top_k(-SquaredDistances(query, train), k);
    indices           = top.second;
    squared_distances = -top.first;
    if (inverse_distance) {
      const auto equal     = nn::equal(squared_distances, 0.0);
      const auto any_equal = nn::any(equal, {1}, true);
      const auto inverse =
          nn::reciprocal(nn::sqrt(nn::maximum(squared_distances, nn::scalar_like(1e-20, squared_distances))));
      weights = nn::where(any_equal, nn::cast(equal, squared_distances.dtype()), inverse);
    } else {
      weights = nn::ones_like(squared_distances);
    }
    weights = weights / nn::sum(weights, {1}, true);
  }
  nn::Tensor Classify(const nn::Tensor& labels, int classes) const {
    if (classes <= 0) throw std::invalid_argument("positive class count required");
    const auto votes  = nn::one_hot(nn::gather(labels, indices), classes, 1.0, 0.0, weights.dtype());
    const auto scores = nn::sum(votes * nn::expand_dims(weights, -1), {1});
    return nn::argmax(scores, 1);
  }
  nn::Tensor Regress(const nn::Tensor& targets) const { return nn::sum(nn::gather(targets, indices) * weights, {1}); }
};

}  // namespace ml
