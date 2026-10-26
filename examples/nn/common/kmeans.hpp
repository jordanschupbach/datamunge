// datamunge::nn port of tflow_examples src/common/ml/kmeans.h.
// Lloyd k-means with nn distances, assignments and segmented means.
// Random initialization samples distinct row indices. k-means++ samples each
// next row proportional to squared distance from its nearest current center.
// If all distances are zero, choose an unused row uniformly. Empty clusters
// retain their old centers; this keeps the Lloyd objective nonincreasing even
// with repeated points or more centers than distinct observations.
// The host controls iterations and seeded sampling; numerical updates are ops.
#pragma once

#include <algorithm>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ml {
namespace nn = datamunge::nn;

/// ||x_i - y_j||^2 for x [N, D], y [K, D] -> [N, K] (tflow ml/kernels.h).
inline nn::Tensor SquaredDistances(const nn::Tensor& x, const nn::Tensor& y) {
  return nn::sum(nn::square(nn::expand_dims(x, 1) - nn::expand_dims(y, 0)), {2});
}

struct KMeansStep {
  nn::Tensor centers, labels, squared_distances, counts;
  double     inertia = 0;
};

class KMeans {
 public:
  explicit KMeans(int dimensions) : dimensions_(dimensions) {
    if (dimensions <= 0) throw std::invalid_argument("positive feature width required");
  }

  KMeansStep Step(const nn::Tensor& x, const nn::Tensor& centers) const {
    Validate(x);
    if (centers.dtype() != nn::DType::Float32 || centers.dim() != 2 || centers.size(1) != dimensions_ || centers.size(0) < 1)
      throw std::invalid_argument("centers must be nonempty float32 [K,D]");
    const std::int64_t k         = centers.size(0);
    const auto         distances = SquaredDistances(x, centers);
    KMeansStep         out;
    out.labels            = nn::argmin(distances, 1);
    out.squared_distances = nn::min(distances, {1});
    out.inertia           = nn::sum(out.squared_distances, {0}).item<float>();
    const auto sums       = nn::unsorted_segment_sum(x, out.labels, k);
    out.counts            = nn::unsorted_segment_sum(nn::ones_like(out.squared_distances), out.labels, k);
    const auto column     = nn::expand_dims(out.counts, 1);
    out.centers = nn::where(nn::greater(column, 0.0), sums / nn::maximum(column, nn::scalar_like(1.0, column)), centers);
    return out;
  }

  nn::Tensor Initialize(const nn::Tensor& x, int k, uint64_t seed, bool plus_plus) const {
    Validate(x);
    const int64_t n = x.size(0);
    if (k <= 0 || k > n) throw std::invalid_argument("k must be in [1, rows]");
    std::mt19937_64     rng(seed);
    std::vector<size_t> candidates(n);
    std::iota(candidates.begin(), candidates.end(), 0);
    std::shuffle(candidates.begin(), candidates.end(), rng);
    std::vector<size_t> selected;
    const auto          values = x.to_vector<float>();
    nn::Tensor          centers;
    for (int i = 0; i < k; ++i) {
      size_t next = candidates[i];
      if (plus_plus && i > 0) {
        const auto          step = Step(x, centers);
        const auto          d2   = step.squared_distances.to_vector<float>();
        std::vector<double> weights(d2.begin(), d2.end());
        for (size_t row : selected) weights[row] = 0;
        if (std::accumulate(weights.begin(), weights.end(), 0.) > 0)
          next = std::discrete_distribution<size_t>(weights.begin(), weights.end())(rng);
        else
          for (size_t row : candidates)
            if (std::find(selected.begin(), selected.end(), row) == selected.end()) {
              next = row;
              break;
            }
      }
      selected.push_back(next);
      std::vector<float> c;
      for (size_t row : selected) c.insert(c.end(), values.begin() + row * dimensions_, values.begin() + (row + 1) * dimensions_);
      centers = nn::from_vector(c, {i + 1, dimensions_});
    }
    return centers;
  }

 private:
  void Validate(const nn::Tensor& x) const {
    if (x.dtype() != nn::DType::Float32 || x.dim() != 2 || x.size(1) != dimensions_ || x.size(0) < 1)
      throw std::invalid_argument("data must be nonempty float32 [N,D]");
    if (!nn::all(nn::is_finite(x)).item<bool>()) throw std::invalid_argument("k-means requires finite observations");
  }
  int dimensions_;
};

}  // namespace ml
