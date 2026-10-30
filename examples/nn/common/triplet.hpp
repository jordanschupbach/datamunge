#pragma once
#include "common/checks.hpp"
#include "datamunge/nn/metric_learning.hpp"
namespace metric {
namespace nn = datamunge::nn;
using nn::models::MeanTriplet;
using nn::models::MineSemiHard;
using nn::models::TripletHinge;
using nn::models::TripletIndex;
using nn::models::TripletMining;
/// Host copy of an embedding matrix [B, D].
struct HostEmbeddings {
  std::vector<float> values;
  std::int64_t rows = 0, dims = 0;
  HostEmbeddings() = default;
  explicit HostEmbeddings(const nn::Tensor &t) {
    check::Require(t.dim() == 2, "embedding matrix rank");
    rows = t.size(0), dims = t.size(1);
    values = t.contiguous().to_vector<float>();
  }
  [[nodiscard]] float at(std::int64_t r, std::int64_t c) const {
    return values[std::size_t(r * dims + c)];
  }
};

inline double EmbeddingDistance(const HostEmbeddings &f, int a, int b) {
  return nn::models::MetricPairDistance(nn::from_vector(f.values, {f.rows, f.dims}), {a}, {b})
      .item();
}
inline TripletMining MineSemiHard(const HostEmbeddings &f, const std::vector<int> &labels,
                                  float margin = .2f) {
  return nn::models::MineSemiHard(nn::from_vector(f.values, {f.rows, f.dims}), labels, margin);
}
} // namespace metric
