// Squared-L2 triplet hinge and explicit online strict semi-hard selection: a
// port of tflow's common/ml/triplet.h. Mining runs on a frozen (host) copy of
// the current batch embeddings; only the selected distances are differentiated.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <limits>
#include <vector>

#include "common/checks.hpp"

namespace metric {

namespace nn = datamunge::nn;

using TripletIndex = std::array<int, 3>;

/// Per-triplet max(0, |a-p|^2 - |a-n|^2 + margin) for embeddings [B, D].
inline nn::Tensor TripletHinge(const nn::Tensor& embeddings, const std::vector<TripletIndex>& indices, float margin = .2f) {
  std::vector<std::int64_t> a, p, n;
  for (const auto& t : indices) a.push_back(t[0]), p.push_back(t[1]), n.push_back(t[2]);
  if (indices.empty()) return nn::zeros({0}, embeddings.dtype());
  const nn::Tensor ea = nn::gather(embeddings, nn::from_vector(a)), ep = nn::gather(embeddings, nn::from_vector(p)),
                   en = nn::gather(embeddings, nn::from_vector(n));
  return nn::relu(nn::sum(nn::squared_difference(ea, ep), {1}) - nn::sum(nn::squared_difference(ea, en), {1}) +
                  double(margin));
}

/// Mean hinge (0 with zero gradients when there are no triplets).
inline nn::Tensor MeanTriplet(const nn::Tensor& embeddings, const std::vector<TripletIndex>& indices, float margin = .2f) {
  if (indices.empty()) return nn::sum(embeddings) * 0.0;
  return nn::sum(TripletHinge(embeddings, indices, margin)) / double(indices.size());
}

/// Host copy of an embedding matrix [B, D].
struct HostEmbeddings {
  std::vector<float> values;
  std::int64_t       rows = 0, dims = 0;
  HostEmbeddings() = default;
  explicit HostEmbeddings(const nn::Tensor& t) {
    check::Require(t.dim() == 2, "embedding matrix rank");
    rows = t.size(0), dims = t.size(1);
    values = t.contiguous().to_vector<float>();
  }
  [[nodiscard]] float at(std::int64_t r, std::int64_t c) const { return values[std::size_t(r * dims + c)]; }
};

inline double EmbeddingDistance(const HostEmbeddings& f, int a, int b) {
  check::Require(a >= 0 && b >= 0 && a < f.rows && b < f.rows, "embedding distance shape/index");
  double d = 0;
  for (std::int64_t c = 0; c < f.dims; ++c) {
    const double diff = double(f.at(a, c)) - f.at(b, c);
    check::Require(std::isfinite(diff), "finite embedding coordinate");
    d += diff * diff;
  }
  return d;
}

struct TripletMining {
  std::vector<TripletIndex> triplets;
  int                       positive_pairs = 0, without_semihard = 0;
};

/// Every ordered positive pair with its closest strictly semi-hard negative
/// (d(a,p) < d(a,n) < d(a,p) + margin); no fallback.
inline TripletMining MineSemiHard(const HostEmbeddings& features, const std::vector<int>& labels, float margin = .2f) {
  check::Require(features.rows == std::int64_t(labels.size()) && std::isfinite(margin) && margin > 0,
                 "valid semi-hard inputs");
  const int           count = int(labels.size());
  std::vector<double> distances(std::size_t(count) * count);
  for (int a = 0; a < count; ++a)
    for (int b = 0; b < count; ++b) distances[a * count + b] = EmbeddingDistance(features, a, b);
  TripletMining out;
  for (int a = 0; a < count; ++a)
    for (int p = 0; p < count; ++p)
      if (a != p && labels[a] == labels[p]) {
        ++out.positive_pairs;
        const double positive = distances[a * count + p];
        int          chosen   = -1;
        double       best     = INFINITY;
        for (int n = 0; n < count; ++n)
          if (labels[n] != labels[a]) {
            const double negative = distances[a * count + n];
            if (negative > positive && negative < positive + margin && negative < best) chosen = n, best = negative;
          }
        if (chosen >= 0)
          out.triplets.push_back({a, p, chosen});
        else
          ++out.without_semihard;
      }
  return out;
}
inline TripletMining MineSemiHard(const nn::Tensor& features, const std::vector<int>& labels, float margin = .2f) {
  return MineSemiHard(HostEmbeddings(features), labels, margin);
}

}  // namespace metric
