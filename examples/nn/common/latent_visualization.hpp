// Port of tflow's common/ml/latent_visualization.h: fit-only PCA projections and
// neighborhood retrieval diagnostics for frozen autoencoder representations.
#pragma once

#include "common/autoencoder.hpp"
#include "common/pca.hpp"

namespace ae {

/// [n, ...] float32/float64 values as float64 [n, features].
inline nn::Tensor DoubleMatrix(const nn::Tensor& input) {
  return nn::cast(nn::reshape(input, {input.size(0), input.numel() / input.size(0)}), nn::DType::Float64).contiguous();
}

struct LatentProjection {
  nn::Tensor mean, components, variances, fit, validation, test, fit_reconstruction, validation_reconstruction,
      test_reconstruction;
};

/// PCA centering/SVD on the fit rows only; frozen transforms of the held-out pools.
inline LatentProjection FitLatentPca(const nn::Tensor& fit, const nn::Tensor& validation, const nn::Tensor& test,
                                     int retained) {
  nn::NoGradGuard          guard;
  ml::PrincipalComponents pca(DoubleMatrix(fit), retained);
  auto                     vs = pca.Transform(DoubleMatrix(validation)), ts = pca.Transform(DoubleMatrix(test));
  auto c = [](const nn::Tensor& t) { return t.contiguous(); };
  return {c(pca.mean),  c(pca.components), c(pca.variances), c(pca.scores), c(vs), c(ts), c(pca.Reconstruct(pca.scores)),
          c(pca.Reconstruct(vs)), c(pca.Reconstruct(ts))};
}

inline nn::Tensor FirstColumns(const nn::Tensor& input, int dimensions) {
  check::Require(input.dtype() == nn::DType::Float64 && dimensions > 0 && dimensions <= input.size(1),
                 "positive retained columns");
  return nn::slice(input, {0, 0}, {input.size(0), dimensions}).contiguous();
}

/// Euclidean top-`neighbors` fitting rows per query (ties: smaller fitting row), int32 [nq, k].
inline nn::Tensor NearestFitRows(const nn::Tensor& fit_raw, const nn::Tensor& query_raw, int neighbors = 10) {
  const auto fit = DoubleMatrix(fit_raw), query = DoubleMatrix(query_raw);
  const auto nf = fit.size(0), nq = query.size(0), d = fit.size(1);
  check::Require(d == query.size(1) && neighbors > 0 && neighbors <= nf, "aligned Euclidean retrieval dimensions");
  const auto       f = fit.to_vector<double>(), q = query.to_vector<double>();
  std::vector<int> out(static_cast<std::size_t>(nq * neighbors));
  for (std::int64_t i = 0; i < nq; ++i) {
    std::vector<std::pair<double, int>> distance;
    for (std::int64_t r = 0; r < nf; ++r) {
      double sum = 0;
      for (std::int64_t j = 0; j < d; ++j) {
        const double delta = q[i * d + j] - f[r * d + j];
        sum += delta * delta;
      }
      distance.emplace_back(sum, int(r));
    }
    std::partial_sort(distance.begin(), distance.begin() + neighbors, distance.end());
    for (int k = 0; k < neighbors; ++k) out[i * neighbors + k] = distance[k].second;
  }
  return nn::from_vector(out, {nq, std::int64_t(neighbors)});
}

/// Fraction of reference neighbours also present in the candidate lists.
inline double NeighborOverlap(const nn::Tensor& reference, const nn::Tensor& candidate) {
  check::Require(reference.shape() == candidate.shape(), "same neighbor list dimensions");
  const auto r = reference.to_vector<int>(), c = candidate.to_vector<int>();
  const auto rows = reference.size(0), k = reference.size(1);
  int        hits = 0;
  for (std::int64_t i = 0; i < rows; ++i)
    for (std::int64_t a = 0; a < k; ++a)
      for (std::int64_t b = 0; b < k; ++b)
        if (r[i * k + a] == c[i * k + b]) {
          ++hits;
          break;
        }
  return double(hits) / double(reference.numel());
}

/// Posthoc 1-NN digit-label accuracy.
inline double NeighborLabelAccuracy(const nn::Tensor& neighbors, const nn::Tensor& fit_labels, const nn::Tensor& query_labels) {
  const auto n = neighbors.to_vector<int>(), f = fit_labels.contiguous().to_vector<int>(),
             q = query_labels.contiguous().to_vector<int>();
  const auto rows = neighbors.size(0), k = neighbors.size(1);
  int        hits = 0;
  for (std::int64_t i = 0; i < rows; ++i) hits += f[n[i * k]] == q[i];
  return double(hits) / double(rows);
}

}  // namespace ae
