// Port of tflow's common/nn/generative_metrics.h and the GenerativeMetrics part
// of common/ml/generative_metrics.h: the exact low-rank Gaussian Frechet
// distance (FID formula on any feature matrices) and the probability-space
// Inception score, both in float64. tflow's FrozenMetricInception (running the
// original frozen ImageNet Inception GraphDef) has no datamunge counterpart:
// loading GraphDef files is out of scope, so callers supply features.
#pragma once

#include <cmath>
#include <map>
#include <string>

#include <datamunge/nn/nn.hpp>

#include "common/checks.hpp"

namespace gan {

namespace gm_nn = datamunge::nn;

/// Every Frechet field of tflow's GaussianFrechet (float64 scalars + the
/// singular values of the [n_a, n_b] cross matrix).
inline std::map<std::string, gm_nn::Tensor> Frechet(const gm_nn::Tensor& x, const gm_nn::Tensor& y) {
  namespace nn = gm_nn;
  check::Require(x.dim() == 2 && y.dim() == 2 && x.size(0) > 1 && y.size(0) > 1 && x.size(1) == y.size(1),
                 "FID requires at least two examples and equal feature width");
  const nn::Tensor a = nn::cast(x, nn::DType::Float64), b = nn::cast(y, nn::DType::Float64);
  for (const auto* t : {&a, &b})
    for (double v : t->contiguous().to_vector<double>()) check::Require(std::isfinite(v), "finite metric input");
  const nn::Tensor am = nn::mean(a, {0}), bm = nn::mean(b, {0});
  const nn::Tensor av = (a - am) / std::sqrt(double(a.size(0) - 1)), bv = (b - bm) / std::sqrt(double(b.size(0) - 1));
  const auto       svd = nn::svd(nn::matmul(av, bv, false, true), false);
  const nn::Tensor mt = nn::sum(nn::square(am - bm)), ta = nn::sum(nn::square(av)), tb = nn::sum(nn::square(bv)),
                   ct  = nn::sum(svd.s);
  const nn::Tensor raw = mt + (ta + tb) - 2.0 * ct;
  check::Require(raw.item() >= -1e-7, "only round-off negative FID may clamp");
  return {{"fid", nn::maximum(raw, nn::scalar_like(0.0, raw))},
          {"raw_fid", raw},
          {"mean_term", mt},
          {"trace_a", ta},
          {"trace_b", tb},
          {"cross_trace", ct},
          {"cross_singular_values", svd.s}};
}

/// exp(mean_i KL(p(y|x_i) || p(y))) from row-normalized class probabilities.
inline double InceptionScore(const gm_nn::Tensor& probability) {
  namespace nn = gm_nn;
  check::Require(probability.dim() == 2 && probability.size(0) > 0 && probability.size(1) > 1,
                 "nonempty class probability matrix");
  const nn::Tensor p    = nn::cast(probability, nn::DType::Float64);
  const auto       v    = p.contiguous().to_vector<double>();
  const auto       cols = p.size(1);
  for (std::int64_t i = 0; i < p.size(0); ++i) {
    double sum = 0;
    for (std::int64_t j = 0; j < cols; ++j) {
      const double x = v[i * cols + j];
      check::Require(std::isfinite(x), "finite metric input");
      check::Require(x >= 0 && x <= 1, "probability not logits");
      sum += x;
    }
    check::Near(sum, 1, 3e-6, "row-normalized probabilities");
  }
  const nn::Tensor marginal = nn::mean(p, {0});
  const nn::Tensor logp     = nn::log(nn::maximum(p, nn::scalar_like(1e-300, p)));
  const nn::Tensor logm     = nn::log(nn::maximum(marginal, nn::scalar_like(1e-300, marginal)));
  return std::exp(nn::mean(nn::sum(p * (logp - logm), {1})).item());
}

}  // namespace gan
