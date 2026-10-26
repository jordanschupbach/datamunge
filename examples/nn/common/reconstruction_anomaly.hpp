// Port of tflow's common/ml/reconstruction_anomaly.h: nominal-validation thresholds and
// tie-correct ranking metrics for reconstruction anomaly stress tests.
#pragma once

#include "common/autoencoder.hpp"

namespace ae {

/// Per-case mean squared residual over 784 pixels, float64 [n].
inline nn::Tensor ReconstructionScores(const nn::Tensor& x, const nn::Tensor& y) {
  check::Require(x.numel() == y.numel() && x.size(0) == y.size(0), "aligned percase reconstruction scores");
  const auto          a = x.contiguous().to_vector<float>(), b = y.contiguous().to_vector<float>();
  const auto          n = x.size(0);
  std::vector<double> out(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    double sum = 0;
    for (int j = 0; j < 784; ++j) {
      const double delta = double(a[i * 784 + j]) - double(b[i * 784 + j]);
      sum += delta * delta;
    }
    out[i] = sum / 784;
  }
  return nn::from_vector(out, {n});
}

/// Nearest-rank 95th percentile of nominal validation scores.
inline double NominalThreshold(const nn::Tensor& scores) {
  check::Require(scores.dtype() == nn::DType::Float64 && scores.numel() > 0, "nonempty nominal validation scores");
  auto values = scores.contiguous().to_vector<double>();
  for (auto v : values) check::Require(std::isfinite(v), "finite calibration scores");
  std::sort(values.begin(), values.end());
  return values[std::size_t(std::ceil(.95 * double(values.size()))) - 1];
}

struct ReconstructionRanking {
  double auroc, average_precision, normal_fpr, anomaly_tpr;
};

/// AUROC with half credit for ties, average precision with tied groups, strict score > threshold.
inline ReconstructionRanking RankAnomalies(const nn::Tensor& normal_scores, const nn::Tensor& anomaly_scores,
                                           double threshold) {
  const auto normal = normal_scores.contiguous().to_vector<double>(), anomaly = anomaly_scores.contiguous().to_vector<double>();
  const int  n = int(normal.size()), m = int(anomaly.size());
  check::Require(n > 0 && m > 0, "nonempty nominal/anomaly scoring pools");
  double                              wins = 0, fp = 0, tp = 0;
  std::vector<std::pair<double, int>> ranked;
  for (int i = 0; i < n; ++i) {
    const double a = normal[i];
    fp += a > threshold;
    ranked.emplace_back(a, 0);
    for (int j = 0; j < m; ++j) {
      const double b = anomaly[j];
      wins += b > a ? 1 : b == a ? .5 : 0;
    }
  }
  for (int j = 0; j < m; ++j) {
    tp += anomaly[j] > threshold;
    ranked.emplace_back(anomaly[j], 1);
  }
  std::sort(ranked.begin(), ranked.end(), [](auto a, auto b) { return a.first > b.first; });
  double ap        = 0;
  int    positives = 0, seen = 0;
  for (std::size_t i = 0; i < ranked.size();) {
    std::size_t end   = i;
    int         added = 0;
    while (end < ranked.size() && ranked[end].first == ranked[i].first) {
      added += ranked[end].second;
      ++end;
    }
    positives += added;
    seen += int(end - i);
    ap += double(added) / m * double(positives) / seen;
    i = end;
  }
  return {wins / (double(n) * m), ap, fp / n, tp / m};
}

/// Paired stress transformations: 0 white 8x8 top-left patch, 1 darkening x.35, 2 shift 6 right.
inline nn::Tensor AnomalyCorruption(const nn::Tensor& original, int kind) {
  check::Require(kind >= 0 && kind < 3, "patch/darkening/shift stress controls");
  const auto         x = original.contiguous().to_vector<float>();
  std::vector<float> out(x.size());
  for (std::int64_t b = 0; b < original.size(0); ++b)
    for (int y = 0; y < 28; ++y)
      for (int c = 0; c < 28; ++c) {
        const auto i     = b * 784 + y * 28 + c;
        float      value = x[i];
        if (kind == 0 && y < 8 && c < 8)
          value = 1;
        else if (kind == 1)
          value *= .35f;
        else if (kind == 2)
          value = c >= 6 ? x[b * 784 + y * 28 + c - 6] : 0;
        out[i] = value;
      }
  return nn::from_vector(out, original.shape());
}

}  // namespace ae
