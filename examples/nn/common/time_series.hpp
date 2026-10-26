// Fully convolutional time-series classifier (8/5/3 temporal kernels, BN,
// ReLU, global average pooling) and real GunPoint training with fit-only
// saved normalization and frozen-BN deployment (port of tflow's
// common/nn/time_series.h and common/ml/time_series.h).
#pragma once

#include "common/sentence.hpp"

namespace series {

namespace nn = datamunge::nn;
using nn::Tensor;
using sentence::Range;
using sentence::Rows;

struct SeriesConfig {
  int   base = 128;
  float mean = 0, standard_deviation = 1;
};
struct SeriesOutput {
  Tensor              logits, normalized, features;
  std::vector<Tensor> stages;
};

/// How SeriesCnn treats batch normalization: frozen moving statistics,
/// batch statistics (optionally updating the moving averages), or
/// tflow's calibration ops (each layer's moving mean / variance set to the
/// population statistics of the input, layer by layer through the frozen network).
enum class BnMode { Infer, Train, TrainNoUpdate, Calibrate };

/// signal [N, T, 1]. Parameters "conv<i>/W|b", "bn<i>/gamma|beta", "head/W|b";
/// state "normalization/mean|std" (scalars) and "bn<i>/moving_mean|moving_var".
inline SeriesOutput SeriesCnn(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& signal, const SeriesConfig& config = {},
                              BnMode mode = BnMode::Infer) {
  if (config.base < 1 || !std::isfinite(config.mean) || !std::isfinite(config.standard_deviation) || config.standard_deviation <= 0)
    throw std::invalid_argument("time-series configuration");
  SeriesOutput out;
  const Tensor mean   = state.get("normalization/mean", {}, nn::init::constant(config.mean));
  const Tensor stddev = state.get("normalization/std", {}, nn::init::constant(config.standard_deviation));
  out.normalized      = nn::div(nn::sub(signal, mean.detach()), stddev.detach());
  Tensor    x         = out.normalized;
  const int kernels[] = {8, 5, 3};
  for (int layer = 0; layer < 3; ++layer) {
    const int         width = config.base * (layer == 1 ? 2 : 1);
    const std::string bn    = "bn" + std::to_string(layer + 1);
    x                       = nn::conv1d(p, "conv" + std::to_string(layer + 1), x, width, kernels[layer]);
    nn::BatchNormOptions options;
    options.momentum     = .99;
    options.eps          = 1e-3;
    options.update_stats = mode == BnMode::Train;
    if (mode == BnMode::Calibrate) {
      nn::batch_norm(p, state, bn, x, false, options);  // creates the tensors on first use
      nn::NoGradGuard guard;
      const Tensor    population_mean = nn::mean(x, {0, 1});
      const Tensor    population_variance = nn::mean(nn::square(nn::sub(x, population_mean)), {0, 1});
      state.set(bn + "/moving_mean", population_mean.detach());
      state.set(bn + "/moving_var", population_variance.detach());
    }
    x = nn::relu(nn::batch_norm(p, state, bn, x, mode == BnMode::Train || mode == BnMode::TrainNoUpdate, options));
    out.stages.push_back(x);
  }
  out.features = x;
  out.logits   = nn::dense(p, "head", nn::mean(x, {1}), 2);
  return out;
}

// ---- data ----------------------------------------------------------------------------------

struct SeriesData {
  Tensor signals, labels, source_ids;
  [[nodiscard]] std::int64_t size() const { return labels.size(0); }
};
struct SeriesSplits {
  std::vector<SeriesData> pools;
  SeriesConfig            config;
};

inline SeriesData SeriesRows(const SeriesData& d, const Rows& rows) {
  return {sentence::Select(d.signals, rows), sentence::Select(d.labels, rows), sentence::Select(d.source_ids, rows)};
}

inline SeriesSplits LoadSeries(const std::filesystem::path& root) {
  const auto    a = nn::load_npz((root / "time_series.npz").string());
  SeriesSplits  result;
  std::set<int> all;
  for (const std::string split : {"fit", "validation", "test"}) {
    SeriesData d{sentence::Npz(a, split + "_signals"), sentence::Npz(a, split + "_labels"), sentence::Npz(a, split + "_source_ids")};
    const std::int64_t n = d.signals.size(0);
    check::Require(n > 0 && d.signals.dtype() == nn::DType::Float32 && d.signals.shape() == nn::Shape{n, 150, 1},
                   "GunPoint raw univariate signal shape");
    for (const auto* t : {&d.labels, &d.source_ids})
      check::Require(t->dtype() == nn::DType::Int32 && t->shape() == nn::Shape{n}, "aligned series labels/source IDs");
    const auto labels = d.labels.to_vector<int>(), ids = d.source_ids.to_vector<int>();
    int        positive = 0;
    for (std::int64_t i = 0; i < n; ++i) {
      check::Require(labels[i] >= 0 && labels[i] < 2 && ids[i] >= 0 && ids[i] < 200 && all.insert(ids[i]).second,
                     "series source IDs unique/disjoint,label range");
      check::Require(split == "test" ? ids[i] >= 50 : ids[i] < 50, "official train/test source boundary");
      positive += labels[i];
    }
    for (float v : d.signals.to_vector<float>()) check::Require(std::isfinite(v), "finite observed time samples");
    std::cout << " GunPoint " << split << ":" << n << " signals,point=" << positive << std::endl;
    result.pools.push_back(d);
  }
  const auto x = result.pools[0].signals.to_vector<float>();
  double     mean = 0, variance = 0;
  for (float v : x) mean += v;
  mean /= x.size();
  for (float v : x) variance += std::pow(v - mean, 2);
  variance /= x.size();
  check::Require(variance > 1e-12, "nonconstant fitting signal population");
  result.config = {16, float(mean), float(std::sqrt(variance))};
  std::cout << " fitting-only global normalization mean=" << mean << ",std=" << std::sqrt(variance) << std::endl;
  return result;
}

// ---- model -------------------------------------------------------------------------------------

class SeriesModel {
 public:
  nn::ParameterStore             parameters{419}, state{421};
  SeriesConfig                   config;
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit SeriesModel(SeriesConfig c = {16, 0, 1}, bool training = true) : config(c) {
    {
      nn::NoGradGuard guard;
      SeriesCnn(parameters, state, nn::zeros({1, 150, 1}), c);
    }
    if (training) optimizer = std::make_unique<nn::Optimizer>(parameters, .001, replay::Adam());
  }
  SeriesOutput Forward(const SeriesData& d, BnMode mode) { return SeriesCnn(parameters, state, d.signals, config, mode); }
  /// Sequential fitting-only population BN calibration, without labels or gradients.
  void CalibrateFit(const SeriesData& fitting) {
    nn::NoGradGuard guard;
    Forward(fitting, BnMode::Calibrate);
  }
  /// Frozen-BN evaluation CE (no gradient).
  double Loss(const SeriesData& d) {
    nn::NoGradGuard guard;
    const double    value = sentence::SentenceCE(Forward(d, BnMode::Infer).logits, d.labels).item<double>();
    check::Require(std::isfinite(value) && value >= 0, "finite time-series CE");
    return value;
  }
  Tensor Predict(const SeriesData& d) {
    nn::NoGradGuard guard;
    return Forward(d, BnMode::Infer).logits.detach();
  }
  replay::Map Snapshot() const { return sentence::Snapshot({&parameters, &state}); }
  void        Save(const std::string& path) const { sentence::SaveStores(path, parameters, &state); }
  void        Restore(const std::string& path) { sentence::RestoreStores(path, parameters, &state); }
  void        Export(const std::filesystem::path& path) const { sentence::ExportArrays(path, parameters, &state); }
};

inline void SeriesMetrics(const std::string& name, const SeriesData& d, const Tensor& logits) {
  sentence::SentenceMetrics(name, d.labels, logits);
}

/// Euclidean 1-NN one-hot pseudo-logits or negative class-centroid MSE.
inline Tensor SeriesReference(const SeriesData& fit, const SeriesData& test, bool centroid) {
  const std::int64_t               T = 150;
  const auto                       fx = fit.signals.to_vector<float>(), tx = test.signals.to_vector<float>();
  const auto                       fy = fit.labels.to_vector<int>();
  std::vector<std::vector<double>> centers(2, std::vector<double>(T));
  int                              count[2] = {0, 0};
  for (std::int64_t n = 0; n < fit.size(); ++n) {
    ++count[fy[n]];
    for (int t = 0; t < T; ++t) centers[fy[n]][t] += fx[n * T + t];
  }
  for (int c = 0; c < 2; ++c) {
    check::Require(count[c] > 0, "both fitting classes for baselines");
    for (double& x : centers[c]) x /= count[c];
  }
  std::vector<float> out(static_cast<std::size_t>(2 * test.size()));
  for (std::int64_t n = 0; n < test.size(); ++n) {
    if (centroid) {
      for (int c = 0; c < 2; ++c) {
        double distance = 0;
        for (int t = 0; t < T; ++t) distance += std::pow(tx[n * T + t] - centers[c][t], 2);
        out[2 * n + c] = static_cast<float>(-distance / T);
      }
    } else {
      double best = INFINITY;
      int    prediction = 0;
      for (std::int64_t i = 0; i < fit.size(); ++i) {
        double distance = 0;
        for (int t = 0; t < T; ++t) distance += std::pow(tx[n * T + t] - fx[i * T + t], 2);
        if (distance < best) {
          best = distance;
          prediction = fy[i];
        }
      }
      out[2 * n] = prediction == 0 ? 1.f : 0.f;
      out[2 * n + 1] = prediction == 1 ? 1.f : 0.f;
    }
  }
  return nn::from_vector(out, {test.size(), 2});
}

/// x with axis 1 reversed for row 0 (the tests' time-reversal control).
inline SeriesData Reversed(const SeriesData& d) {
  SeriesData out = d;
  out.signals = nn::reverse(d.signals, {1}).contiguous();
  return out;
}

}  // namespace series
