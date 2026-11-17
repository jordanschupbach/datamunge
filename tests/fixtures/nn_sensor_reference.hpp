#pragma once
// Independent host metrics and standardized nearest/centroid oracle.
#include "nn_sequence_check_support.hpp"
namespace sensor_reference {
namespace nn = dm::nn;
using dm::nn::models::SensorConfig;
using dm::nn::models::SeriesData;
using nn::Tensor;
inline Tensor Npz(const nn::NamedTensors& arrays, const std::string& key) {
  for (const auto& [name, value] : arrays)
    if (name == key)
      return value;
  throw std::runtime_error("missing array " + key);
}
struct SensorSplits {
  std::vector<SeriesData> pools;
  SensorConfig            config;
  Tensor                  mean, stddev, weights;
  int                     time = 0;
};

inline SensorSplits LoadSensor(const std::filesystem::path& root) {
  const auto arrays = nn::load_npz((root / "signals.npz").string());
  const auto cfg    = Npz(arrays, "config");
  check::Require(
      cfg.dtype() == nn::DType::Int32 && cfg.shape() == nn::Shape{4}, "sensor shape metadata");
  const auto   c = cfg.to_vector<int>();
  SensorSplits result;
  result.time        = c[0];
  result.config      = {c[1], c[2], 8};
  const int channels = result.config.channels, classes = result.config.classes, train_count = c[3];
  check::Require(
      (result.time == 96 && channels == 1 && classes == 2 && train_count == 100) ||
          (result.time == 100 && channels == 6 && classes == 4 && train_count == 40),
      "pinned ECG200/BasicMotions shapes");
  std::set<int> prior;
  for (const std::string split : {"fit", "validation", "test"}) {
    SeriesData d{
        Npz(arrays, split + "_signals"),
        Npz(arrays, split + "_labels"),
        Npz(arrays, split + "_source_ids")};
    const std::int64_t n = d.signals.size(0);
    check::Require(
        n > 0 && d.signals.dtype() == nn::DType::Float32 &&
            d.signals.shape() == nn::Shape{n, result.time, channels},
        "observed multichannel sample matrix");
    for (const auto* t : {&d.labels, &d.source_ids})
      check::Require(
          t->dtype() == nn::DType::Int32 && t->shape() == nn::Shape{n},
          "aligned source/label vectors");
    std::vector<int> counts(classes);
    const auto       labels = d.labels.to_vector<int>(), ids = d.source_ids.to_vector<int>();
    for (std::int64_t i = 0; i < n; ++i) {
      check::Require(
          labels[i] >= 0 && labels[i] < classes && ids[i] >= 0 && ids[i] < 2 * train_count &&
              prior.insert(ids[i]).second,
          "unique/disjoint sensor source IDs and label range");
      check::Require(
          split == "test" ? ids[i] >= train_count : ids[i] < train_count, "official test boundary");
      ++counts[labels[i]];
    }
    for (float v : d.signals.to_vector<float>())
      check::Require(std::isfinite(v), "finite observed signal values");
    std::cout << " " << root.filename().string() << " " << split << ":" << n
              << " records,source-label-index counts=";
    for (int x : counts)
      std::cout << x << ',';
    std::cout << '\n';
    result.pools.push_back(d);
  }
  const auto&        fit = result.pools[0];
  const auto         x   = fit.signals.to_vector<float>();
  const std::int64_t N = fit.size(), T = result.time;
  std::vector<float> means(channels), stds(channels), weights(classes);
  std::vector<int>   counts(classes);
  for (int label : fit.labels.to_vector<int>())
    ++counts[label];
  for (int ch = 0; ch < channels; ++ch) {
    double mean = 0, variance = 0;
    for (std::int64_t n = 0; n < N; ++n)
      for (std::int64_t t = 0; t < T; ++t)
        mean += x[(n * T + t) * channels + ch];
    mean /= N * T;
    for (std::int64_t n = 0; n < N; ++n)
      for (std::int64_t t = 0; t < T; ++t)
        variance += std::pow(x[(n * T + t) * channels + ch] - mean, 2);
    variance /= N * T;
    check::Require(variance > 1e-12, "nonconstant fitting channel");
    means[ch] = static_cast<float>(mean);
    stds[ch]  = static_cast<float>(std::sqrt(variance));
  }
  for (int k = 0; k < classes; ++k) {
    check::Require(counts[k] > 0, "all fitting classes represented");
    weights[k] = static_cast<float>(double(N) / (classes * counts[k]));
  }
  result.mean        = nn::from_vector(means, {channels});
  result.stddev      = nn::from_vector(stds, {channels});
  result.weights     = nn::from_vector(weights, {classes});
  result.config.base = 8;
  return result;
}

struct SensorScore {
  double              accuracy = 0, balanced = 0, macro_f1 = 0, nll = 0, auc0 = 0;
  std::vector<int>    confusion;
  std::vector<double> recall;
};

/// P(score of a class-0 row > score of another row), ties counted half.
inline double SensorAuc0(const std::vector<double>& score, const Tensor& labels_tensor) {
  const auto labels = labels_tensor.to_vector<int>();
  double     total  = 0;
  int        pairs  = 0;
  for (std::size_t i = 0; i < labels.size(); ++i)
    if (labels[i] == 0)
      for (std::size_t j = 0; j < labels.size(); ++j)
        if (labels[j] != 0) {
          total += score[i] > score[j] ? 1 : score[i] == score[j] ? .5 : 0;
          ++pairs;
        }
  check::Require(pairs > 0, "AUC both source classes present");
  return total / pairs;
}

inline SensorScore ScoreSensor(const Tensor& logits, const Tensor& labels_tensor, int classes) {
  const auto count = static_cast<int>(labels_tensor.numel());
  check::Require(logits.shape() == nn::Shape{count, classes}, "classification score shape");
  const auto  z      = logits.to_vector<double>();
  const auto  labels = labels_tensor.to_vector<int>();
  SensorScore out;
  out.confusion.resize(classes * classes);
  out.recall.resize(classes);
  std::vector<double> margin;
  for (int n = 0; n < count; ++n) {
    const int label = labels[n];
    check::Require(label >= 0 && label < classes, "metric source class range");
    int    prediction = 0;
    double maximum = -INFINITY, sum = 0;
    for (int c = 0; c < classes; ++c) {
      const double v = z[n * classes + c];
      check::Require(std::isfinite(v), "finite sensor logit");
      maximum = std::max(maximum, v);
      if (v > z[n * classes + prediction])
        prediction = c;
    }
    for (int c = 0; c < classes; ++c)
      sum += std::exp(z[n * classes + c] - maximum);
    out.nll += maximum + std::log(sum) - z[n * classes + label];
    ++out.confusion[label * classes + prediction];
    out.accuracy += label == prediction;
    if (classes == 2)
      margin.push_back(z[n * 2] - z[n * 2 + 1]);
  }
  for (int c = 0; c < classes; ++c) {
    int truth = 0, predicted = 0;
    for (int j = 0; j < classes; ++j) {
      truth += out.confusion[c * classes + j];
      predicted += out.confusion[j * classes + c];
    }
    check::Require(truth > 0, "every test class has support");
    const double tp = out.confusion[c * classes + c];
    out.recall[c]   = tp / truth;
    out.balanced += out.recall[c];
    out.macro_f1 += truth + predicted ? 2 * tp / (truth + predicted) : 0;
  }
  out.accuracy /= count;
  out.nll /= count;
  out.balanced /= classes;
  out.macro_f1 /= classes;
  if (classes == 2)
    out.auc0 = SensorAuc0(margin, labels_tensor);
  return out;
}

inline void
PrintSensor(const std::string& name, const Tensor& logits, const SeriesData& d, int classes) {
  const auto score = ScoreSensor(logits, d.labels, classes);
  std::cout << " " << name << ":accuracy=" << 100 * score.accuracy
            << "%,balanced=" << 100 * score.balanced << "%,macroF1=" << score.macro_f1
            << ",CE=" << score.nll << ",per-source-class recall=";
  for (double recall : score.recall)
    std::cout << recall << ',';
  if (classes == 2)
    std::cout << "class0AUROC=" << score.auc0;
  std::cout << ",confusion=";
  for (int value : score.confusion)
    std::cout << value << ',';
  std::cout << '\n';
}

/// Standardized 1-NN one-hot pseudo-logits or negative standardized centroid MSE.
inline Tensor SensorReference(
    const SeriesData& fit,
    const SeriesData& test,
    const Tensor&     stddev,
    int               classes,
    bool              centroid) {
  const std::int64_t L = fit.signals.size(1), C = fit.signals.size(2), D = L * C;
  const auto         fx = fit.signals.to_vector<float>(), tx = test.signals.to_vector<float>(),
             sd                       = stddev.to_vector<float>();
  const auto                       fy = fit.labels.to_vector<int>();
  std::vector<std::vector<double>> centers(classes, std::vector<double>(D));
  std::vector<int>                 counts(classes);
  for (std::int64_t n = 0; n < fit.size(); ++n) {
    ++counts[fy[n]];
    for (std::int64_t i = 0; i < D; ++i)
      centers[fy[n]][i] += fx[n * D + i];
  }
  for (int k = 0; k < classes; ++k)
    for (double& x : centers[k])
      x /= counts[k];
  std::vector<float> out(static_cast<std::size_t>(test.size() * classes));
  for (std::int64_t n = 0; n < test.size(); ++n) {
    if (centroid) {
      for (int k = 0; k < classes; ++k) {
        double distance = 0;
        for (std::int64_t i = 0; i < D; ++i)
          distance += std::pow((tx[n * D + i] - centers[k][i]) / sd[i % C], 2);
        out[n * classes + k] = static_cast<float>(-distance / D);
      }
    } else {
      double best       = INFINITY;
      int    prediction = 0;
      for (std::int64_t m = 0; m < fit.size(); ++m) {
        double distance = 0;
        for (std::int64_t i = 0; i < D; ++i)
          distance += std::pow((tx[n * D + i] - fx[m * D + i]) / sd[i % C], 2);
        if (distance < best) {
          best       = distance;
          prediction = fy[m];
        }
      }
      for (int k = 0; k < classes; ++k)
        out[n * classes + k] = k == prediction ? 1.f : 0.f;
    }
  }
  return nn::from_vector(out, {test.size(), classes});
}

} // namespace sensor_reference
