// Three strided temporal residual stages for one- or multi-channel signals,
// and real ECG / multichannel smartwatch classification with fitting-only
// state (port of tflow's common/nn/ecg_sensor.h and common/ml/ecg_sensor.h).
#pragma once

#include "common/time_series.hpp"

namespace sensor {

namespace nn = datamunge::nn;
using nn::Tensor;
using sentence::Range;
using sentence::Rows;
using series::SeriesData;
using series::SeriesRows;

struct SensorConfig {
  int channels = 1, classes = 2, base = 64;
};
struct SensorOutput {
  Tensor              logits, normalized, features;
  std::vector<Tensor> stages;
};

/// signal [N, T, C]. State "normalization/mean|std" [C]; per stage s
/// "stage<s>/first" (k5, stride 2, ReLU), "stage<s>/second" (k3) and the
/// strided 1x1 "stage<s>/projection", all He-uniform SAME; GAP and "head".
inline SensorOutput SensorCnn(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& signal, const SensorConfig& config = {}) {
  if (config.channels < 1 || config.classes < 2 || config.base < 1) throw std::invalid_argument("ECG/sensor model dimensions");
  SensorOutput result;
  const Tensor mean = state.get("normalization/mean", {config.channels}, nn::init::zeros());
  const Tensor stddev = state.get("normalization/std", {config.channels}, nn::init::ones());
  result.normalized = nn::div(nn::sub(signal, mean.detach()), stddev.detach());
  Tensor x = result.normalized;
  for (int block = 0; block < 3; ++block) {
    const auto           name = "stage" + std::to_string(block + 1);
    const int            width = config.base * (1 << block);
    nn::ConvLayerOptions first;
    first.stride = 2;
    first.init = nn::init::he_uniform();
    first.activation = nn::Activation::Relu;
    Tensor               hidden = nn::conv1d(p, name + "/first", x, width, 5, first);
    nn::ConvLayerOptions second;
    second.init = nn::init::he_uniform();
    hidden = nn::conv1d(p, name + "/second", hidden, width, 3, second);
    nn::ConvLayerOptions projection;
    projection.stride = 2;
    projection.init = nn::init::he_uniform();
    const Tensor skip = nn::conv1d(p, name + "/projection", x, width, 1, projection);
    x = nn::relu(nn::add(hidden, skip));
    result.stages.push_back(x);
  }
  result.features = nn::mean(x, {1});
  result.logits = nn::dense(p, "head", result.features, config.classes);
  return result;
}

struct SensorSplits {
  std::vector<SeriesData> pools;
  SensorConfig            config;
  Tensor                  mean, stddev, weights;
  int                     time = 0;
};

inline SensorSplits LoadSensor(const std::filesystem::path& root) {
  const auto arrays = nn::load_npz((root / "signals.npz").string());
  const auto cfg = sentence::Npz(arrays, "config");
  check::Require(cfg.dtype() == nn::DType::Int32 && cfg.shape() == nn::Shape{4}, "sensor shape metadata");
  const auto   c = cfg.to_vector<int>();
  SensorSplits result;
  result.time = c[0];
  result.config = {c[1], c[2], 8};
  const int channels = result.config.channels, classes = result.config.classes, train_count = c[3];
  check::Require((result.time == 96 && channels == 1 && classes == 2 && train_count == 100) ||
                     (result.time == 100 && channels == 6 && classes == 4 && train_count == 40),
                 "pinned ECG200/BasicMotions shapes");
  std::set<int> prior;
  for (const std::string split : {"fit", "validation", "test"}) {
    SeriesData d{sentence::Npz(arrays, split + "_signals"), sentence::Npz(arrays, split + "_labels"), sentence::Npz(arrays, split + "_source_ids")};
    const std::int64_t n = d.signals.size(0);
    check::Require(n > 0 && d.signals.dtype() == nn::DType::Float32 && d.signals.shape() == nn::Shape{n, result.time, channels},
                   "observed multichannel sample matrix");
    for (const auto* t : {&d.labels, &d.source_ids})
      check::Require(t->dtype() == nn::DType::Int32 && t->shape() == nn::Shape{n}, "aligned source/label vectors");
    std::vector<int> counts(classes);
    const auto       labels = d.labels.to_vector<int>(), ids = d.source_ids.to_vector<int>();
    for (std::int64_t i = 0; i < n; ++i) {
      check::Require(labels[i] >= 0 && labels[i] < classes && ids[i] >= 0 && ids[i] < 2 * train_count && prior.insert(ids[i]).second,
                     "unique/disjoint sensor source IDs and label range");
      check::Require(split == "test" ? ids[i] >= train_count : ids[i] < train_count, "official test boundary");
      ++counts[labels[i]];
    }
    for (float v : d.signals.to_vector<float>()) check::Require(std::isfinite(v), "finite observed signal values");
    std::cout << " " << root.filename().string() << " " << split << ":" << n << " records,source-label-index counts=";
    for (int x : counts) std::cout << x << ',';
    std::cout << '\n';
    result.pools.push_back(d);
  }
  const auto&         fit = result.pools[0];
  const auto          x = fit.signals.to_vector<float>();
  const std::int64_t  N = fit.size(), T = result.time;
  std::vector<float>  means(channels), stds(channels), weights(classes);
  std::vector<int>    counts(classes);
  for (int label : fit.labels.to_vector<int>()) ++counts[label];
  for (int ch = 0; ch < channels; ++ch) {
    double mean = 0, variance = 0;
    for (std::int64_t n = 0; n < N; ++n)
      for (std::int64_t t = 0; t < T; ++t) mean += x[(n * T + t) * channels + ch];
    mean /= N * T;
    for (std::int64_t n = 0; n < N; ++n)
      for (std::int64_t t = 0; t < T; ++t) variance += std::pow(x[(n * T + t) * channels + ch] - mean, 2);
    variance /= N * T;
    check::Require(variance > 1e-12, "nonconstant fitting channel");
    means[ch] = static_cast<float>(mean);
    stds[ch] = static_cast<float>(std::sqrt(variance));
  }
  for (int k = 0; k < classes; ++k) {
    check::Require(counts[k] > 0, "all fitting classes represented");
    weights[k] = static_cast<float>(double(N) / (classes * counts[k]));
  }
  result.mean = nn::from_vector(means, {channels});
  result.stddev = nn::from_vector(stds, {channels});
  result.weights = nn::from_vector(weights, {classes});
  result.config.base = 8;
  return result;
}

/// -mean_n sum_c onehot * log_softmax * class_weight (denominator: raw row count).
inline Tensor SensorCE(const Tensor& logits, const Tensor& labels, const Tensor& weights, int classes) {
  const Tensor target = nn::one_hot(labels, classes, 1.0, 0.0, logits.dtype());
  return nn::neg(nn::mean(nn::sum(nn::mul(nn::mul(target, nn::log_softmax(logits)), weights), {1})));
}

class SensorModel {
 public:
  nn::ParameterStore             parameters, state{593};
  SensorConfig                   config;
  Tensor                         weights;
  std::unique_ptr<nn::Optimizer> optimizer;

  SensorModel(const SensorConfig& cfg, const Tensor& mean = {}, const Tensor& stddev = {}, const Tensor& class_weights = {},
              bool training = true, std::uint64_t seed = 587)
      : parameters(seed), config(cfg), weights(class_weights) {
    {
      nn::NoGradGuard guard;
      SensorCnn(parameters, state, nn::zeros({1, 8, cfg.channels}), cfg);
    }
    if (training) {
      check::Require(mean.defined() && mean.shape() == nn::Shape{cfg.channels} && stddev.shape() == mean.shape() &&
                         class_weights.shape() == nn::Shape{cfg.classes},
                     "fitting-only normalization/class weights");
      state.set("normalization/mean", mean);
      state.set("normalization/std", stddev);
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, replay::Adam());
    }
  }
  Tensor Logits(const SeriesData& d) { return SensorCnn(parameters, state, d.signals, config).logits; }
  /// Weighted CE without gradients.
  double Loss(const SeriesData& d) {
    nn::NoGradGuard guard;
    const double    value = SensorCE(Logits(d), d.labels, weights, config.classes).item<double>();
    check::Require(std::isfinite(value) && value >= 0, "finite weighted sensor objective");
    return value;
  }
  Tensor Predict(const SeriesData& d) {
    nn::NoGradGuard guard;
    return Logits(d).detach();
  }
  replay::Map Snapshot() const { return sentence::Snapshot({&parameters, &state}); }
  void        Save(const std::string& path) const { sentence::SaveStores(path, parameters, &state); }
  void        Restore(const std::string& path) { sentence::RestoreStores(path, parameters, &state); }
  void        Export(const std::filesystem::path& path) const { sentence::ExportArrays(path, parameters, &state); }
};

struct SensorScore {
  double              accuracy = 0, balanced = 0, macro_f1 = 0, nll = 0, auc0 = 0;
  std::vector<int>    confusion;
  std::vector<double> recall;
};

/// P(score of a class-0 row > score of another row), ties counted half.
inline double SensorAuc0(const std::vector<double>& score, const Tensor& labels_tensor) {
  const auto labels = labels_tensor.to_vector<int>();
  double     total = 0;
  int        pairs = 0;
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
  const auto          z = logits.to_vector<double>();
  const auto          labels = labels_tensor.to_vector<int>();
  SensorScore         out;
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
      if (v > z[n * classes + prediction]) prediction = c;
    }
    for (int c = 0; c < classes; ++c) sum += std::exp(z[n * classes + c] - maximum);
    out.nll += maximum + std::log(sum) - z[n * classes + label];
    ++out.confusion[label * classes + prediction];
    out.accuracy += label == prediction;
    if (classes == 2) margin.push_back(z[n * 2] - z[n * 2 + 1]);
  }
  for (int c = 0; c < classes; ++c) {
    int truth = 0, predicted = 0;
    for (int j = 0; j < classes; ++j) {
      truth += out.confusion[c * classes + j];
      predicted += out.confusion[j * classes + c];
    }
    check::Require(truth > 0, "every test class has support");
    const double tp = out.confusion[c * classes + c];
    out.recall[c] = tp / truth;
    out.balanced += out.recall[c];
    out.macro_f1 += truth + predicted ? 2 * tp / (truth + predicted) : 0;
  }
  out.accuracy /= count;
  out.nll /= count;
  out.balanced /= classes;
  out.macro_f1 /= classes;
  if (classes == 2) out.auc0 = SensorAuc0(margin, labels_tensor);
  return out;
}

inline void PrintSensor(const std::string& name, const Tensor& logits, const SeriesData& d, int classes) {
  const auto score = ScoreSensor(logits, d.labels, classes);
  std::cout << " " << name << ":accuracy=" << 100 * score.accuracy << "%,balanced=" << 100 * score.balanced << "%,macroF1=" << score.macro_f1
            << ",CE=" << score.nll << ",per-source-class recall=";
  for (double recall : score.recall) std::cout << recall << ',';
  if (classes == 2) std::cout << "class0AUROC=" << score.auc0;
  std::cout << ",confusion=";
  for (int value : score.confusion) std::cout << value << ',';
  std::cout << '\n';
}

/// Standardized 1-NN one-hot pseudo-logits or negative standardized centroid MSE.
inline Tensor SensorReference(const SeriesData& fit, const SeriesData& test, const Tensor& stddev, int classes, bool centroid) {
  const std::int64_t L = fit.signals.size(1), C = fit.signals.size(2), D = L * C;
  const auto         fx = fit.signals.to_vector<float>(), tx = test.signals.to_vector<float>(), sd = stddev.to_vector<float>();
  const auto         fy = fit.labels.to_vector<int>();
  std::vector<std::vector<double>> centers(classes, std::vector<double>(D));
  std::vector<int>                 counts(classes);
  for (std::int64_t n = 0; n < fit.size(); ++n) {
    ++counts[fy[n]];
    for (std::int64_t i = 0; i < D; ++i) centers[fy[n]][i] += fx[n * D + i];
  }
  for (int k = 0; k < classes; ++k)
    for (double& x : centers[k]) x /= counts[k];
  std::vector<float> out(static_cast<std::size_t>(test.size() * classes));
  for (std::int64_t n = 0; n < test.size(); ++n) {
    if (centroid) {
      for (int k = 0; k < classes; ++k) {
        double distance = 0;
        for (std::int64_t i = 0; i < D; ++i) distance += std::pow((tx[n * D + i] - centers[k][i]) / sd[i % C], 2);
        out[n * classes + k] = static_cast<float>(-distance / D);
      }
    } else {
      double best = INFINITY;
      int    prediction = 0;
      for (std::int64_t m = 0; m < fit.size(); ++m) {
        double distance = 0;
        for (std::int64_t i = 0; i < D; ++i) distance += std::pow((tx[n * D + i] - fx[m * D + i]) / sd[i % C], 2);
        if (distance < best) {
          best = distance;
          prediction = fy[m];
        }
      }
      for (int k = 0; k < classes; ++k) out[n * classes + k] = k == prediction ? 1.f : 0.f;
    }
  }
  return nn::from_vector(out, {test.size(), classes});
}

inline double MeanAbsChange(const Tensor& a, const Tensor& b) {
  return nn::mean(nn::abs(nn::sub(nn::cast(a, nn::DType::Float64), nn::cast(b, nn::DType::Float64)))).item<double>();
}

/// Cross-check against a tflow run directory (<tf>/<dataset>/{model_arrays,test_outputs}.npz):
/// our loaded data must equal TF's saved test/fit arrays bit for bit, and TF's
/// final weights loaded into our SensorCnn must reproduce TF's test logits.
inline void CompareWithTflow(const std::filesystem::path& root, const std::filesystem::path& tf_dir) {
  const auto  data = LoadSensor(root);
  const auto  saved = replay::ToMap(nn::load_npz((tf_dir / "test_outputs.npz").string()));
  const auto& fit = data.pools[0];
  const auto& test = data.pools[2];
  check::Require(replay::SameBits(test.signals, saved.at("signals")) && replay::SameBits(test.labels, saved.at("labels")) &&
                     replay::SameBits(test.source_ids, saved.at("source_ids")) && replay::SameBits(fit.signals, saved.at("fit_signals")) &&
                     replay::SameBits(fit.labels, saved.at("fit_labels")) && replay::SameBits(data.weights, saved.at("class_weights")),
                 "test/fit signals, labels, source IDs and class weights identical to tflow's");
  SensorModel model(data.config, {}, {}, {}, false);
  const auto  arrays = replay::ToMap(nn::load_npz((tf_dir / "model_arrays.npz").string()));
  for (std::size_t i = 0; i < model.parameters.size(); ++i) model.parameters.set(model.parameters.names()[i], arrays.at("p" + std::to_string(i)));
  for (std::size_t i = 0; i < model.state.size(); ++i) model.state.set(model.state.names()[i], arrays.at("s" + std::to_string(i)));
  const double norm_error = std::max(replay::MaxAbsDiff(model.state.at("normalization/mean"), data.mean),
                                     replay::MaxAbsDiff(model.state.at("normalization/std"), data.stddev));
  const Tensor logits = model.Predict(test);
  const double error = replay::MaxAbsDiff(logits, saved.at("logits"));
  const auto   ours = ScoreSensor(logits, test.labels, data.config.classes), theirs = ScoreSensor(saved.at("logits"), test.labels, data.config.classes);
  std::cout << " tflow cross-check " << root.filename().string() << ": data identical, fit normalization max diff=" << norm_error
            << ", TF final weights -> our logits max diff vs TF logits=" << error << ", accuracy ours/TF=" << ours.accuracy << "/" << theirs.accuracy
            << ", balanced=" << ours.balanced << "/" << theirs.balanced << '\n';
  check::Near(norm_error, 0, 1e-6, "fit-only normalization equals tflow's saved state");
  check::Near(error, 0, 2e-5, "TF's trained weights reproduce TF's test logits in our SensorCnn");
}

}  // namespace sensor
