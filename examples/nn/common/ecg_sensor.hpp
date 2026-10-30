#pragma once
#include "common/time_series.hpp"
#include <datamunge/nn/sensor_evaluation.hpp>
namespace sensor{namespace nn=datamunge::nn;using nn::Tensor;using sentence::Range;using sentence::Rows;using series::SeriesData;using series::SeriesRows;
using datamunge::nn::models::SensorConfig;
using datamunge::nn::models::SensorOutput;
using datamunge::nn::models::SensorCnn;
using datamunge::nn::models::SensorSplits;
using datamunge::nn::models::LoadSensor;
using datamunge::nn::models::SensorCE;
using datamunge::nn::models::SensorScore;
using datamunge::nn::models::SensorAuc0;
using datamunge::nn::models::ScoreSensor;
using datamunge::nn::models::SensorReference;
using datamunge::nn::models::MeanAbsChange;
class SensorModel {
 public:
  nn::ParameterStore             parameters;
  nn::ParameterStore             state{593};
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

}
