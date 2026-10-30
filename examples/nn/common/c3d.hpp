// C3D (Tran et al. 2015): eight 3x3x3 convolutions, five Caffe-ceil max pools,
// two dense/dropout layers, and short real-video training on a KTH subset.
// A port of tflow's common/nn/c3d.h (model) and common/ml/c3d.h (data,
// training, scoring and controls).
//
// TF -> datamunge: the model is a function of an nn::ParameterStore instead
// of a graph; Conv3DLayer is the nn::conv3d layer; the Caffe "ceil" pool
// (PadV2 with -inf on the odd trailing edge, then a VALID 2x2x2 MaxPool3D) is
// nn::max_pool3d with SAME padding, which for window = stride = 2 pads only at
// the end and never lets padding win (c3d.cpp checks the two agree). Session
// checkpoints are ParameterStore .npz files, and the "deployed" model is a
// fresh store restored from that file.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <datamunge/nn/c3d_models.hpp>
#include <datamunge/nn/video_data.hpp>
#include <datamunge/nn/sensor_evaluation.hpp>
#include "common/checks.hpp"
#include "common/paths.hpp"

namespace c3d {

namespace nn = datamunge::nn;
using nn::Padding;
using nn::Tensor;

// ---- model (tflow common/nn/c3d.h) ---------------------------------------------

using datamunge::nn::models::C3dConfig;
using datamunge::nn::models::C3dOutput;
using datamunge::nn::models::C3dPool;
using datamunge::nn::models::C3d;
using datamunge::nn::models::VideoData;
using datamunge::nn::models::VideoRows;
using datamunge::nn::models::LoadVideo;
using datamunge::nn::models::VideoFeatures;
using datamunge::nn::models::VideoNearest;
using Score=datamunge::nn::models::SensorScore;
inline Score ScoreLogits(const Tensor&z,const Tensor&labels,int classes){return datamunge::nn::models::ScoreSensor(z,labels,classes);}
inline void PrintVideo(const std::string& name, const VideoData& d, const Tensor& logits) {
  const auto r = ScoreLogits(logits, d.labels, 3);
  std::cout << " " << name << ":accuracy=" << 100 * r.accuracy << "%,macroF1=" << r.macro_f1 << ",CE=" << r.nll
            << ",class recalls=";
  for (double x : r.recall) std::cout << x << ',';
  std::cout << ",confusion=";
  for (int x : r.confusion) std::cout << x << ',';
  std::cout << '\n';
}

// ---- training ------------------------------------------------------------------------

const C3dConfig kCompact{4, 64, 3, 32};

/// The compact model: parameters (seed 643) and dropout masks (seed 631).
struct C3dModel {
  nn::ParameterStore parameters{643};
  nn::Generator      dropout{631};
  C3dModel() {
    nn::NoGradGuard guard;  // create all 22 tensors
    C3d(parameters, nn::zeros({1, 16, 32, 32, 3}), kCompact);
  }
  Tensor Logits(const Tensor& rgb, bool training) {
    return C3d(parameters, rgb, kCompact, training, &dropout).logits;
  }
  /// Mean CE (tflow SensorCE with unit class weights).
  Tensor Loss(const VideoData& d, bool training) { return nn::sparse_softmax_cross_entropy(Logits(d.rgb, training), d.labels); }
  /// Inference logits [N, 3] in batches of 2.
  Tensor Predict(const VideoData& d) {
    nn::NoGradGuard     guard;
    std::vector<Tensor> parts;
    for (std::int64_t start = 0; start < d.size(); start += 2) {
      std::vector<std::int64_t> rows;
      for (std::int64_t n = start; n < std::min<std::int64_t>(start + 2, d.size()); ++n) rows.push_back(n);
      parts.push_back(Logits(nn::take_rows(d.rgb, rows), false));
    }
    return nn::concat(parts, 0);
  }
};

// ---- generic compact video-classifier harness ----------------------------------------
// tflow's ml/factorized_video.h and ml/i3d.h repeat TrainC3d around models with
// batch-norm moving statistics. A `Model` here provides
//   nn::ParameterStore parameters, state;   // learned tensors / moving statistics
//   Tensor Logits(const Tensor& rgb, bool training, bool update_stats);
// and creates every tensor in its constructor. Checkpoints are two .npz files
// (best.npz learned, best_state.npz moving statistics).

/// Inference logits [N, classes] in batches of 2 (moving statistics, no dropout).
template <class Model>
Tensor VideoModelPredict(Model& model, const VideoData& d) {
  nn::NoGradGuard     guard;
  std::vector<Tensor> parts;
  for (std::int64_t start = 0; start < d.size(); start += 2) {
    std::vector<std::int64_t> rows;
    for (std::int64_t n = start; n < std::min<std::int64_t>(start + 2, d.size()); ++n) rows.push_back(n);
    parts.push_back(model.Logits(nn::take_rows(d.rgb, rows), false, false));
  }
  return nn::concat(parts, 0);
}

/// Every learned and moving value, in creation order (for immutability checks).
template <class Model>
std::vector<double> VideoModelSnapshot(const Model& model) {
  auto values = model.parameters.flatten();
  const auto stats = model.state.flatten();
  values.insert(values.end(), stats.begin(), stats.end());
  return values;
}

struct VideoTrainSpec {
  std::string directory;                 // under output/nn/models
  std::string title;                     // e.g. "R(2+1)D-18 compact"
  std::string note;                      // closing caveat line
  bool        require_improvement{false};  // C3D requires fit CE to improve; the others only report it
  bool        tsv{false};                // also write model_arrays.tsv (name/kind per array)
};

/// tflow's shared short training loop: batch 2, Adam .001, validation-CE checkpoint
/// selection, fresh restore, determinism/single-batch/time-order controls, baselines, artifacts.

/// Loads a TF run's model_arrays.npz (p0.. = learned then moving tensors in creation
/// order; names checked against model_arrays.tsv when present) into `model` and
/// replays its saved test clips. Returns the largest |ours - TF| over the trained,
/// reversed-frame and repeated-first-frame logits of test_outputs.npz.
template <class Model>
double ReplayTfVideo(const std::filesystem::path& dir, Model& model) {
  std::map<std::string, Tensor> arrays, outputs;
  for (auto& [name, t] : nn::load_npz((dir / "model_arrays.npz").string())) arrays[name] = t;
  for (auto& [name, t] : nn::load_npz((dir / "test_outputs.npz").string())) outputs[name] = t;
  const std::size_t learned = model.parameters.size(), total = learned + model.state.size();
  check::Require(arrays.size() == total, "TF model_arrays tensor count equals ours");
  std::vector<std::string> tf_names;
  if (std::ifstream tsv(dir / "model_arrays.tsv"); tsv) {
    std::string key, name, kind;
    std::getline(tsv, key);
    while (tsv >> key >> name >> kind) tf_names.push_back(name);
    check::Require(tf_names.size() == total, "TF model_arrays.tsv inventory");
  }
  for (std::size_t i = 0; i < total; ++i) {
    const bool         is_learned = i < learned;
    nn::ParameterStore& store     = is_learned ? model.parameters : model.state;
    const std::string&  name      = store.names()[is_learned ? i : i - learned];
    if (!tf_names.empty()) check::Require(tf_names[i] == name, "TF/ours tensor name " + name);
    const Tensor& value = arrays.at("p" + std::to_string(i));
    check::Require(value.shape() == store.at(name).shape(), "TF/ours tensor shape " + name);
    store.set(name, value);
  }
  VideoData d;
  d.rgb = outputs.at("rgb");
  double worst = 0;
  const auto compare = [&](const Tensor& rgb, const std::string& key) {
    d.rgb            = rgb;
    const auto ours  = VideoModelPredict(model, d).template to_vector<float>();
    const auto theirs = outputs.at(key).template to_vector<float>();
    check::Require(ours.size() == theirs.size(), "replayed logit count");
    double error = 0;
    for (std::size_t i = 0; i < ours.size(); ++i) error = std::max(error, std::abs(double(ours[i]) - theirs[i]));
    std::cout << "  replay " << key << " max|ours-TF|=" << error << "\n";
    worst = std::max(worst, error);
  };
  const Tensor rgb = outputs.at("rgb");
  compare(rgb, "logits");
  compare(nn::reverse(rgb, {1}).contiguous(), "reverse");
  compare(nn::tile(nn::slice_axis(rgb, 1, 0, 1), {1, 16, 1, 1, 1}).contiguous(), "repeat_first");
  return worst;
}

}  // namespace c3d
