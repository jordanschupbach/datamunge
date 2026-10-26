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

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace c3d {

namespace nn = datamunge::nn;
using nn::Padding;
using nn::Tensor;

// ---- model (tflow common/nn/c3d.h) ---------------------------------------------

struct C3dConfig {
  int base = 64, dense = 4096, classes = 487, image = 112;
};

struct C3dOutput {
  Tensor              logits, flat, fc6, fc7;
  std::vector<Tensor> conv, pools;
};

/// Caffe-ceil max pooling of [N, T, H, W, C]: 2x2x2 windows (1x2x2 for the
/// first pool, which keeps time), odd trailing edges pooled over the real samples only.
inline Tensor C3dPool(const Tensor& x, bool first) {
  const std::int64_t temporal = first ? 1 : 2;
  return nn::max_pool3d(x, {temporal, 2, 2}, {temporal, 2, 2}, Padding::Same);
}

/// C3D on RGB clips in [0, 1] of shape [N, 16, image, image, 3]. With
/// `training`, dropout (rate 1 - keep) follows fc6 and fc7, drawn from `dropout_gen`.
inline C3dOutput C3d(nn::ParameterStore& p, const Tensor& rgb, const C3dConfig& cfg = {}, bool training = false,
                     nn::Generator* dropout_gen = nullptr, double keep = .5) {
  if (cfg.base < 1 || cfg.dense < 1 || cfg.classes < 2 || cfg.image < 16)
    throw std::invalid_argument("C3D configuration");
  C3dOutput   result;
  Tensor      x       = rgb * 2.0 - 1.0;
  const int   multiples[] = {1, 2, 4, 4, 8, 8, 8, 8};
  const char* names[]     = {"conv1a", "conv2a", "conv3a", "conv3b", "conv4a", "conv4b", "conv5a", "conv5b"};
  for (int layer = 0; layer < 8; ++layer) {
    nn::ConvLayerOptions opt;
    opt.init       = nn::init::he_uniform();
    opt.activation = nn::Activation::Relu;
    x              = nn::conv3d(p, names[layer], x, cfg.base * multiples[layer], 3, opt);
    result.conv.push_back(x);
    if (layer == 0 || layer == 1 || layer == 3 || layer == 5 || layer == 7) {
      x = C3dPool(x, layer == 0);
      result.pools.push_back(x);
    }
  }
  const auto& s = x.shape();
  result.flat   = nn::reshape(x, {s[0], s[1] * s[2] * s[3] * s[4]});
  result.fc6    = nn::dense(p, "fc6", result.flat, cfg.dense, nn::Activation::Relu);
  Tensor feature = result.fc6;
  if (training) feature = nn::dropout(feature, 1 - keep, true, *dropout_gen);
  result.fc7 = nn::dense(p, "fc7", feature, cfg.dense, nn::Activation::Relu);
  feature    = result.fc7;
  if (training) feature = nn::dropout(feature, 1 - keep, true, *dropout_gen);
  result.logits = nn::dense(p, "fc8", feature, cfg.classes);
  return result;
}

// ---- data (tflow common/ml/c3d.h) ------------------------------------------------

struct VideoData {
  Tensor rgb, labels, person, video, segment;
  [[nodiscard]] std::int64_t size() const { return rgb.size(0); }
};

inline VideoData VideoRows(const VideoData& d, const std::vector<std::int64_t>& rows) {
  return {nn::take_rows(d.rgb, rows), nn::take_rows(d.labels, rows), nn::take_rows(d.person, rows),
          nn::take_rows(d.video, rows), nn::take_rows(d.segment, rows)};
}

/// fit / validation / test pools of action_video.npz, with tflow's data checks:
/// 16 RGB frames in [0, 1] per clip, person- and video-disjoint splits, unique clips, real motion.
inline std::vector<VideoData> LoadVideo(const std::filesystem::path& root) {
  const auto file = root / "action_video.npz";
  check::Require(std::filesystem::exists(file), "missing " + file.string() +
                                                    "; run `just action-video-assets` in tflow_examples, or pass data=PATH");
  std::map<std::string, Tensor> arrays;
  for (auto& [name, t] : nn::load_npz(file.string())) arrays[name] = t;
  std::vector<VideoData>             result;
  std::set<int>                      prior_people, prior_videos;
  std::set<std::pair<int, int>>      prior_clips;
  for (const std::string split : {"fit", "validation", "test"}) {
    VideoData d{arrays.at(split + "_rgb"), arrays.at(split + "_label"), arrays.at(split + "_person"),
                arrays.at(split + "_video_id"), arrays.at(split + "_segment")};
    const std::int64_t n = d.size();
    check::Require(d.rgb.dtype() == nn::DType::Float32 && d.rgb.shape() == nn::Shape({n, 16, 32, 32, 3}),
                   "sixteen actual RGB video frames/clip");
    for (const Tensor* t : {&d.labels, &d.person, &d.video, &d.segment})
      check::Require(t->dtype() == nn::DType::Int32 && t->shape() == nn::Shape({n}), "video annotation alignment");
    const auto labels = d.labels.to_vector<int>(), person = d.person.to_vector<int>(), video = d.video.to_vector<int>(),
               segment = d.segment.to_vector<int>();
    const auto      rgb = d.rgb.to_vector<float>();
    std::set<int>   people, videos;
    std::vector<int> counts(3);
    const std::size_t frame = 32 * 32 * 3, clip = 16 * frame;
    for (std::int64_t i = 0; i < n; ++i) {
      const auto k = static_cast<std::size_t>(i);
      check::Require(labels[k] >= 0 && labels[k] < 3 && video[k] >= 0 && video[k] < 18 && segment[k] >= 0 &&
                         segment[k] < 2 && !prior_people.count(person[k]) && !prior_videos.count(video[k]) &&
                         prior_clips.insert({video[k], segment[k]}).second,
                     "person/source-video heldout pools and unique annotated clips");
      people.insert(person[k]);
      videos.insert(video[k]);
      ++counts[static_cast<std::size_t>(labels[k])];
      double motion = 0;
      for (std::size_t j = 0; j < clip; ++j) {
        const double value = rgb[k * clip + j];
        check::Require(std::isfinite(value) && value >= 0 && value <= 1, "finite observed video pixels");
        if (j >= frame) motion += std::abs(value - rgb[k * clip + j - frame]);
      }
      check::Require(motion > 1e-6, "real training/evaluation clips contain temporal change");
    }
    prior_people.insert(people.begin(), people.end());
    prior_videos.insert(videos.begin(), videos.end());
    check::Require(n == (split == "fit" ? 24 : 6), "pinned small video pool counts");
    std::cout << " KTH " << split << ":" << n << " clips/" << videos.size() << " videos/" << people.size()
              << " people,class counts=";
    for (int x : counts) std::cout << x << ',';
    std::cout << '\n';
    result.push_back(d);
  }
  return result;
}

// ---- scoring ------------------------------------------------------------------------

struct Score {
  double              accuracy = 0, macro_f1 = 0, nll = 0;
  std::vector<int>    confusion;
  std::vector<double> recall;
};

/// tflow's ScoreSensor: accuracy, macro F1, mean CE, recalls and confusion of [N, classes] logits.
inline Score ScoreLogits(const Tensor& logits, const Tensor& labels, int classes) {
  const std::int64_t count = labels.numel();
  check::Require(logits.shape() == nn::Shape({count, classes}), "classification score shape");
  const auto z = logits.to_vector<float>();
  const auto y = labels.to_vector<int>();
  Score      out;
  out.confusion.resize(static_cast<std::size_t>(classes * classes));
  out.recall.resize(static_cast<std::size_t>(classes));
  for (std::int64_t n = 0; n < count; ++n) {
    const int label = y[static_cast<std::size_t>(n)];
    check::Require(label >= 0 && label < classes, "metric source class range");
    const float* row = z.data() + n * classes;
    int          prediction = 0;
    double       maximum = -INFINITY, sum = 0;
    for (int c = 0; c < classes; ++c) {
      check::Require(std::isfinite(row[c]), "finite logit");
      maximum = std::max<double>(maximum, row[c]);
      if (row[c] > row[prediction]) prediction = c;
    }
    for (int c = 0; c < classes; ++c) sum += std::exp(row[c] - maximum);
    out.nll += maximum + std::log(sum) - row[label];
    ++out.confusion[static_cast<std::size_t>(label * classes + prediction)];
    out.accuracy += label == prediction;
  }
  for (int c = 0; c < classes; ++c) {
    int truth = 0, predicted = 0;
    for (int j = 0; j < classes; ++j) {
      truth += out.confusion[static_cast<std::size_t>(c * classes + j)];
      predicted += out.confusion[static_cast<std::size_t>(j * classes + c)];
    }
    check::Require(truth > 0, "every test class has support");
    const double tp = out.confusion[static_cast<std::size_t>(c * classes + c)];
    out.recall[static_cast<std::size_t>(c)] = tp / truth;
    out.macro_f1 += truth + predicted ? 2 * tp / (truth + predicted) : 0;
  }
  out.accuracy /= count;
  out.nll /= count;
  out.macro_f1 /= classes;
  return out;
}

inline void PrintVideo(const std::string& name, const VideoData& d, const Tensor& logits) {
  const auto r = ScoreLogits(logits, d.labels, 3);
  std::cout << " " << name << ":accuracy=" << 100 * r.accuracy << "%,macroF1=" << r.macro_f1 << ",CE=" << r.nll
            << ",class recalls=";
  for (double x : r.recall) std::cout << x << ',';
  std::cout << ",confusion=";
  for (int x : r.confusion) std::cout << x << ',';
  std::cout << '\n';
}

/// 4x4 spatial grid of mean colour (or mean absolute frame difference with `motion`) per clip: [N, 48].
inline std::vector<std::vector<double>> VideoFeatures(const VideoData& d, bool motion) {
  const auto                       rgb = d.rgb.to_vector<float>();
  std::vector<std::vector<double>> out(static_cast<std::size_t>(d.size()), std::vector<double>(48, 0.0));
  const auto at = [&](std::int64_t n, int t, int y, int x, int c) {
    return static_cast<double>(rgb[static_cast<std::size_t>((((n * 16 + t) * 32 + y) * 32 + x) * 3 + c)]);
  };
  for (std::int64_t n = 0; n < d.size(); ++n)
    for (int gy = 0; gy < 4; ++gy)
      for (int gx = 0; gx < 4; ++gx)
        for (int c = 0; c < 3; ++c) {
          double sum = 0;
          for (int t = motion ? 1 : 0; t < 16; ++t)
            for (int y = gy * 8; y < (gy + 1) * 8; ++y)
              for (int x = gx * 8; x < (gx + 1) * 8; ++x)
                sum += motion ? std::abs(at(n, t, y, x, c) - at(n, t - 1, y, x, c)) : at(n, t, y, x, c);
          // Stored as float, as tflow's feature tensor is.
          out[static_cast<std::size_t>(n)][static_cast<std::size_t>((gy * 4 + gx) * 3 + c)] =
              static_cast<float>(sum / ((motion ? 15 : 16) * 64));
        }
  return out;
}

/// Standardised 1-nearest-neighbour one-hot "logits" from fit clips.
inline Tensor VideoNearest(const VideoData& fit, const VideoData& test, bool motion) {
  const auto          reference = VideoFeatures(fit, motion), query = VideoFeatures(test, motion);
  const auto          labels    = fit.labels.to_vector<int>();
  std::vector<double> deviation(48);
  for (std::size_t c = 0; c < 48; ++c) {
    double mean = 0, variance = 0;
    for (const auto& r : reference) mean += r[c];
    mean /= static_cast<double>(reference.size());
    for (const auto& r : reference) variance += std::pow(r[c] - mean, 2);
    deviation[c] = std::max(1e-6, std::sqrt(variance / static_cast<double>(reference.size())));
  }
  std::vector<float> out(query.size() * 3, 0.f);
  for (std::size_t n = 0; n < query.size(); ++n) {
    double best       = INFINITY;
    int    prediction = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
      double distance = 0;
      for (std::size_t c = 0; c < 48; ++c) distance += std::pow((query[n][c] - reference[i][c]) / deviation[c], 2);
      if (distance < best) {
        best       = distance;
        prediction = labels[i];
      }
    }
    out[n * 3 + static_cast<std::size_t>(prediction)] = 1;
  }
  return nn::from_vector(out, {static_cast<std::int64_t>(query.size()), 3});
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
