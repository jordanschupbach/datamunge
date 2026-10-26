// The small learned digit classifier of the 1.11 interpretability ports, its
// raw-class-score input derivatives and the short MNIST training/gallery harness:
// a port of tflow's common/nn/saliency.h and common/ml/saliency.h.
//
// Data: the balanced 1000 fit / 200 validation / 200 test MNIST pools of
// saliency_data.npz (ae::LoadSaliency). vanilla_saliency trains the classifier
// and writes output/nn/models/vanilla_saliency/{best,model_arrays,initial_arrays}.npz;
// the other interpretability ports restore it from there (source=PATH), or from
// tflow's own record (source=tflow = tflow's output/models/vanilla_saliency,
// whose model_arrays.npz/initial_arrays.npz use the same parameter names), which
// makes every map directly comparable with TF's saved arrays.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "common/autoencoder.hpp"
#include "common/checks.hpp"
#include "common/ecg_sensor.hpp"
#include "common/paths.hpp"
#include "common/tf_replay.hpp"

namespace sal {

namespace nn = datamunge::nn;
using nn::Tensor;
using ae::SaliencyData;
using Arrays = std::map<std::string, Tensor>;

// ---- host helpers ----------------------------------------------------------------------

inline std::vector<float> F(const Tensor& t) { return t.contiguous().to_vector<float>(); }
inline std::vector<int>   I(const Tensor& t) { return t.contiguous().to_vector<int>(); }
inline Tensor             FromF(const std::vector<float>& v, const nn::Shape& s) { return nn::from_vector(v, s); }
inline Tensor             FromI(const std::vector<int>& v) { return nn::from_vector(v, {std::int64_t(v.size())}); }
inline Tensor             Scalar(double v) { return nn::scalar(v, nn::DType::Float64); }
inline std::string        Bytes(const Tensor& t) { return ae::Bytes(t); }

inline SaliencyData Pick(const SaliencyData& d, const std::vector<std::int64_t>& rows) { return ae::Rows(d, rows); }
inline SaliencyData Images(const Tensor& images) { return {images, {}, {}}; }

/// tflow data::numpy::LoadNpz as a name -> tensor map.
inline Arrays LoadArrays(const std::filesystem::path& file) {
  check::Require(std::filesystem::exists(file), file.string() + " missing");
  Arrays out;
  for (auto& [name, value] : nn::load_npz(file.string())) out[name] = value.contiguous();
  return out;
}
inline void SaveArrays(const std::filesystem::path& file, const Arrays& arrays) {
  nn::save_npz(file.string(), nn::NamedTensors(arrays.begin(), arrays.end()));
}

// ---- model (common/nn/saliency.h) ------------------------------------------------------

/// The post-activation function of conv1/conv2/hidden: ReLU, or guided_backprop's GuidedRelu.
using Gate = std::function<Tensor(const Tensor&)>;

struct SaliencyCnnOutput {
  Tensor conv1, pool1, conv2, pool2, hidden, logits;
};

/// conv5x5x8 ReLU (He uniform, SAME) / avg2 / conv3x3x16 ReLU / avg2 / FC32 ReLU / 10 raw logits.
/// Only the convolutional part (conv1, pool1, conv2, pool2): any input size. TF's graphs run
/// just these nodes when they fetch conv features of resized (20x20, 24x24) images.
inline SaliencyCnnOutput SaliencyConvFeatures(nn::ParameterStore& p, const Tensor& image, const Gate& gate = {}) {
  const Gate           relu = gate ? gate : Gate([](const Tensor& x) { return nn::relu(x); });
  nn::ConvLayerOptions opt;
  opt.init = nn::init::he_uniform();
  SaliencyCnnOutput out;
  out.conv1 = relu(nn::conv2d(p, "conv1", image, 8, 5, opt));
  out.pool1 = nn::avg_pool2d(out.conv1, 2, 2, nn::Padding::Valid);
  out.conv2 = relu(nn::conv2d(p, "conv2", out.pool1, 16, 3, opt));
  out.pool2 = nn::avg_pool2d(out.conv2, 2, 2, nn::Padding::Valid);
  return out;
}

inline SaliencyCnnOutput SaliencyCnn(nn::ParameterStore& p, const Tensor& image, const Gate& gate = {}) {
  const Gate relu = gate ? gate : Gate([](const Tensor& x) { return nn::relu(x); });
  auto       out = SaliencyConvFeatures(p, image, gate);
  out.hidden = relu(nn::dense(p, "hidden", nn::reshape(out.pool2, {out.pool2.size(0), 784}), 32));
  out.logits = nn::dense(p, "head", out.hidden, 10);
  return out;
}

/// Selected raw logit per row: sum(logits * onehot(target)).
inline Tensor SaliencyScore(const Tensor& logits, const Tensor& target, int classes = 10) {
  return nn::sum(nn::mul(logits, nn::one_hot(target, classes, 1.0, 0.0, logits.dtype())), {1});
}

/// d sum_b score_b / d image, ordinary backprop through `forward` (logits of the image).
inline Tensor VanillaInputGradient(const std::function<Tensor(const Tensor&)>& forward, const Tensor& image,
                                   const Tensor& target, int classes = 10) {
  Tensor x = image.detach().clone();
  x.requires_grad_(true);
  return nn::grad(nn::sum(SaliencyScore(forward(x), target, classes)), {x})[0].detach();
}

// ---- classifier wrapper (common/ml/saliency.h) -----------------------------------------

inline void CheckTargets(const Tensor& classes, std::int64_t n) {
  check::Require(classes.dtype() == nn::DType::Int32 && classes.shape() == nn::Shape{n}, "one explicit fixed target per image");
  for (int c : I(classes)) check::Require(c >= 0 && c < 10, "target class0..9");
}

class SaliencyModel {
 public:
  nn::ParameterStore             parameters;
  Gate                           gate;
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit SaliencyModel(bool training = true, std::uint64_t seed = 1103, Gate g = {})
      : parameters(seed), gate(std::move(g)) {
    {
      nn::NoGradGuard guard;
      SaliencyCnn(parameters, nn::zeros({1, 28, 28, 1}));
    }
    if (training) optimizer = std::make_unique<nn::Optimizer>(parameters, .001, replay::Adam());
  }
  SaliencyCnnOutput Forward(const Tensor& images) { return SaliencyCnn(parameters, images, gate); }
  Tensor            Predict(const Tensor& images) {
    nn::NoGradGuard guard;
    return Forward(images).logits.detach().contiguous();
  }
  Tensor Predict(const SaliencyData& d) { return Predict(d.images); }
  /// RAW selected-logit input gradient, standard (or gated) backprop; finite everywhere.
  Tensor Gradient(const Tensor& images, const Tensor& classes) {
    CheckTargets(classes, images.size(0));
    auto g = VanillaInputGradient([&](const Tensor& x) { return Forward(x).logits; }, images, classes).contiguous();
    for (float v : F(g)) check::Require(std::isfinite(v), "finite every input derivative");
    return g;
  }
  Tensor Gradient(const SaliencyData& d, const Tensor& classes) { return Gradient(d.images, classes); }
  Tensor Loss(const SaliencyData& d) {
    return sensor::SensorCE(Forward(d.images).logits, d.labels, nn::ones({10}), 10);
  }
  std::vector<Tensor> ParameterGradients(const SaliencyData& d) { return nn::grad(Loss(d), parameters.parameters()); }
  double              Update(const SaliencyData& d) {
    const Tensor loss = Loss(d);
    optimizer->step(nn::grad(loss, parameters.parameters()));
    const double value = loss.item<double>();
    check::Require(std::isfinite(value) && value >= 0, "finite classifier CE");
    return value;
  }
  std::vector<std::string> Snapshot() const { return ae::Snapshot(parameters); }
  Arrays                   Export() const {
    Arrays out;
    for (const auto& name : parameters.names()) out[name] = parameters.at(name).detach().clone();
    return out;
  }
  void Export(const std::filesystem::path& file) const { SaveArrays(file, Export()); }
  /// Writes exact source tensors into all eight parameters (shape/finiteness checked).
  void Assign(const Arrays& weights) {
    check::Require(parameters.count() == 26826 && parameters.size() == 8 && weights.size() == 8, "same classifier parameter inventory");
    for (const auto& name : parameters.names()) {
      const auto it = weights.find(name);
      check::Require(it != weights.end() && it->second.dtype() == nn::DType::Float32 &&
                         it->second.shape() == parameters.at(name).shape(),
                     "verified trained/initial tensor shape");
      for (float v : F(it->second)) check::Require(std::isfinite(v), "finite source parameters");
      parameters.set(name, it->second.clone());
    }
  }
};

/// tflow SaliencyPredict: logits of all rows in batches of 32.
inline Tensor SaliencyPredict(SaliencyModel& model, const SaliencyData& d) {
  std::vector<Tensor> parts;
  const auto          n = d.images.size(0);
  for (std::int64_t start = 0; start < n; start += 32)
    parts.push_back(model.Predict(nn::slice_axis(d.images, 0, start, std::min<std::int64_t>(start + 32, n)).contiguous()));
  return nn::concat(parts, 0).contiguous();
}

inline void SaliencyMetrics(const std::string& name, const SaliencyData& d, const Tensor& logits) {
  const auto m = sensor::ScoreSensor(logits, d.labels, 10);
  std::cout << name << ":accuracy=" << 100 * m.accuracy << "%,macroF1=" << m.macro_f1 << ",NLL=" << m.nll << '\n';
}

/// Fit-only 7x7 mean-intensity 1-nearest-neighbour reference (one-hot "logits").
inline Tensor SaliencyNearest(const SaliencyData& fit, const SaliencyData& test) {
  auto feature = [](const Tensor& x) {
    const auto                       v = F(x);
    const auto                       n = x.size(0);
    std::vector<std::vector<double>> result(std::size_t(n), std::vector<double>(49));
    for (std::int64_t b = 0; b < n; ++b)
      for (int y = 0; y < 28; ++y)
        for (int a = 0; a < 28; ++a) result[b][(y / 4) * 7 + a / 4] += v[b * 784 + y * 28 + a] / 16;
    return result;
  };
  const auto       a = feature(fit.images), b = feature(test.images);
  const auto       labels = I(fit.labels);
  std::vector<float> out(b.size() * 10, 0.f);
  for (std::size_t i = 0; i < b.size(); ++i) {
    double best = INFINITY;
    int    label = 0;
    for (std::size_t j = 0; j < a.size(); ++j) {
      double distance = 0;
      for (int k = 0; k < 49; ++k) distance += std::pow(b[i][k] - a[j][k], 2);
      if (distance < best) {
        best  = distance;
        label = labels[j];
      }
    }
    out[i * 10 + label] = 1;
  }
  return FromF(out, {std::int64_t(b.size()), 10});
}

// ---- gallery rendering -----------------------------------------------------------------

inline std::uint8_t SaliencyByte(double x) { return std::uint8_t(std::lround(std::clamp(x, 0., 255.))); }

/// An RGB uint8 canvas (tflow's DT_UINT8 [h, w, 3] pixel tensors).
struct Canvas {
  int                       h, w;
  std::vector<std::uint8_t> px;
  Canvas(int height, int width, std::uint8_t fill = 220) : h(height), w(width), px(std::size_t(height) * width * 3, fill) {}
  std::uint8_t& at(int y, int x, int c) { return px[(std::size_t(y) * w + x) * 3 + c]; }
  void          Set(int y, int x, int c, double value) { at(y, x, c) = SaliencyByte(value); }
  void          Gray(int y, int x, double value) {
    for (int c = 0; c < 3; ++c) Set(y, x, c, value);
  }
};

/// tflow SaliencyPng: native PNG of the canvas.
inline void SaliencyPng(const std::filesystem::path& path, const Canvas& canvas) {
  std::vector<float> values(canvas.px.begin(), canvas.px.end());
  nn::save_png(FromF(values, {canvas.h, canvas.w, 3}), path.string(), 0.0, 255.0);
  check::Require(std::filesystem::exists(path), "write saliencyPNG");
}

/// Red positive / blue negative / white zero for a value already divided by the shared peak.
inline std::array<double, 3> SignedColor(double g) {
  return {255 * (1 + std::min(0., g)), 255 * (1 - std::abs(g)), 255 * (1 - std::max(0., g))};
}

// ---- shared selection helpers ----------------------------------------------------------

/// First test record of every digit 0..9.
inline std::vector<std::int64_t> FirstPerDigit(const SaliencyData& test) {
  const auto                labels = I(test.labels);
  std::vector<std::int64_t> rows;
  for (int c = 0; c < 10; ++c)
    for (std::size_t i = 0; i < labels.size(); ++i)
      if (labels[i] == c) {
        rows.push_back(std::int64_t(i));
        break;
      }
  check::Require(rows.size() == 10, "one held-out case per digit");
  return rows;
}

/// Predicted (top) and runner-up classes per row, stable ranking as tflow.
inline std::pair<Tensor, Tensor> TopTwo(const Tensor& logits) {
  const auto       z = F(logits);
  const auto       n = logits.size(0);
  std::vector<int> targets(n), runners(n);
  for (std::int64_t b = 0; b < n; ++b) {
    std::vector<int> ranking(10);
    std::iota(ranking.begin(), ranking.end(), 0);
    std::stable_sort(ranking.begin(), ranking.end(), [&](int a, int c) { return z[b * 10 + a] > z[b * 10 + c]; });
    targets[b] = ranking[0];
    runners[b] = ranking[1];
  }
  return {FromI(targets), FromI(runners)};
}

/// source=PATH resolution: "tflow" means tflow's recorded output/models/vanilla_saliency.
inline std::filesystem::path SourceDir(const std::string& value) {
  return value == "tflow" ? replay::TfModels("vanilla_saliency") : std::filesystem::path(value);
}
inline std::filesystem::path DefaultSource() { return paths::OutputRoot() / "models/vanilla_saliency"; }

/// The verified trained classifier and the actual initial weights of a vanilla_saliency run.
struct SourceWeights {
  Arrays trained, initial;
  bool   ours = false;  // our own run (best.npz present) or tflow's record
};

/// tflow's restore checks: the restored checkpoint equals model_arrays.npz for all eight
/// tensors, and a fresh seed-1103 model reproduces initial_arrays.npz. TF restores its
/// SaveV2 `best` checkpoint; ours restores best.npz. For tflow's record (no best.npz) the
/// trained model loads model_arrays.npz and the fresh model loads initial_arrays.npz, since
/// datamunge's initializers do not reproduce TF's random streams.
inline SourceWeights RestoreSource(const std::filesystem::path& source, SaliencyModel& model, SaliencyModel& initial) {
  SourceWeights w{LoadArrays(source / "model_arrays.npz"), LoadArrays(source / "initial_arrays.npz"),
                  std::filesystem::exists(source / "best.npz")};
  if (w.ours)
    model.parameters.load((source / "best.npz").string());
  else
    model.Assign(w.trained);
  if (!w.ours) initial.Assign(w.initial);
  const auto before = model.Snapshot(), initial_before = initial.Snapshot();
  const auto& names = model.parameters.names();
  for (std::size_t i = 0; i < names.size(); ++i) {
    check::Require(before[i] == Bytes(w.trained.at(names[i])), "all8trained tensors match source");
    check::Require(initial_before[i] == Bytes(w.initial.at(names[i])), "all8actual initial tensors match source");
  }
  return w;
}

/// Copies data/source manifests next to the artifacts when they exist.
inline void CopyManifests(const std::filesystem::path& directory,
                          const std::vector<std::pair<std::filesystem::path, std::string>>& files) {
  for (const auto& [from, to] : files)
    if (std::filesystem::exists(from))
      std::filesystem::copy_file(from, directory / to, std::filesystem::copy_options::overwrite_existing);
}

// ---- vanilla saliency training and gallery (ml::TrainVanillaSaliency) -------------------

inline void SaliencyGallery(const std::filesystem::path& directory, const SaliencyData& examples, const Tensor& targets,
                            const Tensor& runners, const Tensor& gradients, const Tensor& alternative, const Tensor& random,
                            double peak) {
  check::Require(peak > 0, "nonzero global visualization scale");
  std::ofstream html(directory / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Vanilla class-score saliency</title><style>body{font:16px "
          "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:28px "
          "0}</style><h1>Vanilla input gradients of raw digit-class logits</h1><p>Left to right:input,signed predicted-class "
          "gradient,absolute predicted-class gradient,absolute runner-up gradient,absolute fresh-initialization gradient for the "
          "SAME predicted target. All heatmaps share ONE absolute scale across every case/target/model. Red=positive,blue=negative,"
          "white=zero in signed maps;absolute zero=black. Gradient units:logit per[0,1]pixel. Targets fixed after checkpoint "
          "selection;no softmax/loss/gradient-times-input/smoothing/guided gates.</p><p>Ten cases:first selected original-test "
          "record per true digit,including classification errors. Gradients measure local sensitivity,not causal importance or "
          "segmentation. Fresh weights/target changes are controls,not a proof of explanation faithfulness.</p><p>Shared absolute "
          "maximum="
       << peak << ";raw signed arrays and class IDs in saliency.npz.</p>";
  const auto image = F(examples.images), g = F(gradients), alt = F(alternative), rnd = F(random);
  const auto source = I(examples.source), labels = I(examples.labels), t = I(targets), r = I(runners);
  for (int b = 0; b < 10; ++b) {
    Canvas pixels(112, 5 * 116 - 4);
    for (int y = 0; y < 112; ++y)
      for (int x = 0; x < 112; ++x) {
        const int  i = b * 784 + (y / 4) * 28 + x / 4;
        const auto signed_rgb = SignedColor(g[i] / peak);
        for (int c = 0; c < 3; ++c) {
          pixels.Set(y, x, c, 255 * image[i]);
          pixels.Set(y, 116 + x, c, signed_rgb[c]);
          pixels.Set(y, 232 + x, c, 255 * std::abs(double(g[i])) / peak);
          pixels.Set(y, 348 + x, c, 255 * std::abs(double(alt[i])) / peak);
          pixels.Set(y, 464 + x, c, 255 * std::abs(double(rnd[i])) / peak);
        }
      }
    const std::string name = "case" + std::to_string(b) + ".png";
    SaliencyPng(directory / name, pixels);
    html << "<section><h2>Source" << source[b] << ",true " << labels[b] << ",predicted " << t[b] << ",runner-up " << r[b]
         << "</h2><img src='" << name << "'></section>";
  }
  check::Require(bool(html), "write saliency gallery");
}

}  // namespace sal
