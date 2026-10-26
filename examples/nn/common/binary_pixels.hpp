// Shared harness of the binary 8x8 MNIST autoregressive density ports
// (pixel_cnn, gated_pixel_cnn, made_density, pixel_rnn): the program body that
// tflow repeats in each of those .cc files, plus the model-independent part of
// tflow's common/ml/{pixel_cnn,gated_pixel_cnn,made_density,pixel_rnn}.h
// (teacher-forced Bernoulli likelihood, causal logit/input Jacobian, prefix-only
// ancestral sampling with MT19937 uniforms).
//
// Data: the 1000/200/200 MNIST pools of tflow's saliency assets (ae::LoadSaliency),
// half-pixel bilinear 28 -> 8 in double then float32 threshold >= .5, as tflow.
//
// tflow -> datamunge: session graphs -> eager calls, AddSymbolicGradients ->
// nn::grad, Adam train_op -> nn::Optimizer (Adam .003), SaveV2 -> final.npz.
// TF parity: init=auto (default) replays TF's recorded initial_arrays.npz from
// tflow's output/models/<name>[/<variant>] so the whole run follows TF's; every
// record is then compared with TF's and the replayed first-step records must
// match. Independently of the trajectory, TF's FINAL weights (model_arrays.npz)
// are loaded into a fresh model which must reproduce TF's test outputs.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common/autoencoder.hpp"
#include "common/tf_replay.hpp"

namespace bpix {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using Map = replay::Map;

// ---- small helpers -----------------------------------------------------------------------------

inline Tensor Floats(const std::vector<float>& v, const nn::Shape& s) { return nn::from_vector(v, s); }

/// tflow's Png: every 8x8 image as a 64x64 tile (8x nearest), `columns` per row,
/// gray 220 background, floor(v * 255 + .5); `rgb` images use (v + 1) * 127.5
/// with clamping when `signed_range`, else 255 clamp(v, 0, 1).
inline void Png(const fs::path& path, const Tensor& images, int columns, int mode = 0) {
  const auto n = images.size(0), c = images.size(3);
  const auto v = images.contiguous().to_vector<float>();
  const std::int64_t rows = (n + columns - 1) / columns, H = rows * 64, W = columns * 64;
  std::vector<float> out(static_cast<std::size_t>(H * W * 3), 220.f);
  for (std::int64_t b = 0; b < n; ++b)
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x)
        for (int k = 0; k < 3; ++k) {
          const double value = v[static_cast<std::size_t>(((b * 8 + y / 8) * 8 + x / 8) * c + (c == 1 ? 0 : k))];
          double       byte  = mode == 0   ? std::floor(value * 255 + .5)
                               : mode == 1 ? std::floor(std::clamp((value + 1) * 127.5, 0., 255.) + .5)
                                           : std::floor(255 * std::clamp(value, 0., 1.) + .5);
          out[static_cast<std::size_t>(((b / columns * 64 + y) * W + b % columns * 64 + x) * 3 + k)] = float(byte);
        }
  nn::save_png(Floats(out, {H, W, 3}), path.string(), 0, 255);
}

/// Writes a [H, W, 3] uint8-valued byte image.
inline void PngBytes(const fs::path& path, const std::vector<float>& bytes, std::int64_t h, std::int64_t w) {
  nn::save_png(Floats(bytes, {h, w, 3}), path.string(), 0, 255);
}

inline void RequireFinite(const Map& m, const std::string& what) {
  for (const auto& [k, v] : m)
    if (v.dtype() == nn::DType::Float32)
      for (float x : v.contiguous().to_vector<float>()) check::Require(std::isfinite(x), what);
}

inline std::string Bits(const Tensor& t) {
  const auto c = t.contiguous();
  switch (c.dtype()) {
    case nn::DType::Float32: {
      const auto v = c.to_vector<float>();
      return {reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float)};
    }
    case nn::DType::Float64: {
      const auto v = c.to_vector<double>();
      return {reinterpret_cast<const char*>(v.data()), v.size() * sizeof(double)};
    }
    default: {
      const auto v = c.to_vector<std::int32_t>();
      return {reinterpret_cast<const char*>(v.data()), v.size() * sizeof(std::int32_t)};
    }
  }
}

inline bool SameBits(const Map& a, const Map& b) {
  if (a.size() != b.size()) return false;
  for (const auto& [k, v] : a)
    if (!b.count(k) || Bits(v) != Bits(b.at(k)) || v.shape() != b.at(k).shape()) return false;
  return true;
}

// ---- options ----------------------------------------------------------------------------------

struct Options {
  int         epochs = 3, threads = 4;
  fs::path    data = paths::Dataset("saliency"), out, tf;
  std::string init = "auto";
  bool        verbose = false;
};

/// epochs=N data=DIR out=DIR init=auto|tf|nn|PATH tf=DIR threads=N verbose=0|1
inline Options ParseOptions(int argc, char** argv, const std::string& name, const fs::path& data) {
  Options o;
  o.data = data;
  o.tf   = replay::TfModels(name);
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto        e = a.find('=');
    check::Require(e != std::string::npos, "expected key=value (epochs= data= out= init= tf= threads= verbose=)");
    const std::string k = a.substr(0, e), v = a.substr(e + 1);
    if (k == "epochs") o.epochs = std::stoi(v);
    else if (k == "data") o.data = v;
    else if (k == "out") o.out = v;
    else if (k == "init") o.init = v;
    else if (k == "tf") o.tf = v;
    else if (k == "threads") o.threads = std::stoi(v);
    else if (k == "verbose") o.verbose = v != "0";
    else throw std::invalid_argument("epochs=N data=PATH out=PATH init=auto|tf|nn|PATH tf=DIR threads=N verbose=0|1");
  }
  check::Require(o.epochs > 0 && o.epochs <= 100, "three cheap default epochs or explicit 1..100 override");
  if (o.out.empty()) o.out = paths::OutputRoot() / "models" / name;
  fs::create_directories(o.out);
  if (o.threads > 0) nn::set_num_threads(o.threads);
  return o;
}

/// The initial arrays to replay (empty: nn's generator). `tf` is the TF run directory.
inline fs::path InitFile(const std::string& init, const fs::path& tf) {
  if (init == "nn") return {};
  if (init == "auto" || init == "tf") {
    const auto f = tf / "initial_arrays.npz";
    if (init == "tf") check::Require(fs::exists(f), "init=tf needs " + f.string());
    return fs::exists(f) ? f : fs::path();
  }
  check::Require(fs::exists(init), "init file " + init + " missing");
  return init;
}

/// Max |ours - TF| over the arrays both records share (-1 when TF's file is missing);
/// `relative`: max |ours - TF| / (1 + |TF|).
inline double Compare(const fs::path& tf, const std::string& file, const Map& ours, bool verbose = false, bool relative = false) {
  if (!fs::exists(tf / file)) {
    std::printf("  vs TF %-28s (no TF record)\n", file.c_str());
    return -1;
  }
  const auto theirs = replay::ToMap(nn::load_npz((tf / file).string()));
  double     m      = 0;
  int        shared = 0;
  for (const auto& [k, v] : ours) {
    if (!theirs.count(k) || theirs.at(k).numel() != v.numel()) continue;
    const auto   x = nn::cast(v, nn::DType::Float64).to_vector<double>(), y = nn::cast(theirs.at(k), nn::DType::Float64).to_vector<double>();
    double       d = 0;
    for (std::size_t i = 0; i < x.size(); ++i) d = std::max(d, std::abs(x[i] - y[i]) / (relative ? 1 + std::abs(y[i]) : 1.));
    if (verbose) std::printf("      %-24s %.3g\n", k.c_str(), d);
    m = std::max(m, d);
    ++shared;
  }
  std::printf("  vs TF %-28s max |ours - TF|%s = %.3g (%d arrays)\n", file.c_str(), relative ? "/(1+|TF|)" : "", m, shared);
  return m;
}

/// Fraction of equal elements of `key` in ours vs TF's record.
inline double Agreement(const fs::path& tf, const std::string& file, const Map& ours, const std::string& key) {
  if (!fs::exists(tf / file)) return -1;
  const auto theirs = replay::ToMap(nn::load_npz((tf / file).string()));
  if (!theirs.count(key)) return -1;
  const auto a = nn::cast(ours.at(key), nn::DType::Float64).to_vector<double>();
  const auto b = nn::cast(theirs.at(key), nn::DType::Float64).to_vector<double>();
  if (a.size() != b.size()) return -1;
  std::size_t same = 0;
  for (std::size_t i = 0; i < a.size(); ++i) same += a[i] == b[i];
  return double(same) / double(a.size());
}

// ---- data -------------------------------------------------------------------------------------

/// tflow's half-pixel bilinear 28 -> 8 resize (double accumulation, float result).
inline Tensor Resize8(const Tensor& images) {
  const auto n = images.size(0);
  const auto x = images.contiguous().to_vector<float>();
  std::vector<float> out(static_cast<std::size_t>(n * 64));
  for (std::int64_t b = 0; b < n; ++b)
    for (int y = 0; y < 8; ++y)
      for (int col = 0; col < 8; ++col) {
        const double py = (y + .5) * 3.5 - .5, px = (col + .5) * 3.5 - .5;
        const int    iy = int(std::floor(py)), ix = int(std::floor(px));
        double       value = 0;
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            value += (dy ? py - iy : 1 - py + iy) * (dx ? px - ix : 1 - px + ix) *
                     x[static_cast<std::size_t>((b * 28 + iy + dy) * 28 + ix + dx)];
        out[static_cast<std::size_t>(b * 64 + y * 8 + col)] = float(value);
      }
  return Floats(out, {n, 8, 8, 1});
}

struct BinaryData {
  std::vector<ae::SaliencyData> pools;
  std::vector<Tensor>           gray, binary;  // [n, 8, 8, 1] float
  Map                           record;        // tflow's data.npz
};

inline BinaryData LoadBinary(const fs::path& data) {
  BinaryData d;
  d.pools = ae::LoadSaliency(data);
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    auto              gray  = Resize8(d.pools[k].images);
    auto              g     = gray.to_vector<float>();
    for (auto& v : g) v = v >= .5f ? 1.f : 0.f;
    auto x = Floats(g, gray.shape());
    d.gray.push_back(gray);
    d.binary.push_back(x);
    d.record[split + "_original"]   = d.pools[k].images;
    d.record[split + "_gray"]       = gray;
    d.record[split + "_images"]     = x;
    d.record[split + "_labels"]     = d.pools[k].labels;
    d.record[split + "_source_ids"] = d.pools[k].source;
  }
  return d;
}

inline Tensor Rows(const Tensor& x, const std::vector<std::int64_t>& rows) { return nn::take_rows(x, rows); }

// ---- model base -------------------------------------------------------------------------------

/// A binary-pixel autoregressive density: Network() returns the model's named
/// fields with the logits [N, 8, 8, 1] first. `order(j)` is the generation
/// ordinal of pixel j (raster, or reversed for MADE's reverse model).
class BinaryModel {
 public:
  nn::ParameterStore             store;
  std::unique_ptr<nn::Optimizer> optimizer;
  bool                           reverse = false, uses_classes = false;

  explicit BinaryModel(std::uint64_t seed) : store(seed) {}
  virtual ~BinaryModel() = default;
  BinaryModel(const BinaryModel&)            = delete;
  BinaryModel& operator=(const BinaryModel&) = delete;

  virtual std::vector<std::pair<std::string, Tensor>> Network(const Tensor& x, const Tensor& classes) = 0;

  /// Builds every parameter once (call from the derived constructor) and Adam(.003) when training.
  void Build(bool training) {
    {
      nn::NoGradGuard guard;
      Network(nn::zeros({1, 8, 8, 1}), nn::zeros({1}, nn::DType::Int32));
    }
    if (training) Train();
  }
  void Train() { optimizer = std::make_unique<nn::Optimizer>(store, .003, replay::Adam()); }

  int Order(int j) const { return reverse ? 63 - j : j; }

  void Validate(const Tensor& x, const Tensor& classes) const {
    if (!uses_classes) return;
    check::Require(classes.defined() && classes.dim() == 1 && classes.size(0) == x.size(0), "aligned class conditioning");
    for (int c : nn::cast(classes, nn::DType::Int32).to_vector<int>())
      check::Require(c >= 0 && c < 10, "valid 0..9 class conditions");
  }

  static Tensor PixelNll(const Tensor& logits, const Tensor& x) { return nn::sub(nn::softplus(logits), nn::mul(x, logits)); }

  Map Evaluate(const Tensor& x, const Tensor& classes = {}) {
    Validate(x, classes);
    nn::NoGradGuard guard;
    auto            fields = Network(x, classes);
    const Tensor    logits = fields[0].second, nll = PixelNll(logits, x);
    Map             out;
    out["logits"]      = logits;
    out["probability"] = nn::sigmoid(logits);
    for (std::size_t i = 1; i < fields.size(); ++i) out[fields[i].first] = fields[i].second;
    out["per_image_nll"] = nn::sum(nll, {1, 2, 3});
    out["loss"]          = nn::mean(nll);
    for (auto& [k, v] : out) v = v.detach().contiguous();
    RequireFinite(out, "finite model fields");
    return out;
  }

  Tensor Probability(const Tensor& x, const Tensor& classes = {}) {
    Validate(x, classes);
    nn::NoGradGuard guard;
    return nn::sigmoid(Network(x, classes)[0].second).contiguous();
  }

  /// d sum_b logits[b, site] / d image.
  Tensor CausalGradient(const Tensor& x, const Tensor& classes, int site) {
    check::Require(site >= 0 && site < 64, "valid pixel index");
    Validate(x, classes);
    Tensor input = x.detach().clone();
    input.requires_grad_(true);
    const Tensor logits = nn::reshape(Network(input, classes)[0].second, {-1, 64});
    return nn::grad(nn::sum(nn::slice_axis(logits, 1, site, site + 1)), {input})[0].detach().contiguous();
  }

  std::vector<Tensor> RawGradients(const Tensor& x, const Tensor& classes) {
    Validate(x, classes);
    const Tensor    loss = nn::mean(PixelNll(Network(x, classes)[0].second, x));
    nn::GradOptions go;
    go.allow_unused = true;
    auto       g  = nn::grad(loss, store.parameters(), go);
    const auto ps = store.parameters();
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = g[i].defined() ? g[i].detach() : nn::zeros_like(ps[i]);
    return g;
  }
  Map  Gradient(const Tensor& x, const Tensor& classes = {}) { return replay::Named(store, RawGradients(x, classes)); }
  void Update(const Tensor& x, const Tensor& classes = {}) { optimizer->step(RawGradients(x, classes)); }

  Map Arrays() const { return replay::Arrays(store); }

  /// tflow's Sample: n images drawn pixel by pixel in model order from an
  /// all-zero canvas (the first `prefix_size` ordinals copied from `prefix`),
  /// u = (mt19937() + .5) / 2^32 bank [n, 64] in ordinal order; `common_four`
  /// shares the first four rows' uniforms across groups of four.
  Map Sample(int n, std::uint32_t seed, const Tensor& classes = {}, const Tensor* prefix = nullptr, int prefix_size = 0,
             bool common_four = false) {
    check::Require(n > 0 && prefix_size >= 0 && prefix_size <= 64, "valid ancestral sample count/prefix");
    std::vector<float>  images(static_cast<std::size_t>(n * 64), 0.f), conditionals(static_cast<std::size_t>(n * 64), 0.f);
    std::vector<double> uniform(static_cast<std::size_t>(n * 64));
    std::mt19937        rng(seed);
    for (auto& u : uniform) u = (double(rng()) + .5) / 4294967296.;
    if (common_four) {
      check::Require(n % 4 == 0, "common-four-noise group");
      for (int b = 4; b < n; ++b)
        for (int j = 0; j < 64; ++j) uniform[b * 64 + j] = uniform[(b % 4) * 64 + j];
    }
    if (prefix) {
      check::Require(prefix->shape() == nn::Shape({n, 8, 8, 1}), "aligned prefix batch");
      const auto p = prefix->contiguous().to_vector<float>();
      for (int b = 0; b < n; ++b)
        for (int j = 0; j < prefix_size; ++j) images[b * 64 + Order(j)] = p[b * 64 + Order(j)];
    }
    for (int j = prefix_size; j < 64; ++j) {
      const int  pixel = Order(j);
      const auto p     = Probability(Floats(images, {n, 8, 8, 1}), classes).to_vector<float>();
      for (int b = 0; b < n; ++b) {
        const float q          = p[b * 64 + pixel];
        conditionals[b * 64 + j] = q;
        images[b * 64 + pixel] = uniform[b * 64 + j] < q ? 1.f : 0.f;
      }
    }
    return {{"images", Floats(images, {n, 8, 8, 1})},
            {"uniform", nn::from_vector(uniform, {n, 64})},
            {"conditionals", Floats(conditionals, {n, 64})}};
  }
};

// ---- the program ------------------------------------------------------------------------------

/// What differs between the binary density examples.
struct Spec {
  std::string   name, variant;          // tflow output/models/<name>[/<variant>]
  std::int64_t  parameters = 0, tensors = 0, masked = 0;
  std::uint64_t batch_seed = 0;
  std::uint32_t sample_seed = 0, completion_seed = 0, epoch_sample_seed = 0;
  std::string   title, html;            // index.html title / body (after the h1)
  std::string   tag;                    // console tag, e.g. "maskedCNN"
  std::string   metrics_prefix;         // leading JSON members ("\"conditional\":true,")
};

/// Extra (example-specific) checks on the deployed model; returns extra JSON members.
using Extra = std::function<std::string(BinaryModel& deployed, const BinaryData& data, const fs::path& root, Map& outputs)>;
using Factory = std::function<std::unique_ptr<BinaryModel>(bool training)>;

inline Tensor ClassesOf(const BinaryData& d, int split, const std::vector<std::int64_t>& rows) {
  return nn::take_rows(d.pools[split].labels, rows);
}
inline Tensor CycleClasses(int n, bool grouped = false) {
  std::vector<std::int32_t> v(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) v[i] = grouped ? i / 4 : i % 10;
  return nn::from_vector(v, {n});
}

}  // namespace bpix
