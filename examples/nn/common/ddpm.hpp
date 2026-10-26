// Shared support for the 3.4 diffusion ports: a port of tflow's common/nn/ddpm.h
// and common/ml/ddpm.h (cosine 32-level DDPM schedule, compact epsilon MLP,
// clipped-x0 ancestral reverse transition, DdpmModel) plus the program helpers
// that tflow repeats in ddpm/ddim/diffusion_guidance/conditional_diffusion
// (8x8 [-1, 1] MNIST pools, tflow's mt19937 Gaussian banks and timestep draws,
// 64x64-tile PNG sheets, trajectory sampling).
//
// tflow -> datamunge: graph/session models become classes over a
// nn::ParameterStore; AddSymbolicGradients -> nn::grad; the Adam train_op ->
// nn::Optimizer (Adam .003); SaveV2 "final" bundles -> final.npz. Parameter
// names match tflow's ("hidden1/W", ...), so TF's initial_arrays.npz /
// model_arrays.npz load directly (init=auto replays TF's run).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/binary_pixels.hpp"

namespace diffusion {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using Map = replay::Map;

// ---- random streams and data -----------------------------------------------------------

/// tflow's Gaussian(n, seed[, double]): nn::GaussianNoise({n, 8, 8, 1}, 1, seed).
inline Tensor Gaussian8(std::int64_t n, std::uint32_t seed, bool wide = false) {
  const Tensor raw = nn::reshape(ae::GaussianNoise({n, 8, 8, 1}, 1, seed), {n, 8, 8, 1});
  return wide ? raw : nn::cast(raw, nn::DType::Float32);
}

inline Tensor Ints(const std::vector<std::int32_t>& v) { return nn::from_vector(v, {std::int64_t(v.size())}); }
inline Tensor FullInt(std::int64_t n, int value) { return Ints(std::vector<std::int32_t>(std::size_t(n), value)); }

/// tflow's Times(n, seed, cyclic): i % 32, or int(32 (u + .5) / 2^32) from mt19937(seed).
inline Tensor Times(std::int64_t n, std::uint32_t seed, bool cyclic = false) {
  std::vector<std::int32_t> out(static_cast<std::size_t>(n));
  std::mt19937              rng(seed);
  for (std::int64_t i = 0; i < n; ++i) out[i] = cyclic ? int(i % 32) : int(32 * (double(rng()) + .5) / 4294967296.);
  return Ints(out);
}

/// Rows of axis 0.
inline Tensor Rows(const Tensor& x, const std::vector<std::int64_t>& rows) { return nn::take_rows(x, rows); }

/// [n, 8, 8, 1] images of `x` [n0, 8, 8, 1] repeated so row b is row b % k of `x`.
inline Tensor Cycle(const Tensor& x, std::int64_t n) {
  std::vector<std::int64_t> rows(static_cast<std::size_t>(n));
  for (std::int64_t b = 0; b < n; ++b) rows[b] = b % x.size(0);
  return Rows(x, rows);
}

/// The MNIST pools (ae::LoadSaliency) and tflow's continuous 8x8 images:
/// half-pixel bilinear 28 -> 8, then 2 x - 1 (float32), NO threshold.
struct DigitData {
  std::vector<ae::SaliencyData> pools;
  std::vector<Tensor>           images;  // [n, 8, 8, 1] in [-1, 1]
  Map                           record;  // tflow's data.npz (without the noise/time banks)
};

inline DigitData LoadDigits(const fs::path& data) {
  DigitData d;
  d.pools = ae::LoadSaliency(data);
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    d.images.push_back(nn::sub(nn::mul(bpix::Resize8(d.pools[k].images), 2.0), 1.0));
    d.record[split + "_original"]   = d.pools[k].images;
    d.record[split + "_images"]     = d.images.back();
    d.record[split + "_labels"]     = d.pools[k].labels;
    d.record[split + "_source_ids"] = d.pools[k].source;
  }
  return d;
}

inline void Save(const fs::path& file, const Map& m) { nn::save_npz(file.string(), replay::ToNamed(m)); }
inline Map Load(const fs::path& file) { return replay::ToMap(nn::load_npz(file.string())); }

inline void Finite(const Map& fields) {
  for (const auto& [key, x] : fields)
    if (x.dtype() == nn::DType::Float32)
      for (float v : x.contiguous().to_vector<float>()) check::Require(std::isfinite(v), "finite DDPM field:" + key);
}

/// tflow's Png: every [-1, 1] 8x8 image as a 64x64 tile, `columns` per row, gray 220 background.
inline void Png(const fs::path& path, const Tensor& images, int columns) { bpix::Png(path, images, columns, 1); }

inline std::string Bits(const Tensor& t) { return bpix::Bits(t); }

/// Copies of axis-0 slices stacked: [k, ...] from k tensors of equal shape.
inline Tensor Stack(const std::vector<Tensor>& xs) { return nn::stack(xs, 0); }

/// Rows `rows` of x[k] concatenated in order (gallery assembly helper).
inline double MaxAbs(const Tensor& a, const Tensor& b) { return replay::MaxAbsDiff(a, b); }

// ---- schedule --------------------------------------------------------------------------

/// tflow's DdpmSchedule: 32 cosine levels (s = .008, beta cap .999), float64.
struct Schedule {
  std::map<std::string, std::vector<double>> values;
  const std::vector<double>&                 at(const std::string& key) const { return values.at(key); }
};

inline const Schedule& DdpmSchedule() {
  static const Schedule schedule = [] {
    Schedule out;
    auto     f   = [](double t) { return std::pow(std::cos((t + .008) / 1.008 * M_PI / 2), 2); };
    double   bar = 1;
    for (int i = 0; i < 32; ++i) {
      const double beta = std::min(.999, 1 - f((i + 1) / 32.) / f(i / 32.)), alpha = 1 - beta, previous = bar;
      bar *= alpha;
      out.values["beta"].push_back(beta);
      out.values["alpha"].push_back(alpha);
      out.values["alpha_bar"].push_back(bar);
      out.values["alpha_bar_previous"].push_back(previous);
      out.values["sqrt_alpha_bar"].push_back(std::sqrt(bar));
      out.values["sqrt_one_minus_alpha_bar"].push_back(std::sqrt(1 - bar));
      out.values["posterior_variance"].push_back(beta * (1 - previous) / (1 - bar));
      out.values["posterior_coefficient1"].push_back(beta * std::sqrt(previous) / (1 - bar));
      out.values["posterior_coefficient2"].push_back((1 - previous) * std::sqrt(alpha) / (1 - bar));
    }
    return out;
  }();
  return schedule;
}

inline Map ScheduleArrays() {
  Map out;
  for (const auto& [k, v] : DdpmSchedule().values) out[k] = nn::from_vector(v, {32});
  return out;
}

/// tflow's DdpmExtract: float(schedule[key][time]) as float32 [n, 1, 1, 1].
inline Tensor Extract(const std::string& key, const Tensor& time) {
  const auto&        table = DdpmSchedule().at(key);
  const auto         t     = nn::cast(time, nn::DType::Int64).to_vector<std::int64_t>();
  std::vector<float> out(t.size());
  for (std::size_t i = 0; i < t.size(); ++i) {
    check::Require(t[i] >= 0 && t[i] < 32, "timestep in 0..31");
    out[i] = float(table[std::size_t(t[i])]);
  }
  return nn::from_vector(out, {std::int64_t(t.size()), 1, 1, 1});
}

/// q(x_t | x_0) = sqrt(alpha_bar) x0 + sqrt(1 - alpha_bar) eps.
inline Tensor Noised(const Tensor& clean, const Tensor& noise, const Tensor& time) {
  return nn::add(nn::mul(Extract("sqrt_alpha_bar", time), clean), nn::mul(Extract("sqrt_one_minus_alpha_bar", time), noise));
}

// ---- epsilon MLP -----------------------------------------------------------------------

/// [sin, cos](time / maximum_time * pi {1, 2, 4, 8}) as [n, 8].
inline Tensor TimeEmbedding(const Tensor& time, float maximum_time = 31.f) {
  const Tensor normalized = nn::div(nn::reshape(nn::cast(time, nn::DType::Float32), {-1, 1}), double(maximum_time));
  const Tensor phase =
      nn::mul(normalized, nn::from_vector(std::vector<float>{float(M_PI), float(2 * M_PI), float(4 * M_PI), float(8 * M_PI)}, {1, 4}));
  return nn::concat({nn::sin(phase), nn::cos(phase)}, 1);
}

/// Creates tflow's three dense layers in tflow's order (W ~ N(0, .1), b = 0).
inline void CreateMlp(nn::ParameterStore& p, std::int64_t in, std::int64_t out) {
  for (const std::string name : {"hidden1", "hidden2", "output"}) {
    const std::int64_t i = name == "hidden1" ? in : 32, o = name == "output" ? out : 32;
    p.get(name + "/W", {i, o}, nn::init::normal(.1));
    p.get(name + "/b", {o}, nn::init::zeros());
  }
}

inline Tensor Dense(const nn::ParameterStore& p, const std::string& name, const Tensor& x, bool silu) {
  const Tensor y = nn::bias_add(nn::matmul(x, p.at(name + "/W")), p.at(name + "/b"));
  return silu ? nn::mul(y, nn::sigmoid(y)) : y;  // TF kSilu: x * sigmoid(x)
}

struct PredictorPath {
  Tensor epsilon, time_embedding, hidden1, hidden2;
};

/// tflow's DdpmPredictor: flat 64 pixels + 8 time features -> 32 SiLU -> 32 SiLU -> 64.
inline PredictorPath DdpmPredictor(const nn::ParameterStore& p, const Tensor& state, const Tensor& time) {
  const Tensor embedding = TimeEmbedding(time);
  const Tensor encoded   = nn::concat({nn::reshape(state, {-1, 64}), embedding}, 1);
  const Tensor h1 = Dense(p, "hidden1", encoded, true), h2 = Dense(p, "hidden2", h1, true);
  return {nn::reshape(Dense(p, "output", h2, false), {-1, 8, 8, 1}), embedding, h1, h2};
}

struct ReversePath {
  Tensor value, mean, variance, x0_raw, x0_clipped;
};

/// tflow's DdpmReverse: x0 from epsilon, clip [-1, 1], posterior mean c1 x0 + c2 x_t,
/// fixed posterior variance, value = mean + sqrt(variance) z.
inline ReversePath DdpmReverse(const Tensor& state, const Tensor& time, const Tensor& epsilon, const Tensor& innovation) {
  const Tensor signal = Extract("sqrt_alpha_bar", time), sigma = Extract("sqrt_one_minus_alpha_bar", time);
  const Tensor raw      = nn::div(nn::sub(state, nn::mul(sigma, epsilon)), signal);
  const Tensor clipped  = nn::minimum(nn::maximum(raw, nn::scalar_like(-1, raw)), nn::scalar_like(1, raw));
  const Tensor mean     = nn::add(nn::mul(Extract("posterior_coefficient1", time), clipped),
                                  nn::mul(Extract("posterior_coefficient2", time), state));
  const Tensor variance = Extract("posterior_variance", time);
  return {nn::add(mean, nn::mul(nn::sqrt(variance), innovation)), mean, variance, raw, clipped};
}

/// d sum(f(state)[:, first pixel]) / d state.
template <class F>
inline Tensor FirstPixelInputGradient(const Tensor& x, F&& f) {
  Tensor s = x.detach().clone();
  s.requires_grad_();
  const Tensor y = f(s);
  return nn::grad(nn::sum(nn::slice_axis(nn::reshape(y, {y.size(0), -1}), 1, 0, 1)), {s})[0];
}

inline nn::OptimizerConfig Adam() {
  nn::OptimizerConfig c;
  c.kind = nn::OptimizerKind::Adam;
  return c;
}

inline Map Detached(const Map& m) {
  Map out;
  for (const auto& [k, v] : m) out[k] = v.detach();
  return out;
}

/// Gradients of `loss` w.r.t. every parameter of `p`, by name.
inline Map ParameterGradients(const nn::ParameterStore& p, const Tensor& loss) {
  return replay::Named(p, nn::grad(loss, p.parameters()));
}

/// tflow's ml::DdpmModel (seed 11003, 5504 parameters / 6 tensors).
class DdpmModel {
 public:
  nn::ParameterStore             store{11003};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit DdpmModel(bool training = true) {
    CreateMlp(store, 72, 64);
    if (training) optimizer = std::make_unique<nn::Optimizer>(store, .003, Adam());
  }
  Tensor Loss(const Tensor& x, const Tensor& e, const Tensor& t, Map* fields = nullptr) {
    const Tensor noised = Noised(x, e, t);
    const auto   path   = DdpmPredictor(store, noised, t);
    const Tensor loss   = nn::mean(nn::square(nn::sub(path.epsilon, e)));
    if (fields)
      *fields = Detached({{"noised", noised}, {"epsilon", path.epsilon}, {"time_embedding", path.time_embedding},
                          {"hidden1", path.hidden1}, {"hidden2", path.hidden2}, {"loss", loss}});
    return loss;
  }
  Map Evaluate(const Tensor& x, const Tensor& e, const Tensor& t) {
    nn::NoGradGuard guard;
    Map             fields;
    Loss(x, e, t, &fields);
    return fields;
  }
  Map  Gradient(const Tensor& x, const Tensor& e, const Tensor& t) { return ParameterGradients(store, Loss(x, e, t)); }
  void Update(const Tensor& x, const Tensor& e, const Tensor& t) {
    optimizer->step(nn::grad(Loss(x, e, t), store.parameters()));
  }
  /// State-only ancestral transition (no labels/targets).
  Map Reverse(const Tensor& x, const Tensor& t, const Tensor& z) {
    Map out;
    {
      nn::NoGradGuard guard;
      const auto      path = DdpmPredictor(store, x, t);
      const auto      r    = DdpmReverse(x, t, path.epsilon, z);
      out = {{"epsilon", path.epsilon}, {"next", r.value},          {"mean", r.mean},
             {"variance", r.variance},  {"x0_raw", r.x0_raw},       {"x0_clipped", r.x0_clipped}};
    }
    out["input_gradient"] = FirstPixelInputGradient(x, [&](const Tensor& s) { return DdpmPredictor(store, s, t).epsilon; });
    return out;
  }
  Map                      Arrays() const { return replay::Arrays(store); }
  std::vector<std::string> Snapshot() const {
    std::vector<std::string> out;
    for (const auto& [k, v] : Arrays()) out.push_back(Bits(v));
    return out;
  }
};

/// Finds a frozen DDPM-family checkpoint in `dir`: final.npz (our ports) or
/// model_arrays.npz (tflow's restored final weights).
inline fs::path FrozenFile(const fs::path& dir) {
  if (fs::exists(dir / "final.npz")) return dir / "final.npz";
  if (fs::exists(dir / "model_arrays.npz")) return dir / "model_arrays.npz";
  return {};
}

/// Resolves a frozen-model option: "ours" (our port's output), "tf" (tflow's
/// recorded run), "auto" (ours when present, else TF) or a directory.
inline fs::path FrozenDir(const std::string& choice, const fs::path& ours, const fs::path& tf) {
  if (choice == "ours") return ours;
  if (choice == "tf") return tf;
  if (choice == "auto") return FrozenFile(ours).empty() ? tf : ours;
  return choice;
}

/// key=value options shared by the diffusion ports.
struct Options {
  int                                epochs = 3, threads = 4;
  fs::path                           data = paths::Dataset("saliency"), out, tf;
  std::string                        init = "auto";
  std::map<std::string, std::string> extra;  // ddpm= / guidance=
};

inline Options ParseOptions(int argc, char** argv, const std::string& name, const std::vector<std::string>& extra_keys,
                            const std::string& usage) {
  Options o;
  o.tf = replay::TfModels(name);
  for (const auto& k : extra_keys) o.extra[k] = "auto";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto        e = a.find('=');
    if (e == std::string::npos) throw std::invalid_argument(usage);
    const std::string k = a.substr(0, e), v = a.substr(e + 1);
    if (k == "epochs") o.epochs = std::stoi(v);
    else if (k == "data") o.data = v;
    else if (k == "out") o.out = v;
    else if (k == "init") o.init = v;
    else if (k == "tf") o.tf = v;
    else if (k == "threads") o.threads = std::stoi(v);
    else if (o.extra.count(k)) o.extra[k] = v;
    else throw std::invalid_argument(usage);
  }
  check::Require(o.epochs > 0 && o.epochs <= 100, "three cheap default epochs or explicit 1..100 override");
  if (o.out.empty()) o.out = paths::OutputRoot() / "models" / name;
  fs::create_directories(o.out);
  if (o.threads > 0) nn::set_num_threads(o.threads);
  return o;
}

/// Gathers per-step outputs: saved[key] grows to [steps, ...].
struct StepStack {
  std::map<std::string, std::vector<Tensor>> parts;
  void add(const Map& m) {
    for (const auto& [k, v] : m) parts[k].push_back(v);
  }
  Map stacked() const {
    Map out;
    for (const auto& [k, v] : parts) out[k] = nn::stack(v, 0);
    return out;
  }
};

/// Gallery: images[b * columns + j] = xs[j] row rows[b] (all [n, 8, 8, 1]).
inline Tensor Gallery(const std::vector<std::vector<Tensor>>& cells) {
  std::vector<Tensor> flat;
  for (const auto& row : cells)
    for (const auto& c : row) flat.push_back(nn::reshape(c, {1, 8, 8, 1}));
  return nn::concat(flat, 0);
}

inline Tensor Row(const Tensor& x, std::int64_t b) { return nn::slice_axis(x, 0, b, b + 1); }
/// Element `s` of a stacked [steps, ...] tensor, without the leading axis.
inline Tensor At(const Tensor& stacked, std::int64_t s) {
  nn::Shape shape(stacked.shape().begin() + 1, stacked.shape().end());
  return nn::reshape(nn::slice_axis(stacked, 0, s, s + 1), shape);
}

}  // namespace diffusion
