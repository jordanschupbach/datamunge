// Data/model helpers of the 3.3 normalizing-flow density ports (nice_density,
// real_nvp_density, glow_density, maf_density, iaf_density, spline_density,
// planar_radial_density, cnf_density, flow_density_2d): the program body tflow
// repeats in each of those .cc files, plus the model-independent part of tflow's
// common/ml/<flow>_density.h classes (exact change-of-variables NLL, actual
// input Jacobians of the forward and inverse maps, latent-only inverse sampling,
// Adam .003 updates, weights, gradients).
//
// Data: SYNTHETIC analytic banana u ~ N(0, I), x0 = u0, x1 = u1 + .5 (u0^2 - 1),
// 512 fit / 128 validation / 128 test rows from tflow's Box-Muller noise (seeds
// 9203/9209/9221, bit-identical) plus 1024 prior draws (seed 9257).
// flow_density_2d adds a third base coordinate that picks a component of an
// eight-Gaussian ring or a checkerboard.
//
// tflow -> datamunge: session graphs -> eager calls, AddSymbolicGradients ->
// nn::grad (input Jacobians by reverse mode on a differentiable copy of the
// input), Adam train_op -> nn::Optimizer (Adam .003), SaveV2 -> final.npz.
// TF parity: init=auto (default) replays TF's recorded initial_arrays.npz from
// tflow's output/models/<name>[/<variant>] so the whole run follows TF's, and the
// records are compared with TF's. Independently of the trajectory, TF's FINAL
// weights (model_arrays.npz) are loaded into a fresh model which must reproduce
// TF's test outputs, samples and density grid. tflow's independent_audit.json is
// written by a separate tflow script and is not reproduced.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/tf_replay.hpp"

namespace flowd {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using Map   = replay::Map;
using Named = std::vector<std::pair<std::string, Tensor>>;

// ---- small helpers -------------------------------------------------------------------------------

/// x[:, k:k+n] (TF Slice {0, k}, {-1, n}).
inline Tensor Col(const Tensor& x, std::int64_t k, std::int64_t n = 1) { return nn::slice_axis(x, 1, k, k + n); }

/// tflow nn::Dense after p.AddNormal(name/W, {in, out}, stddev) and
/// p.AddConstant(name/b, {out}, bias): x W + b, optionally tanh.
inline Tensor Dense(nn::ParameterStore& p, const std::string& name, const Tensor& x, std::int64_t in, std::int64_t out,
                    double stddev, double bias, bool tanh = false) {
  const Tensor w = p.get(name + "/W", {in, out}, nn::init::normal(stddev));
  const Tensor b = p.get(name + "/b", {out}, nn::init::constant(bias));
  const Tensor y = nn::bias_add(nn::matmul(x, w), b);
  return tanh ? nn::tanh(y) : y;
}

/// ops::Mul(Reshape(Slice(x, {0,0}, {-1,1}), {-1}), 0): float zeros [n] tied to x.
inline Tensor ZeroLogdet(const Tensor& x) { return nn::reshape(Col(x, 0), {-1}) * 0.0; }

/// tflow nn::GaussianNoise({n, d}, 1, seed): Box-Muller pairs from std::mt19937(seed), float64 [n, d].
inline Tensor GaussianNoise(std::int64_t n, std::int64_t d, std::uint32_t seed) {
  std::vector<double> bank(static_cast<std::size_t>(n * d));
  std::mt19937        rng(seed);
  constexpr double    tau = 6.283185307179586476925286766559;
  for (std::size_t i = 0; i < bank.size(); i += 2) {
    const double u = (double(rng()) + .5) / 4294967296., v = (double(rng()) + .5) / 4294967296.;
    const double radius = std::sqrt(-2 * std::log(u));
    bank[i]             = radius * std::cos(tau * v);
    if (i + 1 < bank.size()) bank[i + 1] = radius * std::sin(tau * v);
  }
  return nn::from_vector(bank, {n, d});
}
inline Tensor Floats(const Tensor& x) { return nn::cast(x, nn::DType::Float32).contiguous(); }

inline std::string Bits(const Tensor& t) {
  const auto c = t.contiguous();
  if (c.dtype() == nn::DType::Float64) {
    const auto v = c.to_vector<double>();
    return {reinterpret_cast<const char*>(v.data()), v.size() * sizeof(double)};
  }
  const auto v = c.to_vector<float>();
  return {reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float)};
}
inline bool SameBits(const Map& a, const Map& b) {
  if (a.size() != b.size()) return false;
  for (const auto& [k, v] : a)
    if (!b.count(k) || v.shape() != b.at(k).shape() || Bits(v) != Bits(b.at(k))) return false;
  return true;
}

/// Element b of a per-example field, or its single value (NICE's global logdet).
inline double At(const std::vector<float>& v, std::size_t b) { return v.size() == 1 ? v[0] : v[b]; }
inline double Mean(const Tensor& x) {
  const auto v = x.to_vector<float>();
  double     s = 0;
  for (float f : v) s += f;
  return s / double(v.size());
}

// ---- the model interface -------------------------------------------------------------------------

struct FlowPath {
  Tensor latent, unscaled, logdet;  // logdet [n] (or a scalar for NICE)
  Named  fields;                    // per-stage diagnostics in tflow's Evaluate names
};
struct FlowInverse {
  Tensor value, logdet;
};

/// tflow's ml::<Flow>Density classes without the session plumbing.
class FlowModel {
 public:
  nn::ParameterStore             store;
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit FlowModel(std::uint64_t seed) : store(seed) {}
  virtual ~FlowModel() = default;

  virtual FlowPath    Forward(const Tensor& x) = 0;
  virtual FlowInverse Inverse(const Tensor& z) = 0;
  /// Extra Evaluate fields that differentiate model internals (conditioner / ODE field Jacobians).
  virtual void Extra(const Tensor& /*x*/, const FlowPath& /*path*/, Map& /*out*/) {}
  /// One-time data-dependent initialisation before training (Glow ActNorm).
  virtual void Prepare(const Tensor& /*fit*/, const fs::path& /*root*/) {}

  /// Creates every parameter (call from the derived constructor).
  void Build(bool training) {
    {
      nn::NoGradGuard none;
      const Tensor    probe = nn::zeros({1, 2});
      Forward(probe);
      Inverse(probe);
    }
    if (training) optimizer = std::make_unique<nn::Optimizer>(store, .003, replay::Adam());
  }

  static Tensor BaseLogp(const Tensor& z) {
    return -(0.5 * nn::sum(nn::square(z), {1}) + double(float(std::log(2 * M_PI))));
  }
  Tensor Loss(const Tensor& x) {
    auto path = Forward(x);
    return -nn::mean(BaseLogp(path.latent) + path.logdet, {0});
  }

  /// tflow Evaluate: density, logdets, reconstruction, actual input Jacobian rows and stage fields.
  Map Evaluate(const Tensor& x) {
    Tensor input = x.detach().clone();
    input.requires_grad_();
    const auto path = Forward(input);
    const auto rec  = Inverse(path.latent);
    const auto base = BaseLogp(path.latent);
    const auto logp = base + path.logdet;
    Map        out{{"latent", path.latent},
                   {"unscaled", path.unscaled},
                   {"reconstruction", rec.value},
                   {"base_log_probability", base},
                   {"log_probability", logp},
                   {"forward_logdet", path.logdet},
                   {"inverse_logdet", rec.logdet},
                   {"loss", -nn::mean(logp, {0})}};
    for (const auto& [k, v] : path.fields) out[k] = v;
    for (int k = 0; k < 2; ++k)
      out["jacobian_row" + std::to_string(k)] =
          nn::grad(nn::sum(Col(path.latent, k)), {input}, nn::GradOptions{true, false, false})[0];
    Extra(input, path, out);
    for (auto& [k, v] : out) {
      v = v.detach().contiguous();
      if (v.dtype() == nn::DType::Float32)
        for (float f : v.to_vector<float>()) check::Require(std::isfinite(f), "finite " + k + " field");
    }
    return out;
  }

  /// tflow Sample: latent-only inverse, its roundtrip, actual inverse Jacobian rows and logdet.
  Map Sample(const Tensor& z) {
    Tensor prior = z.detach().clone();
    prior.requires_grad_();
    const auto inv       = Inverse(prior);
    const auto roundtrip = Forward(inv.value).latent;
    Map        out{{"prior", z}, {"samples", inv.value}, {"prior_roundtrip", roundtrip}, {"inverse_logdet", inv.logdet}};
    for (int k = 0; k < 2; ++k)
      out["inverse_jacobian_row" + std::to_string(k)] =
          nn::grad(nn::sum(Col(inv.value, k)), {prior}, nn::GradOptions{true, false, false})[0];
    for (auto& [k, v] : out) v = v.detach().contiguous();
    return out;
  }

  Map Gradient(const Tensor& x) { return replay::Named(store, nn::grad(Loss(x), store.parameters())); }
  void Update(const Tensor& x) { optimizer->minimize(Loss(x)); }
  Map  Arrays() const { return replay::Arrays(store); }
};

// ---- data ----------------------------------------------------------------------------------------

struct Data {
  std::vector<Tensor> pools;  // fit [512, 2], validation [128, 2], test [128, 2] float32
  Tensor              prior;  // [1024, 2] float32
  Map                 arrays; // tflow's data.npz
};

/// kind -1: the banana of the 3.3 examples (2-D base); 0..2: flow_density_2d's banana,
/// eight Gaussians and checkerboard (3-D base, component floor(8 Phi(base2))).
inline double Phi(double z) { return .5 * std::erfc(-z / std::sqrt(2.)); }
inline Data MakeData(int kind) {
  Data d;
  for (int k = 0; k < 3; ++k) {
    const int  n = k ? 128 : 512, seed = k == 0 ? 9203 : k == 1 ? 9209 : 9221;
    const int  dims = kind < 0 ? 2 : 3;
    const auto base = GaussianNoise(n, dims, seed);
    const auto bv   = base.to_vector<double>();
    std::vector<float>        x(n * 2);
    std::vector<std::int32_t> ids(n), components(n);
    for (int b = 0; b < n; ++b) {
      const double u = bv[b * dims], v = bv[b * dims + 1];
      if (kind <= 0) {
        x[b * 2]     = float(u);
        x[b * 2 + 1] = float(v + .5 * (u * u - 1));
      }
      if (kind >= 0) {
        const int component = std::min(7, int(8 * Phi(bv[b * dims + 2])));
        components[b]       = kind ? component : -1;
        if (kind == 1) {
          x[b * 2]     = float(2 * std::cos(2 * M_PI * component / 8) + .15 * u);
          x[b * 2 + 1] = float(2 * std::sin(2 * M_PI * component / 8) + .15 * v);
        } else if (kind == 2) {
          const int row = component / 2, col = 2 * (component % 2) + row % 2;
          x[b * 2]     = float(col - 2 + Phi(u));
          x[b * 2 + 1] = float(row - 2 + Phi(v));
        }
      }
      ids[b] = (kind < 0 ? 0 : kind * 100000) + k * 10000 + b;
    }
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    const Tensor      points = nn::from_vector(x, {n, 2});
    d.arrays[split + "_base"] = base;
    if (kind >= 0) d.arrays[split + "_components"] = nn::from_vector(components, {n});
    d.arrays[split + "_points"]     = points;
    d.arrays[split + "_source_ids"] = nn::from_vector(ids, {n});
    d.pools.push_back(points);
  }
  d.prior                  = Floats(GaussianNoise(1024, 2, 9257));
  d.arrays["sample_prior"] = d.prior;
  return d;
}

inline bool CheckerSupport(double x, double y) {
  if (x < -2 || x >= 2 || y < -2 || y >= 2) return false;
  return (int(std::floor(x + 2)) + int(std::floor(y + 2))) % 2 == 0;
}
/// Analytic log density of each dataset (kind -1 and 0: banana).
inline double TrueLog(double x, double y, int kind) {
  if (kind <= 0) {
    const double z = y - .5 * (x * x - 1);
    return -std::log(2 * M_PI) - .5 * (x * x + z * z);
  }
  if (kind == 2) return CheckerSupport(x, y) ? -std::log(8.) : -INFINITY;
  double logs[8], maximum = -INFINITY;
  for (int j = 0; j < 8; ++j) {
    const double a = x - 2 * std::cos(2 * M_PI * j / 8), b = y - 2 * std::sin(2 * M_PI * j / 8);
    logs[j]        = -std::log(8 * 2 * M_PI * .15 * .15) - .5 * (a * a + b * b) / (.15 * .15);
    maximum        = std::max(maximum, logs[j]);
  }
  double sum = 0;
  for (auto value : logs) sum += std::exp(value - maximum);
  return maximum + std::log(sum);
}

/// tflow's Plot: true / learned density (top), held-out + generated and latent + prior (bottom).
inline void Plot(const fs::path& path, const Tensor& model, const std::vector<double>& truth, const Tensor& heldout,
                 const Tensor& generated, const Tensor& latent, const Tensor& prior, double density_scale) {
  std::vector<float> rgb(768 * 768 * 3, 245.f);
  auto               px = [&](int y, int x, int c) -> float& { return rgb[(std::size_t(y) * 768 + x) * 3 + c]; };
  const auto         mv = model.to_vector<float>();
  for (int panel = 0; panel < 2; ++panel)
    for (int y = 0; y < 384; ++y)
      for (int x = 0; x < 384; ++x) {
        const int    row = (y / 3) * 128 + x / 3;
        const double p = std::exp(panel ? double(mv[row]) : truth[row]), t = std::clamp(p / density_scale, 0., 1.);
        const double color[] = {1 - t, 1 - .7 * t, 1};
        for (int c = 0; c < 3; ++c) px(y, panel * 384 + x, c) = float(std::floor(255 * color[c] + .5));
      }
  for (int panel = 0; panel < 2; ++panel) {
    const int sx = panel * 384, sy = 384;
    for (int j = 0; j < 384; ++j)
      for (int c = 0; c < 3; ++c) px(sy + 192, sx + j, c) = px(sy + j, sx + 192, c) = 190;
  }
  auto scatter = [&](const Tensor& values, int panel, int radius, const std::array<int, 3>& color) {
    const auto v = values.to_vector<float>();
    for (std::size_t b = 0; b < v.size() / 2; ++b) {
      const int x = int(std::floor((double(v[b * 2]) + 4) * 48)), y = int(std::floor((4 - double(v[b * 2 + 1])) * 48));
      if (x < 0 || x >= 384 || y < 0 || y >= 384) continue;
      for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
          if (x + dx >= 0 && x + dx < 384 && y + dy >= 0 && y + dy < 384)
            for (int c = 0; c < 3; ++c) px(384 + y + dy, panel * 384 + x + dx, c) = float(color[c]);
    }
  };
  scatter(generated, 0, 1, {30, 90, 190});
  scatter(heldout, 0, 2, {230, 100, 20});
  scatter(prior, 1, 1, {170, 170, 170});
  scatter(latent, 1, 2, {130, 40, 160});
  nn::save_png(nn::from_vector(rgb, {768, 768, 3}), path.string(), 0, 255);
}

// ---- options ---------------------------------------------------------------------------------------

struct Options {
  int         epochs = 3, threads = 4;
  fs::path    out, tf;
  std::string init = "auto";
  bool        verbose = false;
};

/// epochs=N out=DIR init=auto|tf|nn|PATH tf=DIR threads=N verbose=0|1
inline Options ParseOptions(int argc, char** argv, const std::string& name) {
  Options o;
  o.tf = replay::TfModels(name);
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto        e = a.find('=');
    const std::string usage = "epochs=N out=PATH init=auto|tf|nn|PATH tf=DIR threads=N verbose=0|1";
    if (e == std::string::npos) throw std::invalid_argument(usage);
    const std::string k = a.substr(0, e), v = a.substr(e + 1);
    if (k == "epochs") o.epochs = std::stoi(v);
    else if (k == "out") o.out = v;
    else if (k == "init") o.init = v;
    else if (k == "tf") o.tf = v;
    else if (k == "threads") o.threads = std::stoi(v);
    else if (k == "verbose") o.verbose = v != "0";
    else throw std::invalid_argument(usage);
  }
  check::Require(o.epochs > 0 && o.epochs <= 100, "three cheap default epochs or explicit 1..100 override");
  if (o.out.empty()) o.out = paths::OutputRoot() / "models" / name;
  fs::create_directories(o.out);
  if (o.threads > 0) nn::set_num_threads(o.threads);
  return o;
}

/// The initial arrays to replay (empty: nn's generator).
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

/// max |ours - TF| / (1 + |TF|) over the arrays both records share (-1 when TF's file is missing).
inline double Compare(const fs::path& tf, const std::string& file, const Map& ours, bool verbose = false) {
  if (!fs::exists(tf / file)) {
    std::printf("  vs TF %-30s (no TF record)\n", file.c_str());
    return -1;
  }
  const auto theirs = replay::ToMap(nn::load_npz((tf / file).string()));
  double     m      = 0;
  int        shared = 0;
  std::string worst;
  for (const auto& [k, v] : ours) {
    if (!theirs.count(k)) continue;
    const auto a = v.to_vector<double>(), b = theirs.at(k).to_vector<double>();
    if (a.size() != b.size()) {
      std::printf("  vs TF %-30s %s: size %zu vs %zu\n", file.c_str(), k.c_str(), a.size(), b.size());
      m = INFINITY;
      continue;
    }
    ++shared;
    double d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (std::isinf(a[i]) && a[i] == b[i]) continue;
      d = std::max(d, std::abs(a[i] - b[i]) / (1 + std::abs(b[i])));
    }
    if (verbose) std::printf("    %-34s %.3g\n", k.c_str(), d);
    if (d >= m) worst = k;
    m = std::max(m, d);
  }
  std::printf("  vs TF %-30s %2d/%2zu arrays, max rel diff %.3g%s%s\n", file.c_str(), shared, theirs.size(), m,
              worst.empty() ? "" : " at ", worst.c_str());
  return m;
}
inline double Compare(const fs::path& tf, const fs::path& root, const std::string& file, bool verbose = false) {
  return Compare(tf, file, replay::ToMap(nn::load_npz((root / file).string())), verbose);
}

// ---- the program body --------------------------------------------------------------------------------

struct Spec {
  std::string   name, variant, tag, title, html;
  std::int64_t  parameters = 0, tensors = 0;
  std::uint64_t batch_seed = 0;
  int           kind = -1;              // -1 banana (3.3), 0..2 flow_density_2d datasets
  bool          global_logdet = false;  // NICE: one logdet for every point, no per-point checks
  bool          numerical = false;      // CNF: RK4 inverse is numerical, not closed form
  double        density_scale = .2;
  std::string   protocol;               // source_protocol.json
  std::vector<std::string> extra_files; // fixture npz files to compare with TF's
  std::function<std::unique_ptr<FlowModel>(bool training)> make;
  /// Before training (fixtures, mask inventories).
  std::function<void(const fs::path& root)> before;
  /// After TF's initial arrays were loaded into the training model (Glow re-derives its ActNorm).
  std::function<void(FlowModel& model, const Data& data)> replayed;
  /// After restore, on the deployed model (planar/radial inverse gradients).
  std::function<void(FlowModel& deployed, const Data& data, const fs::path& root)> after;
};

struct Result {
  double test_nll, true_nll, fit_nll, diagnostic;
};

/// The banana protocol shared by the single-dataset examples.
inline std::string BananaProtocol() {
  return "{\"distribution\":\"SYNTHETICanalyticbanana\",\"base\":\"standardGaussian2D\",\"mapping\":\"x0=u0;x1=u1+.5*(u0^2-1)\","
         "\"true_forward_logdet\":0,\"fit\":512,\"validation\":128,\"test\":128,\"seeds\":[9203,9209,9221],\"sample_prior_seed\":9257,"
         "\"sample_prior_count\":1024,\"source_id_offsets\":[0,10000,20000]}\n";
}

}  // namespace flowd
