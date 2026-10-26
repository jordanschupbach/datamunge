// Shared support for the 3.5 energy-based model ports (restricted_boltzmann,
// deep_belief, deep_boltzmann, energy_langevin, modern_hopfield): a port of
// tflow's common/ml/binary_energy_data.h (synthetic single-bar corpus, explicit
// mt19937 uniform banks, 12-pixel binary tiles) plus the program helpers those
// examples repeat (option parsing, .npz records, TF record comparison).
//
// Every random draw of these examples is an explicit uniform/Gaussian bank from
// std::mt19937(seed), so the ports reproduce tflow's data, Gibbs uniforms and
// Langevin innovations bit for bit; only the weight initialisation differs
// (nn's generator), which init=auto removes by replaying TF's initial arrays.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/binary_pixels.hpp"

namespace ebm {

namespace nn = datamunge::nn;
namespace fs = std::filesystem;
using nn::Tensor;
using Map = replay::Map;

// ---- random banks and data -------------------------------------------------------------

/// tflow's ml::BinaryUniform: (mt19937() + .5) / 2^32 as [n, dimensions] float32 (float64 when wide).
inline Tensor BinaryUniform(std::int64_t n, std::int64_t dimensions, std::uint32_t seed, bool wide = false) {
  std::vector<double> u(static_cast<std::size_t>(n * dimensions));
  std::mt19937        rng(seed);
  for (auto& x : u) x = (double(rng()) + .5) / 4294967296.;
  const Tensor t = nn::from_vector(u, {n, dimensions});
  return wide ? t : nn::cast(t, nn::DType::Float32);
}

/// tflow's nn::GaussianNoise bank [n, dimensions] (float32, or float64 when wide).
inline Tensor Gaussian(std::int64_t n, std::uint32_t seed, bool wide = false, std::int64_t dimensions = 2) {
  const Tensor raw = nn::reshape(ae::GaussianNoise({n, dimensions}, 1, seed), {n, dimensions});
  return wide ? raw : nn::cast(raw, nn::DType::Float32);
}

inline Tensor Ints(const std::vector<std::int32_t>& v) { return nn::from_vector(v, {std::int64_t(v.size())}); }

/// tflow's ml::BinaryBars: uniformly one of 4 vertical / 2 horizontal bars of a 2x4
/// grid, each bit flipped with probability .05. Keys points/uniform/components/source_ids.
inline Map BinaryBars(std::int64_t n, std::uint32_t seed, int offset) {
  const auto                u = BinaryUniform(n, 9, seed, true).to_vector<double>();
  std::vector<float>        values(static_cast<std::size_t>(n * 8));
  std::vector<std::int32_t> components(n), ids(n);
  for (std::int64_t b = 0; b < n; ++b) {
    const int c   = int(6 * u[b * 9]);
    components[b] = c;
    ids[b]        = offset + int(b);
    for (int j = 0; j < 8; ++j) {
      const bool bit         = c < 4 ? j % 4 == c : j / 4 == c - 4;
      values[b * 8 + j] = float(bit != (u[b * 9 + j + 1] < .05));
    }
  }
  return {{"points", nn::from_vector(values, {n, 8})},
          {"uniform", nn::from_vector(u, {n, 9})},
          {"components", Ints(components)},
          {"source_ids", Ints(ids)}};
}

/// tflow's nn::RbmBinaryStates: all 2^bits binary rows, bit j of row b = (b >> j) & 1.
inline Tensor BinaryStates(int bits) {
  check::Require(bits > 0 && bits <= 12, "tiny exact binary enumeration only");
  std::vector<float> s(static_cast<std::size_t>((1 << bits) * bits));
  for (int b = 0; b < (1 << bits); ++b)
    for (int j = 0; j < bits; ++j) s[b * bits + j] = float((b >> j) & 1);
  return nn::from_vector(s, {std::int64_t(1) << bits, bits});
}

/// tflow's ml::BinaryTiles: every row of `values` [n, bits] as a (bits / width) x width
/// grid of 12-pixel cells, `columns` tiles per row, 4-pixel gray-220 gutters.
inline void BinaryTiles(const fs::path& path, const Tensor& values, int columns, int width = 4) {
  const auto n = values.size(0), bits = values.size(1);
  check::Require(bits % width == 0, "binary tiles rectangular");
  const std::int64_t height = bits / width, px = width * 12 + 4, py = height * 12 + 4;
  const std::int64_t H = ((n + columns - 1) / columns) * py, W = columns * px;
  const auto         v = values.contiguous().to_vector<float>();
  std::vector<float> rgb(static_cast<std::size_t>(H * W * 3), 220.f);
  for (std::int64_t b = 0; b < n; ++b)
    for (std::int64_t y = 0; y < height * 12; ++y)
      for (std::int64_t x = 0; x < width * 12; ++x) {
        const double p = v[static_cast<std::size_t>(b * bits + (y / 12) * width + x / 12)];
        check::Require(std::isfinite(p) && p >= 0 && p <= 1, "binary tiles probabilities");
        for (int c = 0; c < 3; ++c)
          rgb[static_cast<std::size_t>(((b / columns * py + y) * W + b % columns * px + x) * 3 + c)] = float(std::floor(p * 255 + .5));
      }
  bpix::PngBytes(path, rgb, H, W);
}

// ---- records -----------------------------------------------------------------------------

inline Tensor Rows(const Tensor& x, const std::vector<std::int64_t>& rows) { return nn::take_rows(x, rows); }
inline std::vector<std::int64_t> Range(std::int64_t n) {
  std::vector<std::int64_t> r(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) r[i] = i;
  return r;
}
inline void        Save(const fs::path& file, const Map& m) { nn::save_npz(file.string(), replay::ToNamed(m)); }
inline Map         Load(const fs::path& file) { return replay::ToMap(nn::load_npz(file.string())); }
inline std::string Bits(const Tensor& t) { return bpix::Bits(t); }
inline double      Scalar(const Map& m, const std::string& k) { return nn::cast(m.at(k), nn::DType::Float64).item<double>(); }

inline Map Detached(const Map& m) {
  Map out;
  for (const auto& [k, v] : m) out[k] = v.detach();
  return out;
}
inline std::vector<std::string> Snapshot(const Map& arrays) {
  std::vector<std::string> out;
  for (const auto& [k, v] : arrays) out.push_back(Bits(v));
  return out;
}
/// Bit-identical copies of every array of `ours` in TF's record `file` (missing file: true).
inline bool SameAsTf(const fs::path& tf, const std::string& file, const Map& ours) {
  if (!fs::exists(tf / file)) return true;
  const auto theirs = Load(tf / file);
  for (const auto& [k, v] : ours)
    if (!theirs.count(k) || Bits(theirs.at(k)) != Bits(v) || theirs.at(k).shape() != v.shape()) return false;
  return true;
}
/// Fraction of equal elements over the shared keys of `file` (all, or only `keys`;
/// -1 without a TF record). Used for sampled binary states.
inline double Agreement(const fs::path& tf, const std::string& file, const Map& ours, const std::vector<std::string>& keys = {}) {
  if (!fs::exists(tf / file)) return -1;
  const auto  theirs = Load(tf / file);
  std::size_t same = 0, total = 0;
  for (const auto& [k, v] : ours) {
    if (!keys.empty() && std::find(keys.begin(), keys.end(), k) == keys.end()) continue;
    if (!theirs.count(k) || theirs.at(k).numel() != v.numel()) continue;
    const auto a = nn::cast(v, nn::DType::Float64).to_vector<double>(), b = nn::cast(theirs.at(k), nn::DType::Float64).to_vector<double>();
    for (std::size_t i = 0; i < a.size(); ++i) same += a[i] == b[i];
    total += a.size();
  }
  return total ? double(same) / double(total) : -1;
}

inline nn::OptimizerConfig Sgd() { return {}; }
inline nn::OptimizerConfig Adam() {
  nn::OptimizerConfig c;
  c.kind = nn::OptimizerKind::Adam;
  return c;
}

/// Gradients of `loss` w.r.t. every parameter of `p`, by name.
inline Map ParameterGradients(const nn::ParameterStore& p, const Tensor& loss) {
  return replay::Named(p, nn::grad(loss, p.parameters()));
}

/// Overwrites the parameters of `p` from `arrays` (names `prefix + name`).
inline void SetFrom(nn::ParameterStore& p, const Map& arrays, const std::string& prefix = "") {
  for (const auto& name : p.names()) {
    check::Require(arrays.count(prefix + name) > 0, "array " + prefix + name + " present");
    p.set(name, nn::cast(arrays.at(prefix + name), p.at(name).dtype()));
  }
}

/// Finds a frozen checkpoint in `dir`: final.npz (our ports) or model_arrays.npz (TF's restored weights).
inline fs::path FrozenFile(const fs::path& dir) {
  if (fs::exists(dir / "final.npz")) return dir / "final.npz";
  if (fs::exists(dir / "model_arrays.npz")) return dir / "model_arrays.npz";
  return {};
}
/// "ours" (our port's output), "tf" (tflow's recorded run), "auto" (ours when present, else TF) or a directory.
inline fs::path FrozenDir(const std::string& choice, const fs::path& ours, const fs::path& tf) {
  if (choice == "ours") return ours;
  if (choice == "tf") return tf;
  if (choice == "auto") return FrozenFile(ours).empty() ? tf : ours;
  return choice;
}

// ---- options -----------------------------------------------------------------------------

struct Options {
  int                                epochs = 3, threads = 4;
  fs::path                           out, tf;
  std::string                        init = "auto";
  std::map<std::string, std::string> extra;  // source= / data=
};

/// key=value options: epochs out init tf threads + `extra` (name -> default).
inline Options ParseOptions(int argc, char** argv, const std::string& name, const std::map<std::string, std::string>& extra,
                            const std::string& usage) {
  Options o;
  o.tf    = replay::TfModels(name);
  o.extra = extra;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto        e = a.find('=');
    if (e == std::string::npos) throw std::invalid_argument(usage);
    const std::string k = a.substr(0, e), v = a.substr(e + 1);
    if (k == "epochs") o.epochs = std::stoi(v);
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

/// Bit-identity of our data record with TF's data.npz on `keys`.
inline void CheckData(const fs::path& tf, const Map& data, const std::vector<std::string>& keys) {
  if (!fs::exists(tf / "data.npz")) return;
  const auto theirs = Load(tf / "data.npz");
  for (const auto& k : keys)
    check::Require(theirs.count(k) && Bits(theirs.at(k)) == Bits(data.at(k)), "data bit-identical to TF's data.npz: " + k);
  std::cout << "data: " << keys.size() << " arrays bit-identical to TF's data.npz\n";
}

/// Prints ours vs TF per epoch for history column `column`.
inline void CompareHistory(const fs::path& ours, const fs::path& tf, int column, const std::string& what) {
  if (!fs::exists(tf / "history.tsv")) return;
  const auto a = replay::History(ours / "history.tsv", column), b = replay::History(tf / "history.tsv", column);
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    std::printf("  epoch %zu %s ours %.7f TF %.7f\n", i + 1, what.c_str(), a[i], b[i]);
}

}  // namespace ebm
