// Small helpers the 4.2 sequence-task ports share for working with tflow's
// recorded runs: named-array maps (tflow's `std::map<std::string, tf::Tensor>`
// batches), row selection, exact/approximate comparisons, loading TF's
// initial_arrays.npz into a ParameterStore and reading history.tsv columns.
//
// A tflow run directory (tflow's output/models/<name>) holds initial_arrays.npz,
// first_batch.npz, initial_gradients.npz, history.tsv and model_arrays.npz.
// Replaying the run from TF's own initial weights with the same batch order
// (nn::BatchOrder reproduces tflow's) and the same optimizer checks the port
// numerically, independently of the random initialisation.
#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <datamunge/nn/nn.hpp>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace replay {

namespace nn = datamunge::nn;
using nn::Tensor;
using Map = std::map<std::string, Tensor>;

/// tflow's recorded output directory for an example (its output/models/<name>).
inline std::filesystem::path TfModels(const std::string& name) {
  return paths::DataRoot().parent_path() / "models" / name;
}

inline Tensor Get(const nn::NamedTensors& arrays, const std::string& key) {
  for (const auto& [name, t] : arrays)
    if (name == key) return t;
  throw std::runtime_error("missing array " + key);
}

inline Map ToMap(const nn::NamedTensors& arrays) { return {arrays.begin(), arrays.end()}; }
inline nn::NamedTensors ToNamed(const Map& m) { return {m.begin(), m.end()}; }

/// Rows `rows` (axis 0) of every array.
inline Map Rows(const Map& m, const std::vector<std::int64_t>& rows) {
  Map out;
  for (const auto& [k, v] : m) out[k] = nn::take_rows(v, rows);
  return out;
}

/// {key: arrays["<split>_<key>"]} for each key.
inline Map Split(const nn::NamedTensors& arrays, const std::string& split, const std::vector<std::string>& keys) {
  Map out;
  for (const auto& k : keys) out[k] = Get(arrays, split + "_" + k);
  return out;
}

inline double MaxAbsDiff(const Tensor& a, const Tensor& b) {
  const auto x = a.to_vector<double>(), y = b.to_vector<double>();
  check::Require(x.size() == y.size(), "compared tensors have the same size");
  double m = 0;
  for (std::size_t i = 0; i < x.size(); ++i) m = std::max(m, std::abs(x[i] - y[i]));
  return m;
}

/// Bit-identical values (dtype and shape included).
inline bool SameBits(const Tensor& a, const Tensor& b) {
  if (a.dtype() != b.dtype() || a.shape() != b.shape()) return false;
  auto same = [](const auto& x, const auto& y) {
    return x.size() == y.size() && (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])) == 0);
  };
  switch (a.dtype()) {
    case nn::DType::Float32: return same(a.to_vector<float>(), b.to_vector<float>());
    case nn::DType::Float64: return same(a.to_vector<double>(), b.to_vector<double>());
    case nn::DType::Int32: return same(a.to_vector<std::int32_t>(), b.to_vector<std::int32_t>());
    default: return same(a.to_vector<std::int64_t>(), b.to_vector<std::int64_t>());
  }
}

inline bool SameBits(const Map& a, const Map& b) {
  if (a.size() != b.size()) return false;
  for (const auto& [k, v] : a)
    if (!b.count(k) || !SameBits(v, b.at(k))) return false;
  return true;
}

/// Copies of every parameter, by name.
inline Map Arrays(const nn::ParameterStore& p) {
  Map out;
  for (const auto& name : p.names()) out[name] = p.at(name).detach().clone();
  return out;
}

/// Gradients aligned with p.names(), by name.
inline Map Named(const nn::ParameterStore& p, const std::vector<Tensor>& g) {
  Map out;
  for (std::size_t i = 0; i < g.size(); ++i) out[p.names()[i]] = g[i].defined() ? g[i].detach() : nn::zeros_like(p.parameters()[i]).detach();
  return out;
}

/// Overwrites every parameter of `p` from an .npz (e.g. TF's initial_arrays.npz),
/// which must hold exactly p's names.
inline void LoadInto(nn::ParameterStore& p, const std::filesystem::path& file) {
  const auto arrays = nn::load_npz(file.string());
  check::Require(arrays.size() == p.size(), "array file covers exactly the model's parameters: " + file.string());
  for (const auto& [name, value] : arrays) {
    check::Require(p.contains(name), "array " + name + " is a model parameter");
    p.set(name, nn::cast(value, p.at(name).dtype()));
  }
}

/// max |a[k] - b[k]| over the keys of `a` (all must exist in `b`).
inline double MaxDiff(const Map& a, const Map& b) {
  double m = 0;
  for (const auto& [k, v] : a) {
    check::Require(b.count(k) > 0, "reference has array " + k);
    m = std::max(m, MaxAbsDiff(v, b.at(k)));
  }
  return m;
}
inline double MaxDiff(const Map& a, const std::filesystem::path& file) { return MaxDiff(a, ToMap(nn::load_npz(file.string()))); }

/// Column `column` (0-based) of a tab-separated history file with a header line.
inline std::vector<double> History(const std::filesystem::path& file, int column = 1) {
  std::ifstream       in(file);
  std::string         line;
  std::vector<double> out;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string       cell;
    for (int c = 0; c <= column && std::getline(ss, cell, '\t'); ++c)
      if (c == column) out.push_back(std::stod(cell));
  }
  return out;
}

inline nn::OptimizerConfig Adam() {
  nn::OptimizerConfig c;
  c.kind = nn::OptimizerKind::Adam;
  return c;
}

/// Copies `files` from `from` to `to` when present.
inline void CopyAssets(const std::filesystem::path& from, const std::filesystem::path& to,
                       const std::vector<std::string>& files) {
  for (const auto& f : files)
    if (std::filesystem::exists(from / f))
      std::filesystem::copy_file(from / f, to / f, std::filesystem::copy_options::overwrite_existing);
}

}  // namespace replay
