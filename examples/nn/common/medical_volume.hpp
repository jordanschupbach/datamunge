// Compact residual 3D CNN for real 28-cube CT nodule classification
// (NoduleMNIST3D subset): model, data, short training and controls.
// A port of tflow's common/nn/medical_volume.h and common/ml/medical_volume.h.
//
// TF -> datamunge: the model is a function of an nn::ParameterStore plus a
// "state" store holding the two fit-only normalization scalars; Conv3DLayer is
// the nn::conv3d layer (He uniform, SAME); TF's head dropout takes a keep
// probability (.8), nn::dropout a rate (.2). Session checkpoints are .npz
// files and the "deployed" model is a fresh pair of stores restored from them.
// nn's .npz reader widens the uint8 voxels to int32, so the loader checks
// their integer range instead of the uint8 dtype.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common/c3d.hpp"
#include "common/checks.hpp"
#include "common/paths.hpp"

namespace medical_volume {

namespace nn = datamunge::nn;
using nn::Padding;
using nn::Tensor;

// ---- model (tflow common/nn/medical_volume.h) -----------------------------------------

struct MedicalVolumeOutput {
  Tensor              normalized, global, logits;
  std::vector<Tensor> stages;
};

/// Custom residual block (not a named medical model): relu(conv3x3x3 s2) -> conv3x3x3,
/// plus a 1x1x1 stride-2 projection, then ReLU. "<name>/{first,second,projection}/{W,b}".
inline Tensor MedicalResidual(nn::ParameterStore& p, const std::string& name, const Tensor& input, std::int64_t out) {
  nn::ConvLayerOptions opt;
  opt.init   = nn::init::he_uniform();
  opt.stride = 2;
  const Tensor first = nn::relu(nn::conv3d(p, name + "/first", input, out, 3, opt));
  opt.stride         = 1;
  const Tensor second = nn::conv3d(p, name + "/second", first, out, 3, opt);
  opt.stride          = 2;
  const Tensor shortcut = nn::conv3d(p, name + "/projection", input, out, 1, opt);
  return nn::relu(second + shortcut);
}

/// Three residual stages (base, 2 base, 4 base) on [N, 28, 28, 28, 1] intensities, global
/// average pool and a two-logit head. With `training`, head dropout (keep .8) from `dropout_gen`.
inline MedicalVolumeOutput MedicalVolume(nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& volume,
                                         int base = 4, bool training = false, nn::Generator* dropout_gen = nullptr) {
  if (base < 1) throw std::invalid_argument("positive medical volume width");
  MedicalVolumeOutput out;
  const Tensor mean   = state.get("normalization/mean", {}, nn::init::zeros()).detach();
  const Tensor stddev = state.get("normalization/stddev", {}, nn::init::ones()).detach();
  out.normalized      = (volume - mean) / stddev;
  Tensor x            = out.normalized;
  for (int i = 0; i < 3; ++i) {
    x = MedicalResidual(p, "stage" + std::to_string(i), x, std::int64_t{base} << i);
    out.stages.push_back(x);
  }
  out.global = nn::mean(x, {1, 2, 3});
  x          = out.global;
  if (training) x = nn::dropout(x, .2, true, *dropout_gen);
  out.logits = nn::dense(p, "head", x, 2);
  return out;
}

// ---- data (tflow common/ml/medical_volume.h) ----------------------------------------------

struct MedicalData {
  Tensor voxels, labels, source;
  [[nodiscard]] std::int64_t size() const { return voxels.size(0); }
};

inline MedicalData MedicalRows(const MedicalData& d, const std::vector<std::int64_t>& rows) {
  return {nn::take_rows(d.voxels, rows), nn::take_rows(d.labels, rows), nn::take_rows(d.source, rows)};
}

/// fit (64) / validation (16) / test (64) balanced pools of medical_volume.npz, with tflow's
/// checks: official source-ID split boundaries, disjoint IDs, no byte-identical cubes.
inline std::vector<MedicalData> LoadMedical(const std::filesystem::path& root) {
  const auto file = root / "medical_volume.npz";
  check::Require(std::filesystem::exists(file), "missing " + file.string() +
                                                    "; run `just medical-volume-assets` in tflow_examples, or pass data=PATH");
  std::map<std::string, Tensor> arrays;
  for (auto& [name, t] : nn::load_npz(file.string())) arrays[name] = t;
  std::vector<MedicalData> result;
  std::set<int>            ids;
  std::set<std::vector<int>> unique;
  for (const std::string split : {"fit", "validation", "test"}) {
    const Tensor&      bytes = arrays.at(split + "_voxels");
    const std::int64_t n     = split == "validation" ? 16 : 64;
    check::Require(bytes.shape() == nn::Shape({n, 28, 28, 28, 1}), "real28cube grayscale CT observations");
    MedicalData d{Tensor(), arrays.at(split + "_labels"), arrays.at(split + "_source_ids")};
    check::Require(d.labels.dtype() == nn::DType::Int32 && d.source.dtype() == nn::DType::Int32 &&
                       d.labels.shape() == nn::Shape({n}) && d.source.shape() == nn::Shape({n}),
                   "aligned CT labels/sourceIDs");
    const auto voxels = nn::cast(bytes, nn::DType::Int32).to_vector<int>();
    const auto labels = d.labels.to_vector<int>(), source = d.source.to_vector<int>();
    int        counts[2] = {};
    for (std::int64_t b = 0; b < n; ++b) {
      const int id = source[static_cast<std::size_t>(b)], label = labels[static_cast<std::size_t>(b)];
      check::Require(label >= 0 && label < 2 && ids.insert(id).second, "valid labels/disjoint sourceIDs");
      check::Require(split == "fit"          ? id >= 0 && id < 1158
                     : split == "validation" ? id >= 1158 && id < 1323
                                             : id >= 1323 && id < 1633,
                     "official source split boundaries");
      ++counts[label];
      const auto begin = voxels.begin() + b * 21952;
      check::Require(unique.insert(std::vector<int>(begin, begin + 21952)).second, "no byte-identical CT cubes across pools");
    }
    check::Require(counts[0] == n / 2 && counts[1] == n / 2, "balanced small CT subset");
    std::vector<float> intensity(voxels.size());
    for (std::size_t i = 0; i < voxels.size(); ++i) {
      check::Require(voxels[i] >= 0 && voxels[i] <= 255, "uint8 CT intensities");
      intensity[i] = static_cast<float>(voxels[i]) / 255.f;
    }
    d.voxels = nn::from_vector(intensity, bytes.shape());
    result.push_back(d);
  }
  return result;
}

// ---- model wrapper, scoring and controls ---------------------------------------------------

/// Parameters seed 1049, normalization store seed 1051, dropout seed 1031.
struct MedicalModel {
  nn::ParameterStore parameters{1049}, state{1051};
  nn::Generator      dropout{1031};
  MedicalModel() {
    nn::NoGradGuard guard;  // create all 20 learned and 2 normalization tensors
    MedicalVolume(parameters, state, nn::zeros({1, 28, 28, 28, 1}));
  }
  /// Fit-only intensity moments.
  void Normalize(double mean, double stddev) {
    check::Require(std::isfinite(mean) && std::isfinite(stddev) && stddev > 0, "fitting-only CT moments");
    state.set("normalization/mean", nn::full({}, mean));
    state.set("normalization/stddev", nn::full({}, stddev));
  }
  Tensor Logits(const Tensor& voxels, bool training) {
    return MedicalVolume(parameters, state, voxels, 4, training, &dropout).logits;
  }
  /// Inference logits [N, 2] in batches of 4.
  Tensor Predict(const MedicalData& d) {
    nn::NoGradGuard     guard;
    std::vector<Tensor> parts;
    for (std::int64_t start = 0; start < d.size(); start += 4) {
      std::vector<std::int64_t> rows;
      for (std::int64_t i = start; i < std::min<std::int64_t>(start + 4, d.size()); ++i) rows.push_back(i);
      parts.push_back(Logits(nn::take_rows(d.voxels, rows), false));
    }
    return nn::concat(parts, 0);
  }
  std::vector<double> Snapshot() const {
    auto       values = parameters.flatten();
    const auto stats  = state.flatten();
    values.insert(values.end(), stats.begin(), stats.end());
    return values;
  }
  void Save(const std::filesystem::path& prefix) const {
    parameters.save(prefix.string() + ".npz");
    state.save(prefix.string() + "_state.npz");
  }
  void Restore(const std::filesystem::path& prefix) {
    parameters.load(prefix.string() + ".npz");
    state.load(prefix.string() + "_state.npz");
  }
};

inline void MedicalMetrics(const std::string& name, const MedicalData& d, const Tensor& logits) {
  const auto score = c3d::ScoreLogits(logits, d.labels, 2);
  std::cout << name << ":accuracy=" << 100 * score.accuracy << "%,macroF1=" << score.macro_f1 << ",NLL=" << score.nll
            << ",confusion=";
  for (int v : score.confusion) std::cout << v << ',';
  std::cout << std::endl;
}

/// One-hot 1-nearest-neighbour "logits" on 4x4x4 block-mean intensities of the fit pool.
inline Tensor MedicalBaseline(const MedicalData& fit, const MedicalData& test) {
  const auto features = [](const Tensor& x) {
    const auto                       v = x.to_vector<float>();
    const std::int64_t               n = x.size(0);
    std::vector<std::vector<double>> out(static_cast<std::size_t>(n), std::vector<double>(64));
    for (std::int64_t b = 0; b < n; ++b)
      for (int z = 0; z < 28; ++z)
        for (int y = 0; y < 28; ++y)
          for (int a = 0; a < 28; ++a)
            out[static_cast<std::size_t>(b)][static_cast<std::size_t>((z / 7 * 4 + y / 7) * 4 + a / 7)] +=
                v[static_cast<std::size_t>(((b * 28 + z) * 28 + y) * 28 + a)] / 343;
    return out;
  };
  const auto         a = features(fit.voxels), b = features(test.voxels);
  const auto         labels = fit.labels.to_vector<int>();
  std::vector<float> out(b.size() * 2, 0.f);
  for (std::size_t i = 0; i < b.size(); ++i) {
    double best  = INFINITY;
    int    label = 0;
    for (std::size_t j = 0; j < a.size(); ++j) {
      double distance = 0;
      for (int k = 0; k < 64; ++k) distance += std::pow(a[j][static_cast<std::size_t>(k)] - b[i][static_cast<std::size_t>(k)], 2);
      if (distance < best) {
        best  = distance;
        label = labels[j];
      }
    }
    out[i * 2 + static_cast<std::size_t>(label)] = 1;
  }
  return nn::from_vector(out, {static_cast<std::int64_t>(b.size()), 2});
}

// ---- training (tflow ml::TrainMedical) ---------------------------------------------------

}  // namespace medical_volume
