// SRCNN, ESPCN, VDSR, EDSR and RCAN with explicit subpixel channel ordering,
// and short five-model x2 SR training with honest common-crop bicubic
// comparisons: a port of tflow's common/nn/super_resolution.h and
// common/ml/super_resolution.h (namespace sr; the data/metric half,
// common/ml/sr_data.h, is common/sr_data.hpp).
// TF -> datamunge: parameters are an nn::ParameterStore (seed 17, datamunge's
// generator); SRConv / SRPixelShuffle are reused from
// common/adversarial_super_resolution.hpp (srgan::, identical to tflow's);
// the checkpoint is best.npz; PNGs via nn::save_png; test_outputs.npz via
// nn::save_npz. nn::depth_to_space is TF's DCR order, so (like tflow) the CRD
// shuffle is an explicit reshape/transpose.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "common/adversarial_super_resolution.hpp"
#include "common/sr_data.hpp"
#include <datamunge/nn/super_resolution_models.hpp>

namespace sr {

using datamunge::nn::models::kSRMean;
using datamunge::nn::models::SRChannelAttention;
using datamunge::nn::models::SRChannelGate;
using datamunge::nn::models::SRConfig;
using datamunge::nn::models::SRConv;
using datamunge::nn::models::SRKind;
using datamunge::nn::models::SRLayer;
using datamunge::nn::models::SRName;
using datamunge::nn::models::SROutputs;
using datamunge::nn::models::SRPixelShuffle;
using datamunge::nn::models::SRRcab;
using datamunge::nn::models::SRResidual;
using datamunge::nn::models::SRUpsample;
using datamunge::nn::models::SuperResolution;
using datamunge::nn::models::VdsrClippedGradient;
// ---------------------------------------------------------------- training
class SRModel {
public:
  nn::ParameterStore parameters{17};
  SRConfig config;
  std::unique_ptr<nn::Optimizer> optimizer;
  double peak = 0, rate = .001;
  std::int64_t clipped = 0;
  explicit SRModel(SRKind kind) : config(SRConfig::Compact(kind)) {
    const int side = config.PreUpsampled() ? 32 : 16;
    {
      nn::NoGradGuard guard;
      SuperResolution(parameters, nn::zeros({1, side, side, config.Channels()}), config);
    }
    nn::OptimizerConfig oc;
    oc.kind = kind == SRKind::kVdsr ? nn::OptimizerKind::Momentum : nn::OptimizerKind::Adam;
    rate = kind == SRKind::kVdsr ? .1 : .001;
    optimizer = std::make_unique<nn::Optimizer>(parameters, rate, oc);
  }
  Tensor Loss(const Tensor &x, const Tensor &y) {
    const Tensor diff = SuperResolution(parameters, x, config).image - y;
    return config.kind == SRKind::kEdsr || config.kind == SRKind::kRcan
               ? nn::mean(nn::abs(diff))
               : nn::mean(nn::square(diff));
  }
  double Run(const Tensor &x, const Tensor &y, bool update = false) {
    double value = 0;
    if (update) {
      const Tensor loss = Loss(x, y);
      value = loss.item();
      auto gradients = nn::grad(loss, parameters.parameters());
      if (config.kind == SRKind::kVdsr) {
        const double bound = double(.01f / float(rate));
        for (auto &g : gradients) {
          peak = std::max(peak, nn::max(nn::abs(g)).item());
          clipped += nn::count_nonzero(nn::greater(nn::abs(g), bound)).item<std::int64_t>();
          g = VdsrClippedGradient(g, rate);
        }
      }
      optimizer->step(gradients);
    } else {
      nn::NoGradGuard guard;
      value = Loss(x, y).item();
    }
    check::Require(std::isfinite(value) && value >= 0, "finite reconstruction loss");
    return value;
  }
  Tensor Predict(const Tensor &x) {
    nn::NoGradGuard guard;
    const Tensor out = SuperResolution(parameters, x, config).image;
    check::Require(nn::all(nn::is_finite(out)).item<bool>(), "finite raw SR pixels");
    return out;
  }
  void Save(const std::string &path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i)
      arrays.push_back({parameters.names()[i], parameters.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string &path) {
    nn::NoGradGuard guard;
    const auto arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size(), "complete SR checkpoint");
    for (const auto &[name, value] : arrays)
      parameters.set(name, value);
  }
};

inline std::vector<std::int64_t> SRRows(std::int64_t start, std::int64_t end) {
  std::vector<std::int64_t> rows;
  for (std::int64_t i = start; i < end; ++i)
    rows.push_back(i);
  return rows;
}
inline double SRLoss(SRModel &model, const SRPrepared &data) {
  double sum = 0;
  for (std::int64_t row = 0; row < data.x.size(0); row += 8) {
    const auto rows = SRRows(row, std::min<std::int64_t>(row + 8, data.x.size(0)));
    sum += model.Run(SRBatch(data.x, rows), SRBatch(data.y, rows)) * double(rows.size());
  }
  return sum / double(data.x.size(0));
}
inline Tensor SRPredict(SRModel &model, const SRPrepared &data) {
  std::vector<Tensor> parts;
  for (std::int64_t row = 0; row < data.x.size(0); row += 8) {
    const auto rows = SRRows(row, std::min<std::int64_t>(row + 8, data.x.size(0)));
    const Tensor out = model.Predict(SRBatch(data.x, rows));
    check::Require(out.size(0) == std::int64_t(rows.size()) &&
                       out.numel() ==
                           std::int64_t(rows.size()) * (data.y.numel() / data.y.size(0)),
                   "SR image alignment");
    parts.push_back(out);
  }
  return nn::concat(parts, 0).contiguous();
}

inline void SRPreview(const SRData &data, const Tensor &prediction,
                      const std::filesystem::path &path) {
  const auto tiles = nn::models::SRPreviewTiles(data, prediction);
  nn::save_image_grid(nn::slice_axis(tiles, 0, 0, std::min<std::int64_t>(8, tiles.size(0))),
                      path.string(), 1, 0, 1, 0, 0, 4);
}

struct SRRunSummary {
  double initial = 0, learned = 0, best = 0, seconds = 0;
  int selected = -1;
  SRMetrics metrics;
};

} // namespace sr
