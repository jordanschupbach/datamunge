// Short supervised hourglass training, BN restoration and image-only deployment:
// a port of tflow's common/ml/hourglass.h (stacked_hourglass).
// TF -> datamunge: weights live in nn::ParameterStore "pose" (seed 173) and the
// moving BN statistics in a second store "bn_state" (seed 179), both with
// tflow's names and datamunge's generator. A training step runs the
// training-mode network once (BN moving averages update during that forward,
// as TF's update ops do in the same session run) and takes one Adam step.
// The checkpoint is best.npz ("pose/<name>" and "bn_state/<name>") instead of
// a SaveV2 bundle; the deployment model is a second HourglassModel built
// without an optimizer whose Predict() takes RGB alone (TF inspects the
// GraphDef placeholders). The preview PNG is written with nn::save_png.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common/pose_data.hpp"

namespace pose {

inline HourglassConfig CompactHourglassConfig() {
  HourglassConfig config;
  config.size  = 128;
  config.width = 32;
  config.stem  = 8;
  return config;
}

class HourglassModel {
 public:
  nn::ParameterStore             parameters{173}, state{179};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit HourglassModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      StackedHourglass(parameters, state, nn::zeros({1, 128, 128, 3}), CompactHourglassConfig(), false);
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  /// {total, stack0, stack1}. `training` uses batch BN statistics (without
  /// moving-average updates unless `update`); `update` also takes one Adam step.
  std::array<double, 3> Loss(const PoseData& data, bool update = false, bool training = false) {
    check::Require(bool(optimizer), "pose losses require training graph");
    const auto                       truth = PoseHeatmaps(data.joints);
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const auto net = StackedHourglass(parameters, state, data.rgb, CompactHourglassConfig(), update || training, update);
    const auto l0 = PoseHeatmapMSE(net.heatmaps[0], truth.heatmap, truth.mask, 32), l1 = PoseHeatmapMSE(net.heatmaps[1], truth.heatmap, truth.mask, 32);
    const nn::Tensor      total = l0 + l1;
    std::array<double, 3> result{total.item(), l0.item(), l1.item()};
    for (double v : result) check::Require(std::isfinite(v) && v >= 0, "finite both-stack losses");
    if (update) optimizer->step(nn::grad(total, parameters.parameters()));
    return result;
  }
  /// Final-stack heatmaps [N, 32, 32, 17] from RGB alone (moving BN statistics).
  nn::Tensor Predict(const nn::Tensor& images) {
    nn::NoGradGuard  guard;
    const nn::Tensor out = StackedHourglass(parameters, state, images, CompactHourglassConfig(), false).heatmaps.back().contiguous();
    for (float v : out.to_vector<float>()) check::Require(std::isfinite(v), "finite unbounded linear heatmap values");
    return out;
  }
  [[nodiscard]] std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto& v : parameters.parameters()) out.push_back(v.to_vector<float>());
    for (const auto& v : state.parameters()) out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string& path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i) arrays.push_back({"pose/" + parameters.names()[i], parameters.parameters()[i]});
    for (std::size_t i = 0; i < state.size(); ++i) arrays.push_back({"bn_state/" + state.names()[i], state.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string& path) {
    nn::NoGradGuard guard;
    const auto      arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size() + state.size(), "complete hourglass/BN checkpoint");
    for (const auto& [name, value] : arrays) {
      if (name.rfind("pose/", 0) == 0)
        parameters.set(name.substr(5), value);
      else if (name.rfind("bn_state/", 0) == 0)
        state.set(name.substr(9), value);
      else
        throw std::runtime_error("unexpected hourglass checkpoint array " + name);
    }
  }
};

inline nn::Tensor HourglassPredict(HourglassModel& model, const PoseData& data) {
  std::vector<nn::Tensor> parts;
  for (std::int64_t i = 0; i < data.rgb.size(0); i += 2) {
    std::vector<std::size_t> rows;
    for (std::int64_t j = i; j < std::min<std::int64_t>(i + 2, data.rgb.size(0)); ++j) rows.push_back(std::size_t(j));
    parts.push_back(model.Predict(PoseBatch(data.rgb, rows)));
  }
  return nn::concat(parts, 0).contiguous();
}

inline double HourglassValidation(HourglassModel& model, const PoseData& data) {
  double total = 0;
  for (std::int64_t i = 0; i < data.rgb.size(0); ++i) total += model.Loss(PoseBatch(data, {std::size_t(i)}))[0];
  return total / double(data.rgb.size(0));
}

inline void PrintPose(const std::string& label, const PoseMetrics& score) {
  std::cout << " " << label << ":PCK@.2box=" << 100 * score.pck << "%,NME=" << score.normalized_error << ",mean=" << score.pixels << " crop-px ("
            << score.joints << " annotated joints;COCO1+2)\n";
}

inline void PosePreview(const PoseData& data, const nn::Tensor& decoded, const nn::Tensor& heatmap, const std::filesystem::path& file) {
  const std::int64_t count = data.rgb.size(0);
  const auto         rgb = data.rgb.to_vector<float>(), heat = heatmap.to_vector<float>(), joints = data.joints.to_vector<float>(),
             dec = decoded.to_vector<float>();
  std::vector<float> pixels(std::size_t(count * 128 * 512 * 3), 0.f);
  auto view = [&](std::int64_t y, std::int64_t x, int c) -> float& { return pixels[std::size_t((y * 512 + x) * 3 + c)]; };
  for (std::int64_t n = 0; n < count; ++n)
    for (int y = 0; y < 128; ++y)
      for (int x = 0; x < 128; ++x) {
        for (int col = 0; col < 3; ++col)
          for (int c = 0; c < 3; ++c)
            view(n * 128 + y, col * 128 + x, c) = float(std::clamp(int(std::lround(255 * rgb[std::size_t(((n * 128 + y) * 128 + x) * 3 + c)])), 0, 255));
        float maximum = -INFINITY;
        for (int j = 0; j < 17; ++j) maximum = std::max(maximum, heat[std::size_t(((n * 32 + y / 4) * 32 + x / 4) * 17 + j)]);
        const float v = std::clamp(maximum, 0.f, 1.f);
        view(n * 128 + y, 384 + x, 0) = float(std::lround(255 * v));
        view(n * 128 + y, 384 + x, 1) = 0;
        view(n * 128 + y, 384 + x, 2) = float(std::lround(255 * (1 - v)));
      }
  const int edges[][2] = {{5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 6}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}, {0, 1}, {1, 3}, {0, 2}, {2, 4}, {0, 5}, {0, 6}};
  auto dot = [&](std::int64_t n, int col, long x, long y, int channel) {
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx)
        if (x + dx >= 0 && x + dx < 128 && y + dy >= 0 && y + dy < 128)
          for (int c = 0; c < 3; ++c) view(n * 128 + y + dy, col * 128 + x + dx, c) = c == channel ? 255.f : 0.f;
  };
  for (std::int64_t n = 0; n < count; ++n)
    for (int kind = 0; kind < 2; ++kind) {
      const auto& values = kind ? dec : joints;
      auto        at     = [&](int j, int c) { return double(values[std::size_t((n * 17 + j) * 3 + c)]); };
      for (const auto& edge : edges) {
        if (!kind && (at(edge[0], 2) <= 0 || at(edge[1], 2) <= 0)) continue;
        const double x0 = at(edge[0], 0), y0 = at(edge[0], 1), x1 = at(edge[1], 0), y1 = at(edge[1], 1);
        const int    steps = std::max(1, int(std::ceil(std::hypot(x1 - x0, y1 - y0))));
        for (int i = 0; i <= steps; ++i) dot(n, kind + 1, std::lround(x0 + (x1 - x0) * i / steps), std::lround(y0 + (y1 - y0) * i / steps), kind ? 0 : 1);
      }
      for (int j = 0; j < 17; ++j)
        if (kind || at(j, 2) > 0) dot(n, kind + 1, std::lround(at(j, 0)), std::lround(at(j, 1)), kind ? 0 : 1);
    }
  nn::save_png(nn::from_vector(pixels, {count * 128, 512, 3}), file.string(), 0, 255);
}

}  // namespace pose
