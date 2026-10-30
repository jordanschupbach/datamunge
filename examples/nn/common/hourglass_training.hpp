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
#include <datamunge/nn/image_io.hpp>
#include <datamunge/nn/pose_visualization.hpp>

namespace pose {

class HourglassModel {
public:
  nn::ParameterStore parameters{173}, state{179};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit HourglassModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      StackedHourglass(parameters, state, nn::zeros({1, 128, 128, 3}),
                       CompactHourglassConfig(), false);
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  /// {total, stack0, stack1}. `training` uses batch BN statistics (without
  /// moving-average updates unless `update`); `update` also takes one Adam step.
  std::array<double, 3> Loss(const PoseData &data, bool update = false,
                             bool training = false) {
    check::Require(bool(optimizer), "pose losses require training graph");
    const auto truth = PoseHeatmaps(data.joints);
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update)
      guard = std::make_unique<nn::NoGradGuard>();
    const auto net = StackedHourglass(parameters, state, data.rgb, CompactHourglassConfig(),
                                      update || training, update);
    const auto l0 = PoseHeatmapMSE(net.heatmaps[0], truth.heatmap, truth.mask, 32),
               l1 = PoseHeatmapMSE(net.heatmaps[1], truth.heatmap, truth.mask, 32);
    const nn::Tensor total = l0 + l1;
    std::array<double, 3> result{total.item(), l0.item(), l1.item()};
    for (double v : result)
      check::Require(std::isfinite(v) && v >= 0, "finite both-stack losses");
    if (update)
      optimizer->step(nn::grad(total, parameters.parameters()));
    return result;
  }
  /// Final-stack heatmaps [N, 32, 32, 17] from RGB alone (moving BN statistics).
  nn::Tensor Predict(const nn::Tensor &images) {
    nn::NoGradGuard guard;
    const nn::Tensor out =
        StackedHourglass(parameters, state, images, CompactHourglassConfig(), false)
            .heatmaps.back()
            .contiguous();
    check::Require(nn::all(nn::is_finite(out)).item<bool>(),
                   "finite unbounded linear heatmap values");
    return out;
  }
  [[nodiscard]] std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto &v : parameters.parameters())
      out.push_back(v.to_vector<float>());
    for (const auto &v : state.parameters())
      out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string &path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i)
      arrays.push_back({"pose/" + parameters.names()[i], parameters.parameters()[i]});
    for (std::size_t i = 0; i < state.size(); ++i)
      arrays.push_back({"bn_state/" + state.names()[i], state.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string &path) {
    nn::NoGradGuard guard;
    const auto arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size() + state.size(),
                   "complete hourglass/BN checkpoint");
    for (const auto &[name, value] : arrays) {
      if (name.rfind("pose/", 0) == 0)
        parameters.set(name.substr(5), value);
      else if (name.rfind("bn_state/", 0) == 0)
        state.set(name.substr(9), value);
      else
        throw std::runtime_error("unexpected hourglass checkpoint array " + name);
    }
  }
};

inline nn::Tensor HourglassPredict(HourglassModel &model, const PoseData &data) {
  std::vector<nn::Tensor> parts;
  for (std::int64_t i = 0; i < data.rgb.size(0); i += 2) {
    std::vector<std::size_t> rows;
    for (std::int64_t j = i; j < std::min<std::int64_t>(i + 2, data.rgb.size(0)); ++j)
      rows.push_back(std::size_t(j));
    parts.push_back(model.Predict(PoseBatch(data.rgb, rows)));
  }
  return nn::concat(parts, 0).contiguous();
}

inline double HourglassValidation(HourglassModel &model, const PoseData &data) {
  double total = 0;
  for (std::int64_t i = 0; i < data.rgb.size(0); ++i)
    total += model.Loss(PoseBatch(data, {std::size_t(i)}))[0];
  return total / double(data.rgb.size(0));
}

inline void PrintPose(const std::string &label, const PoseMetrics &score) {
  std::cout << " " << label << ":PCK@.2box=" << 100 * score.pck
            << "%,NME=" << score.normalized_error << ",mean=" << score.pixels << " crop-px ("
            << score.joints << " annotated joints;COCO1+2)\n";
}

inline void PosePreview(const PoseData &data, const nn::Tensor &decoded,
                        const nn::Tensor &heatmap, const std::filesystem::path &file) {
  nn::save_image_grid(nn::models::PosePreviewTiles(data, decoded, heatmap), file.string(), 1,
                      0, 1, 0, 0, 1);
}
} // namespace pose
