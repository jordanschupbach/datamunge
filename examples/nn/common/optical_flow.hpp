// Shared short supervised optical-flow training and pair-only restored
// inference: a port of tflow's common/ml/optical_flow.h (flownets, pwcnet).
// TF -> datamunge: parameters are an nn::ParameterStore ("flow", seed 163,
// datamunge's generator); the checkpoint is best.npz; the deployment model
// is a second FlowModel without an optimizer (TF inspects the GraphDef for
// "two image inputs only"); the preview PNG is written with nn::save_png.
#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <set>

#include "common/flow_data.hpp"
#include "common/flow_losses.hpp"
#include <datamunge/nn/flow_visualization.hpp>
#include <datamunge/nn/nn.hpp>

namespace flow {

using FlowBuilder = std::function<FlowResult(nn::ParameterStore &, const nn::Tensor &first,
                                             const nn::Tensor &second)>;

class FlowModel {
public:
  nn::ParameterStore parameters{163};
  FlowBuilder builder;
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit FlowModel(FlowBuilder build, bool training = true) : builder(std::move(build)) {
    {
      nn::NoGradGuard guard;
      builder(parameters, nn::zeros({1, 64, 64, 3}), nn::zeros({1, 64, 64, 3}));
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  /// Inference-mode {total, level2..level6}; the example owns training/autograd/Adam.
  std::array<double, 6> Loss(const FlowPrepared &data) {
    nn::NoGradGuard guard;
    const auto net = builder(parameters, data.data.first, data.data.second);
    const auto losses = FlowObjective(net, data.target.flow, data.target.valid);
    std::array<double, 6> result;
    for (std::size_t i = 0; i < 6; ++i) {
      result[i] = losses[i].item();
      check::Require(std::isfinite(result[i]) && result[i] >= -1e-7,
                     "finite every smooth endpoint objective");
    }
    return result;
  }
  nn::Tensor Predict(const nn::Tensor &image1, const nn::Tensor &image2) {
    nn::NoGradGuard guard;
    const nn::Tensor out = builder(parameters, image1, image2).flow;
    check::Require(out.shape() == nn::Shape({image1.size(0), 64, 64, 2}),
                   "input-pixel dx/dy field shape");
    for (float v : out.to_vector<float>())
      check::Require(std::isfinite(v), "finite signed physical optical flow");
    return out;
  }
  std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto &v : parameters.parameters())
      out.push_back(v.to_vector<float>());
    return out;
  }
};

inline std::vector<std::size_t> FlowRows(std::int64_t start, std::int64_t end) {
  std::vector<std::size_t> rows;
  for (std::int64_t i = start; i < end; ++i)
    rows.push_back(std::size_t(i));
  return rows;
}

inline double FlowValidation(FlowModel &model, const FlowPrepared &data) {
  double total = 0;
  for (std::int64_t row = 0; row < data.data.first.size(0); row += 2) {
    const auto rows = FlowRows(row, std::min<std::int64_t>(row + 2, data.data.first.size(0)));
    total += double(rows.size()) * model.Loss(FlowBatch(data, rows))[0];
  }
  return total / double(data.data.first.size(0));
}

inline nn::Tensor FlowPredict(FlowModel &model, const FlowData &data) {
  std::vector<nn::Tensor> parts;
  for (std::int64_t row = 0; row < data.first.size(0); row += 2) {
    const auto rows = FlowRows(row, std::min<std::int64_t>(row + 2, data.first.size(0)));
    parts.push_back(model.Predict(FlowBatch(data.first, rows), FlowBatch(data.second, rows)));
  }
  return nn::concat(parts, 0).contiguous();
}

using datamunge::nn::models::FlowColor;
/// Tensor-rendered source eight-row scene-interleaved optical-flow comparison.
inline void FlowPreview(const FlowData &data, const nn::Tensor &prediction,
                        const nn::Tensor &warped, const std::filesystem::path &path) {
  std::vector<std::size_t> ids;
  for (int row = 0; row < 8; ++row)
    ids.push_back((row % 3) * 8 + row / 3);
  nn::save_image_grid(datamunge::nn::models::FlowPreviewTiles(FlowBatch(data, ids),
                                                              FlowBatch(prediction, ids),
                                                              FlowBatch(warped, ids)),
                      path.string(), 1, 0, 1, 0, 0, 2);
}

inline void PrintFlowMetrics(const char *label, const FlowMetrics &value) {
  std::cout << " " << label << ":EPE=" << value.epe << " input-px,AAE=" << value.aae
            << "deg,bad1/bad3=" << 100 * value.bad1 << "/" << 100 * value.bad3
            << "%,Fl-all=" << 100 * value.fl_all << "% (" << value.images << " patches,"
            << value.pixels << " GT pixels)\n";
}

} // namespace flow
