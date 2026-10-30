// Short RGB-only metric depth learning on real room-disjoint sensor pairs: a
// port of tflow's common/ml/monocular_depth.h (the network of
// common/nn/monocular_depth.h lives in depth_losses.hpp, which the losses need).
// TF -> datamunge: the deployment model is a second DepthModel built without an
// optimizer or depth/mask inputs (TF inspects GraphDef placeholders); the
// checkpoint is best.npz with the parameter names.
#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include "common/depth_data.hpp"
#include "common/depth_losses.hpp"
#include <datamunge/nn/nn.hpp>

namespace depth {

class DepthModel {
public:
  nn::ParameterStore parameters{139};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit DepthModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      MonocularDepth(parameters, nn::zeros({1, 48, 64, 3}));
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  std::array<double, 4> Loss(const DepthData &data, bool update = false) {
    check::Require(bool(optimizer), "depth objectives exist only in training model");
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update)
      guard = std::make_unique<nn::NoGradGuard>();
    const auto losses =
        DepthObjective(MonocularDepth(parameters, data.rgb), data.depth, data.valid);
    std::array<double, 4> result;
    for (int i = 0; i < 4; ++i) {
      result[std::size_t(i)] = losses[std::size_t(i)].item();
      check::Require(std::isfinite(result[std::size_t(i)]) && result[std::size_t(i)] >= 0,
                     "finite nonnegative every depth loss component");
    }
    if (update)
      optimizer->step(nn::grad(losses[0], parameters.parameters()));
    return result;
  }
  std::pair<nn::Tensor, nn::Tensor> Predict(const nn::Tensor &image) {
    nn::NoGradGuard guard;
    const auto net = MonocularDepth(parameters, image);
    const nn::Tensor d = net.depth.contiguous(), c = net.coarse.contiguous();
    for (const auto &value : {d, c}) {
      check::Require(value.shape() == nn::Shape({image.size(0), 48, 64, 1}),
                     "aligned coarse/refined depth shapes");
      for (float v : value.to_vector<float>())
        check::Require(std::isfinite(v) && v >= .1f, "positive finite raw depth in meters");
    }
    return {d, c};
  }
  std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto &v : parameters.parameters())
      out.push_back(v.to_vector<float>());
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
    check::Require(arrays.size() == parameters.size(), "complete monocular checkpoint");
    for (const auto &[name, value] : arrays)
      parameters.set(name, value);
  }
};

inline std::vector<std::size_t> DepthRows(std::int64_t start, std::int64_t end) {
  std::vector<std::size_t> rows;
  for (std::int64_t i = start; i < end; ++i)
    rows.push_back(std::size_t(i));
  return rows;
}

inline double DepthValidation(DepthModel &model, const DepthData &data) {
  double sum = 0;
  const std::int64_t n = data.rgb.size(0);
  for (std::int64_t row = 0; row < n; row += 4) {
    const auto rows = DepthRows(row, std::min<std::int64_t>(row + 4, n));
    sum += double(rows.size()) * model.Loss(DepthBatch(data, rows))[0];
  }
  return sum / double(n);
}

inline std::pair<nn::Tensor, nn::Tensor> DepthPredict(DepthModel &model,
                                                      const DepthData &data) {
  std::vector<nn::Tensor> depth, coarse;
  const std::int64_t n = data.rgb.size(0);
  for (std::int64_t row = 0; row < n; row += 4) {
    const auto batch = model.Predict(
        DepthBatch(data.rgb, DepthRows(row, std::min<std::int64_t>(row + 4, n))));
    depth.push_back(batch.first), coarse.push_back(batch.second);
  }
  return {nn::concat(depth, 0).contiguous(), nn::concat(coarse, 0).contiguous()};
}

using datamunge::nn::models::DepthColor;
/// Tensor-rendered source five-panel sensor/prediction preview,first eight rooms.
inline void DepthPreview(const DepthData &data, const nn::Tensor &prediction,
                         const std::filesystem::path &path) {
  const auto ids = DepthRows(0, std::min<std::int64_t>(8, data.rgb.size(0)));
  nn::save_image_grid(datamunge::nn::models::DepthPreviewTiles(DepthBatch(data, ids),
                                                               DepthBatch(prediction, ids)),
                      path.string(), 1, 0, 1, 0, 0, 2);
}

inline void PrintDepthMetrics(const char *label, const DepthMetrics &metrics) {
  std::cout << " " << label << ":AbsRel=" << metrics.abs_rel << ",SqRel=" << metrics.sq_rel
            << ",RMSE=" << metrics.rmse << "m,logRMSE=" << metrics.log_rmse
            << ",log10=" << metrics.log10 << ",SILog=" << metrics.silog
            << ",delta1/2/3=" << metrics.delta1 << "/" << metrics.delta2 << "/"
            << metrics.delta3 << " (" << metrics.images << " rooms," << metrics.pixels
            << " observed pixels)\n";
}

} // namespace depth
