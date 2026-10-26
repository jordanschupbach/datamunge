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

namespace depth {

class DepthModel {
 public:
  nn::ParameterStore             parameters{139};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit DepthModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      MonocularDepth(parameters, nn::zeros({1, 48, 64, 3}));
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  std::array<double, 4> Loss(const DepthData& data, bool update = false) {
    check::Require(bool(optimizer), "depth objectives exist only in training model");
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const auto            losses = DepthObjective(MonocularDepth(parameters, data.rgb), data.depth, data.valid);
    std::array<double, 4> result;
    for (int i = 0; i < 4; ++i) {
      result[std::size_t(i)] = losses[std::size_t(i)].item();
      check::Require(std::isfinite(result[std::size_t(i)]) && result[std::size_t(i)] >= 0, "finite nonnegative every depth loss component");
    }
    if (update) optimizer->step(nn::grad(losses[0], parameters.parameters()));
    return result;
  }
  std::pair<nn::Tensor, nn::Tensor> Predict(const nn::Tensor& image) {
    nn::NoGradGuard guard;
    const auto      net = MonocularDepth(parameters, image);
    const nn::Tensor d = net.depth.contiguous(), c = net.coarse.contiguous();
    for (const auto& value : {d, c}) {
      check::Require(value.shape() == nn::Shape({image.size(0), 48, 64, 1}), "aligned coarse/refined depth shapes");
      for (float v : value.to_vector<float>()) check::Require(std::isfinite(v) && v >= .1f, "positive finite raw depth in meters");
    }
    return {d, c};
  }
  std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto& v : parameters.parameters()) out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string& path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i) arrays.push_back({parameters.names()[i], parameters.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string& path) {
    nn::NoGradGuard guard;
    const auto      arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size(), "complete monocular checkpoint");
    for (const auto& [name, value] : arrays) parameters.set(name, value);
  }
};

inline std::vector<std::size_t> DepthRows(std::int64_t start, std::int64_t end) {
  std::vector<std::size_t> rows;
  for (std::int64_t i = start; i < end; ++i) rows.push_back(std::size_t(i));
  return rows;
}

inline double DepthValidation(DepthModel& model, const DepthData& data) {
  double             sum = 0;
  const std::int64_t n   = data.rgb.size(0);
  for (std::int64_t row = 0; row < n; row += 4) {
    const auto rows = DepthRows(row, std::min<std::int64_t>(row + 4, n));
    sum += double(rows.size()) * model.Loss(DepthBatch(data, rows))[0];
  }
  return sum / double(n);
}

inline std::pair<nn::Tensor, nn::Tensor> DepthPredict(DepthModel& model, const DepthData& data) {
  std::vector<nn::Tensor> depth, coarse;
  const std::int64_t      n = data.rgb.size(0);
  for (std::int64_t row = 0; row < n; row += 4) {
    const auto batch = model.Predict(DepthBatch(data.rgb, DepthRows(row, std::min<std::int64_t>(row + 4, n))));
    depth.push_back(batch.first), coarse.push_back(batch.second);
  }
  return {nn::concat(depth, 0).contiguous(), nn::concat(coarse, 0).contiguous()};
}

inline std::array<double, 3> DepthColor(double depth) {
  const double t = std::clamp((depth - .1) / 9.9, 0., 1.);
  return {std::clamp(1.5 - std::abs(4 * t - 3), 0., 1.), std::clamp(1.5 - std::abs(4 * t - 2), 0., 1.), std::clamp(1.5 - std::abs(4 * t - 1), 0., 1.)};
}

/// RGB | sensor depth | prediction | |error|/5 (red) | valid mask, 2x scaled.
inline void DepthPreview(const DepthData& data, const nn::Tensor& prediction, const std::filesystem::path& path) {
  const int  rows = int(std::min<std::int64_t>(8, data.rgb.size(0)));
  const auto rgb = data.rgb.to_vector<float>(), d = data.depth.to_vector<float>(), v = data.valid.to_vector<float>(), p = prediction.to_vector<float>();
  std::vector<float> image(std::size_t(rows) * 96 * 640 * 3);
  for (int n = 0; n < rows; ++n)
    for (int y = 0; y < 48; ++y)
      for (int x = 0; x < 64; ++x) {
        const std::size_t q     = std::size_t((n * 48 + y) * 64 + x);
        const bool        valid = v[q] > 0;
        const double      truth = d[q], estimated = p[q];
        const auto        target_color = DepthColor(truth), predicted_color = DepthColor(estimated);
        for (int c = 0; c < 3; ++c) {
          const double values[] = {rgb[q * 3 + std::size_t(c)], valid ? target_color[std::size_t(c)] : 0, predicted_color[std::size_t(c)],
                                   valid && c == 0 ? std::abs(estimated - truth) / 5 : 0, valid ? 1. : 0.};
          for (int col = 0; col < 5; ++col)
            for (int dy = 0; dy < 2; ++dy)
              for (int dx = 0; dx < 2; ++dx)
                image[(std::size_t((n * 48 + y) * 2 + dy) * 640 + std::size_t((col * 64 + x) * 2 + dx)) * 3 + std::size_t(c)] =
                    float(std::lround(255 * std::clamp(values[col], 0., 1.)));
        }
      }
  nn::save_png(nn::from_vector(image, {rows * 96, 640, 3}), path.string(), 0, 255);
}

inline void PrintDepthMetrics(const char* label, const DepthMetrics& metrics) {
  std::cout << " " << label << ":AbsRel=" << metrics.abs_rel << ",SqRel=" << metrics.sq_rel << ",RMSE=" << metrics.rmse << "m,logRMSE=" << metrics.log_rmse
            << ",log10=" << metrics.log10 << ",SILog=" << metrics.silog << ",delta1/2/3=" << metrics.delta1 << "/" << metrics.delta2 << "/"
            << metrics.delta3 << " (" << metrics.images << " rooms," << metrics.pixels << " observed pixels)\n";
}

}  // namespace depth
