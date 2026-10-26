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

namespace flow {

using FlowBuilder = std::function<FlowResult(nn::ParameterStore&, const nn::Tensor& first, const nn::Tensor& second)>;

class FlowModel {
 public:
  nn::ParameterStore             parameters{163};
  FlowBuilder                    builder;
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit FlowModel(FlowBuilder build, bool training = true) : builder(std::move(build)) {
    {
      nn::NoGradGuard guard;
      builder(parameters, nn::zeros({1, 64, 64, 3}), nn::zeros({1, 64, 64, 3}));
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  /// {total, level2..level6}; `update` takes one Adam step on the total.
  std::array<double, 6> Loss(const FlowPrepared& data, bool update = false) {
    check::Require(bool(optimizer), "flow objective requires training graph");
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const auto            net    = builder(parameters, data.data.first, data.data.second);
    const auto            losses = FlowObjective(net, data.target.flow, data.target.valid);
    std::array<double, 6> result;
    for (std::size_t i = 0; i < 6; ++i) {
      result[i] = losses[i].item();
      check::Require(std::isfinite(result[i]) && result[i] >= -1e-7, "finite every smooth endpoint objective");
    }
    if (update) optimizer->step(nn::grad(losses[0], parameters.parameters()));
    return result;
  }
  nn::Tensor Predict(const nn::Tensor& image1, const nn::Tensor& image2) {
    nn::NoGradGuard  guard;
    const nn::Tensor out = builder(parameters, image1, image2).flow;
    check::Require(out.shape() == nn::Shape({image1.size(0), 64, 64, 2}), "input-pixel dx/dy field shape");
    for (float v : out.to_vector<float>()) check::Require(std::isfinite(v), "finite signed physical optical flow");
    return out;
  }
  std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto& v : parameters.parameters()) out.push_back(v.to_vector<float>());
    return out;
  }
};

inline std::vector<std::size_t> FlowRows(std::int64_t start, std::int64_t end) {
  std::vector<std::size_t> rows;
  for (std::int64_t i = start; i < end; ++i) rows.push_back(std::size_t(i));
  return rows;
}

inline double FlowValidation(FlowModel& model, const FlowPrepared& data) {
  double total = 0;
  for (std::int64_t row = 0; row < data.data.first.size(0); row += 2) {
    const auto rows = FlowRows(row, std::min<std::int64_t>(row + 2, data.data.first.size(0)));
    total += double(rows.size()) * model.Loss(FlowBatch(data, rows))[0];
  }
  return total / double(data.data.first.size(0));
}

inline nn::Tensor FlowPredict(FlowModel& model, const FlowData& data) {
  std::vector<nn::Tensor> parts;
  for (std::int64_t row = 0; row < data.first.size(0); row += 2) {
    const auto rows = FlowRows(row, std::min<std::int64_t>(row + 2, data.first.size(0)));
    parts.push_back(model.Predict(FlowBatch(data.first, rows), FlowBatch(data.second, rows)));
  }
  return nn::concat(parts, 0).contiguous();
}

/// Middlebury-style colour wheel, saturation |v| / 4 px.
inline std::array<double, 3> FlowColor(double u, double v) {
  const double saturation = std::min(1., std::hypot(u, v) / 4), hue = (std::atan2(v, u) / 6.2831853071795864769 + 1) * 6;
  const double part = hue - std::floor(hue), p = 1 - saturation, q = 1 - saturation * part, t = 1 - saturation * (1 - part);
  switch (int(std::floor(hue)) % 6) {
    case 0: return {1, t, p};
    case 1: return {q, 1, p};
    case 2: return {p, 1, t};
    case 3: return {p, q, 1};
    case 4: return {t, p, 1};
    default: return {1, p, q};
  }
}

/// 8 rows x (frame1 | frame2 | GT colour | predicted colour | EPE/5 | warped frame2), 2x scaled.
inline void FlowPreview(const FlowData& data, const nn::Tensor& prediction, const nn::Tensor& warped, const std::filesystem::path& path) {
  const int  H = 1024, W = 768;
  const auto va = data.valid.to_vector<float>(), fl = data.flow.to_vector<float>(), pr = prediction.to_vector<float>(),
             f1 = data.first.to_vector<float>(), f2 = data.second.to_vector<float>(), wa = warped.to_vector<float>();
  std::vector<float> image(std::size_t(H) * W * 3);
  for (int row = 0; row < 8; ++row) {
    const int n = (row % 3) * 8 + row / 3;
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x) {
        const std::size_t px    = std::size_t((n * 64 + y) * 64 + x);
        const bool        valid = va[px] > 0;
        const double      u = fl[px * 2], v = fl[px * 2 + 1], pu = pr[px * 2], pv = pr[px * 2 + 1];
        const auto        truth_color = FlowColor(u, v), estimate = FlowColor(pu, pv);
        for (int c = 0; c < 3; ++c) {
          const double values[] = {f1[px * 3 + std::size_t(c)], f2[px * 3 + std::size_t(c)], valid ? truth_color[std::size_t(c)] : 0,
                                   estimate[std::size_t(c)],    valid && c == 0 ? std::hypot(pu - u, pv - v) / 5 : 0,
                                   wa[px * 3 + std::size_t(c)]};
          for (int col = 0; col < 6; ++col)
            for (int dy = 0; dy < 2; ++dy)
              for (int dx = 0; dx < 2; ++dx)
                image[(std::size_t((row * 64 + y) * 2 + dy) * W + std::size_t((col * 64 + x) * 2 + dx)) * 3 + std::size_t(c)] =
                    float(std::lround(255 * std::clamp(values[col], 0., 1.)));
        }
      }
  }
  nn::save_png(nn::from_vector(image, {H, W, 3}), path.string(), 0, 255);
}

inline void PrintFlowMetrics(const char* label, const FlowMetrics& value) {
  std::cout << " " << label << ":EPE=" << value.epe << " input-px,AAE=" << value.aae << "deg,bad1/bad3=" << 100 * value.bad1 << "/"
            << 100 * value.bad3 << "%,Fl-all=" << 100 * value.fl_all << "% (" << value.images << " patches," << value.pixels << " GT pixels)\n";
}

}  // namespace flow
