// Short DnCNN training with fresh AWGN, validation-selected BN state and
// deployment checks: a port of tflow's common/ml/denoising.h.
// TF -> datamunge: the checkpoint is one .npz holding "dncnn/<name>" weights
// and "statistics/<name>" moving BN state (TF SaveV2 bundle "best"); the
// deployment model is a second DnModel without an optimizer, and the
// "one noisy input, inference-only BN, no optimizer" graph inspection becomes
// the corresponding structural checks on that model.
#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>

#include "common/cifar.hpp"
#include "common/denoising_data.hpp"
#include "common/dncnn.hpp"

namespace denoise {

class DnModel {
 public:
  nn::ParameterStore             parameters{17}, state{18};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit DnModel(bool train = true) {
    {
      nn::NoGradGuard guard;
      DnCNN(parameters, state, nn::zeros({1, 32, 32, 1}), 16, false);
    }
    if (train) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  /// Residual half-MSE; train_mode uses batch statistics, `update` also steps Adam and the moving BN state.
  double Loss(const DnData& batch, bool train_mode = false, bool update = false) {
    check::Require(bool(optimizer) && (!update || train_mode), "training graph required for residual objective/update");
    double loss = 0;
    if (update) {
      const nn::Tensor objective = DnResidualLoss(DnCNN(parameters, state, batch.noisy, 16, true, true).noise, batch.noise);
      loss = objective.item();
      optimizer->step(nn::grad(objective, parameters.parameters()));
    } else {
      nn::NoGradGuard guard;
      loss = DnResidualLoss(DnCNN(parameters, state, batch.noisy, 16, train_mode, false).noise, batch.noise).item();
    }
    check::Require(std::isfinite(loss) && loss >= 0, "finite nonnegative noise objective");
    return loss;
  }
  nn::Tensor Predict(const nn::Tensor& noisy) {
    nn::NoGradGuard  guard;
    const nn::Tensor out = DnCNN(parameters, state, noisy, 16, false).clean;
    check::Require(out.shape() == noisy.shape(), "unchanged denoising coordinates");
    for (float v : out.to_vector<float>()) check::Require(std::isfinite(v), "finite raw restored pixels");
    return out;
  }
  std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto& v : parameters.parameters()) out.push_back(v.to_vector<float>());
    for (const auto& v : state.parameters()) out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string& path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i) arrays.push_back({"dncnn/" + parameters.names()[i], parameters.parameters()[i]});
    for (std::size_t i = 0; i < state.size(); ++i) arrays.push_back({"statistics/" + state.names()[i], state.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string& path) {
    nn::NoGradGuard guard;
    const auto      arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size() + state.size(), "complete DnCNN checkpoint");
    for (const auto& [name, value] : arrays) {
      if (name.rfind("dncnn/", 0) == 0)
        parameters.set(name.substr(6), value);
      else if (name.rfind("statistics/", 0) == 0)
        state.set(name.substr(11), value);
      else
        throw std::runtime_error("unexpected DnCNN checkpoint array " + name);
    }
  }
};

inline std::vector<std::size_t> Range(std::int64_t start, std::int64_t end) {
  std::vector<std::size_t> rows;
  for (std::int64_t i = start; i < end; ++i) rows.push_back(std::size_t(i));
  return rows;
}

inline double DnLoss(DnModel& model, const DnData& data) {
  double sum = 0;
  for (std::int64_t row = 0; row < data.clean.size(0); row += 8) {
    const auto rows = Range(row, std::min<std::int64_t>(row + 8, data.clean.size(0)));
    sum += double(rows.size()) * model.Loss(DnBatch(data, rows));
  }
  return sum / double(data.clean.size(0));
}

inline nn::Tensor DnPredict(DnModel& model, const nn::Tensor& noisy) {
  std::vector<nn::Tensor> parts;
  for (std::int64_t row = 0; row < noisy.size(0); row += 8)
    parts.push_back(model.Predict(DnBatch(noisy, Range(row, std::min<std::int64_t>(row + 8, noisy.size(0))))));
  return nn::concat(parts, 0).contiguous();
}

/// noisy | box | prediction | clean | 4x |error| (red), 3x scaled, 8 rows.
inline void DnPreview(const DnData& data, const nn::Tensor& restored, const nn::Tensor& box, const std::filesystem::path& path) {
  const int          rows = int(std::min<std::int64_t>(8, data.clean.size(0))), scale = 3, H = rows * 32 * scale, W = 5 * 32 * scale;
  std::vector<float> image(std::size_t(H) * W * 3);
  const auto c = data.clean.to_vector<float>(), p = restored.to_vector<float>(), n = data.noisy.to_vector<float>(), b = box.to_vector<float>();
  for (int r = 0; r < rows; ++r)
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x) {
        const std::size_t i     = std::size_t((r * 32 + y) * 32 + x);
        const double      clean = c[i], predicted = p[i];
        const double      values[] = {n[i], b[i], predicted, clean, 4 * std::abs(predicted - clean)};
        for (int col = 0; col < 5; ++col)
          for (int dy = 0; dy < scale; ++dy)
            for (int dx = 0; dx < scale; ++dx)
              for (int ch = 0; ch < 3; ++ch)
                image[(std::size_t((r * 32 + y) * scale + dy) * W + std::size_t((col * 32 + x) * scale + dx)) * 3 + std::size_t(ch)] =
                    float(std::lround(255 * std::clamp(col == 4 && ch != 0 ? 0. : values[col], 0., 1.)));
      }
  nn::save_png(nn::from_vector(image, {H, W, 3}), path.string(), 0, 255);
}

}  // namespace denoise
