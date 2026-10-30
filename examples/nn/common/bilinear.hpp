// Two-stream VGG16 bilinear pooling, smoothed signed-root and unit
// normalization, and the short two-stream bilinear flower training with
// RGB-only restoration: a port of tflow's common/nn/bilinear.h and
// common/ml/bilinear.h.
// TF -> datamunge: checkpoints are .npz parameter stores (best.npz); the
// deployment model is a second BilinearModel without an optimizer (TF checks
// the GraphDef placeholders for "RGB only"); PNGs use nn::save_png.
#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "common/fine_grained.hpp"

namespace fine {

using nn::models::BilinearCNN;
using nn::models::BilinearResult;
using nn::models::BilinearRoot;
using nn::models::BilinearUnit;
using nn::models::FlowerCrossEntropy;
using nn::models::SpatialBilinear;
/// Compact (divisor 16) six-way bilinear classifier, Adam .001.
class BilinearModel {
public:
  nn::ParameterStore parameters;
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit BilinearModel(bool training = true, std::uint64_t seed = 233) : parameters(seed) {
    {
      nn::NoGradGuard guard;
      BilinearCNN(parameters, nn::zeros({1, 64, 64, 3}), 16, 6);
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  double Loss(const FlowerData &data, bool update = false) {
    check::Require(bool(optimizer), "training graph for flower loss");
    double value = 0;
    if (update) {
      const nn::Tensor loss =
          FlowerCrossEntropy(BilinearCNN(parameters, data.rgb, 16, 6).logits, data.labels, 6);
      value = loss.item();
      optimizer->step(nn::grad(loss, parameters.parameters()));
    } else {
      nn::NoGradGuard guard;
      value =
          FlowerCrossEntropy(BilinearCNN(parameters, data.rgb, 16, 6).logits, data.labels, 6)
              .item();
    }
    check::Require(std::isfinite(value) && value >= 0, "finite flower cross entropy");
    return value;
  }
  nn::Tensor Predict(const nn::Tensor &rgb) {
    nn::NoGradGuard guard;
    return nn::softmax(BilinearCNN(parameters, rgb, 16, 6).logits);
  }
  std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto &v : parameters.parameters())
      out.push_back(v.to_vector<float>());
    return out;
  }
};

inline nn::Tensor FlowerPredict(BilinearModel &model, const FlowerData &data) {
  std::vector<nn::Tensor> parts;
  for (std::int64_t n = 0; n < data.rgb.size(0); n += 4) {
    std::vector<std::size_t> rows;
    for (std::int64_t i = n; i < std::min<std::int64_t>(n + 4, data.rgb.size(0)); ++i)
      rows.push_back(std::size_t(i));
    parts.push_back(model.Predict(FlowerRows(data.rgb, rows)));
  }
  return nn::concat(parts, 0).contiguous();
}

inline void PrintFlowers(const std::string &name, const FlowerMetrics &score) {
  std::cout << " " << name << ":accuracy=" << 100 * score.accuracy
            << "%,macro=" << 100 * score.macro_accuracy << "%,NLL=" << score.nll
            << ",Brier=" << score.brier << "\n";
}

/// 6x8 contact sheet: green/red border for correct/wrong, probability bars below.
inline void FlowerPreview(const FlowerData &data, const nn::Tensor &probability,
                          const std::filesystem::path &path) {
  nn::save_image_grid(nn::models::FlowerPredictionTiles(data, probability), path.string(), 8,
                      0, 1, 0);
}
} // namespace fine
