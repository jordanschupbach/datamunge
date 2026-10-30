// Lightness-only encoder-decoder colorizer with spatial skips and global
// context, plus its short Lab-regression training harness: a port of tflow's
// common/nn/colorization.h and common/ml/colorization.h.
// TF -> datamunge: the deployment model is a second ColorModel built without
// an optimizer or target (TF inspects the GraphDef placeholders for the same
// property); the checkpoint is best.npz with the parameter names.
#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "common/colorization_data.hpp"
#include <datamunge/nn/nn.hpp>

namespace color {

using datamunge::nn::models::Colorization;
using datamunge::nn::models::ColorizationLoss;
using datamunge::nn::models::ColorizationResult;
using datamunge::nn::models::ColorResize;

class ColorModel {
public:
  nn::ParameterStore parameters{31};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit ColorModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      Colorization(parameters, nn::zeros({1, 32, 32, 1}));
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  double Loss(const nn::Tensor &l, const nn::Tensor &ab, bool update = false) {
    check::Require(bool(optimizer), "loss only in training model");
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update)
      guard = std::make_unique<nn::NoGradGuard>();
    const nn::Tensor loss = ColorizationLoss(Colorization(parameters, l).chroma, ab);
    const double value = loss.item();
    check::Require(std::isfinite(value) && value >= 0, "finite chroma objective");
    if (update)
      optimizer->step(nn::grad(loss, parameters.parameters()));
    return value;
  }
  nn::Tensor Predict(const nn::Tensor &l) {
    nn::NoGradGuard guard;
    const nn::Tensor out = Colorization(parameters, l).chroma.contiguous();
    check::Require(out.shape() == nn::Shape({l.size(0), 32, 32, 2}),
                   "colorization prediction shape");
    for (float v : out.to_vector<float>())
      check::Require(std::isfinite(v) && std::abs(v) <= 1, "bounded finite chroma");
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
    check::Require(arrays.size() == parameters.size(), "complete colorizer checkpoint");
    for (const auto &[name, value] : arrays)
      parameters.set(name, value);
  }
  std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto &v : parameters.parameters())
      out.push_back(v.to_vector<float>());
    return out;
  }
};

inline std::vector<std::size_t> ColorRows(std::int64_t start, std::int64_t end) {
  std::vector<std::size_t> rows;
  for (std::int64_t i = start; i < end; ++i)
    rows.push_back(std::size_t(i));
  return rows;
}

inline double ColorLoss(ColorModel &model, const ColorData &data) {
  double sum = 0;
  const std::int64_t n = data.lightness.size(0);
  for (std::int64_t row = 0; row < n; row += 8) {
    const auto rows = ColorRows(row, std::min<std::int64_t>(row + 8, n));
    sum += double(rows.size()) *
           model.Loss(ColorBatch(data.lightness, rows), ColorBatch(data.chroma, rows));
  }
  return sum / double(n);
}

inline nn::Tensor ColorPredict(ColorModel &model, const ColorData &data) {
  std::vector<nn::Tensor> parts;
  const std::int64_t n = data.lightness.size(0);
  for (std::int64_t row = 0; row < n; row += 8)
    parts.push_back(model.Predict(
        ColorBatch(data.lightness, ColorRows(row, std::min<std::int64_t>(row + 8, n)))));
  return nn::concat(parts, 0).contiguous();
}

/// Tensor-rendered grayscale/mean/learned/target preview,first eight rows.
inline void ColorPreview(const ColorData &data, const nn::Tensor &learned,
                         const nn::Tensor &gray, const nn::Tensor &mean,
                         const std::filesystem::path &path) {
  const auto ids = ColorRows(0, std::min<std::int64_t>(8, data.rgb.size(0)));
  ColorData d{ColorBatch(data.lightness, ids), ColorBatch(data.chroma, ids),
              ColorBatch(data.rgb, ids)};
  nn::save_image_grid(datamunge::nn::models::ColorPreviewTiles(d, ColorBatch(learned, ids),
                                                               ColorBatch(gray, ids),
                                                               ColorBatch(mean, ids)),
                      path.string(), 1, 0, 1, 0, 0, 3);
}
} // namespace color
