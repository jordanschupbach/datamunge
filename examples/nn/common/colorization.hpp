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

namespace color {

/// TF ResizeBilinear(half_pixel_centers) of `input` to the spatial size of `like`.
inline nn::Tensor ColorResize(const nn::Tensor& input, const nn::Tensor& like) {
  return nn::resize_bilinear(input, {like.size(1), like.size(2)}, false, true);
}

struct ColorizationResult {
  nn::Tensor              chroma;
  std::vector<nn::Tensor> stages;
};

/// Input L/50-1 [N,H,W,1], output tanh(a/128, b/128). No target/color/label input.
inline ColorizationResult Colorization(nn::ParameterStore& p, const nn::Tensor& lightness) {
  ColorizationResult result;
  auto conv = [&](const std::string& name, const nn::Tensor& x, std::int64_t out, std::int64_t stride = 1, std::int64_t kernel = 3) {
    nn::ConvLayerOptions o;
    o.activation = nn::Activation::Relu;
    o.stride     = stride;
    auto y       = nn::conv2d(p, name, x, out, kernel, o);
    result.stages.push_back(y);
    return y;
  };
  auto shallow = conv("encode1", lightness, 16);
  auto middle  = conv("encode2", shallow, 32, 2);
  auto deep    = conv("encode3", middle, 64, 2);
  auto context = conv("bottleneck1", deep, 64);
  context      = conv("bottleneck2", context, 64);
  auto pooled  = nn::mean(context, {1, 2}, true);
  auto global  = conv("global", pooled, 32, 1, 1);
  auto fused   = conv("fusion", nn::concat({context, ColorResize(global, context)}, 3), 64, 1, 1);
  auto decoded = conv("decode2", nn::concat({ColorResize(fused, middle), middle}, 3), 32);
  decoded      = conv("decode1", nn::concat({ColorResize(decoded, shallow), shallow}, 3), 16);
  result.chroma = nn::tanh(nn::conv2d(p, "chroma", decoded, 2, 1));
  result.stages.push_back(result.chroma);
  return result;
}

inline nn::Tensor ColorizationLoss(const nn::Tensor& prediction, const nn::Tensor& target) {
  return nn::mean(nn::square(prediction - target));
}

class ColorModel {
 public:
  nn::ParameterStore             parameters{31};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit ColorModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      Colorization(parameters, nn::zeros({1, 32, 32, 1}));
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  double Loss(const nn::Tensor& l, const nn::Tensor& ab, bool update = false) {
    check::Require(bool(optimizer), "loss only in training model");
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const nn::Tensor loss  = ColorizationLoss(Colorization(parameters, l).chroma, ab);
    const double     value = loss.item();
    check::Require(std::isfinite(value) && value >= 0, "finite chroma objective");
    if (update) optimizer->step(nn::grad(loss, parameters.parameters()));
    return value;
  }
  nn::Tensor Predict(const nn::Tensor& l) {
    nn::NoGradGuard  guard;
    const nn::Tensor out = Colorization(parameters, l).chroma.contiguous();
    check::Require(out.shape() == nn::Shape({l.size(0), 32, 32, 2}), "colorization prediction shape");
    for (float v : out.to_vector<float>()) check::Require(std::isfinite(v) && std::abs(v) <= 1, "bounded finite chroma");
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
    check::Require(arrays.size() == parameters.size(), "complete colorizer checkpoint");
    for (const auto& [name, value] : arrays) parameters.set(name, value);
  }
  std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto& v : parameters.parameters()) out.push_back(v.to_vector<float>());
    return out;
  }
};

inline std::vector<std::size_t> ColorRows(std::int64_t start, std::int64_t end) {
  std::vector<std::size_t> rows;
  for (std::int64_t i = start; i < end; ++i) rows.push_back(std::size_t(i));
  return rows;
}

inline double ColorLoss(ColorModel& model, const ColorData& data) {
  double             sum = 0;
  const std::int64_t n   = data.lightness.size(0);
  for (std::int64_t row = 0; row < n; row += 8) {
    const auto rows = ColorRows(row, std::min<std::int64_t>(row + 8, n));
    sum += double(rows.size()) * model.Loss(ColorBatch(data.lightness, rows), ColorBatch(data.chroma, rows));
  }
  return sum / double(n);
}

inline nn::Tensor ColorPredict(ColorModel& model, const ColorData& data) {
  std::vector<nn::Tensor> parts;
  const std::int64_t      n = data.lightness.size(0);
  for (std::int64_t row = 0; row < n; row += 8) parts.push_back(model.Predict(ColorBatch(data.lightness, ColorRows(row, std::min<std::int64_t>(row + 8, n)))));
  return nn::concat(parts, 0).contiguous();
}

/// grayscale | fit-mean chroma | learned | truth, 3x scaled, 8 rows.
inline void ColorPreview(const ColorData& data, const nn::Tensor& learned, const nn::Tensor& gray, const nn::Tensor& mean,
                         const std::filesystem::path& path) {
  const int  rows = int(std::min<std::int64_t>(8, data.rgb.size(0))), scale = 3, W = 4 * 32 * scale;
  const std::vector<float> columns[] = {gray.to_vector<float>(), mean.to_vector<float>(), learned.to_vector<float>(), data.rgb.to_vector<float>()};
  std::vector<float> image(std::size_t(rows) * 32 * scale * W * 3);
  for (int n = 0; n < rows; ++n)
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x)
        for (int col = 0; col < 4; ++col)
          for (int c = 0; c < 3; ++c)
            for (int dy = 0; dy < scale; ++dy)
              for (int dx = 0; dx < scale; ++dx)
                image[(std::size_t((n * 32 + y) * scale + dy) * W + std::size_t((col * 32 + x) * scale + dx)) * 3 + std::size_t(c)] =
                    float(std::lround(255 * std::clamp(columns[col][std::size_t(((n * 32 + y) * 32 + x) * 3 + c)], 0.f, 1.f)));
  nn::save_png(nn::from_vector(image, {rows * 32 * scale, W, 3}), path.string(), 0, 255);
}

}  // namespace color
