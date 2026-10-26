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

/// Spatial mean of the outer products of two aligned feature maps: [B, Ca, Cb].
inline nn::Tensor SpatialBilinear(const nn::Tensor& first, const nn::Tensor& second) {
  const std::int64_t batch = first.size(0);
  const nn::Tensor   a = nn::reshape(first, {batch, -1, first.size(-1)}), b = nn::reshape(second, {batch, -1, second.size(-1)});
  return nn::matmul(a, b, true) / double(a.size(1));
}

/// x / sqrt(|x| + epsilon) (smoothed signed square root).
inline nn::Tensor BilinearRoot(const nn::Tensor& input, float epsilon = 1e-5f) {
  return input * nn::rsqrt(nn::abs(input) + double(epsilon));
}

/// Rows divided by max(|row|, 1e-6) (sqrt of a 1e-12 floor on the squared norm).
inline nn::Tensor BilinearUnit(const nn::Tensor& input) {
  return input / nn::sqrt(nn::maximum(nn::sum(nn::square(input), {1}, true), nn::scalar_like(double(1e-12f), input)));
}

struct BilinearResult {
  nn::Tensor              feature_a, feature_b, matrix, descriptor, logits;
  std::vector<nn::Tensor> stages;
};

/// Two independent VGG16 conv stacks (widths / divisor) on rgb - .5, bilinear
/// pooling, root, unit and a dense head "head".
inline BilinearResult BilinearCNN(nn::ParameterStore& p, const nn::Tensor& rgb, int divisor = 1, int classes = 102) {
  if (divisor < 1 || 64 % divisor || classes < 2) throw std::invalid_argument("bilinear CNN divisor must divide64 and classes>=2");
  BilinearResult       result;
  const int            widths[] = {64, 128, 256, 512, 512}, layers[] = {2, 2, 3, 3, 3};
  nn::ConvLayerOptions config;
  config.activation = nn::Activation::Relu;
  config.init       = nn::init::he_normal();
  for (int stream = 0; stream < 2; ++stream) {
    nn::Tensor x = rgb - .5;
    for (int block = 0; block < 5; ++block) {
      for (int layer = 0; layer < layers[block]; ++layer) {
        x = nn::conv2d(p, "stream" + std::to_string(stream) + "/block" + std::to_string(block) + "/conv" + std::to_string(layer), x,
                       widths[block] / divisor, 3, config);
        result.stages.push_back(x);
      }
      if (block < 4) {
        x = nn::max_pool2d(x, 2, 2);
        result.stages.push_back(x);
      }
    }
    (stream == 0 ? result.feature_a : result.feature_b) = x;
  }
  const std::int64_t width = 512 / divisor;
  result.matrix            = SpatialBilinear(result.feature_a, result.feature_b);
  result.stages.push_back(result.matrix);
  result.descriptor = BilinearUnit(BilinearRoot(nn::reshape(result.matrix, {-1, width * width})));
  result.stages.push_back(result.descriptor);
  result.logits = nn::dense(p, "head", result.descriptor, classes);
  result.stages.push_back(result.logits);
  return result;
}

/// -mean(sum(one_hot(labels) * log_softmax(logits))).
inline nn::Tensor FlowerCrossEntropy(const nn::Tensor& logits, const nn::Tensor& labels, int classes) {
  return -nn::mean(nn::sum(nn::one_hot(labels, classes, 1.0, 0.0, logits.dtype()) * nn::log_softmax(logits), {1}));
}

/// Compact (divisor 16) six-way bilinear classifier, Adam .001.
class BilinearModel {
 public:
  nn::ParameterStore             parameters;
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit BilinearModel(bool training = true, std::uint64_t seed = 233) : parameters(seed) {
    {
      nn::NoGradGuard guard;
      BilinearCNN(parameters, nn::zeros({1, 64, 64, 3}), 16, 6);
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(parameters, .001, config);
    }
  }
  double Loss(const FlowerData& data, bool update = false) {
    check::Require(bool(optimizer), "training graph for flower loss");
    double value = 0;
    if (update) {
      const nn::Tensor loss = FlowerCrossEntropy(BilinearCNN(parameters, data.rgb, 16, 6).logits, data.labels, 6);
      value                 = loss.item();
      optimizer->step(nn::grad(loss, parameters.parameters()));
    } else {
      nn::NoGradGuard guard;
      value = FlowerCrossEntropy(BilinearCNN(parameters, data.rgb, 16, 6).logits, data.labels, 6).item();
    }
    check::Require(std::isfinite(value) && value >= 0, "finite flower cross entropy");
    return value;
  }
  nn::Tensor Predict(const nn::Tensor& rgb) {
    nn::NoGradGuard guard;
    return nn::softmax(BilinearCNN(parameters, rgb, 16, 6).logits);
  }
  std::vector<std::vector<float>> Snapshot() const {
    std::vector<std::vector<float>> out;
    for (const auto& v : parameters.parameters()) out.push_back(v.to_vector<float>());
    return out;
  }
};

inline nn::Tensor FlowerPredict(BilinearModel& model, const FlowerData& data) {
  std::vector<nn::Tensor> parts;
  for (std::int64_t n = 0; n < data.rgb.size(0); n += 4) {
    std::vector<std::size_t> rows;
    for (std::int64_t i = n; i < std::min<std::int64_t>(n + 4, data.rgb.size(0)); ++i) rows.push_back(std::size_t(i));
    parts.push_back(model.Predict(FlowerRows(data.rgb, rows)));
  }
  return nn::concat(parts, 0).contiguous();
}

inline void PrintFlowers(const std::string& name, const FlowerMetrics& score) {
  std::cout << " " << name << ":accuracy=" << 100 * score.accuracy << "%,macro=" << 100 * score.macro_accuracy << "%,NLL=" << score.nll
            << ",Brier=" << score.brier << "\n";
}

/// 6x8 contact sheet: green/red border for correct/wrong, probability bars below.
inline void FlowerPreview(const FlowerData& data, const nn::Tensor& probability, const std::filesystem::path& path) {
  const int          H = 6 * 80, W = 8 * 64;
  std::vector<float> canvas(std::size_t(H) * W * 3, 0.f);
  auto view = [&](int y, int x, int c) -> float& { return canvas[(std::size_t(y) * W + std::size_t(x)) * 3 + std::size_t(c)]; };
  const auto labels = data.labels.to_vector<int>();
  const auto p      = probability.to_vector<float>();
  const auto rgb    = data.rgb.to_vector<float>();
  for (int n = 0; n < 48; ++n) {
    const int row = n / 8, column = n % 8, label = labels[n];
    int       predicted = 0;
    for (int c = 1; c < 6; ++c)
      if (p[n * 6 + c] > p[n * 6 + predicted]) predicted = c;
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x)
        for (int c = 0; c < 3; ++c) {
          const bool border = x < 2 || y < 2 || x >= 62 || y >= 62;
          view(row * 80 + y, column * 64 + x, c) =
              border ? (c == (predicted == label ? 1 : 0) ? 255.f : 0.f)
                     : float(std::clamp(int(std::lround(rgb[((std::size_t(n) * 64 + y) * 64 + x) * 3 + c] * 255)), 0, 255));
        }
    for (int c = 0; c < 6; ++c)
      for (int x = 0; x < 8; ++x) {
        const int height = int(std::lround(p[n * 6 + c] * 14));
        for (int y = 0; y < height; ++y) view(row * 80 + 79 - y, column * 64 + c * 10 + x, c == label ? 1 : 2) = 255;
      }
  }
  nn::save_png(nn::from_vector(canvas, {H, W, 3}), path.string(), 0, 255);
}

}  // namespace fine
