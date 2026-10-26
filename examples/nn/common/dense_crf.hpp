// Exact dense Gaussian/bilateral Potts CRF mean-field inference on small
// grids (port of tflow's common/ml/dense_crf.h).
// https://arxiv.org/abs/1210.5644
// K_ii = 0; each Gaussian is symmetrically normalized by sqrt(degree_i * degree_j).
// Potts mean field simplifies to Q = softmax(logits + K Q), since the
// all-class message is a per-pixel constant. This is O(pixels^2), explicitly
// bounded to 4096 pixels: a numerical/reference postprocessor, not the
// paper's fast permutohedral filtering. Fixed kernels allow gradients
// through the iterations (built from differentiable nn ops).
#pragma once

#include <algorithm>
#include <cmath>
#include <datamunge/nn/image_io.hpp>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "common/checks.hpp"
#include "common/segmentation.hpp"

namespace seg {

struct DenseCrfConfig {
  float spatial_sigma = 3, appearance_sigma = 10, color_sigma = .1f, spatial_weight = .5f, appearance_weight = 2;
  int   iterations = 5;
};

/// Pairwise kernel [P, P] (float32) of an image [H, W, 3] (float32 values) with P = H W <= 4096.
inline nn::Tensor DenseCrfKernel(const nn::Tensor& image, const DenseCrfConfig& cfg = {}) {
  if (image.dim() != 3 || image.shape()[2] != 3 || cfg.spatial_sigma <= 0 || cfg.appearance_sigma <= 0 ||
      cfg.color_sigma <= 0 || cfg.spatial_weight < 0 || cfg.appearance_weight < 0)
    throw std::invalid_argument("invalid dense CRF image/configuration");
  const int h = int(image.shape()[0]), w = int(image.shape()[1]), count = h * w;
  if (count < 1 || count > 4096) throw std::invalid_argument("exact dense CRF requires1..4096pixels");
  const auto         pixels = image.to(nn::DType::Float32).to_vector<float>();
  std::vector<float> kernel(std::size_t(count) * count, 0.f);
  for (int component = 0; component < 2; ++component) {
    std::vector<double> weights(std::size_t(count) * count), degree(static_cast<std::size_t>(count));
    const double        sigma  = component ? cfg.appearance_sigma : cfg.spatial_sigma,
                 weight = component ? cfg.appearance_weight : cfg.spatial_weight;
    for (int i = 0; i < count; ++i)
      for (int j = i + 1; j < count; ++j) {
        const double dy = i / w - j / w, dx = i % w - j % w;
        double       exponent = (dy * dy + dx * dx) / (2 * sigma * sigma);
        if (component)
          for (int c = 0; c < 3; ++c) {
            const double difference = pixels[std::size_t(i * 3 + c)] - pixels[std::size_t(j * 3 + c)];
            exponent += difference * difference / (2 * double(cfg.color_sigma) * cfg.color_sigma);
          }
        const double value = std::exp(-exponent);
        weights[std::size_t(i) * count + j] = weights[std::size_t(j) * count + i] = value;
        degree[std::size_t(i)] += value;
        degree[std::size_t(j)] += value;
      }
    for (int i = 0; i < count; ++i)
      for (int j = 0; j < count; ++j)
        if (i != j)
          kernel[std::size_t(i) * count + j] += float(
              weight * weights[std::size_t(i) * count + j] /
              std::sqrt(std::max(degree[std::size_t(i)] * degree[std::size_t(j)], 1e-30)));
  }
  return nn::from_vector(kernel, {count, count});
}

/// Q after `iterations` mean-field steps; logits [P, classes], kernel [P, P].
inline nn::Tensor DenseCrfMeanField(const nn::Tensor& logits, const nn::Tensor& kernel, int iterations = 5) {
  if (iterations < 1) throw std::invalid_argument("CRF requires a positive iteration count");
  nn::Tensor probabilities = nn::softmax(logits, -1);
  for (int i = 0; i < iterations; ++i) probabilities = nn::softmax(logits + nn::matmul(kernel, probabilities), -1);
  return probabilities;
}

/// Fixed CRF on 32x32 nearest samples of the first four held-out images:
/// before/after mIoU and an image | truth | raw | CRF preview.
inline void DenseCrfPreview(const LogitsFn& logits_of, const SegmentationData& data,
                            const std::filesystem::path& directory) {
  nn::NoGradGuard           guard;
  constexpr int             size = 32;
  const int                 count = std::min(4, data.count), classes = data.classes;
  std::vector<std::int64_t> before(std::size_t(classes) * classes), after(std::size_t(classes) * classes);
  std::vector<float>        preview(std::size_t(count) * size * 4 * size * 3);
  auto put = [&](int n, int r, int column, int k, float value) {
    preview[((std::size_t(n) * size + r) * 4 * size + std::size_t(column)) * 3 + std::size_t(k)] = value;
  };
  for (int n = 0; n < count; ++n) {
    const auto         batch  = SegmentationBatch(data, {n});
    const auto         logits = logits_of(batch.first).to_vector<float>();
    const auto         image  = batch.first.to_vector<float>();
    const auto         labels = batch.second.to_vector<std::int32_t>();
    std::vector<float> small(std::size_t(size) * size * 3), unary(std::size_t(size) * size * classes);
    std::vector<int>   truth(std::size_t(size) * size), raw(std::size_t(size) * size);
    for (int r = 0; r < size; ++r)
      for (int c = 0; c < size; ++c) {
        const int y = std::min(data.height - 1, int((r + .5) * data.height / size)),
                  x = std::min(data.width - 1, int((c + .5) * data.width / size)), index = r * size + c;
        const std::size_t source = std::size_t(y) * data.width + std::size_t(x);
        truth[std::size_t(index)] = labels[source];
        for (int k = 0; k < 3; ++k) small[std::size_t(index * 3 + k)] = image[source * 3 + std::size_t(k)];
        for (int k = 0; k < classes; ++k)
          unary[std::size_t(index * classes + k)] = logits[source * std::size_t(classes) + std::size_t(k)];
        const float* u = unary.data() + std::size_t(index) * classes;
        raw[std::size_t(index)] = int(std::max_element(u, u + classes) - u);
      }
    const auto kernel  = DenseCrfKernel(nn::from_vector(small, {size, size, 3}));
    const auto refined = DenseCrfMeanField(nn::from_vector(unary, {size * size, classes}), kernel).to_vector<float>();
    for (int r = 0; r < size; ++r)
      for (int c = 0; c < size; ++c) {
        const int    index      = r * size + c;
        const float* value      = refined.data() + std::size_t(index) * classes;
        const int    prediction = int(std::max_element(value, value + classes) - value);
        ++before[std::size_t(truth[std::size_t(index)] * classes + raw[std::size_t(index)])];
        ++after[std::size_t(truth[std::size_t(index)] * classes + prediction)];
        for (int k = 0; k < 3; ++k) {
          put(n, r, c, k, float(std::clamp(std::lround(small[std::size_t(index * 3 + k)] * 255), 0L, 255L)));
          put(n, r, size + c, k, kSegmentationPalette[truth[std::size_t(index)] % 4][k]);
          put(n, r, 2 * size + c, k, kSegmentationPalette[raw[std::size_t(index)] % 4][k]);
          put(n, r, 3 * size + c, k, kSegmentationPalette[prediction % 4][k]);
        }
      }
  }
  nn::save_png(nn::from_vector(preview, {count * size, 4 * size, 3}), (directory / "crf_preview.png").string(), 0, 255);
  std::cout << "  fixed CRF on first" << count << " held-out scenes sampled32x32: mIoU " << ConfusionMeanIou(before, classes)
            << " -> " << ConfusionMeanIou(after, classes)
            << "; image/truth/raw/CRF preview: " << (directory / "crf_preview.png").string() << std::endl;
}

}  // namespace seg
