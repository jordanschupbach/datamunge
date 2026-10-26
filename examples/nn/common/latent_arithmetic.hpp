// Port of tflow's common/{nn,ml}/latent_arithmetic.h: batch latent interpolation and explicit
// directional arithmetic (no projection or clipping of codes); fitting-label centroids define
// explicit supervised posthoc directions, not unsupervised semantic factors.
#pragma once

#include "common/autoencoder.hpp"

namespace ae {

/// [B, D] endpoints and [T] weights -> [B, T, D] codes (1 - t) a + t b (float32 or float64).
inline nn::Tensor LatentLerp(const nn::Tensor& a, const nn::Tensor& b, const nn::Tensor& alpha) {
  auto t = nn::reshape(alpha, {1, -1, 1});
  return (1.0 - t) * nn::expand_dims(a, 1) + t * nn::expand_dims(b, 1);
}

/// [B, D] sources, [B, D] directions and [T] scales -> [B, T, D] codes source + s direction.
inline nn::Tensor LatentDirection(const nn::Tensor& source, const nn::Tensor& direction, const nn::Tensor& scale) {
  return nn::expand_dims(source, 1) + nn::reshape(scale, {1, -1, 1}) * nn::expand_dims(direction, 1);
}

/// Per-digit mean [10, D] of float32 codes (double accumulation), every class required.
inline nn::Tensor LatentClassMeans(const nn::Tensor& codes, const nn::Tensor& labels) {
  const auto         d = codes.size(1), n = codes.size(0);
  const auto         x = codes.contiguous().to_vector<float>();
  const auto         l = labels.contiguous().to_vector<int>();
  std::vector<float> means(static_cast<std::size_t>(10 * d), 0.f);
  for (int digit = 0; digit < 10; ++digit) {
    int                 count = 0;
    std::vector<double> sum(static_cast<std::size_t>(d));
    for (std::int64_t b = 0; b < n; ++b)
      if (l[b] == digit) {
        ++count;
        for (std::int64_t j = 0; j < d; ++j) sum[j] += x[b * d + j];
      }
    check::Require(count > 0, "every fitting class has records for centroid direction");
    for (std::int64_t j = 0; j < d; ++j) means[digit * d + j] = float(sum[j] / count);
  }
  return nn::from_vector(means, {10, d});
}

/// [B, T, D] -> [B T, D] for the decoder.
inline nn::Tensor LatentFlatten(const nn::Tensor& codes) {
  check::Require(codes.dim() == 3, "flatten complete batch/path axes for decoder");
  return nn::reshape(codes, {codes.size(0) * codes.size(1), codes.size(2)}).contiguous();
}

}  // namespace ae
