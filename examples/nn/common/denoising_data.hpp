// Genuine natural clean images plus unbounded, independently seeded Gaussian
// noise: a port of tflow's common/ml/denoising_data.h. Grayscale images are
// float32 [N, 32, 32, 1] computed in double from CIFAR uint8 RGB (int32 here,
// see common/cifar.hpp); the noise is tflow's std::mt19937_64 Box-Muller
// stream, so clean, noisy and noise tensors are bit-identical to TF's.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/checks.hpp"

namespace denoise {

namespace nn = datamunge::nn;

inline double DnGray(double r, double g, double b) { return .299 * r + .587 * g + .114 * b; }

/// Grayscale rows of CIFAR int32 RGB images [N, 32, 32, 3] (0..255).
inline nn::Tensor DnImages(const nn::Tensor& images, const std::vector<std::size_t>& rows) {
  check::Require(!rows.empty() && images.dtype() == nn::DType::Int32 && images.dim() == 4 && images.size(1) == 32 &&
                     images.size(2) == 32 && images.size(3) == 3,
                 "CIFAR denoising RGB source");
  const std::int32_t* px = images.contiguous().data_ptr<std::int32_t>();
  std::vector<float>  clean(rows.size() * 1024);
  for (std::size_t n = 0; n < rows.size(); ++n) {
    check::Require(std::int64_t(rows[n]) < images.size(0), "denoising row bounds");
    for (int i = 0; i < 1024; ++i) {
      const std::size_t j = (rows[n] * 1024 + std::size_t(i)) * 3;
      clean[n * 1024 + std::size_t(i)] = float(DnGray(px[j] / 255., px[j + 1] / 255., px[j + 2] / 255.));
    }
  }
  return nn::from_vector(clean, {std::int64_t(rows.size()), 32, 32, 1});
}

/// Rows of a float NHWC tensor.
inline nn::Tensor DnBatch(const nn::Tensor& x, const std::vector<std::size_t>& rows) {
  check::Require(x.dtype() == nn::DType::Float32 && x.dim() == 4 && x.size(0) > 0, "float NHWC denoising batch");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    check::Require(std::int64_t(r) < x.size(0), "denoising batch bounds");
    ids.push_back(std::int64_t(r));
  }
  return nn::take_rows(x, ids).contiguous();
}

struct DnData {
  nn::Tensor clean, noisy, noise;
};

/// noisy = clean + sigma z (no clipping), noise = noisy - clean (float).
inline DnData DnCorrupt(const nn::Tensor& clean, double sigma, std::uint64_t seed) {
  check::Require(clean.dtype() == nn::DType::Float32 && clean.dim() == 4 && clean.size(3) == 1 && clean.numel() > 0,
                 "nonempty grayscale noise source");
  if (!std::isfinite(sigma) || sigma < 0) throw std::invalid_argument("finite nonnegative noise sigma");
  const auto         truth = clean.contiguous().to_vector<float>();
  std::vector<float> noisy(truth.size()), noise(truth.size());
  std::mt19937_64    rng(seed);
  const double       inverse = 1. / 9007199254740992.;
  const std::int64_t count   = clean.numel();
  for (std::int64_t i = 0; i < count; i += 2) {
    const double u = ((rng() >> 11) + 1.) * inverse, v = ((rng() >> 11) + 1.) * inverse;
    const double radius = std::sqrt(-2 * std::log(u)), angle = 6.2831853071795864769 * v;
    const double values[] = {radius * std::cos(angle), radius * std::sin(angle)};
    for (int k = 0; k < 2 && i + k < count; ++k) {
      const float t = truth[std::size_t(i + k)];
      check::Require(std::isfinite(t) && t >= 0 && t <= 1, "clean intensities in[0,1]");
      noisy[std::size_t(i + k)] = float(t + sigma * values[k]);
      noise[std::size_t(i + k)] = noisy[std::size_t(i + k)] - t;
    }
  }
  return {clean, nn::from_vector(noisy, clean.shape()), nn::from_vector(noise, clean.shape())};
}

inline DnData DnBatch(const DnData& source, const std::vector<std::size_t>& rows) {
  return {DnBatch(source.clean, rows), DnBatch(source.noisy, rows), DnBatch(source.noise, rows)};
}

inline int DnReflect(int i, int length) {
  if (length == 1) return 0;
  while (i < 0 || i >= length) i = i < 0 ? -i : 2 * length - 2 - i;
  return i;
}

/// 3x3 mean filter with reflected borders.
inline nn::Tensor DnBoxFilter(const nn::Tensor& image) {
  check::Require(image.dim() == 4 && image.size(3) == 1, "grayscale box filter");
  const auto         in = image.contiguous().to_vector<float>();
  std::vector<float> out(in.size());
  const int          b = int(image.size(0)), h = int(image.size(1)), w = int(image.size(2));
  for (int n = 0; n < b; ++n)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        double value = 0;
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) value += in[std::size_t((n * h + DnReflect(y + dy, h)) * w + DnReflect(x + dx, w))];
        out[std::size_t((n * h + y) * w + x)] = float(value / 9);
      }
  return nn::from_vector(out, image.shape());
}

struct DnMetrics {
  double mse = 0, psnr = 0, clipped_psnr = 0, outside = 0;
};

inline double DnPSNR(double mse) {
  check::Require(std::isfinite(mse) && mse >= 0, "nonnegative finite denoising MSE");
  return mse == 0 ? INFINITY : -10 * std::log10(mse);
}

/// Mean per-image raw / clipped PSNR, pooled MSE and the fraction outside [0, 1].
inline DnMetrics DnScore(const nn::Tensor& estimate, const nn::Tensor& clean) {
  check::Require(estimate.dtype() == nn::DType::Float32 && estimate.shape() == clean.shape() && estimate.dim() == 4 &&
                     estimate.size(0) > 0,
                 "aligned denoised image scores");
  DnMetrics          out;
  const auto         e = estimate.contiguous().to_vector<float>(), c = clean.contiguous().to_vector<float>();
  const std::int64_t pixels = clean.numel() / clean.size(0);
  for (std::int64_t n = 0; n < clean.size(0); ++n) {
    double mse = 0, clipped = 0;
    for (std::int64_t i = 0; i < pixels; ++i) {
      const double value = e[std::size_t(n * pixels + i)], truth = c[std::size_t(n * pixels + i)];
      check::Require(std::isfinite(value) && std::isfinite(truth), "finite denoised and clean pixels");
      mse += (value - truth) * (value - truth);
      const double difference = std::clamp(value, 0., 1.) - truth;
      clipped += difference * difference;
      out.outside += value < 0 || value > 1;
    }
    out.mse += mse / pixels;
    out.psnr += DnPSNR(mse / pixels);
    out.clipped_psnr += DnPSNR(clipped / pixels);
  }
  out.mse /= clean.size(0);
  out.psnr /= clean.size(0);
  out.clipped_psnr /= clean.size(0);
  out.outside /= double(clean.numel());
  return out;
}

}  // namespace denoise
