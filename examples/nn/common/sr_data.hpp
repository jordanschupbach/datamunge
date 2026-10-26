// Port of tflow's common/ml/sr_data.h: explicit antialiased Keys bicubic
// (a = -.5, half-pixel centres, symmetric edges) HR/LR pairs and the
// common-coordinate studio-range BT.601 Y metrics (PSNR, 11x11 Gaussian SSIM)
// on the central 20x20 crop (pixels 6..25) of 32x32 images.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <vector>

#include "common/checks.hpp"
#include "common/cifar.hpp"

namespace sr {

namespace nn = datamunge::nn;
using nn::Tensor;

inline double SRCubic(double x) {
  x = std::abs(x);
  if (x <= 1) return 1.5 * x * x * x - 2.5 * x * x + 1;
  if (x < 2) return -.5 * x * x * x + 2.5 * x * x - 4 * x + 2;
  return 0;
}
inline int SRSymmetric(int index, int length) {
  if (length < 1) throw std::invalid_argument("resampling length");
  if (length == 1) return 0;
  while (index < 0 || index >= length) index = index < 0 ? -index - 1 : 2 * length - 1 - index;
  return index;
}
using SRWeights = std::vector<std::vector<std::pair<int, double>>>;

/// Normalized antialiased Keys contributions of `input` samples to each of `output`.
inline SRWeights SRContributions(int input, int output) {
  if (input < 1 || output < 1) throw std::invalid_argument("positive bicubic dimensions");
  SRWeights    weights(output);
  const double scale = double(output) / input, filter = std::min(1., scale), radius = 2 / filter;
  for (int i = 0; i < output; ++i) {
    const double center = (i + .5) / scale - .5;
    double       sum    = 0;
    for (int j = int(std::ceil(center - radius)); j <= int(std::floor(center + radius)); ++j) {
      const double value = filter * SRCubic(filter * (center - j));
      if (value != 0) {
        weights[i].push_back({SRSymmetric(j, input), value});
        sum += value;
      }
    }
    check::Require(std::isfinite(sum) && std::abs(sum) > 1e-12, "bicubic normalized kernel mass");
    for (auto& item : weights[i]) item.second /= sum;
  }
  return weights;
}

/// Separable bicubic resize of float32 NHWC images (rows after columns, float32 intermediate as tflow).
inline Tensor SRBicubic(const Tensor& image, int oh, int ow) {
  check::Require(image.dim() == 4, "bicubic float NHWC");
  const int  b = int(image.size(0)), h = int(image.size(1)), w = int(image.size(2)), c = int(image.size(3));
  const auto rows = SRContributions(h, oh), columns = SRContributions(w, ow);
  const auto in   = nn::cast(image, nn::DType::Float32).contiguous().to_vector<float>();
  std::vector<float> horizontal(std::size_t(b) * h * ow * c), out(std::size_t(b) * oh * ow * c);
  for (int n = 0; n < b; ++n)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < ow; ++x)
        for (int k = 0; k < c; ++k) {
          double value = 0;
          for (auto [j, a] : columns[x]) value += a * in[((std::size_t(n) * h + y) * w + j) * c + k];
          horizontal[((std::size_t(n) * h + y) * ow + x) * c + k] = float(value);
        }
  for (int n = 0; n < b; ++n)
    for (int y = 0; y < oh; ++y)
      for (int x = 0; x < ow; ++x)
        for (int k = 0; k < c; ++k) {
          double value = 0;
          for (auto [j, a] : rows[y]) value += a * horizontal[((std::size_t(n) * h + j) * ow + x) * c + k];
          out[((std::size_t(n) * oh + y) * ow + x) * c + k] = float(value);
        }
  return nn::from_vector(out, {b, oh, ow, c});
}

inline double SRLuma(double r, double g, double b) { return 16. / 255 + (65.481 * r + 128.553 * g + 24.966 * b) / 255; }

inline Tensor SRBatch(const Tensor& x, const std::vector<std::int64_t>& rows) { return nn::take_rows(x, rows); }

struct SRData {
  Tensor hr, lr, bicubic;
};

/// HR = CIFAR rows / 255, LR = bicubic 16x16, bicubic = LR back to 32x32.
inline SRData SRPairs(const data::CifarDataset& source, const std::vector<std::int64_t>& rows) {
  for (auto r : rows) check::Require(r >= 0 && r < source.size(), "CIFAR SR row bounds");
  const Tensor hr = nn::cast(nn::take_rows(source.images, rows), nn::DType::Float32) / 255.0;
  const Tensor lr = SRBicubic(hr, 16, 16);
  return {hr, lr, SRBicubic(lr, 32, 32)};
}

inline double SRPSNR(double mse) {
  check::Require(std::isfinite(mse) && mse >= 0, "finite nonnegative MSE");
  return mse == 0 ? INFINITY : -10 * std::log10(mse);
}

inline double SRSSIM(const std::vector<double>& a, const std::vector<double>& b, int h, int w) {
  check::Require(h >= 11 && w >= 11 && a.size() == std::size_t(h * w) && a.size() == b.size(), "SSIM valid 11 Gaussian window");
  std::array<double, 11> kernel;
  double                 sum = 0;
  for (int i = 0; i < 11; ++i) {
    kernel[i] = std::exp(-(i - 5) * (i - 5) / (2 * 1.5 * 1.5));
    sum += kernel[i];
  }
  for (auto& v : kernel) v /= sum;
  double score = 0;
  int    count = 0;
  for (int y = 5; y < h - 5; ++y)
    for (int x = 5; x < w - 5; ++x) {
      double ma = 0, mb = 0, aa = 0, bb = 0, ab = 0;
      for (int j = 0; j < 11; ++j)
        for (int k = 0; k < 11; ++k) {
          const double weight = kernel[j] * kernel[k], p = a[(y + j - 5) * w + x + k - 5], q = b[(y + j - 5) * w + x + k - 5];
          ma += weight * p;
          mb += weight * q;
          aa += weight * p * p;
          bb += weight * q * q;
          ab += weight * p * q;
        }
      const double va = std::max(0., aa - ma * ma), vb = std::max(0., bb - mb * mb), cov = ab - ma * mb;
      score += ((2 * ma * mb + .0001) * (2 * cov + .0009)) / ((ma * ma + mb * mb + .0001) * (va + vb + .0009));
      ++count;
    }
  return score / count;
}

struct SRMetrics {
  double psnr = 0, ssim = 0, mse = 0, baseline_psnr = 0, baseline_ssim = 0;
};

/// Mean per-image Y PSNR/SSIM (clipped estimates) over the central 20x20 crop, and the bicubic baseline.
inline SRMetrics SRScore(const Tensor& prediction, const SRData& data) {
  const bool cropped = prediction.size(1) == 20;
  check::Require(prediction.size(0) == data.hr.size(0) && prediction.size(1) == (cropped ? 20 : 32) &&
                     prediction.size(2) == prediction.size(1) && (prediction.size(3) == 1 || prediction.size(3) == 3),
                 "aligned SR predictions");
  SRMetrics  score;
  const int  b = int(prediction.size(0)), channels = int(prediction.size(3)), side = int(prediction.size(1));
  const auto p = prediction.contiguous().to_vector<float>(), hr = data.hr.contiguous().to_vector<float>(),
             bi = data.bicubic.contiguous().to_vector<float>();
  auto at = [](const std::vector<float>& v, int side, int channels, int n, int y, int x, int c) {
    return double(v[((std::size_t(n) * side + y) * side + x) * channels + c]);
  };
  for (int n = 0; n < b; ++n) {
    std::vector<double> estimated(400), truth(400), base(400);
    double              mse = 0, bmse = 0;
    for (int y = 0; y < 20; ++y)
      for (int x = 0; x < 20; ++x) {
        const int gy = y + 6, gx = x + 6, py = cropped ? y : gy, px = cropped ? x : gx;
        double    value = channels == 1 ? at(p, side, 1, n, py, px, 0)
                                        : SRLuma(at(p, side, 3, n, py, px, 0), at(p, side, 3, n, py, px, 1),
                                                 at(p, side, 3, n, py, px, 2));
        check::Require(std::isfinite(value), "finite raw SR estimates");
        value          = std::clamp(value, 0., 1.);
        const double t = SRLuma(at(hr, 32, 3, n, gy, gx, 0), at(hr, 32, 3, n, gy, gx, 1), at(hr, 32, 3, n, gy, gx, 2));
        const double bc = std::clamp(
            SRLuma(at(bi, 32, 3, n, gy, gx, 0), at(bi, 32, 3, n, gy, gx, 1), at(bi, 32, 3, n, gy, gx, 2)), 0., 1.);
        const int index  = y * 20 + x;
        estimated[index] = value;
        truth[index]     = t;
        base[index]      = bc;
        mse += (value - t) * (value - t);
        bmse += (bc - t) * (bc - t);
      }
    score.mse += mse / 400;
    score.psnr += SRPSNR(mse / 400);
    score.baseline_psnr += SRPSNR(bmse / 400);
    score.ssim += SRSSIM(estimated, truth, 20, 20);
    score.baseline_ssim += SRSSIM(base, truth, 20, 20);
  }
  score.mse /= b;
  score.psnr /= b;
  score.baseline_psnr /= b;
  score.ssim /= b;
  score.baseline_ssim /= b;
  check::Require(!std::isnan(score.psnr) && score.psnr >= 0 && std::isfinite(score.ssim) && std::abs(score.ssim) <= 1.00001,
                 "finite held-out PSNR/SSIM");
  return score;
}

}  // namespace sr
