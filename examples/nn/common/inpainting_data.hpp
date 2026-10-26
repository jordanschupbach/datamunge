// Natural RGB context patches, explicit validity masks and hole-only scoring:
// a port of tflow's common/ml/inpainting_data.h (context_encoder,
// partial_inpainting). CIFAR images come from common/cifar.hpp (int32 0..255);
// every tensor here is float32 and bit-identical to tflow's.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <vector>

#include "common/checks.hpp"

namespace inpaint {

namespace nn = datamunge::nn;

struct InpaintData {
  nn::Tensor truth, masked, valid, patch;  // [n,32,32,3] [0,1]; [n,32,32,3] [-1,1]; [n,32,32,1]; [n,16,16,3] [-1,1]
};

/// Central 16x16 hole filled with the ImageNet mean (117,104,123)/255, inputs scaled to [-1, 1].
inline InpaintData ContextPairs(const nn::Tensor& images, const std::vector<std::size_t>& rows) {
  check::Require(!rows.empty() && images.dim() == 4 && images.size(1) == 32 && images.size(2) == 32 && images.size(3) == 3 &&
                     images.dtype() == nn::DType::Int32,
                 "original32px CIFAR RGB context images");
  const std::int64_t  n  = std::int64_t(rows.size());
  const std::int32_t* px = images.contiguous().data_ptr<std::int32_t>();
  std::vector<float>  truth(std::size_t(n) * 3072), masked(std::size_t(n) * 3072), valid(std::size_t(n) * 1024),
      patch(std::size_t(n) * 768);
  const double fill[] = {117. / 255, 104. / 255, 123. / 255};
  for (std::size_t b = 0; b < rows.size(); ++b) {
    check::Require(std::int64_t(rows[b]) < images.size(0), "context image row bounds");
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x) {
        const bool hole = y >= 8 && y < 24 && x >= 8 && x < 24;
        valid[(b * 32 + std::size_t(y)) * 32 + std::size_t(x)] = !hole;
        for (int c = 0; c < 3; ++c) {
          const float       value = float(px[(rows[b] * 1024 + std::size_t(y * 32 + x)) * 3 + std::size_t(c)]) / 255.f;
          const std::size_t i     = ((b * 32 + std::size_t(y)) * 32 + std::size_t(x)) * 3 + std::size_t(c);
          truth[i]                = value;
          masked[i]               = float(2 * (hole ? fill[c] : value) - 1);
          if (hole) patch[((b * 16 + std::size_t(y - 8)) * 16 + std::size_t(x - 8)) * 3 + std::size_t(c)] = 2 * value - 1;
        }
      }
  }
  return {nn::from_vector(truth, {n, 32, 32, 3}), nn::from_vector(masked, {n, 32, 32, 3}), nn::from_vector(valid, {n, 32, 32, 1}),
          nn::from_vector(patch, {n, 16, 16, 3})};
}

inline nn::Tensor InpaintBatch(const nn::Tensor& x, const std::vector<std::size_t>& rows) {
  check::Require(x.dtype() == nn::DType::Float32 && x.dim() == 4 && x.size(0) > 0, "float NHWC inpainting batch");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    check::Require(std::int64_t(r) < x.size(0), "inpainting batch bounds");
    ids.push_back(std::int64_t(r));
  }
  return nn::take_rows(x, ids).contiguous();
}
inline InpaintData InpaintBatch(const InpaintData& data, const std::vector<std::size_t>& rows) {
  return {InpaintBatch(data.truth, rows), InpaintBatch(data.masked, rows), InpaintBatch(data.valid, rows),
          InpaintBatch(data.patch, rows)};
}

/// Visible pixels kept, holes filled with the per-image visible-context mean colour.
inline nn::Tensor ContextFillMean(const InpaintData& data) {
  const auto         t = data.truth.contiguous().to_vector<float>(), m = data.masked.contiguous().to_vector<float>(),
             v = data.valid.contiguous().to_vector<float>();
  const std::int64_t n = data.truth.size(0), H = data.truth.size(1), W = data.truth.size(2);
  std::vector<float> out(t.size());
  for (std::int64_t b = 0; b < n; ++b) {
    double mean[3] = {};
    int    count   = 0;
    for (std::int64_t p = 0; p < H * W; ++p)
      if (v[std::size_t(b * H * W + p)] > 0) {
        ++count;
        for (int c = 0; c < 3; ++c) mean[c] += (m[std::size_t((b * H * W + p) * 3 + c)] + 1) / 2;
      }
    check::Require(count > 0, "mean fill needs observed pixels");
    for (std::int64_t p = 0; p < H * W; ++p)
      for (int c = 0; c < 3; ++c) {
        const std::size_t i = std::size_t((b * H * W + p) * 3 + c);
        out[i]              = v[std::size_t(b * H * W + p)] != 0 ? t[i] : float(mean[c] / count);
      }
  }
  return nn::from_vector(out, data.truth.shape());
}

struct InpaintMetrics {
  double hole_mse = 0, hole_psnr = 0, full_psnr = 0, known_error = 0;
};

inline double InpaintPSNR(double mse) {
  check::Require(std::isfinite(mse) && mse >= 0, "finite nonnegative inpainting MSE");
  return mse == 0 ? INFINITY : -10 * std::log10(mse);
}

/// Hole-only MSE / mean per-image PSNR, full-image PSNR and the max visible-pixel error.
inline InpaintMetrics InpaintScore(const InpaintData& data, const nn::Tensor& output) {
  check::Require(output.shape() == data.truth.shape(), "aligned inpainting prediction");
  InpaintMetrics score;
  const auto     o = output.contiguous().to_vector<float>(), t = data.truth.contiguous().to_vector<float>(),
             v = data.valid.contiguous().to_vector<float>();
  const std::int64_t n = output.size(0);
  for (std::int64_t b = 0; b < n; ++b) {
    double error = 0, hole_error = 0;
    int    holes = 0;
    for (std::int64_t p = 0; p < 1024; ++p) {
      const bool hole = v[std::size_t(b * 1024 + p)] == 0;
      holes += hole;
      for (int c = 0; c < 3; ++c) {
        const std::size_t i     = std::size_t((b * 1024 + p) * 3 + c);
        const double      value = o[i];
        check::Require(std::isfinite(value), "finite inpainting pixels");
        const double difference = value - t[i];
        error += difference * difference;
        if (hole)
          hole_error += difference * difference;
        else
          score.known_error = std::max(score.known_error, std::abs(difference));
      }
    }
    check::Require(holes > 0, "hole metrics require missing pixels");
    score.hole_mse += hole_error / (3 * holes);
    score.hole_psnr += InpaintPSNR(hole_error / (3 * holes));
    score.full_psnr += InpaintPSNR(error / 3072);
  }
  score.hole_mse /= double(n);
  score.hole_psnr /= double(n);
  score.full_psnr /= double(n);
  return score;
}

}  // namespace inpaint
