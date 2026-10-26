// DeepDream input optimization: feature-mean ascent, RMS normalization and bounds,
// host half-pixel bilinear octaves that preserve batch/channel axes; no model
// updates. A port of tflow's common/nn/dream.h.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <vector>

#include "common/checks.hpp"

namespace dream {

namespace nn = datamunge::nn;
using nn::Tensor;

/// Half-pixel bilinear resize of NHWC float32 on the host (double arithmetic, as tflow).
inline Tensor DreamResize(const Tensor& x, int height, int width) {
  check::Require(x.dtype() == nn::DType::Float32 && x.dim() == 4 && height > 0 && width > 0, "positive NHWC float32 octave");
  const int  n = int(x.size(0)), h = int(x.size(1)), w = int(x.size(2)), c = int(x.size(3));
  const auto v = x.contiguous().to_vector<float>();
  auto       at = [&](int b, int y, int a, int ch) { return double(v[((std::size_t(b) * h + y) * w + a) * c + ch]); };
  std::vector<float> out(std::size_t(n) * height * width * c);
  for (int b = 0; b < n; ++b)
    for (int y = 0; y < height; ++y)
      for (int a = 0; a < width; ++a) {
        const double fy = std::clamp((y + .5) * h / height - .5, 0., double(h - 1)),
                     fx = std::clamp((a + .5) * w / width - .5, 0., double(w - 1));
        const int    y0 = int(fy), x0 = int(fx), y1 = std::min(y0 + 1, h - 1), x1 = std::min(x0 + 1, w - 1);
        const double dy = fy - y0, dx = fx - x0;
        for (int ch = 0; ch < c; ++ch)
          out[((std::size_t(b) * height + y) * width + a) * c + ch] =
              float((1 - dy) * ((1 - dx) * at(b, y0, x0, ch) + dx * at(b, y0, x1, ch)) +
                    dy * ((1 - dx) * at(b, y1, x0, ch) + dx * at(b, y1, x1, ch)));
      }
  return nn::from_vector(out, {n, height, width, c});
}

struct DreamEvaluation {
  double score;
  Tensor gradient;
};
using DreamFunction = std::function<DreamEvaluation(const Tensor&)>;
struct DreamStep {
  Tensor image, gradient;
  double before = 0, after = 0, rms = 0, step = 0;
  int    backtracks = 0;
};

/// One RMS-normalized ascent step projected to [0,1] with monotonic backtracking
/// (halving up to 12 times); an explicit no-op if every bounded proposal decreases.
inline DreamStep DreamAscent(const Tensor& image, double learning_rate, const DreamFunction& evaluate) {
  check::Require(learning_rate >= 0 && std::isfinite(learning_rate), "finite nonnegative input step");
  const auto e = evaluate(image);
  check::Require(e.gradient.dtype() == nn::DType::Float32 && e.gradient.shape() == image.shape() && std::isfinite(e.score),
                 "finite scalar feature objective/aligned gradient");
  DreamStep out;
  out.gradient = e.gradient;
  out.before = out.after = e.score;
  const auto x = image.contiguous().to_vector<float>(), g = e.gradient.contiguous().to_vector<float>();
  double     sum = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    check::Require(std::isfinite(g[i]) && x[i] >= 0 && x[i] <= 1, "finite gradient and image in [0,1]");
    sum += double(g[i]) * g[i];
  }
  out.rms   = std::sqrt(sum / double(x.size()));
  out.image = image.contiguous().clone();
  if (learning_rate == 0 || out.rms == 0) return out;
  double step = learning_rate;
  for (int attempt = 0; attempt < 12; ++attempt) {
    std::vector<float> candidate(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      candidate[i] = float(std::clamp(double(x[i]) + step * g[i] / (out.rms + 1e-12), 0., 1.));
    const Tensor c = nn::from_vector(candidate, image.shape());
    const double score = evaluate(c).score;
    check::Require(std::isfinite(score), "finite candidate objective");
    if (score >= e.score) {
      out.image      = c;
      out.after      = score;
      out.step       = step;
      out.backtracks = attempt;
      return out;
    }
    step *= .5;
  }
  out.backtracks = 12;
  return out;  // Explicit no-op if all bounded proposals decrease.
}

}  // namespace dream
