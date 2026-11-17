#pragma once
#include "nn_interpretation_check_support.hpp"
namespace occlusion {

namespace nn = datamunge::nn;
using nn::Tensor;

struct OcclusionSpec {
  int height = 7, width = 7, stride_y = 4, stride_x = 4;
};

/// Patch starts 0, stride, ... plus a final edge anchor so every pixel is covered.
inline auto OcclusionAnchors(int length, int patch, int stride) {
  return nn::models::OcclusionAnchors(length, patch, stride);
}

/// Host arrays (row-major) of every patch/case: rectangles [P, 4] (top, left, h, w),
/// coverage [H, W], logits [N, C] / [P, N, C], scores [N] / [P, N], drops [P, N],
/// overlap means [N, H, W] and the occluded inputs [P, N, H, W, C].
struct OcclusionResult {
  int                patches = 0, n = 0, h = 0, w = 0, c = 0, classes = 0;
  std::vector<int>   rectangles, coverage;
  std::vector<float> original_logits, occluded_logits, original_score, occluded_score,
      occluded_inputs;
  std::vector<double>  drops, pixel_mean;
  [[nodiscard]] double drop(int p, int b) const { return drops[std::size_t(p) * n + b]; }
  [[nodiscard]] double mean(int b, int y, int x) const {
    return pixel_mean[(std::size_t(b) * h + y) * w + x];
  }
  [[nodiscard]] float score(int p, int b) const { return occluded_score[std::size_t(p) * n + b]; }
  [[nodiscard]] float input(int p, int b, int y, int x, int ch) const {
    return occluded_inputs[(((std::size_t(p) * n + b) * h + y) * w + x) * c + ch];
  }
};
using PredictionFunction = std::function<Tensor(const Tensor&)>;

inline OcclusionResult OcclusionSensitivity(
    const Tensor&             input,
    const Tensor&             replacement,
    const Tensor&             target,
    const OcclusionSpec&      spec,
    int                       batch,
    const PredictionFunction& predict) {
  const auto x = nn::models::OcclusionSensitivity(
      input,
      replacement,
      target,
      {spec.height, spec.width, spec.stride_y, spec.stride_x},
      batch,
      predict);
  OcclusionResult out;
  out.patches         = x.patches;
  out.n               = x.n;
  out.h               = x.h;
  out.w               = x.w;
  out.c               = x.c;
  out.classes         = x.classes;
  out.rectangles      = x.rectangles.to_vector<int>();
  out.coverage        = x.coverage.to_vector<int>();
  out.original_logits = x.original_logits.to_vector<float>();
  out.occluded_logits = x.occluded_logits.to_vector<float>();
  out.original_score  = x.original_score.to_vector<float>();
  out.occluded_score  = x.occluded_score.to_vector<float>();
  out.occluded_inputs = x.occluded_inputs.to_vector<float>();
  out.drops           = x.drops.to_vector<double>();
  out.pixel_mean      = x.pixel_mean.to_vector<double>();
  return out;
}

} // namespace occlusion
namespace {
void Checks() {
  std::vector<float> kernel(20), input(40);
  for (int i = 0; i < 20; ++i)
    kernel[i] = .1f * std::sin(.7 * i);
  for (int i = 0; i < 40; ++i)
    input[i] = .1f + .02f * i;
  const Tensor k = sal::FromF(kernel, {1, 4, 5, 1}), x = sal::FromF(input, {2, 4, 5, 1}),
               replace = nn::full({2, 4, 5, 1}, double(.2f)), target = sal::FromI({0, 1});
  auto predict = [&](const Tensor& points) {
    const Tensor score = nn::add(nn::sum(nn::mul(points, k), {1, 2, 3}), double(.3f));
    return nn::stack({score, nn::neg(score)}, 1).contiguous();
  };
  const auto result = occlusion::OcclusionSensitivity(x, replace, target, {2, 2, 2, 2}, 3, predict);
  check::Require(
      result.patches == 6 && result.rectangles[5 * 4] == 2 && result.rectangles[5 * 4 + 1] == 3,
      "rectangular odd-width edge anchor appended");
  int    coverage[4][5]    = {};
  double expected[2][4][5] = {};
  for (int p = 0; p < 6; ++p) {
    const int top = result.rectangles[p * 4], left = result.rectangles[p * 4 + 1];
    for (int y = top; y < top + 2; ++y)
      for (int a = left; a < left + 2; ++a)
        ++coverage[y][a];
    for (int b = 0; b < 2; ++b) {
      double drop = 0;
      for (int y = 0; y < 4; ++y)
        for (int a = 0; a < 5; ++a) {
          const bool  covered = y >= top && y < top + 2 && a >= left && a < left + 2;
          const float value   = covered ? .2f : input[b * 20 + y * 5 + a];
          check::Near(result.input(p, b, y, a, 0), value, 0, "ALL240actual occluded pixels exact");
          if (covered)
            drop +=
                (input[b * 20 + y * 5 + a] - double(.2f)) * kernel[y * 5 + a] * (b == 0 ? 1 : -1);
        }
      check::Near(result.drop(p, b), drop, 2e-7, "ALL12 signed raw target score drops");
      for (int y = top; y < top + 2; ++y)
        for (int a = left; a < left + 2; ++a)
          expected[b][y][a] += drop;
    }
  }
  for (int y = 0; y < 4; ++y)
    for (int a = 0; a < 5; ++a) {
      check::Near(result.coverage[y * 5 + a], coverage[y][a], 0, "ALL20coverage counts");
      for (int b = 0; b < 2; ++b)
        check::Near(
            result.mean(b, y, a),
            expected[b][y][a] / coverage[y][a],
            2e-7,
            "ALL40overlap averages");
    }
  const auto identity = occlusion::OcclusionSensitivity(x, x, target, {2, 2, 2, 2}, 1, predict);
  for (double d : identity.drops)
    check::Near(d, 0, 2e-7, "identityreplacement givesnullscorechange");
  const auto full = occlusion::OcclusionSensitivity(x, replace, target, {4, 5, 4, 5}, 2, predict);
  check::Require(full.patches == 1, "full image replacement one patch");
  const auto replacement_score = sal::F(predict(replace));
  for (int b = 0; b < 2; ++b)
    check::Near(
        full.score(0, b),
        replacement_score[b * 2 + b],
        0,
        "full image score is replacement image score");
  check::Throws(
      [] { occlusion::OcclusionAnchors(5, 2, 3); },
      "stride>patch rejected instead of silently leaving holes");
  std::cout << "occlusion ALL240mutatedpixels/12signeddrops/20coverage/40overlapmeans and "
               "rectangular-edge/identity/full/invalidstride "
               "fixtures passed\n";
}

} // namespace
