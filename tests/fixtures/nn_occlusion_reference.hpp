// Batched finite-patch occlusion scores and the explicit overlap-average projection:
// a port of tflow's common/nn/occlusion.h.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace occlusion_reference {

namespace nn = datamunge::nn;
using nn::Tensor;

struct OcclusionSpec {
  int height = 7, width = 7, stride_y = 4, stride_x = 4;
};

/// Patch starts 0, stride, ... plus a final edge anchor so every pixel is covered.
inline std::vector<int> OcclusionAnchors(int length, int patch, int stride) {
  check::Require(length > 0 && patch > 0 && patch <= length && stride > 0 && stride <= patch,
                 "patch inside image;stride<=patch guarantees full coverage");
  std::vector<int> starts;
  for (int i = 0; i <= length - patch; i += stride) starts.push_back(i);
  if (starts.back() != length - patch) starts.push_back(length - patch);
  return starts;
}

/// Host arrays (row-major) of every patch/case: rectangles [P, 4] (top, left, h, w),
/// coverage [H, W], logits [N, C] / [P, N, C], scores [N] / [P, N], drops [P, N],
/// overlap means [N, H, W] and the occluded inputs [P, N, H, W, C].
struct OcclusionResult {
  int                 patches = 0, n = 0, h = 0, w = 0, c = 0, classes = 0;
  std::vector<int>    rectangles, coverage;
  std::vector<float>  original_logits, occluded_logits, original_score, occluded_score, occluded_inputs;
  std::vector<double> drops, pixel_mean;
  [[nodiscard]] double drop(int p, int b) const { return drops[std::size_t(p) * n + b]; }
  [[nodiscard]] double mean(int b, int y, int x) const { return pixel_mean[(std::size_t(b) * h + y) * w + x]; }
  [[nodiscard]] float  score(int p, int b) const { return occluded_score[std::size_t(p) * n + b]; }
  [[nodiscard]] float  input(int p, int b, int y, int x, int ch) const {
    return occluded_inputs[(((std::size_t(p) * n + b) * h + y) * w + x) * c + ch];
  }
};
using PredictionFunction = std::function<Tensor(const Tensor&)>;

inline OcclusionResult OcclusionSensitivity(const Tensor& input, const Tensor& replacement, const Tensor& target,
                                            const OcclusionSpec& spec, int batch, const PredictionFunction& predict) {
  check::Require(input.dtype() == nn::DType::Float32 && input.dim() == 4 && input.numel() > 0 &&
                     input.shape() == replacement.shape() && replacement.dtype() == nn::DType::Float32 &&
                     target.dtype() == nn::DType::Int32 && target.shape() == nn::Shape{input.size(0)} && batch > 0,
                 "matching nonempty NHWC input/replacement/fixed targets");
  OcclusionResult out;
  out.n = int(input.size(0));
  out.h = int(input.size(1));
  out.w = int(input.size(2));
  out.c = int(input.size(3));
  const int  n = out.n, h = out.h, w = out.w, c = out.c;
  const auto ys = OcclusionAnchors(h, spec.height, spec.stride_y), xs = OcclusionAnchors(w, spec.width, spec.stride_x);
  out.patches = int(ys.size() * xs.size());
  out.coverage.assign(std::size_t(h) * w, 0);
  for (int y : ys)
    for (int x : xs) {
      out.rectangles.insert(out.rectangles.end(), {y, x, spec.height, spec.width});
      for (int a = y; a < y + spec.height; ++a)
        for (int b = x; b < x + spec.width; ++b) ++out.coverage[a * w + b];
    }
  for (int v : out.coverage) check::Require(v > 0, "all image pixels covered including appended edge anchors");
  const Tensor original = predict(input);
  check::Require(original.dtype() == nn::DType::Float32 && original.dim() == 2 && original.size(0) == n,
                 "prediction returns class logits");
  out.classes         = int(original.size(1));
  const int classes   = out.classes;
  out.original_logits = original.contiguous().to_vector<float>();
  const auto x = input.contiguous().to_vector<float>(), r = replacement.contiguous().to_vector<float>();
  const auto t = target.contiguous().to_vector<int>();
  const int  patches = out.patches;
  out.original_score.resize(n);
  out.occluded_score.resize(std::size_t(patches) * n);
  out.occluded_logits.resize(std::size_t(patches) * n * classes);
  out.drops.resize(std::size_t(patches) * n);
  out.pixel_mean.assign(std::size_t(n) * h * w, 0.0);
  const std::int64_t image_size = std::int64_t(h) * w * c;
  out.occluded_inputs.resize(std::size_t(patches) * n * image_size);
  for (int b = 0; b < n; ++b) {
    check::Require(t[b] >= 0 && t[b] < classes, "fixed target class range");
    out.original_score[b] = out.original_logits[b * classes + t[b]];
  }
  for (int start = 0; start < patches * n; start += batch) {
    const int          count = std::min(batch, patches * n - start);
    std::vector<float> points(std::size_t(count) * image_size);
    for (int j = 0; j < count; ++j) {
      const int flat = start + j, patch = flat / n, b = flat % n;
      std::copy_n(x.data() + b * image_size, image_size, points.data() + j * image_size);
      const int top = out.rectangles[patch * 4], left = out.rectangles[patch * 4 + 1];
      for (int y = top; y < top + spec.height; ++y)
        for (int a = left; a < left + spec.width; ++a)
          for (int ch = 0; ch < c; ++ch) {
            const std::int64_t offset    = (std::int64_t(y) * w + a) * c + ch;
            points[j * image_size + offset] = r[b * image_size + offset];
          }
      std::copy_n(points.data() + j * image_size, image_size, out.occluded_inputs.data() + flat * image_size);
    }
    const Tensor z = predict(nn::from_vector(points, {count, h, w, c}));
    check::Require(z.dtype() == nn::DType::Float32 && z.shape() == nn::Shape({count, classes}), "all occluded prediction logits aligned");
    const auto zv = z.contiguous().to_vector<float>();
    for (int j = 0; j < count; ++j) {
      const int flat = start + j, patch = flat / n, b = flat % n;
      for (int cls = 0; cls < classes; ++cls) {
        const float value = zv[j * classes + cls];
        check::Require(std::isfinite(value), "finite every occluded class score");
        out.occluded_logits[(std::size_t(patch) * n + b) * classes + cls] = value;
      }
      const float score                          = zv[j * classes + t[b]];
      out.occluded_score[std::size_t(patch) * n + b] = score;
      const double delta                         = double(out.original_score[b]) - score;
      out.drops[std::size_t(patch) * n + b]      = delta;
      const int top = out.rectangles[patch * 4], left = out.rectangles[patch * 4 + 1];
      for (int y = top; y < top + spec.height; ++y)
        for (int a = left; a < left + spec.width; ++a) out.pixel_mean[(std::size_t(b) * h + y) * w + a] += delta;
    }
  }
  for (int b = 0; b < n; ++b)
    for (int y = 0; y < h; ++y)
      for (int a = 0; a < w; ++a) out.pixel_mean[(std::size_t(b) * h + y) * w + a] /= out.coverage[y * w + a];
  return out;
}

/// The result's arrays as tensors under tflow's names (prefix + "_rectangles", ...).
inline void Export(const OcclusionResult& r, const std::string& prefix, std::map<std::string, Tensor>& arrays) {
  const std::int64_t P = r.patches, N = r.n, H = r.h, W = r.w, C = r.c, K = r.classes;
  arrays[prefix + "_rectangles"]      = nn::from_vector(r.rectangles, {P, 4});
  arrays[prefix + "_coverage"]        = nn::from_vector(r.coverage, {H, W});
  arrays[prefix + "_original_logits"] = nn::from_vector(r.original_logits, {N, K});
  arrays[prefix + "_occluded_logits"] = nn::from_vector(r.occluded_logits, {P, N, K});
  arrays[prefix + "_original_score"]  = nn::from_vector(r.original_score, {N});
  arrays[prefix + "_occluded_score"]  = nn::from_vector(r.occluded_score, {P, N});
  arrays[prefix + "_drops"]           = nn::from_vector(r.drops, {P, N});
  arrays[prefix + "_pixel_mean"]      = nn::from_vector(r.pixel_mean, {N, H, W});
  arrays[prefix + "_occluded_inputs"] = nn::from_vector(r.occluded_inputs, {P, N, H, W, C});
}

}  // namespace occlusion
