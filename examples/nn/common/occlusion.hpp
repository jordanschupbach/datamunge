// Batched finite-patch occlusion scores and the explicit overlap-average projection:
// a port of tflow's common/nn/occlusion.h.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/occlusion.hpp>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace occlusion {

namespace nn = datamunge::nn;
using nn::Tensor;

struct OcclusionSpec {
  int height = 7, width = 7, stride_y = 4, stride_x = 4;
};

/// Patch starts 0, stride, ... plus a final edge anchor so every pixel is covered.
inline auto OcclusionAnchors(int length,int patch,int stride){return nn::models::OcclusionAnchors(length,patch,stride);}

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

inline OcclusionResult OcclusionSensitivity(const Tensor&input,const Tensor&replacement,const Tensor&target,const OcclusionSpec&spec,int batch,const PredictionFunction&predict){const auto x=nn::models::OcclusionSensitivity(input,replacement,target,{spec.height,spec.width,spec.stride_y,spec.stride_x},batch,predict);OcclusionResult out;out.patches=x.patches;out.n=x.n;out.h=x.h;out.w=x.w;out.c=x.c;out.classes=x.classes;
out.rectangles=x.rectangles.to_vector<int>();
out.coverage=x.coverage.to_vector<int>();
out.original_logits=x.original_logits.to_vector<float>();
out.occluded_logits=x.occluded_logits.to_vector<float>();
out.original_score=x.original_score.to_vector<float>();
out.occluded_score=x.occluded_score.to_vector<float>();
out.occluded_inputs=x.occluded_inputs.to_vector<float>();
out.drops=x.drops.to_vector<double>();
out.pixel_mean=x.pixel_mean.to_vector<double>();
return out;}
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
