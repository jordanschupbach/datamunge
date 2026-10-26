// Multi-person COCO18 heatmaps, averaged directed limb fields and unknown
// masks: a port of tflow's common/ml/paf_data.h (host-side target
// construction on std::vector, nn tensors in and out).
#pragma once

#include <array>
#include <cmath>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "common/pose_data.hpp"

namespace pose {

inline constexpr int kPoseEdges[19][2] = {{1, 2},  {1, 5},   {2, 3},   {3, 4},  {5, 6},   {6, 7},   {1, 8},
                                          {8, 9},  {9, 10},  {1, 11},  {11, 12}, {12, 13}, {1, 0},   {0, 14},
                                          {14, 16}, {0, 15}, {15, 17}, {2, 16},  {5, 17}};

/// rgb [N,128,128,3], joints [N,P,18,3] (x, y, visibility), boxes [N,P,4]
/// (centre x, centre y, w, h), counts/image_ids int32 [N].
struct PafData {
  nn::Tensor rgb, joints, boxes, counts, image_ids;
};
struct PafSplits {
  PafData fit, validation, test;
};

inline PafData PafBatch(const PafData& data, const std::vector<std::size_t>& rows) {
  return {PoseBatch(data.rgb, rows), PoseBatch(data.joints, rows), PoseBatch(data.boxes, rows), PoseBatch(data.counts, rows),
          PoseBatch(data.image_ids, rows)};
}

/// COCO17 [N,P,17,3] -> OpenPose COCO18 order with a neck at the midpoint of
/// two labeled shoulders (visibility = min flag). Unlabeled joints are zero.
inline nn::Tensor Coco18(const nn::Tensor& input) {
  check::Require(input.dim() == 4 && input.size(2) == 17 && input.size(3) == 3, "COCO17 person annotations");
  const int          mapping[] = {0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3};
  const std::int64_t N = input.size(0), P = input.size(1);
  const auto         in = input.to_vector<float>();
  std::vector<float> out(std::size_t(N * P * 18 * 3), 0.f);
  auto               src = [&](std::int64_t n, std::int64_t p, int j, int c) { return in[std::size_t(((n * P + p) * 17 + j) * 3 + c)]; };
  for (std::int64_t n = 0; n < N; ++n)
    for (std::int64_t person = 0; person < P; ++person)
      for (int joint = 0; joint < 18; ++joint) {
        float* o = &out[std::size_t(((n * P + person) * 18 + joint) * 3)];
        if (joint == 1) {
          const float a = src(n, person, 5, 2), b = src(n, person, 6, 2);
          if (a > 0 && b > 0) {
            for (int c = 0; c < 2; ++c) o[c] = .5f * (src(n, person, 5, c) + src(n, person, 6, c));
            o[2] = std::min(a, b);
          }
        } else if (src(n, person, mapping[joint], 2) > 0)
          for (int c = 0; c < 3; ++c) o[c] = src(n, person, mapping[joint], c);
      }
  return nn::from_vector(out, {N, P, 18, 3});
}

inline PafSplits LoadPafImages(const std::filesystem::path& root) {
  const auto    arrays = PoseArrays(root / "poses.npz");
  std::set<int> previous;
  auto          read = [&](const std::string& split, std::int64_t count) {
    PafData result{arrays.at(split + "_full_rgb"), Coco18(arrays.at(split + "_full_joints")), arrays.at(split + "_full_boxes"),
                   arrays.at(split + "_full_counts"), arrays.at(split + "_full_image_ids")};
    const std::vector<std::pair<nn::Tensor, nn::Shape>> shapes = {{result.rgb, {count, 128, 128, 3}}, {result.joints, {count, 8, 18, 3}},
                                                                  {result.boxes, {count, 8, 4}},      {result.counts, {count}},
                                                                  {result.image_ids, {count}}};
    for (const auto& item : shapes) check::Require(item.first.shape() == item.second, "full-person pose array geometry");
    check::Require(result.rgb.dtype() == nn::DType::Float32 && result.joints.dtype() == nn::DType::Float32 &&
                       result.boxes.dtype() == nn::DType::Float32 && result.counts.dtype() == nn::DType::Int32 &&
                       result.image_ids.dtype() == nn::DType::Int32,
                   "full-person pose numeric types");
    const auto    counts = result.counts.to_vector<std::int32_t>(), image_ids = result.image_ids.to_vector<std::int32_t>();
    const auto    boxes = result.boxes.to_vector<float>(), joints = result.joints.to_vector<float>();
    std::set<int> ids;
    for (std::int64_t n = 0; n < count; ++n) {
      const int people = counts[std::size_t(n)], id = image_ids[std::size_t(n)];
      check::Require(people > 0 && people <= 8 && !previous.count(id), "person count and disjoint source-image IDs");
      ids.insert(id);
      for (int person = 0; person < people; ++person) {
        for (int c = 0; c < 4; ++c) {
          const float b = boxes[std::size_t((n * 8 + person) * 4 + c)];
          check::Require(std::isfinite(b) && (c < 2 || b > 0), "finite positive person boxes");
        }
        for (int j = 0; j < 18; ++j) {
          const float* o = &joints[std::size_t(((n * 8 + person) * 18 + j) * 3)];
          check::Require(o[2] == 0 || o[2] == 1 || o[2] == 2, "COCO18 visibility");
          if (o[2] > 0)
            for (int c = 0; c < 2; ++c) check::Require(std::isfinite(o[c]), "finite annotated full-image joints");
        }
      }
    }
    for (float v : result.rgb.to_vector<float>()) check::Require(std::isfinite(v) && v >= 0 && v <= 1, "real full-image RGB values");
    previous.insert(ids.begin(), ids.end());
    return result;
  };
  auto fit = read("fit", 6), validation = read("validation", 1), test = read("test", 4);
  return {fit, validation, test};
}

/// heatmap/heatmask [N,side,side,19] (18 joints + background), paf/pafmask [N,side,side,38].
struct PafTargets {
  nn::Tensor heatmap, paf, heatmask, pafmask;
};

/// Max-combined Gaussians, background = 1 - max, limb strips of half-width
/// `thickness` averaged over overlapping people; unknown joints/limbs zero the
/// mask inside that person's box (conservative, not COCO crowd segmentation).
inline PafTargets PreparePafTargets(const PafData& data, int side = 16, int input_size = 128, float sigma = 1.f, float thickness = 1.f) {
  check::Require(side > 0 && input_size > 0 && sigma > 0 && thickness > 0, "PAF target configuration");
  const std::int64_t batch = data.joints.size(0), P = data.joints.size(1);
  const auto         jv = data.joints.to_vector<float>(), bv = data.boxes.to_vector<float>();
  const auto         cv = data.counts.to_vector<std::int32_t>();
  const std::size_t  pixels = std::size_t(batch * side * side);
  std::vector<float> heat(pixels * 19, 0.f), paf(pixels * 38, 0.f), heatmask(pixels * 19, 1.f), pafmask(pixels * 38, 1.f), counts(pixels * 19, 0.f);
  auto               at = [&](std::int64_t n, int y, int x) { return std::size_t((n * side + y) * side + x); };
  for (std::int64_t n = 0; n < batch; ++n)
    for (int person = 0; person < cv[std::size_t(n)]; ++person) {
      std::array<bool, 18>                  annotated{};
      std::array<std::array<double, 2>, 18> points{};
      for (int j = 0; j < 18; ++j) {
        const float* o = &jv[std::size_t(((n * P + person) * 18 + j) * 3)];
        annotated[std::size_t(j)] = o[2] > 0;
        if (annotated[std::size_t(j)])
          for (int c = 0; c < 2; ++c) {
            const float value = o[c];
            annotated[std::size_t(j)] = annotated[std::size_t(j)] && std::isfinite(value) && value >= 0 && value <= input_size - 1;
            points[std::size_t(j)][std::size_t(c)] = (value + .5) * side / input_size - .5;
          }
      }
      const float* box = &bv[std::size_t((n * P + person) * 4)];
      const double bx = box[0], by = box[1], bw = box[2], bh = box[3];
      for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
          const double      ix = (x + .5) * input_size / side - .5, iy = (y + .5) * input_size / side - .5;
          const bool        in_box = std::abs(ix - bx) <= bw / 2 && std::abs(iy - by) <= bh / 2;
          const std::size_t q      = at(n, y, x);
          for (int j = 0; j < 18; ++j) {
            if (!annotated[std::size_t(j)]) {
              if (in_box) heatmask[q * 19 + std::size_t(j)] = 0;
              continue;
            }
            const double dx = x - points[std::size_t(j)][0], dy = y - points[std::size_t(j)][1];
            if (std::abs(dx) <= 3 * sigma && std::abs(dy) <= 3 * sigma)
              heat[q * 19 + std::size_t(j)] =
                  std::max(heat[q * 19 + std::size_t(j)], float(std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))));
          }
          for (int edge = 0; edge < 19; ++edge) {
            const int a = kPoseEdges[edge][0], b = kPoseEdges[edge][1];
            if (!annotated[std::size_t(a)] || !annotated[std::size_t(b)]) {
              if (in_box)
                for (int c = 0; c < 2; ++c) pafmask[q * 38 + std::size_t(2 * edge + c)] = 0;
              continue;
            }
            const auto&  pa = points[std::size_t(a)];
            const auto&  pb = points[std::size_t(b)];
            const double dx = pb[0] - pa[0], dy = pb[1] - pa[1], length = std::hypot(dx, dy);
            if (length < 1e-9) continue;
            const double ux = dx / length, uy = dy / length, rx = x - pa[0], ry = y - pa[1], along = rx * ux + ry * uy,
                         across = std::abs(rx * uy - ry * ux);
            if (along >= 0 && along <= length && across <= thickness) {
              paf[q * 38 + std::size_t(2 * edge)] += float(ux);
              paf[q * 38 + std::size_t(2 * edge + 1)] += float(uy);
              counts[q * 19 + std::size_t(edge)] += 1;
            }
          }
        }
    }
  for (std::size_t q = 0; q < pixels; ++q) {
    float maximum = 0, valid = 1;
    for (int j = 0; j < 18; ++j) {
      maximum = std::max(maximum, heat[q * 19 + std::size_t(j)]);
      valid   = std::min(valid, heatmask[q * 19 + std::size_t(j)]);
    }
    heat[q * 19 + 18]     = 1 - maximum;
    heatmask[q * 19 + 18] = valid;
    for (int edge = 0; edge < 19; ++edge)
      if (counts[q * 19 + std::size_t(edge)] > 0)
        for (int c = 0; c < 2; ++c) paf[q * 38 + std::size_t(2 * edge + c)] /= counts[q * 19 + std::size_t(edge)];
  }
  const nn::Shape s19{batch, side, side, 19}, s38{batch, side, side, 38};
  return {nn::from_vector(heat, s19), nn::from_vector(paf, s38), nn::from_vector(heatmask, s19), nn::from_vector(pafmask, s38)};
}

/// Per-image mean over masked entries, averaged over images with any mask;
/// masked-out predictions (even NaN) get exactly zero derivative.
inline nn::Tensor PafMaskedMSE(const nn::Tensor& prediction, const nn::Tensor& truth, const nn::Tensor& mask) {
  const nn::Tensor active  = nn::greater(mask, .5);
  const nn::Tensor p       = nn::where(active, prediction, 0.0), t = nn::where(active, truth, 0.0);
  const nn::Tensor count   = nn::sum(mask, {1, 2, 3});
  const nn::Tensor present = nn::cast(nn::greater(count, 0.0), prediction.dtype());
  const nn::Tensor mean    = nn::sum(nn::squared_difference(p, t), {1, 2, 3}) / nn::maximum(count, nn::scalar_like(1.0, count));
  return nn::sum(mean * present) / nn::maximum(nn::sum(present), nn::scalar_like(1.0, count));
}

}  // namespace pose
