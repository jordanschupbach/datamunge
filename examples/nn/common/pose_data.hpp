// Source-image-disjoint real COCO17 person crops, Gaussian targets and PCK: a
// port of tflow's common/ml/pose_data.h. The arrays come from tflow's
// human_pose/poses.npz (scripts/fetch_datasets.py human_pose); nn tensors
// replace tf::Tensor (float32 RGB/joints/boxes, int32 ids/counts).
#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/hourglass.hpp"

namespace pose {

struct PoseData {
  nn::Tensor rgb, joints, box_size, image_ids, person_ids;
};
struct PoseSplits {
  PoseData fit, validation, test;
};

/// npz members by name.
inline std::map<std::string, nn::Tensor> PoseArrays(const std::filesystem::path& file) {
  std::map<std::string, nn::Tensor> arrays;
  for (auto& [name, value] : nn::load_npz(file.string())) arrays[name] = value;
  return arrays;
}

/// Rows of `input` (axis 0) in the given order.
inline nn::Tensor PoseBatch(const nn::Tensor& input, const std::vector<std::size_t>& rows) {
  std::vector<std::int64_t> index;
  for (std::size_t row : rows) {
    check::Require(row < std::size_t(input.size(0)), "pose row bounds");
    index.push_back(std::int64_t(row));
  }
  return nn::gather(input, nn::from_vector(index, {std::int64_t(index.size())}), 0).contiguous();
}
inline PoseData PoseBatch(const PoseData& input, const std::vector<std::size_t>& rows) {
  return {PoseBatch(input.rgb, rows), PoseBatch(input.joints, rows), PoseBatch(input.box_size, rows), PoseBatch(input.image_ids, rows),
          PoseBatch(input.person_ids, rows)};
}

inline PoseSplits LoadPoseCrops(const std::filesystem::path& directory) {
  const auto    arrays = PoseArrays(directory / "poses.npz");
  std::set<int> previous;
  auto          read = [&](const std::string& split, std::int64_t count) {
    PoseData result{arrays.at(split + "_rgb"), arrays.at(split + "_joints"), arrays.at(split + "_box_size"), arrays.at(split + "_image_ids"),
                    arrays.at(split + "_person_ids")};
    const std::vector<std::pair<nn::Tensor, nn::Shape>> shapes = {{result.rgb, {count, 128, 128, 3}}, {result.joints, {count, 17, 3}},
                                                                  {result.box_size, {count, 2}},      {result.image_ids, {count}},
                                                                  {result.person_ids, {count}}};
    for (const auto& item : shapes) check::Require(item.first.shape() == item.second, "pose data dimensions");
    check::Require(result.rgb.dtype() == nn::DType::Float32 && result.joints.dtype() == nn::DType::Float32 &&
                       result.box_size.dtype() == nn::DType::Float32 && result.image_ids.dtype() == nn::DType::Int32 &&
                       result.person_ids.dtype() == nn::DType::Int32,
                   "pose numeric dtypes");
    const auto    joints = result.joints.to_vector<float>(), box = result.box_size.to_vector<float>();
    const auto    image_ids = result.image_ids.to_vector<std::int32_t>();
    std::set<int> ids;
    for (std::int64_t n = 0; n < count; ++n) {
      const int id = image_ids[std::size_t(n)];
      check::Require(!previous.count(id), "source images never cross pose splits");
      ids.insert(id);
      int labeled = 0;
      for (int j = 0; j < 17; ++j) {
        const float v = joints[std::size_t((n * 17 + j) * 3 + 2)];
        check::Require(v == 0 || v == 1 || v == 2, "COCO visibility0/1/2");
        if (v > 0) {
          ++labeled;
          for (int c = 0; c < 2; ++c) {
            const float x = joints[std::size_t((n * 17 + j) * 3 + c)];
            check::Require(std::isfinite(x) && x >= 0 && x <= 127, "labeled person-crop keypoint within image");
          }
        }
      }
      check::Require(labeled > 0, "every crop has annotated joints");
      for (int c = 0; c < 2; ++c)
        check::Require(box[std::size_t(n * 2 + c)] > 0 && std::isfinite(box[std::size_t(n * 2 + c)]), "positive person box scale");
    }
    for (float v : result.rgb.to_vector<float>()) check::Require(std::isfinite(v) && v >= 0 && v <= 1, "observed RGB range");
    previous.insert(ids.begin(), ids.end());
    return result;
  };
  auto fit = read("fit", 12), validation = read("validation", 1), test = read("test", 14);
  return {fit, validation, test};
}

struct PoseTargets {
  nn::Tensor heatmap, mask;
};

/// Unnormalised Gaussians (3 sigma window) at half-pixel-mapped joint centres;
/// mask [N, 1, 1, J] is 1 for annotated in-image joints.
inline PoseTargets PoseHeatmaps(const nn::Tensor& joints, int side = 32, int input_size = 128, float sigma = 1.f) {
  check::Require(joints.dim() == 3 && joints.size(2) == 3 && side > 0 && input_size > 0 && sigma > 0, "Gaussian pose target specification");
  const std::int64_t batch = joints.size(0), channels = joints.size(1);
  const auto         jv    = joints.to_vector<float>();
  std::vector<float> heat(std::size_t(batch * side * side * channels), 0.f), mask(std::size_t(batch * channels), 0.f);
  for (std::int64_t n = 0; n < batch; ++n)
    for (std::int64_t j = 0; j < channels; ++j)
      if (jv[std::size_t((n * channels + j) * 3 + 2)] > 0) {
        const float px = jv[std::size_t((n * channels + j) * 3)], py = jv[std::size_t((n * channels + j) * 3 + 1)];
        if (!std::isfinite(px) || !std::isfinite(py) || px < 0 || px > input_size - 1 || py < 0 || py > input_size - 1) continue;
        mask[std::size_t(n * channels + j)] = 1;
        const double cx = (px + .5) * side / input_size - .5, cy = (py + .5) * side / input_size - .5;
        for (int y = 0; y < side; ++y)
          for (int x = 0; x < side; ++x)
            if (std::abs(x - cx) <= 3 * sigma && std::abs(y - cy) <= 3 * sigma)
              heat[std::size_t(((n * side + y) * side + x) * channels + j)] =
                  float(std::exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / (2 * sigma * sigma)));
      }
  return {nn::from_vector(heat, {batch, side, side, channels}), nn::from_vector(mask, {batch, 1, 1, channels})};
}

/// Per-image mean squared error over annotated joint channels, averaged over images with any annotation.
inline nn::Tensor PoseHeatmapMSE(const nn::Tensor& prediction, const nn::Tensor& target, const nn::Tensor& mask, int side) {
  const nn::Tensor active  = nn::greater(nn::broadcast_to(mask, prediction.shape()), .5);
  const nn::Tensor p       = nn::where(active, prediction, 0.0), t = nn::where(active, target, 0.0);
  const nn::Tensor count   = nn::sum(mask, {1, 2, 3});
  const nn::Tensor present = nn::cast(nn::greater(count, 0.0), prediction.dtype());
  const nn::Tensor error   = nn::sum(nn::squared_difference(p, t), {1, 2, 3}) / nn::maximum(count * double(side * side), nn::scalar_like(1.0, count));
  return nn::sum(error * present) / nn::maximum(nn::sum(present), nn::scalar_like(1.0, count));
}

/// Argmax (first maximum in row-major order) per joint channel -> [N, J, 3] (x, y, peak) in input pixels.
inline nn::Tensor DecodePoseHeatmaps(const nn::Tensor& heatmaps, int input_size = 128) {
  check::Require(heatmaps.dim() == 4 && heatmaps.size(1) == heatmaps.size(2), "square keypoint heatmaps");
  const std::int64_t batch = heatmaps.size(0), side = heatmaps.size(1), channels = heatmaps.size(3);
  const auto         h     = heatmaps.to_vector<float>();
  std::vector<float> result(std::size_t(batch * channels * 3));
  for (std::int64_t n = 0; n < batch; ++n)
    for (std::int64_t j = 0; j < channels; ++j) {
      float        best  = -INFINITY;
      std::int64_t index = 0;
      for (std::int64_t i = 0; i < side * side; ++i) {
        const float value = h[std::size_t((n * side * side + i) * channels + j)];
        check::Require(std::isfinite(value), "finite predicted pose heatmaps");
        if (value > best) best = value, index = i;
      }
      const std::size_t o = std::size_t((n * channels + j) * 3);
      result[o]           = (float(index % side) + .5f) * input_size / float(side) - .5f;
      result[o + 1]       = (float(index / side) + .5f) * input_size / float(side) - .5f;
      result[o + 2]       = best;
    }
  return nn::from_vector(result, {batch, channels, 3});
}

struct PoseMetrics {
  double pck = .0, normalized_error = .0, pixels = .0;
  int    joints = 0;
};

/// PCK@.2 of the larger box side, mean normalised and pixel error over annotated joints.
inline PoseMetrics ScorePose(const nn::Tensor& prediction, const PoseData& data) {
  check::Require(prediction.shape() == data.joints.shape(), "decoded joint shape matches COCO17 truth");
  const auto  p = prediction.to_vector<float>(), t = data.joints.to_vector<float>(), box = data.box_size.to_vector<float>();
  PoseMetrics result;
  for (std::int64_t n = 0; n < data.joints.size(0); ++n)
    for (int j = 0; j < 17; ++j)
      if (t[std::size_t((n * 17 + j) * 3 + 2)] > 0) {
        const std::size_t o  = std::size_t((n * 17 + j) * 3);
        const double      dx = double(p[o]) - t[o], dy = double(p[o + 1]) - t[o + 1], error = std::hypot(dx, dy);
        const double      scale = std::max(box[std::size_t(n * 2)], box[std::size_t(n * 2 + 1)]);
        check::Require(std::isfinite(error) && scale > 0, "finite normalized keypoint distance");
        result.pixels += error;
        result.normalized_error += error / scale;
        result.pck += error <= .2 * scale;
        ++result.joints;
      }
  check::Require(result.joints > 0, "PCK needs annotated joints");
  result.pck /= result.joints;
  result.normalized_error /= result.joints;
  result.pixels /= result.joints;
  return result;
}

/// Mean annotated position per joint over the fitting crops ((63.5, 63.5) when never labeled) -> [17, 3].
inline nn::Tensor FitPoseMean(const PoseData& fitting) {
  const auto         t = fitting.joints.to_vector<float>();
  std::vector<float> mean(51, 0.f);
  for (int j = 0; j < 17; ++j) {
    int    count = 0;
    double x = 0, y = 0;
    for (std::int64_t n = 0; n < fitting.joints.size(0); ++n)
      if (t[std::size_t((n * 17 + j) * 3 + 2)] > 0) {
        ++count;
        x += t[std::size_t((n * 17 + j) * 3)];
        y += t[std::size_t((n * 17 + j) * 3 + 1)];
      }
    mean[std::size_t(j * 3)]     = float(count ? x / count : 63.5);
    mean[std::size_t(j * 3 + 1)] = float(count ? y / count : 63.5);
    mean[std::size_t(j * 3 + 2)] = 1;
  }
  return nn::from_vector(mean, {17, 3});
}

inline nn::Tensor ConstantPose(const PoseData& data, const nn::Tensor& mean) {
  const auto         m = mean.to_vector<float>();
  std::vector<float> result;
  for (std::int64_t n = 0; n < data.joints.size(0); ++n) result.insert(result.end(), m.begin(), m.end());
  return nn::from_vector(result, data.joints.shape());
}

}  // namespace pose
