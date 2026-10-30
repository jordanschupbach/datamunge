// PointNet++ CPU sampling/grouping: farthest point sampling, radius (ball)
// query and batched point gathers. A port of tflow's common/nn/point_geometry.h.
//
// TF -> datamunge: tflow registers two custom CPU kernels (TflowPointFps,
// TflowPointBall, no gradient) and builds PointGather from GatherV2. Here FPS
// and the ball query are plain host functions returning int32 index tensors
// (discrete, so nothing to differentiate; their errors are exceptions instead
// of failed op statuses), and PointGather is nn::gather on the flattened
// points with per-cloud offsets, whose gradient scatters (and accumulates
// duplicates) like GatherV2's.
#pragma once

#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <limits>
#include <stdexcept>
#include <vector>

namespace point_geometry_reference {

namespace nn = datamunge::nn;
using nn::Tensor;

/// Farthest point sampling of xyz [B, N, 3] -> int32 [B, count]: starts at point 0;
/// distance ties go to the lowest index (a collapsed cloud repeats the seed).
inline Tensor PointFps(const Tensor& xyz, int count) {
  if (count < 1 || xyz.dim() != 3 || xyz.size(2) != 3 || xyz.size(1) < count)
    throw std::invalid_argument("FPS expects[B,N,3],N>=count");
  const auto         x = nn::cast(xyz, nn::DType::Float32).to_vector<float>();
  for (float v : x)
    if (!std::isfinite(v)) throw std::invalid_argument("finite XYZ required");
  const std::int64_t batch = xyz.size(0), n = xyz.size(1);
  std::vector<int>   indices(static_cast<std::size_t>(batch * count));
  for (std::int64_t b = 0; b < batch; ++b) {
    std::vector<double> nearest(static_cast<std::size_t>(n), std::numeric_limits<double>::infinity());
    int                 selected = 0;
    for (int i = 0; i < count; ++i) {
      indices[static_cast<std::size_t>(b * count + i)] = selected;
      double farthest = -1;
      int    next     = 0;
      for (std::int64_t j = 0; j < n; ++j) {
        double distance = 0;
        for (int axis = 0; axis < 3; ++axis) {
          const double d = double(x[static_cast<std::size_t>((b * n + j) * 3 + axis)]) -
                           x[static_cast<std::size_t>((b * n + selected) * 3 + axis)];
          distance += d * d;
        }
        auto& near = nearest[static_cast<std::size_t>(j)];
        near       = std::min(near, distance);
        if (near > farthest) {
          farthest = near;
          next     = static_cast<int>(j);
        }
      }
      selected = next;
    }
  }
  return nn::from_vector(indices, {batch, count});
}

struct PointNeighborhood {
  Tensor indices, counts;  // int32 [B, S, count] and [B, S]
};

/// Radius query: for each center [B, S, 3] the first `count` points of xyz [B, N, 3] in input
/// order with squared distance < radius^2; unused slots repeat the first hit. An empty
/// neighborhood is an error (centers must come from the input cloud).
inline PointNeighborhood PointBall(const Tensor& xyz, const Tensor& centers, float radius, int count) {
  if (!std::isfinite(radius) || radius <= 0) throw std::invalid_argument("positive finite radius required");
  if (count < 1 || xyz.dim() != 3 || centers.dim() != 3 || xyz.size(0) != centers.size(0) || xyz.size(2) != 3 ||
      centers.size(2) != 3 || xyz.size(1) < 1)
    throw std::invalid_argument("ball query requires compatible[B,N,3]/[B,S,3]");
  const auto x = nn::cast(xyz, nn::DType::Float32).to_vector<float>();
  const auto c = nn::cast(centers, nn::DType::Float32).to_vector<float>();
  for (const auto* values : {&x, &c})
    for (float v : *values)
      if (!std::isfinite(v)) throw std::invalid_argument("finite geometry required");
  const std::int64_t batch = xyz.size(0), n = xyz.size(1), s_count = centers.size(1);
  const double       radius_squared = double(radius) * radius;
  std::vector<int>   out(static_cast<std::size_t>(batch * s_count * count)), counts(static_cast<std::size_t>(batch * s_count));
  for (std::int64_t b = 0; b < batch; ++b)
    for (std::int64_t s = 0; s < s_count; ++s) {
      int*      row   = out.data() + (b * s_count + s) * count;
      int       found = 0;
      for (std::int64_t j = 0; j < n && found < count; ++j) {
        double distance = 0;
        for (int axis = 0; axis < 3; ++axis) {
          const double d = double(x[static_cast<std::size_t>((b * n + j) * 3 + axis)]) -
                           c[static_cast<std::size_t>((b * s_count + s) * 3 + axis)];
          distance += d * d;
        }
        if (distance < radius_squared) {
          if (found == 0)
            for (int k = 0; k < count; ++k) row[k] = static_cast<int>(j);
          row[found++] = static_cast<int>(j);
        }
      }
      if (found == 0)
        throw std::invalid_argument("empty radius neighborhood;model centers must come from input cloud");
      counts[static_cast<std::size_t>(b * s_count + s)] = found;
    }
  return {nn::from_vector(out, {batch, s_count, count}), nn::from_vector(counts, {batch, s_count})};
}

/// Gathers rows of points [B, N, C] with per-cloud int indices [B, S] or [B, S, K]
/// -> [B, S, C] or [B, S, K, C]. Batch offsets keep the clouds separate.
inline Tensor PointGather(const Tensor& points, const Tensor& indices) {
  if (points.dim() != 3 || (indices.dim() != 2 && indices.dim() != 3) || indices.size(0) != points.size(0))
    throw std::invalid_argument("point gather index rank2or3");
  const std::int64_t batch = points.size(0), n = points.size(1), channels = points.size(2);
  const auto         local = nn::cast(indices, nn::DType::Int32).to_vector<int>();
  const std::int64_t per   = indices.numel() / batch;
  std::vector<int>   global(local.size());
  for (std::size_t i = 0; i < local.size(); ++i) {
    const std::int64_t b = static_cast<std::int64_t>(i) / per;
    if (local[i] < 0 || local[i] >= n) throw std::invalid_argument("point index out of range");
    global[i] = static_cast<int>(local[i] + b * n);
  }
  return nn::gather(nn::reshape(points, {batch * n, channels}), nn::from_vector(global, indices.shape()), 0);
}

struct PointGrouped {
  Tensor centers, relative, indices, counts;
};

/// FPS centers, ball query and centroid-relative grouped XYZ [B, S, K, 3].
inline PointGrouped PointGroup(const Tensor& xyz, int centers, float radius, int neighbors) {
  const Tensor fps       = PointFps(xyz.detach(), centers);
  const Tensor centroids = PointGather(xyz, fps);
  const auto   ball      = PointBall(xyz.detach(), centroids.detach(), radius, neighbors);
  const Tensor grouped   = PointGather(xyz, ball.indices);
  return {centroids, grouped - nn::expand_dims(centroids, 2), ball.indices, ball.counts};
}

}  // namespace point_geometry_reference
