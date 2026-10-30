#pragma once
/// @file point_geometry.hpp
/// @brief Tensor numeric farthest-point/radius geometry with frozen categorical choices and differentiable gathers.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Gather batched point features with differentiable duplicate accumulation.
///
/// Tensor batch offsets keep clouds separate. Repeated indices scatter-add
/// adjoints into the same original point. Does not mutate inputs;no fitting.
///
/// @param points Floating[B,N,C],all axes positive.
/// @param indices Int32/64[B,S] or[B,S,K],all axes positive,local IDs0..N-1.
/// @return Floating[B,S,C] or[B,S,K,C],same dtype/graph as points.
/// @throws std::invalid_argument For invalid ranks,empty axes,integer index dtype,batch mismatch or bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto xyz = n::tensor({0., 0., 0., 1., 0., 0., 0., 1., 0., .25, .25, 0.}, {1, 4, 3});
/// const auto gathered = m::PointGather(xyz, n::tensor({0, 3, 3}, {1, 3}).to(n::DType::Int32));
/// @endcode
/// @par References
/// - Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching policy starts FPS at index0,uses lowest-index ties,strict radius,first input-order hits and first-hit repeated padding;discrete choices are frozen in coordinate derivatives.
/// @see PointGroup()
inline Tensor PointGather(const Tensor &points, const Tensor &indices) {
  if (points.dim() != 3 || points.size(0) < 1 || points.size(1) < 1 || points.size(2) < 1 ||
      (indices.dim() != 2 && indices.dim() != 3) || indices.size(0) != points.size(0) ||
      indices.numel() < 1 ||
      (indices.dtype() != nn::DType::Int32 && indices.dtype() != nn::DType::Int64) ||
      !nn::all(nn::logical_and(nn::greater_equal(indices, 0.),
                               nn::less(indices, double(points.size(1)))))
           .item<bool>())
    throw std::invalid_argument(
        "bounded integer point gather index rank2or3 and positive axes required");
  auto offset_shape = nn::Shape(indices.dim(), 1);
  offset_shape[0] = points.size(0);
  const auto offsets =
      (nn::arange(0, double(points.size(0)), 1., nn::DType::Int64) * points.size(1))
          .reshape(offset_shape);
  return nn::gather(points.reshape({points.size(0) * points.size(1), points.size(2)}),
                    indices.to(nn::DType::Int64) + offsets, 0);
}
/// @brief Select deterministic farthest points using tensor distance updates.
///
/// Start index0;nearest squared distances use Double arithmetic after source
/// Float32 conversion. First argmax resolves ties to lowest index. A collapsed
/// cloud repeats seed0. All numeric distances/reductions use tensors.
/// Selections are discrete and never differentiated;O(B*N*count)work/O(B*N)storage.
///
/// @param xyz Finite floating[B,N,3],positive B,N;source ingress rounds to Float32.
/// @param count Requested samples1..N.
/// @return Int32[B,count] detached local index metadata.
/// @throws std::invalid_argument For invalid shape,empty batch,nonfinite coordinates or N<count.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto xyz = n::tensor({0., 0., 0., 1., 0., 0., 0., 1., 0., .25, .25, 0.}, {1, 4, 3});
/// const auto ids = m::PointFps(xyz, 3);
/// @endcode
/// @par References
/// - Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching policy starts FPS at index0,uses lowest-index ties,strict radius,first input-order hits and first-hit repeated padding;discrete choices are frozen in coordinate derivatives.
/// @see PointGroup()
inline Tensor PointFps(const Tensor &xyz, int count) {
  if (count < 1 || xyz.dim() != 3 || xyz.size(0) < 1 || xyz.size(2) != 3 ||
      xyz.size(1) < count)
    throw std::invalid_argument("FPS expects[B,N,3],N>=count and positive B");
  nn::NoGradGuard guard;
  const auto x = xyz.to(nn::DType::Float32).to(nn::DType::Float64);
  if (!nn::all(nn::is_finite(x)).item<bool>())
    throw std::invalid_argument("finite XYZ required");
  const auto batch = x.size(0), n = x.size(1);
  auto selected = nn::zeros({batch}, nn::DType::Int32),
       nearest = nn::full({batch, n}, INFINITY, nn::DType::Float64);
  std::vector<Tensor> ids;
  for (int i = 0; i < count; ++i) {
    ids.push_back(selected);
    const auto center = PointGather(x, selected.reshape({batch, 1}));
    const auto distance = nn::sum(nn::square(x - center), {2});
    nearest = nn::minimum(nearest, distance);
    selected = nn::argmax(nearest, 1).to(nn::DType::Int32);
  }
  return nn::stack(ids, 1);
}
/// @brief Own capped radius-neighborhood identities and unpadded counts.
///
/// Detached categorical tensors own storage;counts exclude repeated padding.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto xyz = n::tensor({0., 0., 0., 1., 0., 0., 0., 1., 0., .25, .25, 0.}, {1, 4, 3});
/// const auto ball =
///     m::PointBall(xyz, m::PointGather(xyz, n::zeros({1, 1}, n::DType::Int32)), 1., 4);
/// @endcode
/// @par References
/// - Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching policy starts FPS at index0,uses lowest-index ties,strict radius,first input-order hits and first-hit repeated padding;discrete choices are frozen in coordinate derivatives.
/// @see PointGroup()
struct PointNeighborhood {
  Tensor indices; ///< Int32[B,S,K] local input-order indices,first-hit padding.
  Tensor counts;  ///< Int32[B,S] min(number of strict-radius hits,K).
};
/// @brief Query strict-radius input-order neighborhoods using tensor distances and ranking.
///
/// Compare Double squared distances after Float32 source conversion. Select
/// first hits in original point order,not distance order;repeat first hit in
/// unused slots. Empty neighborhoods throw. O(B*S*N)distance storage.
/// Discrete membership/ranking is frozen and has no geometry derivative.
///
/// @param xyz Finite floating[B,N,3],positive axes;source Float32 ingress.
/// @param centers Finite floating[B,S,3],same B,positive S;normally gathered source centers.
/// @param radius Finite strictly positive Float32 radius;boundary distance==radius is excluded.
/// @param count Positive output slots;can exceed N,then first-hit padding fills remaining slots.
/// @return Detached indices[B,S,count] and capped distinct counts[B,S].
/// @throws std::invalid_argument For incompatible/empty shapes,nonfinite geometry,invalid radius/count or an empty radius neighborhood.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto xyz = n::tensor({0., 0., 0., 1., 0., 0., 0., 1., 0., .25, .25, 0.}, {1, 4, 3});
/// const auto ball =
///     m::PointBall(xyz, m::PointGather(xyz, n::zeros({1, 1}, n::DType::Int32)), 1., 4);
/// @endcode
/// @par References
/// - Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching policy starts FPS at index0,uses lowest-index ties,strict radius,first input-order hits and first-hit repeated padding;discrete choices are frozen in coordinate derivatives.
/// @see PointGroup()
inline PointNeighborhood PointBall(const Tensor &xyz, const Tensor &centers, float radius,
                                   int count) {
  if (!std::isfinite(radius) || radius <= 0)
    throw std::invalid_argument("positive finite radius required");
  if (count < 1 || xyz.dim() != 3 || centers.dim() != 3 || xyz.size(0) < 1 ||
      xyz.size(0) != centers.size(0) || xyz.size(2) != 3 || centers.size(2) != 3 ||
      xyz.size(1) < 1 || centers.size(1) < 1)
    throw std::invalid_argument("ball query requires compatible positive[B,N,3]/[B,S,3]");
  nn::NoGradGuard guard;
  const auto x = xyz.to(nn::DType::Float32).to(nn::DType::Float64),
             c = centers.to(nn::DType::Float32).to(nn::DType::Float64);
  if (!nn::all(nn::is_finite(x)).item<bool>() || !nn::all(nn::is_finite(c)).item<bool>())
    throw std::invalid_argument("finite geometry required");
  const auto b = x.size(0), n = x.size(1), s = c.size(1);
  const auto distance =
      nn::sum(nn::square(nn::expand_dims(x, 1) - nn::expand_dims(c, 2)), {3});
  const auto inside = nn::less(distance, double(radius) * radius);
  const auto total = nn::sum(inside.to(nn::DType::Int32), {2});
  if (nn::any(nn::equal(total, 0.)).item<bool>())
    throw std::invalid_argument(
        "empty radius neighborhood;model centers must come from input cloud");
  const auto first = nn::argmax(inside.to(nn::DType::Int32), 2).to(nn::DType::Int32);
  const auto candidates = nn::where(
      inside, nn::arange(0, double(n), 1., nn::DType::Int64).reshape({1, 1, n}), double(n));
  auto selected =
      (-nn::top_k(-candidates.to(nn::DType::Float64), std::min<std::int64_t>(count, n)).first)
          .to(nn::DType::Int32);
  selected = nn::where(nn::less(selected, double(n)), selected, nn::expand_dims(first, 2));
  if (count > n)
    selected = nn::concat(
        {selected, nn::broadcast_to(nn::expand_dims(first, 2), {b, s, count - n})}, 2);
  return {selected, nn::minimum(total, nn::full(total.shape(), count, nn::DType::Int32))};
}
/// @brief Own sampled centers,local coordinates and fixed neighborhood metadata.
///
/// Centers/relative coordinates retain input graph;indices/counts are detached.
/// No fitted parameters are introduced by grouping.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto xyz = n::tensor({0., 0., 0., 1., 0., 0., 0., 1., 0., .25, .25, 0.}, {1, 4, 3});
/// const auto grouped = m::PointGroup(xyz, 2, 1., 4);
/// @endcode
/// @par References
/// - Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching policy starts FPS at index0,uses lowest-index ties,strict radius,first input-order hits and first-hit repeated padding;discrete choices are frozen in coordinate derivatives.
/// @see PointGroup()
struct PointGrouped {
  Tensor centers;  ///< Floating[B,S,3] sampled source coordinates.
  Tensor relative; ///< Floating[B,S,K,3] gathered XYZ minus centers.
  Tensor indices;  ///< Detached Int32[B,S,K] original point identities.
  Tensor counts;   ///< Detached Int32[B,S] capped unpadded counts.
};
/// @brief Sample centers and form differentiable centroid-relative XYZ groups.
///
/// FPS/radius queries see detached coordinates. Gather and subtraction retain
/// the graph,so every neighbor contributes and the center receives the negative
/// sum of local-coordinate adjoints. Translation cancels for fixed membership.
/// No derivative is promised through the discrete selection boundaries.
///
/// @param xyz Finite floating[B,N,3],positive axes.
/// @param centers FPS center count1..N.
/// @param radius Finite positive Float32 radius.
/// @param neighbors Positive output neighborhood count,first-hit padded.
/// @return Sampled centers,relative XYZ and discrete neighborhood metadata.
/// @throws std::invalid_argument For invalid geometry,sampling/radius/count or empty neighborhoods.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto xyz = n::tensor({0., 0., 0., 1., 0., 0., 0., 1., 0., .25, .25, 0.}, {1, 4, 3});
/// const auto grouped = m::PointGroup(xyz, 2, 1., 4);
/// @endcode
/// @par References
/// - Qi,Yi,Su and Guibas(2017),PointNet++:Deep Hierarchical Feature Learning on Point Sets in a Metric Space, https://arxiv.org/abs/1706.02413. Source teaching policy starts FPS at index0,uses lowest-index ties,strict radius,first input-order hits and first-hit repeated padding;discrete choices are frozen in coordinate derivatives.
/// @see PointGroup()
inline PointGrouped PointGroup(const Tensor &xyz, int centers, float radius, int neighbors) {
  const auto fps = PointFps(xyz.detach(), centers);
  const auto centroids = PointGather(xyz, fps);
  const auto ball = PointBall(xyz.detach(), centroids.detach(), radius, neighbors);
  return {centroids, PointGather(xyz, ball.indices) - nn::expand_dims(centroids, 2),
          ball.indices, ball.counts};
}
} // namespace datamunge::nn::models
