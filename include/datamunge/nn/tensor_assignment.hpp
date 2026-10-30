#pragma once
/// @file tensor_assignment.hpp
/// @brief Tensor rectangular Hungarian matching with discrete augmenting-path metadata.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Solve minimum-cost rectangular one-to-one assignment using tensor primal/dual updates.
///
/// Successive augmenting paths use Float64 tensor reduced costs, distances,
/// potentials and minima. Host arrays contain only discrete owner/previous/visited
/// indices; scalar extractions select paths, never compute numerical costs.
/// Strict improvements and first-minimum columns preserve deterministic source
/// ties. O(R²*C) arithmetic; tensor dispatch overhead favors small target sets.
/// No gradients, parameter/state mutation, random draws or fitting.
///
/// @param cost Float32/64 [R,C] finite matrix with R<=C; negative costs are supported.
/// @return Detached Int64 [R], one distinct zero-based column per row; empty R yields empty output.
/// @throws std::invalid_argument For invalid shape/dtype, nonfinite costs, nonfinite potential updates, R>C or dimensions beyond int range.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto a =
///     n::models::TensorHungarianAssignment(n::tensor({4., 1., 3., 2., 0., 5.}, {2, 3}));
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872
/// @see MatchDetrTargets()
inline nn::Tensor TensorHungarianAssignment(const nn::Tensor &cost) {
  if (cost.dim() != 2 ||
      (cost.dtype() != nn::DType::Float32 && cost.dtype() != nn::DType::Float64) ||
      cost.shape()[0] > cost.shape()[1] || cost.shape()[1] > std::numeric_limits<int>::max() ||
      !nn::all(nn::less(nn::abs(cost), std::numeric_limits<double>::infinity())).item<bool>())
    throw std::invalid_argument("invalid rectangular assignment costs");
  nn::NoGradGuard guard;
  const int R = int(cost.shape()[0]), C = int(cost.shape()[1]);
  if (!R)
    return nn::zeros({0}, nn::DType::Int64);
  const auto type = nn::DType::Float64;
  const auto matrix = cost.to(type);
  auto u = nn::zeros({R + 1}, type), v = nn::zeros({C + 1}, type);
  std::vector<std::int64_t> owner(C + 1), previous(C + 1);
  const auto scalar = [](std::int64_t i) {
    return nn::full({}, double(i), nn::DType::Int64);
  };
  for (int row = 1; row <= R; ++row) {
    owner[0] = row;
    int current = 0;
    auto distance = nn::full({C + 1}, std::numeric_limits<double>::infinity(), type);
    std::vector<std::int64_t> visited(C + 1, 0);
    do {
      visited[current] = 1;
      const auto seen = nn::from_vector(visited).to(nn::DType::Bool);
      const auto reduced =
          nn::concat(
              {nn::zeros({1}, type),
               nn::slice_axis(matrix, 0, owner[current] - 1, owner[current]).reshape({C})},
              0) -
          nn::gather(u, scalar(owner[current])) - v;
      const auto improve = nn::logical_and(nn::logical_not(seen), nn::less(reduced, distance));
      const auto changed = improve.to_vector<bool>();
      for (int col = 1; col <= C; ++col)
        if (changed[col])
          previous[col] = current;
      distance = nn::where(improve, reduced, distance);
      const auto available = nn::where(
          seen, nn::full({}, std::numeric_limits<double>::infinity(), type), distance);
      const int next = nn::argmin(available).item<int>();
      const auto delta = nn::min(available);
      if (!std::isfinite(delta.item()))
        throw std::invalid_argument("assignment potential overflow");
      const auto mask = seen.to(type);
      u = u + nn::matmul(nn::one_hot(nn::from_vector(owner), R + 1, 1, 0, type),
                         mask.reshape({C + 1, 1}), true, false)
                      .reshape({R + 1}) *
                  delta;
      v = v - mask * delta;
      distance = distance - nn::where(seen, nn::zeros({}, type), delta);
      current = next;
    } while (owner[current] != 0);
    do {
      const int parent = int(previous[current]);
      owner[current] = owner[parent];
      current = parent;
    } while (current != 0);
  }
  std::vector<std::int64_t> assignment(R, -1);
  for (int col = 1; col <= C; ++col)
    if (owner[col])
      assignment[owner[col] - 1] = col - 1;
  return nn::from_vector(assignment);
}
} // namespace datamunge::nn::models
