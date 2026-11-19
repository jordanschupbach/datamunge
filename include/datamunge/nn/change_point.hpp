#pragma once

/// @file change_point.hpp
/// @brief Neural localization of a single offline change point.

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Score candidate boundaries and a no-change class with a shared MLP.
///
/// Candidates range from radius through length minus radius, inclusive.
/// Each window is centered by its left-half mean before scoring. The learned
/// no-change head reads the maximum candidate score. This offline model uses
/// observations on both sides of each boundary.
/// @param store CPU parameter store matching the input dtype.
/// @param name Prefix for the learned layers.
/// @param input Float32 or Float64 CPU matrix [batch, length].
/// @param radius Positive number of observations per side.
/// @param width Positive ReLU hidden width, default 16.
/// @return Differentiable logits [batch, length - 2 * radius + 2], with no change last.
/// @throws std::invalid_argument Incompatible input, window size, or stored layers.
/// @par Example
/// @code
/// dm::nn::ParameterStore p;
/// const auto logits = dm::nn::change_point_logits(p, "detector", dm::nn::ones({2, 10}), 3);
/// std::println("{}", logits.size(1)); // 6
/// @endcode
/// @see dense(), softmax()
[[nodiscard]] inline Tensor change_point_logits(
    ParameterStore& store, const std::string& name, const Tensor& input,
    std::int64_t radius, std::int64_t width = 16) {
  if (!input.defined() || input.dim() != 2 || input.size(0) < 1 ||
      input.device().backend != compute::Backend::CPU || input.dtype() != store.dtype() ||
      radius < 1 || radius > input.size(1) / 2 || width < 1)
    throw std::invalid_argument("change_point_logits: invalid input or window settings");

  const auto rows = input.size(0);
  const auto count = input.size(1) - 2 * radius + 1;
  std::vector<Tensor> segments;
  segments.reserve(count);
  for (std::int64_t i = 0; i < count; ++i)
    segments.push_back(slice(input, {0, i}, {rows, 2 * radius}));
  const auto windows = stack(segments, 1).reshape({-1, 2 * radius});
  const auto left = slice(windows, {0, 0}, {windows.size(0), radius});
  const auto centered = windows - mean(left, {1}, true);
  const auto hidden = dense(store, name + "/hidden", centered, width, Activation::Relu);
  const auto candidates = dense(store, name + "/boundary", hidden, 1).reshape({rows, count});
  const auto no_change = dense(store, name + "/null", max(candidates, {1}, true), 1);
  return concat({candidates, no_change}, 1);
}

}  // namespace datamunge::nn
