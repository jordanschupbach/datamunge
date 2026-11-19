#pragma once

/// @file magnitude_pruning.hpp
/// @brief Stable one-shot and iterative magnitude masks.

#include <datamunge/nn/ops.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Keep a fixed number of the largest-magnitude eligible weights.
///
/// Traversal uses row-major order. Equal magnitudes prefer earlier positions.
/// Pass the previous mask to ensure that removed weights stay ineligible
/// during iterative pruning.
/// @param weights Finite CPU Float32 or Float64 tensor of any rank.
/// @param keep Number of entries to retain, from zero to the eligible count.
/// @param eligible Optional same-shape CPU numeric mask containing only zeros and ones.
/// @return Detached zero-one mask with the weights' shape and dtype.
/// @throws std::invalid_argument Invalid tensor, mask, or retained count.
/// @par Example
/// @code
/// const auto w = dm::nn::tensor({-4., 1., 3.});
/// const auto mask = dm::nn::magnitude_pruning_mask(w, 2);
/// std::println("{}", mask.values_str()); // [1, 0, 1]
/// @endcode
/// @par References
/// - Han et al. (2015), Learning both Weights and Connections for Efficient Neural Network,
///   https://arxiv.org/abs/1506.02626
/// @see prune_conv_channels()
[[nodiscard]] inline Tensor magnitude_pruning_mask(
    const Tensor& weights, std::int64_t keep, const Tensor& eligible = {}) {
  if (!weights.defined() || weights.device().backend != compute::Backend::CPU ||
      (weights.dtype() != DType::Float32 && weights.dtype() != DType::Float64) ||
      keep < 0 || keep > weights.numel() ||
      (eligible.defined() && (eligible.shape() != weights.shape() ||
                             eligible.device().backend != compute::Backend::CPU)))
    throw std::invalid_argument("magnitude_pruning_mask: incompatible weights, mask or count");

  const auto values = weights.to_vector<double>();
  const auto flags = eligible.defined() ? eligible.to_vector<double>() : std::vector<double>{};
  std::vector<std::size_t> order;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i]) || (!flags.empty() && flags[i] != 0. && flags[i] != 1.))
      throw std::invalid_argument("magnitude_pruning_mask: expected finite weights and a binary mask");
    if (flags.empty() || flags[i] == 1.) order.push_back(i);
  }
  if (static_cast<std::size_t>(keep) > order.size())
    throw std::invalid_argument("magnitude_pruning_mask: insufficient eligible weights");
  std::partial_sort(order.begin(), order.begin() + keep, order.end(), [&](auto a, auto b) {
    const auto x = std::abs(values[a]), y = std::abs(values[b]);
    return x == y ? a < b : x > y;
  });
  std::vector<double> mask(values.size());
  for (std::int64_t i = 0; i < keep; ++i) mask[order[i]] = 1.;
  return from_vector(mask).reshape(weights.shape()).to(weights.dtype());
}

} // namespace datamunge::nn
