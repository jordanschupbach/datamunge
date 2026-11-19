#pragma once

/// @file channel_pruning.hpp
/// @brief L1 filter selection and channel compaction for adjacent convolutions.

#include <datamunge/nn/ops.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Compacted filter pair and retained intermediate channel indices.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto p = n::prune_conv_channels(n::ones({1,1,2,3}), n::zeros({3}),
///                                      n::ones({1,1,3,2}), 2);
/// std::println("{}", p.first.size(3)); // 2
/// @endcode
/// @see prune_conv_channels()
struct PrunedConvChannels {
  Tensor first; ///< First kernel with only retained output channels.
  Tensor bias; ///< First-layer bias restricted to retained channels.
  Tensor second; ///< Second kernel with only retained input channels.
  Tensor kept; ///< Int32 original channel indices, ordered by decreasing score.
};

/// @brief Select intermediate channels by first-kernel L1 norm and compact both kernels.
///
/// Ties keep the smaller original channel index first. The result applies to
/// ungrouped convolutions with a channelwise activation between them.
/// @param first Finite CPU Float32 or Float64 kernel [height, width, inputs, hidden].
/// @param bias First convolution's bias [hidden].
/// @param second Finite second kernel [height, width, hidden, outputs].
/// @param keep Number of channels to retain, between one and hidden inclusive.
/// @return Compacted kernels, bias, and selected indices.
/// @throws std::invalid_argument Incompatible shapes, CPU device, dtype, or nonfinite values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto p = n::prune_conv_channels(n::tensor({1., 3.}, {1,1,1,2}),
///     n::zeros({2}), n::ones({1,1,2,1}), 1);
/// std::println("{}", p.kept.item<int>()); // 1
/// @endcode
/// @par References
/// - Li et al. (2017), Pruning Filters for Efficient ConvNets, https://arxiv.org/abs/1608.08710
/// @see gather(), conv2d()
[[nodiscard]] inline PrunedConvChannels prune_conv_channels(
    const Tensor& first, const Tensor& bias, const Tensor& second, std::int64_t keep) {
  if (!first.defined() || !bias.defined() || !second.defined() ||
      first.dim() != 4 || second.dim() != 4 || first.numel() == 0 || second.numel() == 0 ||
      first.size(3) != second.size(2) || bias.shape() != Shape{first.size(3)} ||
      keep < 1 || keep > first.size(3) || first.size(3) > INT32_MAX ||
      (first.dtype() != DType::Float32 && first.dtype() != DType::Float64) ||
      bias.dtype() != first.dtype() || second.dtype() != first.dtype() ||
      first.device().backend != compute::Backend::CPU || bias.device().backend != compute::Backend::CPU ||
      second.device().backend != compute::Backend::CPU ||
      !all(is_finite(first)).item<bool>() || !all(is_finite(bias)).item<bool>() ||
      !all(is_finite(second)).item<bool>())
    throw std::invalid_argument("prune_conv_channels: expected compatible finite CPU kernels and bias");

  const auto score = sum(abs(first.detach().to(DType::Float64)), {0, 1, 2}).to_vector<double>();
  std::vector<std::int32_t> order(score.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return score[a] > score[b]; });
  order.resize(keep);
  const auto kept = from_vector(order);
  return {gather(first, kept, 3), gather(bias, kept), gather(second, kept, 2), kept};
}

} // namespace datamunge::nn
