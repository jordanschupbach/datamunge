#pragma once
/// @file convolution_components.hpp
/// @brief Differentiable patch extraction and convolution filter factorizations.
#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/views.hpp"
#include <algorithm>
#include <stdexcept>

namespace datamunge::nn {
/// @brief Materialise NHWC image patches as rows for a convolution GEMM.
///
/// Rows order batch, output row, output column; columns order kernel row,
/// kernel column, input channel. SAME padding puts an odd extra pixel after
/// the image. Groups must be one. Explicit nonnegative padding, unequal strides
/// and unequal dilation rates are supported. Autograd scatter-adds overlapping
/// patch gradients to their source pixels. No input mutation; O(N Ho Wo kh kw C)
/// output storage/work, plus padding and intermediate view metadata.
/// @param input Nonempty floating [N,H,W,C].
/// @param kernel_height Positive undilated filter height.
/// @param kernel_width Positive undilated filter width.
/// @param options Convolution strides/dilations/padding; groups must be one.
/// @return Floating [N Ho Wo,kh kw C], same dtype as input, differentiable.
/// @throws std::invalid_argument For invalid shape, dtype, geometry or groups.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::tensor({1,2,3,4,5,6,7,8,9},{1,3,3,1});
/// const auto patches = n::im2col2d(x,2,2,n::conv_options(1,n::Padding::Valid));
/// // Four rows: [1,2,4,5], [2,3,5,6], [4,5,7,8], [5,6,8,9].
/// @endcode
/// @par References
/// - Dumoulin and Visin (2016), A guide to convolution arithmetic for deep learning,
/// https://arxiv.org/abs/1603.07285
/// @see sliding_window(), conv2d(), conv2d_backprop_input()
inline Tensor im2col2d(const Tensor &input, std::int64_t kernel_height,
                       std::int64_t kernel_width, const Conv2DOptions &options = {}) {
  if (!input.defined() || input.dim() != 4 || input.numel() < 1 ||
      !is_floating(input.dtype()) || kernel_height < 1 || kernel_width < 1 ||
      options.groups != 1 ||
      (options.padding != Padding::Same && options.padding != Padding::Valid &&
       options.padding != Padding::Explicit))
    throw std::invalid_argument(
        "im2col2d requires floating NHWC input and positive ungrouped geometry");
  std::vector<std::pair<std::int64_t, std::int64_t>> pads(4, {0, 0});
  for (int axis = 0; axis < 2; ++axis) {
    const auto length = input.size(axis + 1),
               kernel = axis == 0 ? kernel_height : kernel_width;
    const auto stride = options.strides[axis], dilation = options.dilations[axis];
    if (stride < 1 || dilation < 1)
      throw std::invalid_argument("positive patch stride/dilation required");
    if (options.padding == Padding::Explicit) {
      pads[axis + 1] = {options.explicit_paddings[2 * axis],
                        options.explicit_paddings[2 * axis + 1]};
      if (pads[axis + 1].first < 0 || pads[axis + 1].second < 0)
        throw std::invalid_argument("nonnegative explicit padding required");
    } else if (options.padding == Padding::Same) {
      const auto outputs = (length + stride - 1) / stride;
      const auto total = std::max<std::int64_t>(0, (outputs - 1) * stride +
                                                       (kernel - 1) * dilation + 1 - length);
      pads[axis + 1] = {total / 2, total - total / 2};
    }
  }
  const auto x = pad(input, pads);
  auto windows =
      sliding_window(x, 1, (kernel_height - 1) * options.dilations[0] + 1, options.strides[0]);
  windows = sliding_window(windows, 2, (kernel_width - 1) * options.dilations[1] + 1,
                           options.strides[1]);
  windows = slice_axis(windows, 4, 0, windows.size(4), options.dilations[0]);
  windows = slice_axis(windows, 5, 0, windows.size(5), options.dilations[1]);
  return reshape(transpose(windows, {0, 1, 2, 4, 5, 3}),
                 {input.size(0) * windows.size(1) * windows.size(2),
                  kernel_height * kernel_width * input.size(3)});
}

/// @brief Insert constant zeros between HWIO filter taps without blocking gradients.
///
/// Materialises the effective filter; only original taps carry input gradients.
/// Unit dilation returns an alias. This reference operation is not an optimised
/// atrous kernel. Cost/storage O(((kh-1)d+1)((kw-1)d+1)Cin Cout).
/// @param kernel Nonempty floating HWIO tensor.
/// @param dilation Positive common spatial dilation rate.
/// @return Same-dtype expanded HWIO filter; differentiable in kernel.
/// @throws std::invalid_argument For invalid kernel/rate or incompatible tensor shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto expanded = n::expand_dilated_kernel(n::ones({2,2,1,1}),2);
/// // Spatial shape 3x3, with four corner ones and five constant zeros.
/// @endcode
/// @par References
/// - Yu and Koltun (2016), Multi-Scale Context Aggregation by Dilated Convolutions,
/// https://arxiv.org/abs/1511.07122
/// @see conv2d(), im2col2d()
inline Tensor expand_dilated_kernel(const Tensor &kernel, std::int64_t dilation) {
  if (!kernel.defined() || kernel.dim() != 4 || kernel.numel() < 1 ||
      !is_floating(kernel.dtype()) || dilation < 1)
    throw std::invalid_argument("need nonempty floating HWIO filter and positive dilation");
  if (dilation == 1)
    return kernel;
  const auto h = kernel.size(0), w = kernel.size(1), in = kernel.size(2), out = kernel.size(3),
             d = dilation;
  const auto taps = reshape(kernel, {h, 1, w, 1, in, out});
  const auto expanded =
      reshape(pad(taps, {{0, 0}, {0, d - 1}, {0, 0}, {0, d - 1}, {0, 0}, {0, 0}}),
              {h * d, w * d, in, out});
  return slice(expanded, {0, 0, 0, 0}, {(h - 1) * d + 1, (w - 1) * d + 1, in, out});
}

/// @brief Compose linear depthwise and pointwise convolutions into one HWIO filter.
///
/// W[a,b,i,o]=sum_m D[a,b,i,m] P[0,0,i*M+m,o]. The spatial-by-output matrix
/// per input channel has rank at most M. No nonlinearity or intermediate bias
/// may intervene for this kernel-only equivalence. Supports rectangular filters,
/// matching floating dtypes, and gradients in both inputs. No mutation; output
/// storage O(kh kw Cin Cout), work O(kh kw Cin M Cout).
/// @param depthwise Nonempty floating [kh,kw,Cin,M].
/// @param pointwise Floating [1,1,Cin*M,Cout] of matching dtype, positive Cout.
/// @return Composed HWIO filter [kh,kw,Cin,Cout].
/// @throws std::invalid_argument For invalid shapes or incompatible dtypes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto combined=n::compose_separable_kernel(n::ones({3,3,2,1}),n::ones({1,1,2,4}));
/// // [3,3,2,4], every weight one.
/// @endcode
/// @par References
/// - Howard et al. (2017), MobileNets, https://arxiv.org/abs/1704.04861
/// @see depthwise_conv2d(), separable_conv2d()
inline Tensor compose_separable_kernel(const Tensor &depthwise, const Tensor &pointwise) {
  if (!depthwise.defined() || !pointwise.defined() || depthwise.dim() != 4 ||
      pointwise.dim() != 4 || depthwise.numel() < 1 || pointwise.numel() < 1 ||
      !is_floating(depthwise.dtype()) || depthwise.dtype() != pointwise.dtype() ||
      pointwise.size(0) != 1 || pointwise.size(1) != 1 ||
      pointwise.size(2) != depthwise.size(2) * depthwise.size(3))
    throw std::invalid_argument("separable filters require matching HWIM and [1,1,I*M,O]");
  const auto h = depthwise.size(0), w = depthwise.size(1), in = depthwise.size(2),
             m = depthwise.size(3), out = pointwise.size(3);
  const auto spatial = transpose(reshape(depthwise, {h * w, in, m}), {1, 0, 2});
  const auto mixing = reshape(pointwise, {in, m, out});
  return reshape(transpose(matmul(spatial, mixing), {1, 0, 2}), {h, w, in, out});
}
} // namespace datamunge::nn
