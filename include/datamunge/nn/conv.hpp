#pragma once

/// @file conv.hpp
/// @brief Convolution, pooling and batch normalization on NHWC tensors.
///
/// Layouts follow TensorFlow: images are [N, H, W, C], 2-D filters are
/// [KH, KW, C_in / groups, C_out], depthwise filters [KH, KW, C, multiplier].
/// Output sizes use TF's padding rules, with k' = d (k - 1) + 1:
///   VALID: out = (in - k') / s + 1
///   SAME:  out = ceil(in / s), zero padding split evenly with the extra at the end.
///
/// All ops are differentiable in every float input, and their gradients are
/// differentiable again (the backward passes are built from these same ops),
/// so gradient penalties and other second-order methods work through them.

#include <array>
#include <cstdint>
#include <utility>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

enum class Padding { Same, Valid, Explicit };

struct Conv2DOptions {
  /// Stride along H and W.
  std::array<std::int64_t, 2> strides{1, 1};
  /// Dilation along H and W.
  std::array<std::int64_t, 2> dilations{1, 1};
  Padding padding{Padding::Same};
  /// Used with Padding::Explicit: top, bottom, left, right.
  std::array<std::int64_t, 4> explicit_paddings{0, 0, 0, 0};
  /// Grouped convolution: C_in and C_out are split into this many groups.
  std::int64_t groups{1};
};

/// Options with the same stride on both axes.
Conv2DOptions conv_options(std::int64_t stride, Padding padding = Padding::Same, std::int64_t dilation = 1);

/// Output length of one spatial axis (TF ConvOutputSize).
std::int64_t conv_output_size(std::int64_t in, std::int64_t kernel, std::int64_t stride, std::int64_t dilation,
                              Padding padding, std::int64_t pad_before = 0, std::int64_t pad_after = 0);

// ---- convolution -------------------------------------------------------------------

/// 2-D cross-correlation (TF Conv2D): x [N, H, W, C], filter [KH, KW, C / groups, O] -> [N, H', W', O].
Tensor conv2d(const Tensor& x, const Tensor& filter, const Conv2DOptions& options = {});

/// Gradient of conv2d with respect to its input (TF Conv2DBackpropInput):
/// grad [N, H', W', O] -> [N, H, W, C] for `input_shape` [N, H, W, C].
Tensor conv2d_backprop_input(const Shape& input_shape, const Tensor& filter, const Tensor& grad,
                             const Conv2DOptions& options = {});

/// Gradient of conv2d with respect to its filter (TF Conv2DBackpropFilter).
Tensor conv2d_backprop_filter(const Tensor& x, const Shape& filter_shape, const Tensor& grad,
                              const Conv2DOptions& options = {});

/// Transposed convolution: the adjoint of conv2d with the same filter,
/// strides and padding. x [N, H, W, C_in], filter [KH, KW, C_out, C_in] (laid
/// out for the forward convolution it inverts) -> `output_shape` [N, H2, W2, C_out].
Tensor conv2d_transpose(const Tensor& x, const Tensor& filter, const Shape& output_shape,
                        const Conv2DOptions& options = {});
/// As above with the conventional output size: H s for SAME, (H - 1) s + k' for VALID.
Tensor conv2d_transpose(const Tensor& x, const Tensor& filter, const Conv2DOptions& options = {});

/// Depthwise convolution (TF DepthwiseConv2dNative): filter [KH, KW, C, M]
/// -> [N, H', W', C M], output channel c M + m.
Tensor depthwise_conv2d(const Tensor& x, const Tensor& filter, const Conv2DOptions& options = {});

/// Depthwise then pointwise (1x1, [1, 1, C M, O]) convolution, TF SeparableConv2D.
Tensor separable_conv2d(const Tensor& x, const Tensor& depthwise_filter, const Tensor& pointwise_filter,
                        const Conv2DOptions& options = {});

/// 1-D convolution: x [N, L, C], filter [K, C, O] -> [N, L', O].
Tensor conv1d(const Tensor& x, const Tensor& filter, std::int64_t stride = 1, Padding padding = Padding::Same,
              std::int64_t dilation = 1);

// ---- pooling -------------------------------------------------------------------------

/// Max pooling over [KH, KW] windows (TF MaxPool). SAME padding ignores the
/// padded positions. The gradient goes to the first maximum of each window.
Tensor max_pool2d(const Tensor& x, std::array<std::int64_t, 2> window, std::array<std::int64_t, 2> strides,
                  Padding padding = Padding::Valid);
/// Square window and stride.
Tensor max_pool2d(const Tensor& x, std::int64_t window, std::int64_t stride, Padding padding = Padding::Valid);

/// Average pooling (TF AvgPool). With SAME padding each output averages only
/// the in-bounds elements of its window.
Tensor avg_pool2d(const Tensor& x, std::array<std::int64_t, 2> window, std::array<std::int64_t, 2> strides,
                  Padding padding = Padding::Valid);
Tensor avg_pool2d(const Tensor& x, std::int64_t window, std::int64_t stride, Padding padding = Padding::Valid);

/// 1-D pooling of x [N, L, C].
Tensor max_pool1d(const Tensor& x, std::int64_t window, std::int64_t stride, Padding padding = Padding::Valid);
Tensor avg_pool1d(const Tensor& x, std::int64_t window, std::int64_t stride, Padding padding = Padding::Valid);

/// [N, H, W, C] -> [N, C] (or [N, 1, 1, C] with keepdims).
Tensor global_avg_pool2d(const Tensor& x, bool keepdims = false);
Tensor global_max_pool2d(const Tensor& x, bool keepdims = false);

// ---- batch normalization --------------------------------------------------------------

struct BatchNormResult {
  /// Normalized output, same shape as x.
  Tensor y;
  /// Per-channel batch mean and unbiased (N - 1) variance, as TF
  /// FusedBatchNormV3 returns them for the moving-average update. These are
  /// not differentiable.
  Tensor mean, variance;
};

/// Training-mode batch normalization over every axis but the last:
/// y = gamma (x - mean) / sqrt(var + eps) + beta with the batch's (biased)
/// mean and variance. gamma and beta are [C].
BatchNormResult batch_norm_train(const Tensor& x, const Tensor& gamma, const Tensor& beta, double eps = 1e-3);

/// Inference-mode batch normalization with given per-channel statistics.
Tensor batch_norm_infer(const Tensor& x, const Tensor& gamma, const Tensor& beta, const Tensor& mean,
                        const Tensor& variance, double eps = 1e-3);

}  // namespace datamunge::nn
