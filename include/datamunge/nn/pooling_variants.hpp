#pragma once

/// @file pooling_variants.hpp
/// @brief Power-mean, generalised-mean (GeM) and log-sum-exp pooling between average and max; adaptive pooling to a
///        fixed output size; and anti-aliased (blur) pooling.
///
/// conv.hpp provides max, average and global pooling. The functions here fill the space between them and add two
/// practical variants. power_mean_pool2d() and logsumexp_pool2d() are smooth families that run from the window
/// average to the window maximum as their parameter grows. gem_pool2d() is the global power mean used for image
/// retrieval (Radenović, Tolias and Chum 2018), with an optional learnable exponent. adaptive_avg_pool2d() and
/// adaptive_max_pool2d() split any input into a fixed grid of nearly equal windows, as in spatial pyramid pooling
/// (He et al. 2014). blur_pool2d() low-pass filters with a binomial kernel before subsampling, which restores much of
/// the shift equivariance that strided pooling loses (Zhang 2019). All tensors are NHWC and every function is
/// differentiable with respect to x.

#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/tensor.hpp"

#include <cstdint>

namespace datamunge::nn {

/// @brief Power-mean pooling: \f$\big(\frac1{|w|}\sum_{i\in w}|x_i|^p\big)^{1/p}\f$ over each window.
///
/// p = 1 is average pooling of |x|, and as p grows the result tends to max pooling of |x|. Unlike PyTorch's
/// LPPool2d this is a mean, not a sum, so it does not grow with the window size. With SAME padding only the real
/// entries of a border window are averaged, as in avg_pool2d(). A window of zeros gives 0 with zero gradient, so the
/// function is safe after ReLU.
///
/// @param x  Input [N, H, W, C].
/// @param p  Exponent, p > 0.
/// @param window  Square window size.
/// @param stride  Stride on both axes.
/// @param padding  Padding::Valid or Padding::Same.
/// @return The pooled tensor [N, H', W', C].
/// @throws std::invalid_argument  If p <= 0.
[[nodiscard]] Tensor power_mean_pool2d(const Tensor& x, double p, std::int64_t window, std::int64_t stride,
                                       Padding padding = Padding::Valid);

/// @brief Generalised-mean (GeM) global pooling with a learnable exponent: \f$\big(\frac1{HW}\sum_{h,w}\max(x,\epsilon)^p\big)^{1/p}\f$ per channel.
///
/// @param x  Input [N, H, W, C].
/// @param p  Scalar exponent tensor (shape [] or [1]); gradients flow to it if it requires them.
/// @param eps  Lower clamp, which keeps the power and its gradient finite.
/// @return Pooled features [N, C].
/// @throws std::invalid_argument  If x is not rank 4 or p has more than one element.
[[nodiscard]] Tensor gem_pool2d(const Tensor& x, const Tensor& p, double eps = 1e-6);

/// @brief GeM global pooling with a fixed exponent p > 0: p = 1 is global average pooling and large p approaches global max.
/// @param x  Input [N, H, W, C].
/// @param p  Exponent, p > 0.
/// @param eps  Lower clamp.
/// @return Pooled features [N, C].
/// @throws std::invalid_argument  If p <= 0 or x is not rank 4.
[[nodiscard]] Tensor gem_pool2d(const Tensor& x, double p, double eps = 1e-6);

/// @brief Log-sum-exp pooling: \f$\frac1\beta\log\big(\frac1{|w|}\sum_{i\in w}e^{\beta x_i}\big)\f$ over each window.
///
/// As \f$\beta\to0\f$ this tends to the window mean, and as \f$\beta\to\infty\f$ to the window maximum. It always lies between the two,
/// and its gradient is the softmax of \f$\beta x\f$ over the window, so every entry receives some gradient. Computed
/// stably by subtracting each window's own maximum, so large beta does not underflow.
///
/// @param x  Input [N, H, W, C].
/// @param beta  Inverse temperature, beta > 0.
/// @param window  Square window size.
/// @param stride  Stride on both axes.
/// @param padding  Padding::Valid or Padding::Same (border windows average their real entries only).
/// @return The pooled tensor [N, H', W', C].
/// @throws std::invalid_argument  If beta <= 0.
[[nodiscard]] Tensor logsumexp_pool2d(const Tensor& x, double beta, std::int64_t window, std::int64_t stride,
                                      Padding padding = Padding::Valid);

/// @brief Adaptive average pooling to a fixed [out_h, out_w] grid.
///
/// Output cell (i, j) averages rows \f$\lfloor iH/h_o\rfloor\f$ to \f$\lceil (i+1)H/h_o\rceil-1\f$ and the corresponding columns (PyTorch's
/// AdaptiveAvgPool2d), so neighbouring windows may overlap by one when H is not a multiple of out_h.
///
/// @param x  Input [N, H, W, C].
/// @param out_h  Output height, 1 <= out_h <= H.
/// @param out_w  Output width, 1 <= out_w <= W.
/// @return The pooled tensor [N, out_h, out_w, C].
/// @throws std::invalid_argument  If x is not rank 4 or the output size is out of range.
[[nodiscard]] Tensor adaptive_avg_pool2d(const Tensor& x, std::int64_t out_h, std::int64_t out_w);

/// @brief Adaptive max pooling to a fixed [out_h, out_w] grid, with the windows of adaptive_avg_pool2d().
/// @param x  Input [N, H, W, C].
/// @param out_h  Output height, 1 <= out_h <= H.
/// @param out_w  Output width, 1 <= out_w <= W.
/// @return The pooled tensor [N, out_h, out_w, C].
/// @throws std::invalid_argument  If x is not rank 4 or the output size is out of range.
[[nodiscard]] Tensor adaptive_max_pool2d(const Tensor& x, std::int64_t out_h, std::int64_t out_w);

/// @brief Anti-aliased downsampling (Zhang 2019): a depthwise binomial blur, then subsampling by `stride`.
///
/// The filter is the outer product of the binomial row of length `filter_size` (for example [1, 2, 1] / 4 for 3),
/// applied with reflect padding so that the output has \f$\lceil H/s\rceil\times\lceil W/s\rceil\f$ cells, like SAME pooling. For Zhang's
/// MaxBlurPool use `blur_pool2d(max_pool2d(x, 2, 1, Padding::Same), 2)`.
///
/// @param x  Input [N, H, W, C].
/// @param stride  Subsampling factor, >= 1.
/// @param filter_size  Binomial filter length, 1 to 7 (1 is plain subsampling).
/// @return The downsampled tensor [N, ceil(H / stride), ceil(W / stride), C].
/// @throws std::invalid_argument  If x is not rank 4, stride < 1, or filter_size is outside [1, 7].
[[nodiscard]] Tensor blur_pool2d(const Tensor& x, std::int64_t stride = 2, std::int64_t filter_size = 3);

}  // namespace datamunge::nn
