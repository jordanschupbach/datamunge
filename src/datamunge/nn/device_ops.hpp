#pragma once
/// @file device_ops.hpp
/// @brief Private native CUDA forward kernels behind the public nn operations.
///
/// Each public operation checks is_native() on its inputs first and forwards GPU
/// tensors here. These paths are forward-only: with gradients enabled, a GPU input
/// that requires gradients is rejected, so inference runs under NoGradGuard.
/// Vulkan tensors are rejected with a diagnostic; there is no host fallback.
#include <datamunge/nn/tensor.hpp>

#include <cstdint>
#include <utility>
#include <vector>
namespace datamunge::nn::detail {
/// @brief True for a defined tensor whose storage is not host memory.
/// @param tensor Any tensor, including an undefined one.
/// @return Whether a public op must take its native device path.
bool is_native(const Tensor& tensor);

/// @brief Resolved 2-D NHWC convolution geometry, as computed by conv2d's validation.
struct DeviceConvGeometry {
  std::int64_t n, h, w, c;     ///< Input batch, height, width and channels.
  std::int64_t kh, kw, o;      ///< Filter height, width and output channels.
  std::int64_t oh, ow;         ///< Output height and width.
  std::int64_t sh, sw, dh, dw; ///< Strides and dilations.
  std::int64_t pt, pl;         ///< Top and left zero padding.
  std::int64_t groups;         ///< Channel groups; the device path requires one.
};

/// @brief a + b for equal shapes or b broadcast over a's trailing axes (either order).
/// @throws std::logic_error Unsupported broadcast, backend or gradient request.
/// @throws std::invalid_argument Mixed contexts or dtypes.
Tensor device_add(const Tensor& a, const Tensor& b);
/// @brief max(x, 0) elementwise, with NaN mapped to zero like the CPU kernel.
Tensor device_relu(const Tensor& x);
/// @brief NHWC convolution with an HWIO filter: im2col patches times a cuBLAS GEMM.
/// @throws std::logic_error Grouped convolution or a build without cuBLAS.
Tensor device_conv2d(const Tensor& x, const Tensor& filter, const DeviceConvGeometry& g);
/// @brief Inference batch norm over the last axis using moving statistics.
Tensor device_batch_norm_infer(
    const Tensor& x,
    const Tensor& gamma,
    const Tensor& beta,
    const Tensor& mean,
    const Tensor& variance,
    double        eps);
/// @brief Mean over the H and W axes of an NHWC tensor.
Tensor device_global_avg_pool2d(const Tensor& x, bool keepdims);
/// @brief Join tensors along `axis`; `shape` is the validated output shape.
/// @throws std::logic_error Unsupported backend, dtype or gradient request.
/// @throws std::invalid_argument Mixed contexts or dtypes.
Tensor device_concat(const std::vector<Tensor>& tensors, std::int64_t axis, const Shape& shape);
/// @brief Constant padding (any finite or infinite value) of a rank <= 4 tensor.
/// @throws std::logic_error Rank above four, unsupported backend/dtype or gradient request.
Tensor device_pad_constant(
    const Tensor& x, const std::vector<std::pair<std::int64_t, std::int64_t>>& paddings,
    double value);

/// @brief Resolved NHWC pooling geometry, as computed by max_pool2d's validation.
struct DevicePoolGeometry {
  std::int64_t n, h, w, c;     ///< Input batch, height, width and channels.
  std::int64_t kh, kw, sh, sw; ///< Window and strides.
  std::int64_t oh, ow, pt, pl; ///< Output size and top/left padding (padded taps are skipped).
};
/// @brief NHWC max pooling with the CPU kernel's tie and NaN rules.
Tensor device_max_pool2d(const Tensor& x, const DevicePoolGeometry& g);
} // namespace datamunge::nn::detail
