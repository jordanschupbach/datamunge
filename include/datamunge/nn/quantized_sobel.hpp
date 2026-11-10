#pragma once
/// @file quantized_sobel.hpp
/// @brief Detached unit-image conditions from 8-bit quantization and raw zero-padded Sobel tensors.
#include <cmath>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {
/// @brief Threshold raw L1 Sobel magnitude after source-compatible 8-bit unit-pixel recovery.
///
/// Finite Float32/64 CPU images[N,H,W,1], all axes positive, values in[0,1];
/// strides allowed. Recover integers as floor(Double(pixel)*255+.5), then apply
/// unnormalized3x3 Sobel cross-correlations with zero padding, including border
/// pixels. Output1 iff abs(gx)+abs(gy)>=threshold(default256), else0. Unlike
/// image::sobel_edges(), this preserves a binary raw-L1 threshold and processes
/// borders against zeros. Threshold0 produces all ones; values>2040 all zeros.
///
/// Uses native tensor quantization/convolution/comparison, without fitting,
/// parameter creation, input mutation or RNG. Always detached owning output in
/// input shape/dtype; no input derivative is defined through this condition
/// construction. O(N*H*W) work/storage; synchronize external input mutation.
/// @param image Finite unit-range Float32/64 CPU[N,H,W,1], positive axes.
/// @param threshold Nonnegative inclusive raw integer magnitude threshold, default256.
/// @return Owning detached binary tensor with input geometry/dtype.
/// @throws std::invalid_argument For incompatible geometry/dtype/range or negative threshold.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto edge=dm::nn::quantized_sobel_edges(dm::nn::ones({1,3,3,1}));
/// // Center is0; border pixels are1 because padding is zero, not replicated.
/// @endcode
/// @par References
/// - Isola,Zhu,Zhou and Efros (2017), Image-to-Image Translation with Conditional
///   Adversarial Networks, https://arxiv.org/abs/1611.07004 . The quantized binary
///   MNIST Sobel condition is this repository's paired teaching-data transform.
/// @see models::Pix2PixGenerate(), conv2d(), floor()
inline Tensor quantized_sobel_edges(const Tensor& image, std::int64_t threshold = 256) {
  if (!image.defined() || image.dim() != 4 || image.shape().back() != 1 || image.numel() < 1 ||
      (image.dtype() != DType::Float32 && image.dtype() != DType::Float64) || threshold < 0)
    throw std::invalid_argument(
        "Sobel conditions need positive floating grayscale NHWC images and nonnegative threshold");
  for (auto size : image.shape())
    if (size <= 0)
      throw std::invalid_argument("Sobel axes must be positive");
  for (double value : image.contiguous().to_vector<double>())
    if (!std::isfinite(value) || value < 0 || value > 1)
      throw std::invalid_argument("Sobel conditions need finite unit pixels");
  NoGradGuard guard;
  auto        integer = floor(cast(image, DType::Float64) * 255. + .5);
  auto        sx = tensor({-1., 0., 1., -2., 0., 2., -1., 0., 1.}, {3, 3, 1, 1}, DType::Float64);
  auto        sy = tensor({-1., -2., -1., 0., 0., 0., 1., 2., 1.}, {3, 3, 1, 1}, DType::Float64);
  auto        magnitude = abs(conv2d(integer, sx)) + abs(conv2d(integer, sy));
  return cast(greater_equal(magnitude, static_cast<double>(threshold)), image.dtype()).detach();
}
} // namespace datamunge::nn
