#pragma once

/// @file image_io.hpp
/// @brief Conversion between `nn::Tensor` and `image::Image`, sample sheets and PNG output.
///
/// Tensors hold images as [H, W, C] (or [H, W] for grayscale) with values in a
/// caller-chosen range [low, high], e.g. 0..1 or -1..1. Converting to bytes
/// maps low -> 0 and high -> 255, clamps, and rounds half away from zero
/// (like tflow's `std::lround(std::clamp(...))`). NaN becomes 0.

#include <cstdint>
#include <string>

#include "datamunge/image/image.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// An 8-bit image from a [H, W], [H, W, C] or [1, H, W, C] tensor with C in
/// {1, 2, 3, 4} (grayscale, grayscale+alpha, RGB, RGBA).
image::Image to_image(const Tensor& x, double low = 0.0, double high = 1.0);

/// The pixels of `img` as [H, W, C] with values 0..255 (float32 by default).
Tensor from_image(const image::Image& img, DType dtype = DType::Float32);

/// Tiles a batch [N, H, W, C] (or [N, H, W]) into one [H', W', C] sheet,
/// row-major with `cols` tiles per row (0 picks ceil(sqrt(N))). Tiles are
/// separated (not surrounded) by `padding` pixels of `pad_value` and each
/// pixel is first repeated `scale` x `scale` times (e.g. 28 -> 112 for MNIST).
/// Values stay in the input's range and dtype.
Tensor image_grid(const Tensor& batch, std::int64_t cols = 0, std::int64_t padding = 2, double pad_value = 0.0,
                  std::int64_t scale = 1);

/// Writes a [H, W], [H, W, C] or [1, H, W, C] tensor in [low, high] as a PNG.
void save_png(const Tensor& x, const std::string& path, double low = 0.0, double high = 1.0);

/// `image_grid` followed by `save_png`; `pad_value` is in [low, high] units.
void save_image_grid(const Tensor& batch, const std::string& path, std::int64_t cols = 0, double low = 0.0,
                     double high = 1.0, std::int64_t padding = 2, double pad_value = 0.0, std::int64_t scale = 1);

}  // namespace datamunge::nn
