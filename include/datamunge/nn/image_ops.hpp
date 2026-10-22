#pragma once

/// @file image_ops.hpp
/// @brief Differentiable image ops on NHWC tensors (TF image ops semantics).
///
/// Images are [N, H, W, C]; ops that take a single image also accept
/// [H, W, C] and return the same rank. Sampling coordinates are computed in
/// float32 exactly as TensorFlow's CPU kernels do, so results match TF
/// bit-for-bit on float32 inputs. Integer inputs are converted to float32.
///
/// Resizes and crop_and_resize are linear in the image. Their gradient is the
/// adjoint map (TF ResizeBilinearGrad / ResizeNearestNeighborGrad /
/// CropAndResizeGradImage), recorded as a primitive whose own gradient is the
/// forward map again, so gradients of any order are available.

#include <cstdint>

#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// Output spatial size {height, width}.
struct ImageSize {
  std::int64_t height{0};
  std::int64_t width{0};
};

/// TF ResizeBilinear. `align_corners` maps the corner pixel centres onto each
/// other; `half_pixel_centers` samples at pixel centres (what
/// tf.image.resize uses). The two flags are mutually exclusive.
Tensor resize_bilinear(const Tensor& images, ImageSize size, bool align_corners = false,
                       bool half_pixel_centers = false);

/// TF ResizeNearestNeighbor, with the same flags as `resize_bilinear`.
/// align_corners rounds the source coordinate; otherwise it is floored.
Tensor resize_nearest(const Tensor& images, ImageSize size, bool align_corners = false,
                      bool half_pixel_centers = false);

enum class ResizeMethod { Bilinear, Nearest };

/// TF CropAndResize. `boxes` is [B, 4] of normalised (y1, x1, y2, x2) (y2 < y1
/// flips the crop), `box_indices` is an integer [B] tensor selecting the image
/// of each box. Returns [B, crop.height, crop.width, C]. Samples falling
/// outside the image get `extrapolation_value`. Differentiable w.r.t.
/// `images` only (boxes are treated as constants).
Tensor crop_and_resize(const Tensor& images, const Tensor& boxes, const Tensor& box_indices, ImageSize crop_size,
                       ResizeMethod method = ResizeMethod::Bilinear, double extrapolation_value = 0.0);

/// TF SpaceToDepth (NHWC, DCR order): [N, H, W, C] -> [N, H/b, W/b, b*b*C];
/// output channel (dy * b + dx) * C + c holds pixel (y*b + dy, x*b + dx).
Tensor space_to_depth(const Tensor& x, std::int64_t block_size);

/// TF DepthToSpace (NHWC, DCR order), the inverse of `space_to_depth`.
Tensor depth_to_space(const Tensor& x, std::int64_t block_size);

/// Repeats every pixel `factor` x `factor` times (the tflow `Upsample2x` for
/// factor 2). Equal to resize_nearest to an exact integer multiple.
Tensor upsample_nearest(const Tensor& x, std::int64_t factor);

/// Mirror-pads the H and W axes by `pad_h` / `pad_w` on both sides (TF
/// MirrorPad on NHWC; Reflect excludes the edge, Symmetric repeats it). A thin
/// wrapper over `nn::pad`.
Tensor mirror_pad(const Tensor& x, std::int64_t pad_h, std::int64_t pad_w, PadMode mode = PadMode::Reflect);

}  // namespace datamunge::nn
