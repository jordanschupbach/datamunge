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
///
/// `grid_sample`, `sample_pixels`, `affine_grid` and the deformable
/// convolution helpers sample at arbitrary (learned) positions, with
/// gradients w.r.t. both the images and the positions.

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

/// Channel order of depth_to_space / space_to_depth (ONNX DepthToSpace modes).
/// DCR (TF's order): channel (dy * b + dx) * C + c. CRD (PyTorch PixelShuffle,
/// tflow's super-resolution pixel shuffle): channel c * b * b + dy * b + dx.
enum class DepthToSpaceMode { DCR, CRD };

/// TF SpaceToDepth (NHWC, DCR order): [N, H, W, C] -> [N, H/b, W/b, b*b*C];
/// output channel (dy * b + dx) * C + c holds pixel (y*b + dy, x*b + dx).
Tensor space_to_depth(const Tensor& x, std::int64_t block_size);

/// TF DepthToSpace (NHWC, DCR order), the inverse of `space_to_depth`.
Tensor depth_to_space(const Tensor& x, std::int64_t block_size);

/// Either channel order; differentiable (a reshape and transpose).
Tensor space_to_depth(const Tensor& x, std::int64_t block_size, DepthToSpaceMode mode);
Tensor depth_to_space(const Tensor& x, std::int64_t block_size, DepthToSpaceMode mode);

/// Repeats every pixel `factor` x `factor` times (the tflow `Upsample2x` for
/// factor 2). Equal to resize_nearest to an exact integer multiple.
Tensor upsample_nearest(const Tensor& x, std::int64_t factor);

/// Mirror-pads the H and W axes by `pad_h` / `pad_w` on both sides (TF
/// MirrorPad on NHWC; Reflect excludes the edge, Symmetric repeats it). A thin
/// wrapper over `nn::pad`.
Tensor mirror_pad(const Tensor& x, std::int64_t pad_h, std::int64_t pad_w, PadMode mode = PadMode::Reflect);

// ---- sampling at arbitrary positions (spatial transformers, deformable conv) ----------
//
// The samplers below read images at real-valued positions. They are
// differentiable w.r.t. the images and the positions, to any order: the
// sampler, its adjoint (the image gradient) and the sampler with derivative
// weights (the position gradient) are recorded as each other's gradients.
// Bilinear sampling is piecewise linear in each coordinate. At a position
// with an integer pixel coordinate (a kink, e.g. every point of an identity
// grid) the position derivative is the mean of the two one-sided slopes,
// (v[x+1] - v[x-1]) / 2, so an identity transform gets an unbiased gradient;
// elsewhere it is the slope of the cell. Nearest sampling has zero position
// gradient.

/// How `grid_sample` interpolates.
enum class GridSampleMode { Bilinear, Nearest };

/// What `grid_sample` reads outside the image. Zeros: taps outside the image
/// read 0. Border: positions are clamped to the outermost pixel centres
/// (PyTorch padding_mode="border"); the position gradient is 0 where clamped.
enum class GridSamplePadding { Zeros, Border };

/// Options of `grid_sample` (PyTorch grid_sample semantics, NHWC).
struct GridSampleOptions {
  GridSampleMode    mode{GridSampleMode::Bilinear};
  GridSamplePadding padding{GridSamplePadding::Zeros};
  /// true: -1 and 1 are the centres of the corner pixels (x = (g + 1) (W - 1) / 2).
  /// false: -1 and 1 are the outer edges of the corner pixels (x = ((g + 1) W - 1) / 2).
  bool align_corners{false};
};

/// Samples `images` [N, H, W, C] at normalised positions `grid` [N, Ho, Wo, 2]
/// whose last axis is (x, y) in [-1, 1] (x along W). Returns [N, Ho, Wo, C].
/// The grid is converted to the images' float dtype. Differentiable w.r.t.
/// images and grid (PyTorch F.grid_sample with NHWC tensors).
Tensor grid_sample(const Tensor& images, const Tensor& grid, const GridSampleOptions& options = {});

/// Samples `images` [N, H, W, C] at pixel positions `coords` [N, ..., 2]
/// whose last axis is (x, y) (pixel centres at integers). Returns
/// [N, ..., C]. Differentiable w.r.t. images and coords.
Tensor sample_pixels(const Tensor& images, const Tensor& coords, GridSampleMode mode = GridSampleMode::Bilinear,
                     GridSamplePadding padding = GridSamplePadding::Zeros);

/// The normalised sampling grid [N, size.height, size.width, 2] of affine
/// transforms `theta` [N, 2, 3] (or [N, 6]): output pixel (i, j) with
/// normalised target coordinates (x_t, y_t) samples (x_s, y_s) = theta (x_t, y_t, 1).
/// Use the same `align_corners` here and in `grid_sample`. Differentiable
/// w.r.t. theta (spatial transformer grid generator, PyTorch F.affine_grid).
Tensor affine_grid(const Tensor& theta, ImageSize size, bool align_corners = false);

/// Geometry of a deformable convolution.
struct DeformableOptions {
  std::int64_t kernel_h{3}, kernel_w{3};
  std::int64_t stride{1};
  std::int64_t pad_h{1}, pad_w{1};
  std::int64_t dilation{1};
};

/// Output spatial size of a deformable convolution on an H x W input.
ImageSize deformable_output_size(std::int64_t height, std::int64_t width, const DeformableOptions& options);

/// Bilinear deformable-convolution samples (Deformable ConvNets v1/v2).
/// `offsets` [N, Ho, Wo, 2K] holds (dy, dx) per tap k = ky * kernel_w + kx
/// (the release's order). Tap k of output position (i, j) reads `images`
/// [N, H, W, C] at row i*stride - pad_h + dilation*ky + dy and column
/// j*stride - pad_w + dilation*kx + dx, with zeros outside. Returns
/// [N, Ho, Wo, K, C]; differentiable w.r.t. images and offsets.
Tensor deformable_sample(const Tensor& images, const Tensor& offsets, const DeformableOptions& options = {});

/// Deformable convolution: `deformable_sample`, optionally multiplied per tap
/// by `mask` [N, Ho, Wo, K] (v2, modulated), then contracted with `filter`
/// [kernel_h, kernel_w, C, Cout]. Returns [N, Ho, Wo, Cout]. With zero
/// offsets and no mask it equals conv2d with explicit padding (pad_h, pad_w).
Tensor deformable_conv2d(const Tensor& images, const Tensor& offsets, const Tensor& filter,
                         const DeformableOptions& options = {}, const Tensor& mask = Tensor());

}  // namespace datamunge::nn
