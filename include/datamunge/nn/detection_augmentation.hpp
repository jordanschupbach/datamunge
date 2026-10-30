#pragma once
/// @file detection_augmentation.hpp
/// @brief Tensor inverse pixel-centre affine/exposure augmentation of grayscale scenes and attached boxes.
#include "datamunge/nn/two_stage_detection_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Transform selected detection scenes by explicit uniform scale, translation and exposure tensors.
///
/// Scale acts about the image centre, then pixel shifts translate. Inverse
/// pixel-centre bilinear sampling uses zero padding, followed by exposure, byte
/// clipping and half-up quantization. Boxes use the matching forward transform
/// and image clipping; only fully off-image/zero-area boxes disappear. Double
/// tensor sampling preserves source geometry; nonfused box arithmetic can differ
/// from source fma by rounding. Randomness and fitting belong to the caller.
/// No mutation, gradients or optimizer. Empty rows produce an empty batch.
///
/// @param source Valid Float32/64 [N,H,W,1] scenes with finite pixels[0,1], positive in-image boxes and aligned nonnegative labels.
/// @param rows Selected source IDs, including repeats; output is renumbered in this order.
/// @param transforms Float32/64 [B,4] scale,shift_y,shift_x,exposure, B=rows.size(); scale/exposure finite positive, shifts finite.
/// @return Detached tensor scenes on the original canvas, quantized to bytes/255, with clipped boxes and attached labels.
/// @throws std::invalid_argument For invalid scene/transform geometry, values, row IDs, labels or overflowing offsets.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::ones({1, 32, 32, 1});
/// data.boxes = {n::tensor({8., 8., 24., 24.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// const auto out =
///     n::models::TransformTensorDetectionScenes(data, {0}, n::tensor({1., 0., 0., 1.}, {1, 4}));
/// @endcode
/// @par References
/// - Synthetic affine convention from examples/nn/common/detection_affine.hpp; pixel sampling uses the tensor image_ops contract.
/// @see MakeTensorDetectionScenes(), CropDetectionPatches()
inline TensorDetectionScenes
TransformTensorDetectionScenes(const TensorDetectionScenes &source,
                               const std::vector<std::int64_t> &rows,
                               const nn::Tensor &transforms) {
  two_stage_detection_detail::ValidateScenes(source, rows, std::numeric_limits<int>::max());
  const auto N = std::int64_t(rows.size()), H = source.images.shape()[1],
             W = source.images.shape()[2];
  const auto type = source.images.dtype(), calculation = nn::DType::Float64;
  if (transforms.shape() != nn::Shape{N, 4} ||
      (transforms.dtype() != nn::DType::Float32 && transforms.dtype() != nn::DType::Float64) ||
      !nn::all(nn::less(nn::abs(transforms), std::numeric_limits<double>::infinity()))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(source.images, 0.),
                               nn::less_equal(source.images, 1.)))
           .item<bool>())
    throw std::invalid_argument("invalid detection affine transforms/pixels");
  nn::NoGradGuard guard;
  TensorDetectionScenes out;
  if (!N) {
    out.images = nn::zeros({0, H, W, 1}, type);
    return out;
  }
  const auto t = nn::unstack(transforms.to(calculation), 1);
  if (nn::any(nn::logical_or(nn::less_equal(t[0], 0.), nn::less_equal(t[3], 0.))).item<bool>())
    throw std::invalid_argument("positive affine scale/exposure required");
  const auto oy = (1. - t[0]) * (double(H) * .5) + t[1],
             ox = (1. - t[0]) * (double(W) * .5) + t[2];
  if (!nn::all(
           nn::less(nn::abs(nn::stack({oy, ox}, 1)), std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("affine offsets overflow");
  const auto sy = (nn::arange(0, double(H)).to(calculation).reshape({1, H, 1}) + .5 -
                   oy.reshape({N, 1, 1})) /
                      t[0].reshape({N, 1, 1}) -
                  .5,
             sx = (nn::arange(0, double(W)).to(calculation).reshape({1, 1, W}) + .5 -
                   ox.reshape({N, 1, 1})) /
                      t[0].reshape({N, 1, 1}) -
                  .5,
             ones = nn::ones({N, H, W}, calculation);
  const auto bytes =
      nn::floor(nn::gather(source.images, nn::from_vector(rows)).to(calculation) * 255. + .5);
  const auto sampled = nn::sample_pixels(bytes, nn::stack({sx * ones, sy * ones}, 3));
  out.images =
      (nn::floor(nn::clip(sampled * t[3].reshape({N, 1, 1, 1}), 0., 255.) + .5) / 255.)
          .to(type);
  for (std::int64_t i = 0; i < N; ++i) {
    const auto &b = source.boxes[rows[i]];
    const auto parts = nn::unstack(b, 1);
    if (nn::any(nn::logical_or(nn::logical_or(nn::less(parts[0], 0.), nn::less(parts[1], 0.)),
                               nn::logical_or(nn::greater(parts[2], double(H)),
                                              nn::greater(parts[3], double(W)))))
            .item<bool>())
      throw std::invalid_argument("affine truth outside source canvas");
    const auto index = nn::full({}, double(i), nn::DType::Int64);
    const auto scale = nn::gather(t[0], index), dy = nn::gather(oy, index),
               dx = nn::gather(ox, index);
    const auto transformed = nn::clip_boxes(
        (b.to(calculation) * scale + nn::stack({dy, dx, dy, dx}, 0)).to(type), H, W);
    const auto keep =
        two_stage_detection_detail::Indices(nn::greater(nn::box_area(transformed), 0.));
    out.boxes.push_back(nn::gather(transformed, keep));
    out.labels.push_back(nn::gather(source.labels[rows[i]], keep));
  }
  return out;
}
} // namespace datamunge::nn::models
