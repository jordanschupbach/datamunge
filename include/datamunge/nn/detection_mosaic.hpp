#pragma once
/// @file detection_mosaic.hpp
/// @brief Tensor fixed-centre four-image mosaic, independent row copying and horizontal box/image flips.
#include "datamunge/nn/detection_augmentation.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Combine four caller-selected grayscale scenes into a same-size fixed-centre mosaic.
///
/// Restores byte values, averages each2x2 source pixel group and rounds half-up
/// to a byte before tiling four quadrants. This is a fixed-centre small-image
/// adaptation, not the random-centre crop mosaic used in full YOLO training.
/// Only selected rows enter the result; caller owns fit/validation isolation and
/// randomness. No gradients, fitting, model/state mutation or source mutation.
/// Empty B yields an empty batch with original canvas dimensions.
///
/// @param source Valid Float32/64 grayscale scenes with even positive H/W>=2, finite pixels[0,1] and aligned positive-area boxes/labels.
/// @param selections Int64 [B,4] source IDs in top-left,top-right,bottom-left,bottom-right order; repeats allowed.
/// @return Detached same-dtype [B,H,W,1] mosaics with half-scaled/translated boxes and attached labels.
/// @throws std::invalid_argument For invalid image dimensions/dtypes/pixels, selections, rows or annotation metadata.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::ones({4, 32, 32, 1});
/// for (int i = 0; i < 4; ++i) {
///   data.boxes.push_back(n::tensor({8., 8., 24., 24.}, {1, 4}));
///   data.labels.push_back(n::full({1}, double(i), n::DType::Int64));
/// }
/// const auto mosaic = n::models::MosaicTensorDetectionScenes(
///     data, n::from_vector(std::vector<std::int64_t>{0, 1, 2, 3}, {1, 4}));
/// @endcode
/// @par References
/// - YOLOv5 v6.0 augmentation reference https://github.com/ultralytics/yolov5/blob/v6.0/utils/augmentations.py; fixed-centre convention from examples/nn/common/detection_mosaic.hpp.
/// @see SelectTensorDetectionScenes(), FlipTensorDetectionScenes()
inline TensorDetectionScenes MosaicTensorDetectionScenes(const TensorDetectionScenes &source,
                                                         const nn::Tensor &selections) {
  if (selections.dim() != 2 || selections.shape()[1] != 4 ||
      selections.dtype() != nn::DType::Int64)
    throw std::invalid_argument("mosaic selections require Int64[B,4]");
  const auto rows = selections.to_vector<std::int64_t>();
  two_stage_detection_detail::ValidateScenes(source, rows, std::numeric_limits<int>::max());
  const auto H = source.images.shape()[1], W = source.images.shape()[2],
             B = selections.shape()[0];
  const auto type = source.images.dtype();
  if (H < 2 || W < 2 || H % 2 || W % 2 ||
      !nn::all(nn::logical_and(nn::greater_equal(source.images, 0.),
                               nn::less_equal(source.images, 1.)))
           .item<bool>())
    throw std::invalid_argument("mosaic requires even images and finite[0,1] pixels");
  nn::NoGradGuard guard;
  TensorDetectionScenes out;
  if (!B) {
    out.images = nn::zeros({0, H, W, 1}, type);
    return out;
  }
  const auto h = H / 2, w = W / 2;
  const auto bytes = nn::floor(
      nn::gather(source.images, selections.reshape({B * 4})).to(nn::DType::Float64) * 255. +
      .5);
  const auto reduced = nn::floor(nn::avg_pool2d(bytes, 2, 2, nn::Padding::Valid) + .5) / 255.;
  out.images = nn::transpose(reduced.reshape({B, 2, 2, h, w, 1}), {0, 1, 3, 2, 4, 5})
                   .reshape({B, H, W, 1})
                   .to(type);
  for (std::int64_t i = 0; i < B; ++i) {
    std::vector<nn::Tensor> boxes, labels;
    for (int q = 0; q < 4; ++q) {
      const auto row = rows[std::size_t(i * 4 + q)];
      const auto offset =
          nn::from_vector(std::vector<double>{double((q / 2) * h), double((q % 2) * w),
                                              double((q / 2) * h), double((q % 2) * w)})
              .to(type);
      boxes.push_back(source.boxes[row] * .5 + offset);
      labels.push_back(source.labels[row]);
    }
    out.boxes.push_back(nn::concat(boxes, 0));
    out.labels.push_back(nn::concat(labels, 0));
  }
  return out;
}
/// @brief Copy selected tensor detection rows into independent data storage.
///
/// No source mutation, numeric preprocessing, random draws or fitting. Empty
/// rows yield an empty batch. Cloned annotations prevent accidental writes
/// through returned tensor handles from changing the source.
///
/// @param source Valid floating grayscale scenes and aligned annotation tensors.
/// @param rows Source IDs in desired output order, including repeats.
/// @return Detached gathered images and cloned boxes/labels with independent mutable storage.
/// @throws std::invalid_argument For invalid scene geometry/annotation metadata or out-of-range row IDs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::ones({1, 32, 32, 1});
/// data.boxes = {n::tensor({8., 8., 24., 24.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// const auto copy = n::models::SelectTensorDetectionScenes(data, {0, 0});
/// @endcode
/// @par References
/// - Detection scene row-copy convention from examples/nn/common/detection_mosaic.hpp.
/// @see MosaicTensorDetectionScenes(), FlipTensorDetectionScenes()
inline TensorDetectionScenes
SelectTensorDetectionScenes(const TensorDetectionScenes &source,
                            const std::vector<std::int64_t> &rows) {
  two_stage_detection_detail::ValidateScenes(source, rows, std::numeric_limits<int>::max());
  nn::NoGradGuard guard;
  TensorDetectionScenes out;
  out.images = nn::gather(source.images, nn::from_vector(rows));
  for (auto row : rows) {
    out.boxes.push_back(source.boxes[row].clone());
    out.labels.push_back(source.labels[row].clone());
  }
  return out;
}
/// @brief Return selected scenes with horizontal pixel reversal and matching box-edge reflection.
///
/// Reflection uses pixel-edge box coordinates and the full image width.
/// Source pixels/annotations are not mutated; applying twice restores values.
/// No gradients, fitted preprocessing, random draws or optimizer.
///
/// @param source Valid floating grayscale scenes with aligned boxes and labels.
/// @param rows Source IDs in desired output order, including repeats.
/// @return Detached independent scenes with x1=width-old_x2,x2=width-old_x1 and unchanged labels.
/// @throws std::invalid_argument For invalid scene/annotation geometry or out-of-range rows.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::ones({1, 32, 32, 1});
/// data.boxes = {n::tensor({8., 4., 24., 12.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// const auto flipped = n::models::FlipTensorDetectionScenes(data, {0});
/// @endcode
/// @par References
/// - Detection horizontal-flip convention from examples/nn/common/detection_mosaic.hpp.
/// @see SelectTensorDetectionScenes(), MosaicTensorDetectionScenes()
inline TensorDetectionScenes FlipTensorDetectionScenes(const TensorDetectionScenes &source,
                                                       const std::vector<std::int64_t> &rows) {
  auto out = SelectTensorDetectionScenes(source, rows);
  nn::NoGradGuard guard;
  out.images = nn::reverse(out.images, {2});
  const double width = double(source.images.shape()[2]);
  for (auto &boxes : out.boxes) {
    const auto b = nn::unstack(boxes, 1);
    boxes = nn::stack({b[0], width - b[3], b[2], width - b[1]}, 1);
  }
  return out;
}
} // namespace datamunge::nn::models
