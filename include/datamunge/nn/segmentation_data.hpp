#pragma once
/// @file segmentation_data.hpp
/// @brief Tensor geometric segmentation scenes, fit-only pixel weights, weighted CE and confusion/IoU metrics.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/losses.hpp"
#include "datamunge/nn/random.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own aligned RGB images and integer shape-class masks.
///
/// No learned state or hidden optimizer. Labels identify shapes, not brightness;
/// background0,circle1,rectangle2,triangle3.
///
/// @throws std::invalid_argument For invalid shapes, dtypes, indices or settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto data = n::models::TensorSegmentationShapes(2, 32, 11);
/// @endcode
/// @par References
/// - Procedural shape/weighted-pixel/mean-IoU conventions from examples/nn/common/segmentation.hpp; tensor RNG and implementation adapted here.
/// @see TensorSegmentationShapes()
struct TensorSegmentationData {
  nn::Tensor images; ///< Float32[N,H,W,3] RGB images in[0,1].
  nn::Tensor labels; ///< Int64[N,H,W] class masks aligned with image pixels.
  int classes = 4;   ///< Foreground/background class count, default4.
};
/// @brief Generate class-independent photometry with overlapping circle/rectangle/triangle labels.
///
/// Each image has1..4 objects, independent centre/axis scales and brightness.
/// Later objects occlude earlier labels. Background/object noise and an image-wide
/// RGB tint are independent of class, preventing a fixed-colour classification
/// shortcut. Pixel centres determine inclusion. Uses the tensor generator, not
/// legacy mt19937, so fixed-seed pixels differ while distribution/conventions
/// remain comparable. Every scene is detached; no fitting or source mutation.
///
/// @param count Positive image count.
/// @param size Square side>=32.
/// @param seed Tensor RNG seed, default11.
/// @return Owned Float32 RGB images and aligned Int64 four-class masks.
/// @throws std::invalid_argument For invalid shapes, dtypes, indices or settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto data = n::models::TensorSegmentationShapes(2, 32, 11);
/// @endcode
/// @par References
/// - Procedural shape/weighted-pixel/mean-IoU conventions from examples/nn/common/segmentation.hpp; tensor RNG and implementation adapted here.
/// @see TensorSegmentationShapes()
inline TensorSegmentationData TensorSegmentationShapes(int count, int size,
                                                       std::uint64_t seed = 11) {
  if (count < 1 || size < 32)
    throw std::invalid_argument("shape scenes need count>0,size>=32");
  nn::NoGradGuard guard;
  nn::Generator rng(seed);
  const auto y = nn::arange(0, double(size)).reshape({size, 1}) + .5,
             x = nn::arange(0, double(size)).reshape({1, size}) + .5;
  std::vector<nn::Tensor> images, labels;
  for (int i = 0; i < count; ++i) {
    auto gray = .1 + nn::uniform({size, size}, -.04, .04, rng);
    auto label = nn::zeros({size, size}, nn::DType::Int64);
    const auto objects = nn::randint({}, 1, 5, rng).item<int>();
    for (int k = 0; k < objects; ++k) {
      const auto type = nn::randint({}, 1, 4, rng).item<int>();
      const auto centre = nn::uniform({2}, .2, .8, rng) * double(size),
                 radius = nn::uniform({2}, .12, .27, rng) * double(size),
                 brightness = nn::uniform({}, .55, .95, rng);
      const auto c = nn::unstack(centre, 0), r = nn::unstack(radius, 0);
      const auto dy = (y - c[0]) / r[0], dx = (x - c[1]) / r[1];
      const auto inside =
          type == 1 ? nn::less(nn::square(dx) + nn::square(dy), 1.)
          : type == 2
              ? nn::logical_and(nn::less(nn::abs(dx), 1.), nn::less(nn::abs(dy), 1.))
              : nn::logical_and(nn::logical_and(nn::greater(dy, -1.), nn::less(dy, 1.)),
                                nn::less(nn::abs(dx), (dy + 1.) * .5));
      gray = nn::where(inside, brightness + nn::uniform({size, size}, -.04, .04, rng), gray);
      label = nn::where(inside, nn::full({}, double(type), nn::DType::Int64), label);
    }
    const auto tint = 1. + nn::uniform({3}, -.04, .04, rng);
    images.push_back(
        nn::clip(gray.reshape({size, size, 1}) * tint.reshape({1, 1, 3}), 0., 1.));
    labels.push_back(label);
  }
  return {nn::stack(images, 0), nn::stack(labels, 0), 4};
}
/// @brief Gather aligned independent segmentation image/mask rows.
///
/// Gathered tensors own storage independently of source. No random draws,
/// fitting or source mutation.
///
/// @param data Valid RGB image and integer mask tensors.
/// @param rows Int64[B] in-range image IDs; repeats and empty B allowed.
/// @return Pair of detached [B,H,W,3] images and [B,H,W] labels.
/// @throws std::invalid_argument For invalid shapes, dtypes, indices or settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto data = n::models::TensorSegmentationShapes(2, 32, 11);
/// const auto batch = n::models::TensorSegmentationBatch(data, n::zeros({1}, n::DType::Int64));
/// @endcode
/// @par References
/// - Procedural shape/weighted-pixel/mean-IoU conventions from examples/nn/common/segmentation.hpp; tensor RNG and implementation adapted here.
/// @see TensorSegmentationData
inline std::pair<nn::Tensor, nn::Tensor>
TensorSegmentationBatch(const TensorSegmentationData &data, const nn::Tensor &rows) {
  if (data.images.dim() != 4 || data.images.shape()[3] != 3 ||
      data.images.dtype() != nn::DType::Float32 ||
      data.labels.shape() !=
          nn::Shape{data.images.shape()[0], data.images.shape()[1], data.images.shape()[2]} ||
      (data.labels.dtype() != nn::DType::Int64 && data.labels.dtype() != nn::DType::Int32) ||
      rows.dim() != 1 || rows.dtype() != nn::DType::Int64 ||
      !nn::all(nn::logical_and(nn::greater_equal(rows, 0.),
                               nn::less(rows, double(data.images.shape()[0]))))
           .item<bool>())
    throw std::invalid_argument("invalid segmentation batch");
  nn::NoGradGuard guard;
  return {nn::gather(data.images, rows), nn::gather(data.labels, rows)};
}
/// @brief Estimate inverse-square-root pixel class weights from fitting masks only.
///
/// A pseudocount1 prevents infinite absent-class weights. The caller must pass
/// fitting masks only; no validation/test data are inspected. No optimizer or
/// model mutation.
///
/// @param labels Nonempty Int32/64 tensor of labels in[0,classes).
/// @param classes At least two classes, including background.
/// @return Detached Float32[classes],1/sqrt((count_c+1)/number_of_pixels).
/// @throws std::invalid_argument For invalid shapes, dtypes, indices or settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto data = n::models::TensorSegmentationShapes(2, 32, 11);
/// const auto weights = n::models::SegmentationClassWeights(data.labels, 4);
/// @endcode
/// @par References
/// - Procedural shape/weighted-pixel/mean-IoU conventions from examples/nn/common/segmentation.hpp; tensor RNG and implementation adapted here.
/// @see SegmentationCrossEntropy()
inline nn::Tensor SegmentationClassWeights(const nn::Tensor &labels, int classes) {
  if (classes < 2 || labels.numel() < 1 ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64) ||
      !nn::all(
           nn::logical_and(nn::greater_equal(labels, 0.), nn::less(labels, double(classes))))
           .item<bool>())
    throw std::invalid_argument("invalid segmentation weight labels");
  nn::NoGradGuard guard;
  const auto count=nn::unsorted_segment_sum(nn::ones({labels.numel()},nn::DType::Int64),labels.reshape({-1}),classes).to(nn::DType::Float32)+1.;
  return 1. / nn::sqrt(count / double(labels.numel()));
}
/// @brief Compute normalized class-weighted pixel cross entropy from logits.
///
/// Stable log-softmax; class sums use tensors. Unlike region losses, this API
/// rejects void labels. Weights may preserve gradients if explicitly learned;
/// fit-only SegmentationClassWeights returns detached constants. No fitting
/// loop or model mutation.
///
/// @param logits Nonempty Float32/64[...,classes] logits.
/// @param labels Aligned Int32/64 leading-shape labels in[0,classes).
/// @param classes At least two classes.
/// @param class_weights Same-dtype[classes] finite nonnegative weights with positive selected sum.
/// @return Scalar sum(w_y*negative_log_probability_y)/sum(w_y).
/// @throws std::invalid_argument For invalid shapes, dtypes, indices or settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto logits = n::zeros({1, 1, 2, 3});
/// const auto labels = n::from_vector(std::vector<std::int64_t>{0, 1}, {1, 1, 2});
/// const auto weights = n::ones({3});
/// const auto loss = n::models::SegmentationCrossEntropy(logits, labels, 3, weights);
/// @endcode
/// @par References
/// - Procedural shape/weighted-pixel/mean-IoU conventions from examples/nn/common/segmentation.hpp; tensor RNG and implementation adapted here.
/// @see SegmentationClassWeights()
inline nn::Tensor SegmentationCrossEntropy(const nn::Tensor &logits, const nn::Tensor &labels,
                                           int classes, const nn::Tensor &class_weights) {
  auto shape = logits.shape();
  if (shape.size() < 2 || logits.numel() < 1 || classes < 2 || shape.back() != classes ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64) ||
      class_weights.shape() != nn::Shape{classes} || class_weights.dtype() != logits.dtype())
    throw std::invalid_argument("invalid pixel CE geometry/dtype");
  shape.pop_back();
  if (labels.shape() != shape ||
      !nn::all(
           nn::logical_and(nn::greater_equal(labels, 0.), nn::less(labels, double(classes))))
           .item<bool>() ||
      !nn::all(
           nn::logical_and(nn::greater_equal(class_weights, 0.),
                           nn::less(class_weights, std::numeric_limits<double>::infinity())))
           .item<bool>())
    throw std::invalid_argument("invalid pixel CE labels/weights");
  const auto flat = labels.reshape({-1}), weights = nn::gather(class_weights, flat);
  if (!(nn::sum(weights).item() > 0))
    throw std::invalid_argument("pixel CE weights sum to zero");
  const auto log_p = nn::log_softmax(logits.reshape({-1, classes}), -1),
             ce = -nn::sum(log_p * nn::one_hot(flat, classes, 1, 0, logits.dtype()), {1});
  return nn::sum(ce * weights) / nn::sum(weights);
}
/// @brief Own detached pixel confusion, accuracy and mean/per-class IoU.
///
/// Truth indexes rows and prediction indexes columns. Empty unions have IoU-1
/// and are excluded from the mean. Zero total yields accuracy/mean IoU0.
///
/// @throws std::invalid_argument For invalid shapes, dtypes, indices or settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto confusion = n::from_vector(std::vector<std::int64_t>{3, 1, 2, 4}, {2, 2});
/// const auto metrics = n::models::SegmentationConfusionMetrics(confusion);
/// @endcode
/// @par References
/// - Procedural shape/weighted-pixel/mean-IoU conventions from examples/nn/common/segmentation.hpp; tensor RNG and implementation adapted here.
/// @see TensorSegmentationShapes()
struct TensorSegmentationMetrics {
  nn::Tensor confusion;      ///< Int64[C,C] truth-row/prediction-column counts.
  nn::Tensor pixel_accuracy; ///< Float64 scalar,trace/total or0 for no pixels.
  nn::Tensor mean_iou;       ///< Float64 scalar,mean across classes with positive union,else0.
  nn::Tensor iou;            ///< Float64[C],per-class IoU or-1 for absent truth/prediction.
};
/// @brief Compute exact count-based pixel accuracy and present-class mean IoU.
///
/// Includes background when its union is positive. Classes present in either
/// truth or predictions contribute; absent classes are excluded. Empty matrices
/// of counts produce accuracy/mean IoU0,all per-class IoU-1. No fitting or mutation.
///
/// @param confusion Nonnegative Int64[C,C] square confusion matrix,C>=2.
/// @return Detached owned confusion and Float64 metrics.
/// @throws std::invalid_argument For invalid shapes, dtypes, indices or settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto confusion = n::from_vector(std::vector<std::int64_t>{3, 1, 2, 4}, {2, 2});
/// const auto metrics = n::models::SegmentationConfusionMetrics(confusion);
/// @endcode
/// @par References
/// - Procedural shape/weighted-pixel/mean-IoU conventions from examples/nn/common/segmentation.hpp; tensor RNG and implementation adapted here.
/// @see SegmentationPredictionMetrics()
inline TensorSegmentationMetrics SegmentationConfusionMetrics(const nn::Tensor &confusion) {
  if (confusion.dim() != 2 || confusion.shape()[0] < 2 ||
      confusion.shape()[0] != confusion.shape()[1] || confusion.dtype() != nn::DType::Int64 ||
      !nn::all(nn::greater_equal(confusion, 0.)).item<bool>())
    throw std::invalid_argument("invalid segmentation confusion");
  nn::NoGradGuard guard;
  const auto C = confusion.shape()[0];
  const auto type = nn::DType::Float64;
  const auto counts = confusion.to(type), diagonal = nn::sum(counts * nn::eye(C, type), {1}),
             actual = nn::sum(counts, {1}), predicted = nn::sum(counts, {0}),
             combined = actual + predicted - diagonal, valid = nn::greater(combined, 0.),
             ious = nn::where(valid, diagonal / nn::maximum(combined, nn::ones({}, type)),
                              nn::full({}, -1., type)),
             mean = nn::sum(nn::where(valid, ious, nn::zeros({}, type))) /
                    nn::maximum(nn::sum(valid.to(type)), nn::ones({}, type)),
             accuracy = nn::sum(diagonal) / nn::maximum(nn::sum(counts), nn::ones({}, type));
  return {confusion.clone(), accuracy, mean, ious};
}
/// @brief Accumulate integer pixel predictions and truth into tensor confusion metrics.
///
/// No model inference or fitting is hidden. The caller supplies argmax logits.
/// Segment sums accumulate exact Int64 counts in O(pixels+classes²) space/time,
/// then feed Float64
/// metrics. No mutation.
///
/// @param predictions Int32/64 tensor, values in[0,classes), empty allowed.
/// @param labels Same-shape Int32/64 truth labels in[0,classes).
/// @param classes At least two foreground/background classes.
/// @return Detached truth-row confusion, accuracy and present-class IoU.
/// @throws std::invalid_argument For invalid shapes, dtypes, indices or settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::from_vector(std::vector<std::int64_t>{0, 1, 1});
/// const auto metrics = n::models::SegmentationPredictionMetrics(y, y, 2);
/// @endcode
/// @par References
/// - Procedural shape/weighted-pixel/mean-IoU conventions from examples/nn/common/segmentation.hpp; tensor RNG and implementation adapted here.
/// @see SegmentationConfusionMetrics()
inline TensorSegmentationMetrics SegmentationPredictionMetrics(const nn::Tensor &predictions,
                                                               const nn::Tensor &labels,
                                                               int classes) {
  if (classes < 2 || predictions.shape() != labels.shape() ||
      (predictions.dtype() != nn::DType::Int32 && predictions.dtype() != nn::DType::Int64) ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64) ||
      !nn::all(nn::logical_and(nn::logical_and(nn::greater_equal(predictions, 0.),
                                               nn::less(predictions, double(classes))),
                               nn::logical_and(nn::greater_equal(labels, 0.),
                                               nn::less(labels, double(classes)))))
           .item<bool>())
    throw std::invalid_argument("invalid segmentation predictions/labels");
  nn::NoGradGuard guard;
  const auto index = labels.reshape({-1}).to(nn::DType::Int64) *
                         nn::full({}, double(classes), nn::DType::Int64) +
                     predictions.reshape({-1}).to(nn::DType::Int64);
  const auto counts=nn::unsorted_segment_sum(nn::ones({index.numel()},nn::DType::Int64),index,std::int64_t(classes)*classes).reshape({classes,classes});
  return SegmentationConfusionMetrics(counts);
}
/// @brief Map four-class shape labels to the source preview RGB palette.
///
/// Background is dark,circle red,rectangle green,triangle blue. Palette mapping
/// is presentation only; no fitting, randomness or source mutation.
///
/// @param labels Int32/64 tensor,values in[0,4).
/// @return Detached Float32 labels.shape+[3] RGB values in[0,1].
/// @throws std::invalid_argument For invalid shapes, dtypes, indices or settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto data = n::models::TensorSegmentationShapes(2, 32, 11);
/// const auto rgb = n::models::ColorSegmentationLabels(data.labels);
/// @endcode
/// @par References
/// - Procedural shape/weighted-pixel/mean-IoU conventions from examples/nn/common/segmentation.hpp; tensor RNG and implementation adapted here.
/// @see TensorSegmentationData
inline nn::Tensor ColorSegmentationLabels(const nn::Tensor &labels) {
  if ((labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64) ||
      !nn::all(nn::logical_and(nn::greater_equal(labels, 0.), nn::less(labels, 4.)))
           .item<bool>())
    throw std::invalid_argument("invalid shape palette labels");
  nn::NoGradGuard guard;
  const auto palette =
      nn::from_vector(std::vector<float>{24, 24, 32, 230, 70, 70, 70, 210, 110, 80, 120, 240},
                      {4, 3}) /
      255.;
  auto shape = labels.shape();
  shape.push_back(3);
  return nn::gather(palette, labels.reshape({-1})).reshape(shape);
}
} // namespace datamunge::nn::models
