#pragma once
/// @file detection_inference.hpp
/// @brief Tensor class-wise suppression with per-image budgets and evaluation-record presentation.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/boxes.hpp"
#include "datamunge/nn/detection_evaluation.hpp"
#include "datamunge/nn/ops.hpp"
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own detached detected boxes, scores and integer image/category identifiers.
///
/// Each field has the same leading detection count D. Result order is increasing
/// image ID, then descending score after suppression;raw decoders may preserve
/// original region/class order. Input order breaks score ties. No fitted state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::SuppressDetectionCandidates(
///     n::tensor({0., 0., 10., 10.}, {1, 4}), n::tensor({.9}),
///     n::from_vector(std::vector<std::int64_t>{0}),
///     n::from_vector(std::vector<std::int64_t>{1}));
/// @endcode
/// @see SuppressDetectionCandidates(), DetectionEvaluationRecords()
struct TensorDetections {
  nn::Tensor boxes;  ///< Float32/64 [D,4] finite ordered half-open pixel boxes.
  nn::Tensor scores; ///< Same floating dtype [D] ranking scores.
  nn::Tensor labels; ///< Int64 [D] nonnegative category IDs.
  nn::Tensor images; ///< Int64 [D] nonnegative image IDs.
};
/// @brief Apply class-wise tensor NMS and a separate detection budget to each image.
///
/// A host map groups only integer row indices by image. All score selection, IoU
/// and class-wise NMS run through tensor kernels. Boxes of different images or
/// categories never suppress each other. Unlike a global batched-NMS cap, the
/// budget resets for each image. No fitting, parameter mutation or gradients.
/// Worst-case overlap work is quadratic within image/class candidate groups.
///
/// @param boxes Float32/64 [D,4] finite ordered half-open coordinates; zero area is allowed.
/// @param scores Floating [D] finite scores, in boxes dtype.
/// @param labels Int64 [D] nonnegative category identifiers.
/// @param images Int64 [D] nonnegative image identifiers; need not be contiguous.
/// @param iou Finite suppression threshold in [0,1], default.3; only strictly larger overlaps suppress.
/// @param score_threshold Finite cutoff, default.05; scores must be strictly greater to be eligible.
/// @param max_per_image Nonnegative output cap applied independently to each image, default100.
/// @return Detached selected tensors, ordered by image then descending score with stable input ties.
/// @throws std::invalid_argument For invalid shapes/dtypes, IDs, nonfinite scores/coordinates, reversed boxes, cutoffs or cap.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::SuppressDetectionCandidates(
///     n::tensor({0., 0., 10., 10., 0., 0., 10., 10.}, {2, 4}), n::tensor({.9, .8}),
///     n::from_vector(std::vector<std::int64_t>{0, 1}),
///     n::from_vector(std::vector<std::int64_t>{3, 3})); // two distinct classes survive
/// @endcode
/// @par References
/// - TensorFlow non_max_suppression, https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression
/// @see nn::batched_non_max_suppression(), DetectionEvaluationRecords()
inline TensorDetections SuppressDetectionCandidates(const nn::Tensor &boxes,
                                                    const nn::Tensor &scores,
                                                    const nn::Tensor &labels,
                                                    const nn::Tensor &images, double iou = .3,
                                                    double score_threshold = .05,
                                                    int max_per_image = 100) {
  if (boxes.dim() != 2 || boxes.shape()[1] != 4 ||
      scores.shape() != nn::Shape{boxes.shape()[0]} || labels.shape() != scores.shape() ||
      images.shape() != scores.shape() ||
      (boxes.dtype() != nn::DType::Float32 && boxes.dtype() != nn::DType::Float64) ||
      scores.dtype() != boxes.dtype() || labels.dtype() != nn::DType::Int64 ||
      images.dtype() != nn::DType::Int64 || !std::isfinite(iou) || iou < 0 || iou > 1 ||
      !std::isfinite(score_threshold) || max_per_image < 0)
    throw std::invalid_argument("invalid detection candidate shapes/dtypes/settings");
  // Tensor finiteness predicate; no host numerical reduction.
  const auto finite = [](const nn::Tensor &x) {
    return nn::all(nn::less(nn::abs(x), std::numeric_limits<double>::infinity())).item<bool>();
  };
  if (!finite(boxes) || !finite(scores) ||
      !nn::all(nn::greater_equal(labels, 0.)).item<bool>() ||
      !nn::all(nn::greater_equal(images, 0.)).item<bool>() ||
      nn::any(nn::less(nn::slice_axis(boxes, 1, 2, 4), nn::slice_axis(boxes, 1, 0, 2)))
          .item<bool>())
    throw std::invalid_argument("invalid detection candidate values");
  nn::NoGradGuard guard;
  std::map<std::int64_t, std::vector<std::int64_t>> groups;
  const auto ids = images.to_vector<std::int64_t>();
  for (std::size_t i = 0; i < ids.size(); ++i)
    groups[ids[i]].push_back(std::int64_t(i));
  std::vector<nn::Tensor> kept;
  for (const auto &[image, rows] : groups) {
    const auto index = nn::from_vector(rows);
    const auto chosen = nn::batched_non_max_suppression(
        nn::gather(boxes, index), nn::gather(scores, index), nn::gather(labels, index), iou,
        max_per_image, score_threshold);
    if (chosen.numel())
      kept.push_back(nn::gather(index, chosen));
  }
  const auto indices = kept.empty() ? nn::zeros({0}, nn::DType::Int64) : nn::concat(kept, 0);
  return {nn::gather(boxes, indices), nn::gather(scores, indices), nn::gather(labels, indices),
          nn::gather(images, indices)};
}
/// @brief Present tensor detections as the original Float32 evaluator record format.
///
/// This is data presentation, not another model implementation or matching pass.
/// No gradients, fitting or mutation. The canonical AP evaluator converts these
/// source-format coordinates to Float64 before metric arithmetic.
///
/// @param detections Consistent tensor detections; finite Float32-representable scores/coordinates and IDs fitting signed int.
/// @return Owning records, preserving order; Float64 coordinates/scores round to Float32 on conversion.
/// @throws std::invalid_argument For inconsistent shapes/dtypes, unrepresentable values or IDs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::SuppressDetectionCandidates(
///     n::tensor({0., 0., 10., 10.}, {1, 4}), n::tensor({.9}),
///     n::from_vector(std::vector<std::int64_t>{0}),
///     n::from_vector(std::vector<std::int64_t>{1}));
/// const auto records = n::models::DetectionEvaluationRecords(d);
/// @endcode
/// @see EvaluateDetectionDataset(), TensorDetections
inline std::vector<DetectionEvaluationPrediction>
DetectionEvaluationRecords(const TensorDetections &detections) {
  const auto &d = detections;
  const auto D = d.scores.numel();
  if (d.scores.dim() != 1 || d.boxes.shape() != nn::Shape{D, 4} ||
      d.labels.shape() != nn::Shape{D} || d.images.shape() != nn::Shape{D} ||
      (d.scores.dtype() != nn::DType::Float32 && d.scores.dtype() != nn::DType::Float64) ||
      d.boxes.dtype() != d.scores.dtype() || d.labels.dtype() != nn::DType::Int64 ||
      d.images.dtype() != nn::DType::Int64)
    throw std::invalid_argument("invalid tensor detection records");
  for (const auto &x : {d.scores, d.boxes})
    if (!nn::all(nn::less_equal(nn::abs(x), double(std::numeric_limits<float>::max())))
             .item<bool>())
      throw std::invalid_argument("evaluation records require Float32-representable values");
  for (const auto &x : {d.labels, d.images})
    if (!nn::all(nn::logical_and(nn::greater_equal(x, 0.),
                                 nn::less_equal(x, double(std::numeric_limits<int>::max()))))
             .item<bool>())
      throw std::invalid_argument("evaluation record IDs must fit signed int");
  const auto boxes = d.boxes.to_vector<float>(), scores = d.scores.to_vector<float>();
  const auto labels = d.labels.to_vector<int>(), images = d.images.to_vector<int>();
  std::vector<DetectionEvaluationPrediction> out;
  for (std::int64_t i = 0; i < D; ++i)
    out.push_back({images[i],
                   labels[i],
                   scores[i],
                   {boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]}});
  return out;
}
} // namespace datamunge::nn::models
