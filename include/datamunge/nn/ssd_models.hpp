#pragma once
/// @file ssd_models.hpp
/// @brief Tensor SSD300/compact models, default boxes, multibox mining/targets and inference.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/classic_cnn_models.hpp"
#include "datamunge/nn/detection_components.hpp"
#include "datamunge/nn/detection_inference.hpp"
#include "datamunge/nn/detection_scene_models.hpp"
#include "datamunge/nn/two_stage_detection_models.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Construct SSD pixel default boxes with square/geometric-mean/reciprocal shapes.
///
/// Level/cell/shape order matches the prediction heads: row, column, then
/// square(size), square(sqrt(size*next_size)), wide/tall ratio2 and optionally
/// wide/tall ratio3. Cell centres are(row+.5)*step. Ratios here are width/height
/// for wide boxes. All square-root and coordinate math is tensor based. No
/// learned state, random draws or fitting. Work/storage O(number of defaults).
///
/// @param cells Nonempty positive square feature-grid sides, one per level.
/// @param sizes Positive finite pixel sides, one per level plus the next coarser side.
/// @param steps Positive finite pixel strides, one per level.
/// @param per_cell Box counts per level, each4 or6.
/// @param dtype Float32 by default, or Float64; all geometry arithmetic follows this dtype.
/// @return Detached [sum(cells²*per_cell),4] ordered half-open pixel boxes, without clipping.
/// @throws std::invalid_argument For inconsistent metadata lengths, nonpositive/nonfinite sizes/steps, invalid cells/counts or dtype.
/// @par Example
/// @code
/// const auto a =
///     dm::nn::models::SsdDefaultBoxes({38, 19, 10, 5, 3, 1}, {21, 45, 99, 153, 207, 261, 315},
///                                     {8, 16, 32, 64, 100, 300}, {4, 6, 6, 6, 4, 4}); // [8732,4]
/// @endcode
/// @par References
/// - Liu et al. (2016), SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
/// @see SSD(), MakeSsdTargets()
inline nn::Tensor SsdDefaultBoxes(const std::vector<int> &cells,
                                  const std::vector<double> &sizes,
                                  const std::vector<double> &steps,
                                  const std::vector<int> &per_cell,
                                  nn::DType dtype = nn::DType::Float32) {
  if (cells.empty() || sizes.size() != cells.size() + 1 || steps.size() != cells.size() ||
      per_cell.size() != cells.size() ||
      (dtype != nn::DType::Float32 && dtype != nn::DType::Float64))
    throw std::invalid_argument("invalid SSD default-box specification");
  for (double size : sizes)
    if (!std::isfinite(size) || size <= 0)
      throw std::invalid_argument("SSD sizes must be finite and positive");
  nn::NoGradGuard guard;
  const auto scales = nn::from_vector(sizes).to(dtype),
             strides = nn::from_vector(steps).to(dtype);
  std::vector<nn::Tensor> all;
  for (std::size_t l = 0; l < cells.size(); ++l) {
    if (cells[l] < 1 || !std::isfinite(steps[l]) || steps[l] <= 0 ||
        (per_cell[l] != 4 && per_cell[l] != 6))
      throw std::invalid_argument("invalid SSD default-box dimensions");
    const auto side = nn::slice_axis(scales, 0, std::int64_t(l), std::int64_t(l) + 1),
               next = nn::slice_axis(scales, 0, std::int64_t(l) + 1, std::int64_t(l) + 2),
               mean = nn::sqrt(side * next);
    std::vector<nn::Tensor> shapes{nn::concat({side, side}, 0).reshape({1, 2}),
                                   nn::concat({mean, mean}, 0).reshape({1, 2})};
    for (int ratio = 2; ratio <= (per_cell[l] == 6 ? 3 : 2); ++ratio) {
      const auto root = nn::sqrt(nn::full({}, double(ratio), dtype));
      const auto h = side / root, w = side * root;
      shapes.push_back(nn::concat({h, w}, 0).reshape({1, 2}));
      shapes.push_back(nn::concat({w, h}, 0).reshape({1, 2}));
    }
    const auto shape = nn::concat(shapes, 0);
    const auto K = std::int64_t(per_cell[l]), C = std::int64_t(cells[l]);
    const auto position = (nn::arange(0, double(C)).to(dtype) + .5) *
                          nn::slice_axis(strides, 0, std::int64_t(l), std::int64_t(l) + 1);
    const auto yy = position.reshape({C, 1, 1}) * nn::ones({C, C, K}, dtype),
               xx = position.reshape({1, C, 1}) * nn::ones({C, C, K}, dtype);
    const auto h = nn::slice_axis(shape, 1, 0, 1).reshape({1, 1, K}),
               w = nn::slice_axis(shape, 1, 1, 2).reshape({1, 1, K});
    all.push_back(nn::stack({yy - h * .5, xx - w * .5, yy + h * .5, xx + w * .5}, 3)
                      .reshape({C * C * K, 4}));
  }
  return nn::concat(all, 0);
}
/// @brief Normalize NHWC channels by their L2 norm and apply a learned per-channel scale.
///
/// Zero channel vectors remain finite zero; their input derivative is scale/1e-6,
/// not zero. The scale is unconstrained during fitting. No random draws, running
/// state, input mutation or optimizer. Work O(numel(x)).
///
/// @param p Mutable Float32/64 learned store; creates <name>/scale with one value per channel.
/// @param name Parameter prefix for the rescaling vector.
/// @param x Floating NHWC tensor, all dimensions positive, same dtype as p.
/// @param initial_scale Finite initializer value, default20; reused entries keep their existing values.
/// @return x/sqrt(max(sum_channels(x²),1e-12))*scale, preserving dtype and gradients.
/// @throws std::invalid_argument For invalid input shape/dtype/scale or incompatible existing leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::SsdNormalize(
///     p, "norm", n::tensor({3., 4., 0., 0.}, {1, 1, 2, 2})); // [12,16,0,0]
/// @endcode
/// @par References
/// - Liu et al. (2016), SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
/// @see SSD()
inline nn::Tensor SsdNormalize(nn::ParameterStore &p, const std::string &name,
                               const nn::Tensor &x, double initial_scale = 20.) {
  if (x.dim() != 4 || x.shape()[0] < 1 || x.shape()[1] < 1 || x.shape()[2] < 1 ||
      x.shape()[3] < 1 || x.dtype() != p.dtype() ||
      (p.dtype() != nn::DType::Float32 && p.dtype() != nn::DType::Float64) ||
      !std::isfinite(initial_scale))
    throw std::invalid_argument("invalid SSD normalization input/settings");
  const auto norm =
      nn::sqrt(nn::maximum(nn::sum(nn::square(x), {3}, true), nn::full({}, 1e-12, x.dtype())));
  return x / norm * p.get(name + "/scale", {x.shape()[3]}, nn::init::constant(initial_scale));
}
/// @brief Retain SSD multi-scale feature maps and aligned prediction tensors.
///
/// Tensor handles own activation storage/graphs; learned leaves belong to the
/// store. Metadata lists align six levels. Background is the final class.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto out = n::models::SSD(p, n::zeros({1, 64, 64, 1}), 11, true);
/// @endcode
/// @par References
/// - Liu et al. (2016), SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
/// @see SSD(), SsdMultiboxLoss()
struct SsdOutputs {
  std::vector<nn::Tensor>
      features; ///< Six floating NHWC maps, fine to coarse; first map is L2-rescaled.
  std::vector<int> cells; ///< Square sides: full38/19/10/5/3/1, compact16/8/4/2/1/1.
  std::vector<int>
      channels; ///< Full512/1024/512/256/256/256 or compact64/128/128/128/128/128.
  std::vector<int> boxes_per_cell; ///< Per-level default counts4/6/6/6/4/4.
  nn::Tensor logits;               ///< Floating [N,A,classes] unnormalized class scores.
  nn::Tensor probabilities;        ///< Floating [N,A,classes] softmax probabilities.
  nn::Tensor deltas;               ///< Floating [N,A,4] class-agnostic normalized box offsets.
  int anchors = 0; ///< Full8732 or compact1536, concatenated level/row/column/default order.
};
/// @brief Build source VGG16 SSD300 or the explicit compact grayscale scene adaptation.
///
/// Full SSD300 uses VGG16 conv4_3 L2 rescaling(initial20), dilation-six fc6,
/// fc7 and four extra blocks, producing8732 predictions and26285486 learned
/// scalars at21 classes. Compact uses stride-two32/64 convolutions, L2 scale10
/// and five128-channel stages, yielding1536 predictions and1167362 learned
/// scalars at11 classes. Rescaling affects the first head only; the backbone
/// continues from the unscaled map. Kernels use He-uniform initialization and
/// biases start at zero; no pretrained weights are loaded. No batch norm, dropout, moving state
/// or fitting. Create every leaf before constructing an optimizer and keep leaf
/// values unchanged until backward.
///
/// @param p Mutable Float32/64 learned parameter store; all leaves are lazily named by source architecture.
/// @param image Floating [N,300,300,3] for full or [N,64,64,1] for compact, N positive, same dtype as p; caller normalizes pixels.
/// @param classes At least2 classes including final background; default21.
/// @param compact False for full SSD300 (default), true for the64-pixel six-level teaching model.
/// @return Six features, grid/channel/default metadata and differentiable aligned logits/probabilities/deltas.
/// @throws std::invalid_argument For invalid geometry/dtype/classes or incompatible existing leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto out = n::models::SSD(p, n::zeros({1, 64, 64, 1}), 11, true);
/// @endcode
/// @par References
/// - Liu et al. (2016), SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
/// @see SsdDefaultBoxes(), SsdMultiboxLoss(), DecodeSsdDetections()
inline SsdOutputs SSD(nn::ParameterStore &p, const nn::Tensor &image, int classes = 21,
                      bool compact = false) {
  if (classes < 2 || classes > std::numeric_limits<int>::max() / 6 || image.dim() != 4 ||
      image.shape()[0] < 1 || image.shape()[1] != (compact ? 64 : 300) ||
      image.shape()[2] != (compact ? 64 : 300) || image.shape()[3] != (compact ? 1 : 3) ||
      image.dtype() != p.dtype() ||
      (p.dtype() != nn::DType::Float32 && p.dtype() != nn::DType::Float64))
    throw std::invalid_argument(
        "SSD requires matching floating full300RGB or compact64gray geometry/classes");
  SsdOutputs result;
  result.boxes_per_cell = {4, 6, 6, 6, 4, 4};
  nn::ConvLayerOptions cfg;
  cfg.init = nn::init::he_uniform();
  cfg.activation = nn::Activation::Relu;
  nn::Tensor h = image;
  if (compact) {
    result.cells = {16, 8, 4, 2, 1, 1};
    result.channels = {64, 128, 128, 128, 128, 128};
    cfg.stride = 2;
    h = nn::conv2d(p, "backbone/conv1", h, 32, 3, cfg);
    h = nn::conv2d(p, "backbone/conv2", h, 64, 3, cfg);
    result.features.push_back(SsdNormalize(p, "normalize", h, 10));
    for (int i = 1; i < 6; ++i) {
      cfg.stride = i == 5 ? 1 : 2;
      h = nn::conv2d(p, "backbone/feature" + std::to_string(i + 1), h, 128, 3, cfg);
      result.features.push_back(h);
    }
  } else {
    result.cells = {38, 19, 10, 5, 3, 1};
    result.channels = {512, 1024, 512, 256, 256, 256};
    const int widths[] = {64, 128, 256, 512, 512};
    const auto depths = VggBlockDepths(16);
    for (int block = 0; block < 5; ++block) {
      for (int layer = 0; layer < depths[std::size_t(block)]; ++layer)
        h = nn::conv2d(p,
                       "backbone/block" + std::to_string(block + 1) + "/conv" +
                           std::to_string(layer + 1),
                       h, widths[block], 3, cfg);
      if (block == 3)
        result.features.push_back(SsdNormalize(p, "normalize", h));
      if (block < 4)
        h = nn::max_pool2d(h, 2, 2, block == 2 ? nn::Padding::Same : nn::Padding::Valid);
    }
    h = nn::max_pool2d(h, 3, 1, nn::Padding::Same);
    auto fc6 = cfg;
    fc6.dilation = 6;
    h = nn::conv2d(p, "backbone/fc6", h, 1024, 3, fc6);
    h = nn::conv2d(p, "backbone/fc7", h, 1024, 1, cfg);
    result.features.push_back(h);
    for (int i = 0; i < 4; ++i) {
      const std::string name = "extra" + std::to_string(i + 1);
      const int hidden = i ? 128 : 256, out = i ? 256 : 512;
      auto reduce = cfg, spatial = cfg;
      h = nn::conv2d(p, name + "/reduce", h, hidden, 1, reduce);
      spatial.stride = i < 2 ? 2 : 1;
      spatial.padding = nn::Padding::Valid;
      if (i < 2)
        h = nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}});
      h = nn::conv2d(p, name + "/spatial", h, out, 3, spatial);
      result.features.push_back(h);
    }
  }
  nn::ConvLayerOptions linear;
  linear.init = nn::init::he_uniform();
  std::vector<nn::Tensor> scores, offsets;
  for (std::size_t i = 0; i < 6; ++i) {
    const int count = result.cells[i] * result.cells[i] * result.boxes_per_cell[i];
    result.anchors += count;
    const auto name = "head" + std::to_string(i + 1);
    scores.push_back(nn::reshape(nn::conv2d(p, name + "/scores", result.features[i],
                                            result.boxes_per_cell[i] * classes, 3, linear),
                                 {-1, count, classes}));
    offsets.push_back(nn::reshape(nn::conv2d(p, name + "/boxes", result.features[i],
                                             result.boxes_per_cell[i] * 4, 3, linear),
                                  {-1, count, 4}));
  }
  result.logits = nn::concat(scores, 1);
  result.probabilities = nn::softmax(result.logits, -1);
  result.deltas = nn::concat(offsets, 1);
  return result;
}

/// @brief Retain SSD objective terms and detached hard-negative selection metadata.
///
/// Classification and regression divide by the same batch-wide positive count
/// floored at one. Per-image mining uses each image's own positive count.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::SsdMultiboxLoss(
///     n::zeros({1, 4, 3}), n::zeros({1, 4, 4}),
///     n::from_vector(std::vector<std::int64_t>{0, 2, 2, 2}, {1, 4}), n::zeros({1, 4, 4}));
/// @endcode
/// @par References
/// - Liu et al. (2016), SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
/// @see SsdMultiboxLoss()
struct SsdMultiboxLossOutputs {
  nn::Tensor loss; ///< Scalar normalized classification plus regression loss, with gradients.
  nn::Tensor classification; ///< Scalar selected-positive/negative cross-entropy contribution.
  nn::Tensor regression; ///< Scalar positive-anchor summed-coordinate Smooth-L1 contribution.
  nn::Tensor
      selected_negatives; ///< Detached Bool [N,A] hardest background selection; stable lower-index ties.
  nn::Tensor positive_counts; ///< Detached Int64 [N] positive counts before mining.
};
/// @brief Compute tensor SSD multibox loss with independent per-image three-to-one hard-negative mining.
///
/// Ranks detached background cross-entropies descending per image, with
/// positives excluded and lower anchor index breaking ties. Selects at most
/// three backgrounds per positive, limited by available backgrounds. Tensor
/// argsort and its inverse implement all ranking/masking; no host numerical
/// sort. The loss sums selected CE and positive Smooth-L1 over the batch, then
/// divides by total positives floored at one. An all-background batch has
/// exact zero loss and zero gradients, not background-only learning. Empty A
/// returns a graph-connected zero. No fitting or parameter mutation.
///
/// @param logits Float32/64 [N,A,C], N positive,C at least2; finite numerical values are caller requirement.
/// @param deltas Floating [N,A,4] predictions, same dtype as logits.
/// @param labels Int64 [N,A] in[0,C-1]; background is C-1.
/// @param targets Floating [N,A,4] encoded offsets, same dtype as logits; finite values are caller requirement.
/// @return Normalized objective terms, detached Bool hard-negative mask and per-image positive counts.
/// @throws std::invalid_argument For inconsistent shapes/dtypes or out-of-range labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::SsdMultiboxLoss(
///     n::zeros({1, 4, 3}), n::zeros({1, 4, 4}),
///     n::from_vector(std::vector<std::int64_t>{0, 2, 2, 2}, {1, 4}), n::zeros({1, 4, 4}));
/// @endcode
/// @par References
/// - Liu et al. (2016), SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
/// @see MakeSsdTargets(), DetectorSmoothL1()
inline SsdMultiboxLossOutputs SsdMultiboxLoss(const nn::Tensor &logits,
                                              const nn::Tensor &deltas,
                                              const nn::Tensor &labels,
                                              const nn::Tensor &targets) {
  if (logits.dim() != 3 || logits.shape()[0] < 1 || logits.shape()[2] < 2 ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid SSD logits");
  const auto N = logits.shape()[0], A = logits.shape()[1], C = logits.shape()[2];
  const auto type = logits.dtype();
  if (deltas.shape() != nn::Shape{N, A, 4} || deltas.dtype() != type ||
      labels.shape() != nn::Shape{N, A} || labels.dtype() != nn::DType::Int64 ||
      targets.shape() != deltas.shape() || targets.dtype() != type ||
      !nn::all(nn::logical_and(nn::greater_equal(labels, 0.), nn::less(labels, double(C))))
           .item<bool>())
    throw std::invalid_argument("invalid SSD loss targets/shapes/dtypes");
  SsdMultiboxLossOutputs out;
  const auto positive = nn::not_equal(labels, double(C - 1));
  out.positive_counts = nn::count_nonzero(positive, {1});
  if (!A) {
    out.classification = nn::sum(logits) * 0.;
    out.regression = nn::sum(deltas) * 0.;
    out.loss = out.classification + out.regression;
    out.selected_negatives = nn::zeros({N, 0}, nn::DType::Bool);
    return out;
  }
  const auto ce =
      -nn::sum(nn::one_hot(labels, C, 1, 0, type) * nn::log_softmax(logits, -1), {2});
  const auto ranked = nn::argsort(
      nn::where(positive, nn::full({}, -std::numeric_limits<double>::infinity(), type),
                ce.detach()),
      true);
  const auto rank = nn::argsort(ranked, false);
  const auto counts = out.positive_counts.reshape({N, 1});
  const auto wanted = nn::minimum(counts * nn::full({}, 3., nn::DType::Int64),
                                  nn::full({}, double(A), nn::DType::Int64) - counts);
  out.selected_negatives = nn::logical_and(nn::logical_not(positive), nn::less(rank, wanted));
  const auto mask = positive.to(type), weights = mask + out.selected_negatives.to(type),
             denominator = nn::maximum(nn::sum(mask), nn::ones({}, type));
  out.classification = nn::sum(ce * weights) / denominator;
  out.regression =
      nn::sum(nn::sum(DetectorSmoothL1(deltas - targets), {2}) * mask) / denominator;
  out.loss = out.classification + out.regression;
  return out;
}
/// @brief Own tensor SSD class labels and encoded box targets.
///
/// Detached data tensors share scene/anchor order. Background is specified by
/// the target builder; no learnable parameters or fitting state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 1;
/// cfg.max_objects = 1;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// const auto anchors =
///     n::models::SsdDefaultBoxes({16, 8, 4, 2, 1, 1}, {8, 16, 24, 32, 48, 64, 80},
///                                {4, 8, 16, 32, 64, 64}, {4, 6, 6, 6, 4, 4});
/// const auto targets = n::models::MakeSsdTargets(anchors, scenes, {0}, 10);
/// @endcode
/// @par References
/// - Liu et al. (2016), SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
/// @see MakeSsdTargets(), SsdMultiboxLoss()
struct TensorSsdTargets {
  nn::Tensor labels; ///< Int64 [N,A] target class, background for unmatched anchors.
  nn::Tensor
      targets; ///< Floating [N,A,4] encoded positive offsets, zeros for unmatched anchors.
};
/// @brief Match default boxes to scene truth using distinct greedy SSD forced assignments.
///
/// Uses tensor IoU and MatchSsd: first assigns distinct anchors greedily, even
/// for zero-overlap pairs, then adds unmatched anchors at IoU>=.5. Matched labels
/// come from selected truths; encoded deltas use variances.1/.1/.2/.2. Empty
/// truth makes every anchor background. All matching and encoding arithmetic
/// uses tensors. No image cropping, random draws, parameter mutation or fitting.
///
/// @param anchors Floating [A,4] finite positive-area boxes, same dtype as scene images; at least one per selected target.
/// @param scenes Floating grayscale scene metadata with aligned positive-area boxes and nonnegative Int64 labels below background; pixels are not read.
/// @param rows Selected source-scene IDs in prediction minibatch order.
/// @param background Positive final class identifier, larger than every selected truth label.
/// @return Detached [N,A] labels and [N,A,4] offsets in anchor dtype; empty rows yield empty leading dimension.
/// @throws std::invalid_argument For invalid anchor/scene geometry/dtype/labels, rows or more targets than anchors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 1;
/// cfg.max_objects = 1;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// const auto anchors =
///     n::models::SsdDefaultBoxes({16, 8, 4, 2, 1, 1}, {8, 16, 24, 32, 48, 64, 80},
///                                {4, 8, 16, 32, 64, 64}, {4, 6, 6, 6, 4, 4});
/// const auto targets = n::models::MakeSsdTargets(anchors, scenes, {0}, 10);
/// @endcode
/// @par References
/// - Liu et al. (2016), SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
/// @see MatchSsd(), SsdMultiboxLoss()
inline TensorSsdTargets MakeSsdTargets(const nn::Tensor &anchors,
                                       const TensorDetectionScenes &scenes,
                                       const std::vector<std::int64_t> &rows, int background) {
  two_stage_detection_detail::ValidateScenes(scenes, rows, background);
  if (anchors.dim() != 2 || anchors.shape()[1] != 4 ||
      anchors.dtype() != scenes.images.dtype())
    throw std::invalid_argument("invalid SSD target anchors");
  SquareDetectionBoxes(anchors, 1.);
  if (nn::any(nn::less_equal(
                  nn::slice_axis(anchors, 1, 2, 4) - nn::slice_axis(anchors, 1, 0, 2), 0.))
          .item<bool>())
    throw std::invalid_argument("SSD target anchors require positive extents");
  nn::NoGradGuard guard;
  const auto A = anchors.shape()[0], N = std::int64_t(rows.size());
  const auto type = anchors.dtype();
  std::vector<nn::Tensor> labels, targets;
  for (auto row : rows) {
    const auto &truth = scenes.boxes[row];
    const auto match = MatchSsd(nn::box_iou(anchors, truth));
    const auto positive = nn::greater_equal(match, 0.);
    nn::Tensor label, target;
    if (truth.shape()[0]) {
      const auto safe = nn::maximum(match, nn::zeros({}, nn::DType::Int64));
      label = nn::where(positive, nn::gather(scenes.labels[row], safe),
                        nn::full({}, double(background), nn::DType::Int64));
      target =
          nn::where(positive.reshape({A, 1}),
                    nn::encode_boxes(anchors, nn::gather(truth, safe)), nn::zeros({}, type));
    } else {
      label = nn::full({A}, double(background), nn::DType::Int64);
      target = nn::zeros({A, 4}, type);
    }
    labels.push_back(label);
    targets.push_back(target);
  }
  return {labels.empty() ? nn::zeros({N, A}, nn::DType::Int64) : nn::stack(labels, 0),
          targets.empty() ? nn::zeros({N, A, 4}, type) : nn::stack(targets, 0)};
}
/// @brief Decode SSD predictions, cap candidates per class, and apply clipped class-wise NMS.
///
/// Every foreground class is considered independently; no argmax shortcut.
/// Top-score ties prefer the higher default-box index, preserving the source
/// std::greater<pair<score,index>> convention. Top candidates are chosen before
/// zero-area decoded boxes are removed. Decoding clips log scales above4, then
/// coordinates to image bounds. NMS does not mix classes/images. No fitting or
/// input mutation; all score/coordinate arithmetic uses tensors.
///
/// @param probabilities Float32/64 [N,A,C], N positive,C at least2, finite; final class is background.
/// @param deltas Floating [N,A,4] finite class-agnostic offsets, same dtype.
/// @param anchors Floating [A,4] finite ordered pixel boxes, same dtype.
/// @param images Int64 [N] nonnegative dataset image IDs.
/// @param height Positive pixel height for clipping.
/// @param width Positive pixel width for clipping.
/// @param before_per_class Nonnegative top candidate cap per image/class, default200.
/// @param score_threshold Finite strict class-score cutoff, default.05.
/// @param iou Finite class-wise suppression cutoff in[0,1], default.45.
/// @param max_per_image Nonnegative final per-image budget, default100.
/// @param suppress Apply final class-wise NMS/per-image cap when true (default); false returns pre-NMS candidates.
/// @return Detached clipped class-specific detections with positive box area, sorted by image then score.
/// @throws std::invalid_argument For invalid shapes/dtypes, nonfinite values/IDs, reversed anchors, dimensions or settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto out = n::models::SSD(p, n::zeros({1, 64, 64, 1}), 11, true);
/// const auto anchors =
///     n::models::SsdDefaultBoxes({16, 8, 4, 2, 1, 1}, {8, 16, 24, 32, 48, 64, 80},
///                                {4, 8, 16, 32, 64, 64}, {4, 6, 6, 6, 4, 4});
/// const auto d =
///     n::models::DecodeSsdDetections(out.probabilities, out.deltas, anchors,
///                                    n::from_vector(std::vector<std::int64_t>{0}), 64, 64);
/// @endcode
/// @par References
/// - Liu et al. (2016), SSD: Single Shot MultiBox Detector, https://arxiv.org/abs/1512.02325; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
/// @see SuppressDetectionCandidates(), SSD()
inline TensorDetections
DecodeSsdDetections(const nn::Tensor &probabilities, const nn::Tensor &deltas,
                    const nn::Tensor &anchors, const nn::Tensor &images, int height, int width,
                    int before_per_class = 200, double score_threshold = .05, double iou = .45,
                    int max_per_image = 100, bool suppress = true) {
  if (probabilities.dim() != 3 || probabilities.shape()[0] < 1 ||
      probabilities.shape()[2] < 2 ||
      (probabilities.dtype() != nn::DType::Float32 &&
       probabilities.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid SSD probability shape/dtype");
  const auto N = probabilities.shape()[0], A = probabilities.shape()[1],
             C = probabilities.shape()[2];
  const auto type = probabilities.dtype();
  if (deltas.shape() != nn::Shape{N, A, 4} || anchors.shape() != nn::Shape{A, 4} ||
      deltas.dtype() != type || anchors.dtype() != type || images.shape() != nn::Shape{N} ||
      images.dtype() != nn::DType::Int64 || height < 1 || width < 1 || before_per_class < 0 ||
      !std::isfinite(score_threshold) || !std::isfinite(iou) || iou < 0 || iou > 1 ||
      max_per_image < 0 || !nn::all(nn::greater_equal(images, 0.)).item<bool>())
    throw std::invalid_argument("invalid SSD decode inputs/settings");
  for (const auto &x : {probabilities, deltas, anchors})
    if (!nn::all(nn::less(nn::abs(x), std::numeric_limits<double>::infinity())).item<bool>())
      throw std::invalid_argument("SSD decoding requires finite values");
  SquareDetectionBoxes(anchors, 1.);
  nn::NoGradGuard guard;
  const auto decoded =
      nn::clip_boxes(nn::decode_boxes(anchors.reshape({1, A, 4}), deltas), height, width);
  std::vector<nn::Tensor> boxes, scores, labels, ids;
  for (std::int64_t image = 0; image < N; ++image) {
    const auto pp = nn::slice_axis(probabilities, 0, image, image + 1).reshape({A, C}),
               bb = nn::slice_axis(decoded, 0, image, image + 1).reshape({A, 4});
    for (std::int64_t label = 0; label < C - 1; ++label) {
      const auto score = nn::slice_axis(pp, 1, label, label + 1).reshape({A});
      const auto order = nn::full({}, double(A - 1), nn::DType::Int64) -
                         nn::argsort(nn::reverse(score, {0}), true);
      const auto top =
          nn::slice_axis(order, 0, 0, std::min<std::int64_t>(before_per_class, A));
      const auto chosen_boxes = nn::gather(bb, top), chosen_scores = nn::gather(score, top);
      const auto valid = two_stage_detection_detail::Indices(
          nn::logical_and(nn::greater(chosen_scores, score_threshold),
                          nn::greater(nn::box_area(chosen_boxes), 0.)));
      if (!valid.numel())
        continue;
      const auto count = valid.numel();
      boxes.push_back(nn::gather(chosen_boxes, valid));
      scores.push_back(nn::gather(chosen_scores, valid));
      labels.push_back(nn::full({count}, double(label), nn::DType::Int64));
      ids.push_back(nn::slice_axis(images, 0, image, image + 1) *
                    nn::ones({count}, nn::DType::Int64));
    }
  }
  if (boxes.empty())
    return SuppressDetectionCandidates(
        nn::zeros({0, 4}, type), nn::zeros({0}, type), nn::zeros({0}, nn::DType::Int64),
        nn::zeros({0}, nn::DType::Int64), iou, score_threshold, max_per_image);
  if (!suppress)
    return {nn::concat(boxes, 0), nn::concat(scores, 0), nn::concat(labels, 0), nn::concat(ids, 0)};
  return SuppressDetectionCandidates(nn::concat(boxes, 0), nn::concat(scores, 0),
                                     nn::concat(labels, 0), nn::concat(ids, 0), iou,
                                     score_threshold, max_per_image);
}
} // namespace datamunge::nn::models
