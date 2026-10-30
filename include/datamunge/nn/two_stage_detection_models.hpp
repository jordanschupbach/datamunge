#pragma once
/// @file two_stage_detection_models.hpp
/// @brief Tensor region heads, multitask losses and shared stride-four detection backbone; no fitting.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/detection_components.hpp"
#include "datamunge/nn/detection_inference.hpp"
#include "datamunge/nn/detection_scene_models.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/losses.hpp"
#include "datamunge/nn/ops.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Retain class probabilities and class-specific box regression tensors.
///
/// Owning tensor handles retain graphs; learned leaves belong to the store.
/// The last class is background and has no box regressor. No parameter fitting.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto r = n::models::RcnnPatchHead(p, n::ones({1, 28, 28, 1}));
/// @endcode
/// @par References
/// - Girshick (2015), Fast R-CNN, https://arxiv.org/abs/1504.08083
/// @see RcnnPatchHead(), DetectionRoiHead(), DetectionRoiLoss()
struct DetectionRoiOutputs {
  nn::Tensor logits;        ///< Floating [K,C] linear class scores, background last.
  nn::Tensor probabilities; ///< Floating [K,C] softmax probabilities across classes.
  nn::Tensor deltas;        ///< Floating [K,C-1,4] class-specific dy/dx/dh/dw offsets.
};
/// @brief Pad each pixel box by a fraction of its height and width on every edge.
///
/// Does not clip at image edges. Zero extents remain zero. Inputs are not
/// mutated; no random state, learned parameters or optimizer.
///
/// @param boxes Float32/64 [...,4] finite ordered pixel-edge coordinates.
/// @param fraction Finite nonnegative per-side padding fraction, default.07.
/// @return Same-shaped differentiable boxes with unchanged centre and extents multiplied by1+2*fraction.
/// @throws std::invalid_argument For malformed/nonfinite/reversed boxes or invalid fraction.
/// @par Example
/// @code
/// const auto padded =
///     dm::nn::models::PadDetectionBoxes(dm::nn::tensor({0., 0., 10., 20.}, {1, 4}), .07);
/// @endcode
/// @par References
/// - Girshick et al. (2014), Rich feature hierarchies for accurate object detection and semantic segmentation, https://arxiv.org/abs/1311.2524
/// @see CropDetectionPatches()
inline nn::Tensor PadDetectionBoxes(const nn::Tensor &boxes, double fraction = .07) {
  if (!std::isfinite(fraction) || fraction < 0)
    throw std::invalid_argument("invalid box padding fraction");
  SquareDetectionBoxes(boxes, 1.);
  const auto c = nn::boxes_to_center(boxes);
  return nn::boxes_from_center(nn::concat(
      {nn::slice_axis(c, -1, 0, 2), nn::slice_axis(c, -1, 2, 4) * (1. + 2. * fraction)}, -1));
}
/// @brief Map pixel-edge boxes to normalized corner-aligned stride-grid RoIs.
///
/// Feature cell i is centred at stride*i+stride/2. The returned coordinates
/// are for TF endpoint-aligned crop_and_resize, distinct from pixel-centre
/// CropDetectionPatches. No clipping, mutation or fitting.
///
/// @param boxes Float32/64 [...,4] finite ordered pixel boxes; coordinates may lie outside the image.
/// @param stride Positive integer feature stride.
/// @param feature_height Feature-grid height, at least2.
/// @param feature_width Feature-grid width, at least2.
/// @return Same-shaped normalized boxes (pixel-stride/2)/stride/(feature_side-1), preserving gradients.
/// @throws std::invalid_argument For invalid boxes, stride or feature sides.
/// @par Example
/// @code
/// const auto roi = dm::nn::models::DetectionFeatureBoxes(
///     dm::nn::tensor({8., 12., 40., 36.}, {1, 4}), 4, 16, 16);
/// @endcode
/// @par References
/// - Girshick (2015), Fast R-CNN, https://arxiv.org/abs/1504.08083
/// @see DetectionRoiHead()
inline nn::Tensor DetectionFeatureBoxes(const nn::Tensor &boxes, int stride,
                                        int feature_height, int feature_width) {
  if (stride < 1 || feature_height < 2 || feature_width < 2)
    throw std::invalid_argument("invalid feature-grid geometry");
  SquareDetectionBoxes(boxes, 1.);
  const auto scale =
      nn::from_vector(
          std::vector<double>{double(feature_height - 1), double(feature_width - 1),
                              double(feature_height - 1), double(feature_width - 1)})
          .to(boxes.dtype());
  return ((boxes - double(stride) * .5) / double(stride)) / scale;
}
/// @brief Build the source four-convolution shared stride-four feature extractor.
///
/// Biased SAME 3x3 He/ReLU convolutions use widths32/32, SAME2x2 pooling,
/// then widths64/64 and another SAME2x2 pool. No batch normalization, dropout,
/// state updates or fitting.64-pixel images yield16x16 features;64992 learned
/// scalars. Work is ordinary convolution; keep store leaves unchanged until backward.
///
/// @param p Mutable Float32/64 learned store; lazily creates backbone/conv1 through conv4.
/// @param images Floating [N,H,W,1], positive dimensions, same dtype as p; caller supplies normalization.
/// @return [N,ceil(H/4),ceil(W/4),64] differentiable ReLU features.
/// @throws std::invalid_argument For invalid image shape/dtype or incompatible existing store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto features = n::models::StrideFourDetectionBackbone(p, n::zeros({1, 64, 64, 1}));
/// @endcode
/// @par References
/// - Girshick (2015), Fast R-CNN, https://arxiv.org/abs/1504.08083
/// @see DetectionRoiHead(), BuildRpn()
inline nn::Tensor StrideFourDetectionBackbone(nn::ParameterStore &p,
                                              const nn::Tensor &images) {
  if (images.dim() != 4 || images.shape()[0] < 1 || images.shape()[1] < 1 ||
      images.shape()[2] < 1 || images.shape()[3] != 1 || images.dtype() != p.dtype() ||
      (p.dtype() != nn::DType::Float32 && p.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid detector backbone input");
  nn::ConvLayerOptions cfg;
  cfg.init = nn::init::he_uniform();
  cfg.activation = nn::Activation::Relu;
  auto h = nn::conv2d(p, "backbone/conv1", images, 32, 3, cfg);
  h = nn::max_pool2d(nn::conv2d(p, "backbone/conv2", h, 32, 3, cfg), 2, 2, nn::Padding::Same);
  h = nn::conv2d(p, "backbone/conv3", h, 64, 3, cfg);
  return nn::max_pool2d(nn::conv2d(p, "backbone/conv4", h, 64, 3, cfg), 2, 2,
                        nn::Padding::Same);
}
/// @brief Apply the shared two-dense-layer class/box head to already pooled RoIs.
///
/// Shares the same parameter names as DetectionRoiHead. Input pooling is the
/// caller's choice; this function performs no sampling, normalization, mutation
/// beyond lazy leaf creation or fitting. Keep store leaves unchanged until backward.
///
/// @param p Mutable Float32/64 learned store; creates head/fc6, fc7, classifier and regressor.
/// @param pooled Floating [K,7,7,Cin], K/Cin positive, same dtype as p.
/// @param classes At least2 classes including final background, default11.
/// @param hidden Positive width of both dense ReLU layers, default256.
/// @return Differentiable class logits/probabilities and class-specific four-offset deltas.
/// @throws std::invalid_argument For invalid shape/dtype/classes/width or incompatible existing leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto h = n::models::PooledDetectionRoiHead(p, n::ones({1, 7, 7, 64}));
/// @endcode
/// @par References
/// - Girshick (2015), Fast R-CNN, https://arxiv.org/abs/1504.08083
/// @see DetectionRoiHead(), DetectionRoiLoss()

inline DetectionRoiOutputs PooledDetectionRoiHead(nn::ParameterStore &p,
                                                  const nn::Tensor &pooled, int classes = 11,
                                                  int hidden = 256) {
  if (pooled.dim() != 4 || pooled.shape()[0] < 1 || pooled.shape()[1] != 7 ||
      pooled.shape()[2] != 7 || pooled.shape()[3] < 1 || pooled.dtype() != p.dtype() ||
      (p.dtype() != nn::DType::Float32 && p.dtype() != nn::DType::Float64) || classes < 2 ||
      classes > std::numeric_limits<int>::max() / 4 || hidden < 1)
    throw std::invalid_argument("invalid pooled RoI head input/settings");
  auto f = nn::dense(p, "head/fc6", nn::reshape(pooled, {-1, 7 * 7 * pooled.shape()[3]}),
                     hidden, nn::Activation::Relu);
  f = nn::dense(p, "head/fc7", f, hidden, nn::Activation::Relu);
  DetectionRoiOutputs out;
  out.logits = nn::dense(p, "head/classifier", f, classes);
  out.probabilities = nn::softmax(out.logits, -1);
  out.deltas =
      nn::reshape(nn::dense(p, "head/regressor", f, 4 * (classes - 1)), {-1, classes - 1, 4});
  return out;
}
/// @brief Pool shared feature RoIs and build a two-dense-layer detector head.
///
/// Uses crop_and_resize to14x14 then VALID2x2 max pooling to7x7. This is
/// the source bilinear pooling adaptation, not original rounded-bin RoI pooling
/// and not ROIAlign. Core crop_and_resize treats boxes as constants: no RoI
/// coordinate gradients. Tensor shapes/finite coordinates/indices are validated;
/// flipped normalized boxes are allowed by the underlying crop op. No fitting,
/// normalization, dropout or moving state; keep learned leaves unchanged until backward.
///
/// @param p Mutable Float32/64 learned store; creates head/fc6, fc7, classifier and regressor.
/// @param features Floating [N,H,W,Cin], all dimensions positive, same dtype as p.
/// @param rois Floating [K,4] normalized corner-aligned crop boxes, K positive; coordinates outside[0,1] extrapolate zero.
/// @param indices Int32/64 [K] valid feature-batch row indices.
/// @param classes At least2 classes including final background; default11.
/// @param hidden Positive width of both dense ReLU layers, default256.
/// @return Class logits/probabilities and class-specific box deltas retaining feature/parameter gradients.
/// @throws std::invalid_argument For invalid settings/input shapes/dtypes/coordinates/indices or incompatible store leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto out = n::models::DetectionRoiHead(p, n::ones({1, 16, 16, 64}),
///                                              n::tensor({0., 0., 1., 1.}, {1, 4}),
///                                              n::from_vector(std::vector<std::int64_t>{0}));
/// @endcode
/// @par References
/// - Girshick (2015), Fast R-CNN, https://arxiv.org/abs/1504.08083
/// @see DetectionFeatureBoxes(), DetectionRoiLoss()
inline DetectionRoiOutputs DetectionRoiHead(nn::ParameterStore &p, const nn::Tensor &features,
                                            const nn::Tensor &rois, const nn::Tensor &indices,
                                            int classes = 11, int hidden = 256) {
  if (features.dim() != 4 || features.shape()[0] < 1 || features.shape()[1] < 1 ||
      features.shape()[2] < 1 || features.shape()[3] < 1 || features.dtype() != p.dtype() ||
      (p.dtype() != nn::DType::Float32 && p.dtype() != nn::DType::Float64) ||
      rois.dim() != 2 || rois.shape()[0] < 1 || rois.shape()[1] != 4 ||
      (rois.dtype() != nn::DType::Float32 && rois.dtype() != nn::DType::Float64) ||
      indices.shape() != nn::Shape{rois.shape()[0]} ||
      (indices.dtype() != nn::DType::Int32 && indices.dtype() != nn::DType::Int64) ||
      classes < 2 || classes > std::numeric_limits<int>::max() / 4 || hidden < 1 ||
      !nn::all(nn::less(nn::abs(rois), std::numeric_limits<double>::infinity()))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(indices, 0.),
                               nn::less(indices, double(features.shape()[0]))))
           .item<bool>())
    throw std::invalid_argument("invalid RoI head inputs/configuration");
  const auto pooled = nn::max_pool2d(nn::crop_and_resize(features, rois, indices, {14, 14}), 2,
                                     2, nn::Padding::Valid);
  return PooledDetectionRoiHead(p, pooled, classes, hidden);
}
/// @brief Build the source compact R-CNN patch classifier and class-specific regressor.
///
/// Two biased SAME3x3 He/ReLU convolutions (32/64), each followed by SAME2x2
/// max pooling, precede a128-unit dense ReLU and independent classification and
/// regression heads. At11 classes there are426931 learned scalars. This source
/// adaptation uses softmax and a joint learned regressor, not paper SVMs and
/// separate ridge fits. No dropout, batch norm, pretrained weights or fitting.
///
/// @param p Mutable Float32/64 learned store; creates conv1/conv2/fc1/classifier/regressor leaves.
/// @param patches Floating [K,28,28,1], K positive, same dtype as p; caller handles padding/crops/normalization.
/// @param classes At least2 classes including final background, default11.
/// @return [K,classes] scores/probabilities and [K,classes-1,4] box deltas with full graphs.
/// @throws std::invalid_argument For invalid patch shape/dtype/classes or incompatible store leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto out = n::models::RcnnPatchHead(p, n::ones({1, 28, 28, 1}));
/// @endcode
/// @par References
/// - Girshick et al. (2014), Rich feature hierarchies for accurate object detection and semantic segmentation, https://arxiv.org/abs/1311.2524
/// @see DetectionRoiLoss(), CropDetectionPatches()
inline DetectionRoiOutputs RcnnPatchHead(nn::ParameterStore &p, const nn::Tensor &patches,
                                         int classes = 11) {
  if (patches.dim() != 4 || patches.shape()[0] < 1 || patches.shape()[1] != 28 ||
      patches.shape()[2] != 28 || patches.shape()[3] != 1 || patches.dtype() != p.dtype() ||
      (p.dtype() != nn::DType::Float32 && p.dtype() != nn::DType::Float64) || classes < 2 ||
      classes > std::numeric_limits<int>::max() / 4)
    throw std::invalid_argument("invalid R-CNN patch input/classes");
  nn::ConvLayerOptions cfg;
  cfg.init = nn::init::he_uniform();
  cfg.activation = nn::Activation::Relu;
  auto h =
      nn::max_pool2d(nn::conv2d(p, "conv1", patches, 32, 3, cfg), 2, 2, nn::Padding::Same);
  h = nn::max_pool2d(nn::conv2d(p, "conv2", h, 64, 3, cfg), 2, 2, nn::Padding::Same);
  const auto f = nn::relu(nn::dense(p, "fc1", nn::reshape(h, {-1, 7 * 7 * 64}), 128));
  DetectionRoiOutputs out;
  out.logits = nn::dense(p, "classifier", f, classes);
  out.probabilities = nn::softmax(out.logits, -1);
  out.deltas =
      nn::reshape(nn::dense(p, "regressor", f, 4 * (classes - 1)), {-1, classes - 1, 4});
  return out;
}
/// @brief Combine mean region classification loss with selected-class Smooth-L1 regression.
///
/// Tensor one-hot selects only the labelled foreground class regressor; unused
/// class-specific outputs have zero gradient. Background contributes only to
/// classification. No parameter mutation, sampling or fitting. O(K*C) class
/// selection/loss work in addition to head computation.
///
/// @param head Floating logits[K,C] and deltas[K,C-1,4], K positive,C at least2; probability field is not used.
/// @param labels Int64 [K] class labels in[0,C-1]; final class is background.
/// @param targets Floating [K,4] normalized regression targets in logits dtype; finite values are caller requirement.
/// @param mask Floating [K] finite nonnegative regression weights in logits dtype; must be zero for background.
/// @return Scalar sparse cross-entropy plus weighted summed-coordinate Smooth-L1/max(sum(mask),1).
/// @throws std::invalid_argument For invalid head/target shapes, dtypes, labels or mask values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto head = n::models::RcnnPatchHead(p, n::ones({1, 28, 28, 1}));
/// const auto loss = n::models::DetectionRoiLoss(
///     head, n::from_vector(std::vector<std::int64_t>{3}), n::zeros({1, 4}), n::ones({1}));
/// @endcode
/// @par References
/// - Girshick (2015), Fast R-CNN, https://arxiv.org/abs/1504.08083
/// @see DetectorSmoothL1(), DetectionRoiHead()
inline nn::Tensor DetectionRoiLoss(const DetectionRoiOutputs &head, const nn::Tensor &labels,
                                   const nn::Tensor &targets, const nn::Tensor &mask) {
  if (head.logits.dim() != 2 || head.logits.shape()[0] < 1 || head.logits.shape()[1] < 2)
    throw std::invalid_argument("invalid RoI loss logits");
  const auto K = head.logits.shape()[0], C = head.logits.shape()[1];
  const auto type = head.logits.dtype();
  if ((type != nn::DType::Float32 && type != nn::DType::Float64) ||
      head.deltas.shape() != nn::Shape{K, C - 1, 4} || head.deltas.dtype() != type ||
      labels.shape() != nn::Shape{K} || labels.dtype() != nn::DType::Int64 ||
      targets.shape() != nn::Shape{K, 4} || targets.dtype() != type ||
      mask.shape() != nn::Shape{K} || mask.dtype() != type ||
      !nn::all(nn::logical_and(nn::greater_equal(labels, 0.), nn::less(labels, double(C))))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(mask, 0.),
                               nn::less(mask, std::numeric_limits<double>::infinity())))
           .item<bool>() ||
      nn::any(nn::logical_and(nn::equal(labels, double(C - 1)), nn::greater(mask, 0.)))
          .item<bool>())
    throw std::invalid_argument("invalid RoI loss targets/mask");
  const auto selected = nn::slice_axis(nn::one_hot(labels, C, 1, 0, type), 1, 0, C - 1);
  const auto picked = nn::sum(head.deltas * selected.reshape({K, C - 1, 1}), {1});
  const auto regression = nn::sum(nn::sum(DetectorSmoothL1(picked - targets), {1}) * mask) /
                          nn::maximum(nn::sum(mask), nn::ones({}, type));
  return nn::sparse_softmax_cross_entropy(head.logits, labels) + regression;
}
/// @brief Own R-CNN warped patches and detached classification/regression targets.
///
/// All fields share leading region count K. Scene indices retain dataset IDs,
/// not positions in a selected minibatch. Background label is10. These are data
/// tensors; no learned parameters, mutable fitting state or optimizer is retained.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 1;
/// cfg.max_objects = 1;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// const auto samples = n::models::MakeRcnnTrainingSamples(scenes, {0});
/// @endcode
/// @par References
/// - Girshick et al. (2014), Rich feature hierarchies for accurate object detection and semantic segmentation, https://arxiv.org/abs/1311.2524
/// @see MakeRcnnTrainingSamples(), DetectionRoiLoss()
struct RcnnTrainingSamples {
  nn::Tensor images;  ///< Floating [K,28,28,1] zero-padded warped patches in [0,1].
  nn::Tensor labels;  ///< Int64 [K], digit0..9 for maximum IoU>=.5, otherwise background10.
  nn::Tensor targets; ///< Floating [K,4] encoded box deltas where IoU>=.6, otherwise zero.
  nn::Tensor mask; ///< Floating [K] regression eligibility, one at IoU>=.6 and zero elsewhere.
  nn::Tensor
      boxes; ///< Floating [K,4] original proposal/ground-truth pixel boxes before7% padding.
  nn::Tensor indices; ///< Int64 [K] source scene indices.
};
/// @brief Build the source R-CNN proposal-plus-ground-truth fine-tuning dataset.
///
/// For each image, component proposals use byte thresholds48/64/96/128/160/200/230
/// and gap3; every ground-truth box is then appended. A candidate takes its
/// maximum-IoU target, first on ties. IoU>=.5 labels foreground; IoU>=.6 enables
/// regression. Crop padding is7% on all edges, then anisotropic28x28 pixel-centre
/// sampling. All model/data geometry uses tensors. No balancing, optimizer or
/// pretraining is hidden here. Empty selections return correctly shaped empty
/// data. Work includes component search, proposal/target IoU and resampling.
///
/// @param scenes Valid normalized tensor scenes with finite positive-area boxes and Int64 digit labels.
/// @param rows Selected valid source-scene indices; only these scenes contribute.
/// @return Detached warped patches, labels, encoded regression targets/masks, pixel boxes and source indices.
/// @throws std::invalid_argument For invalid scene shapes/annotation values, boxes with nonpositive extents or invalid selected rows.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 1;
/// cfg.max_objects = 1;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// const auto samples = n::models::MakeRcnnTrainingSamples(scenes, {0});
/// @endcode
/// @par References
/// - Girshick et al. (2014), Rich feature hierarchies for accurate object detection and semantic segmentation, https://arxiv.org/abs/1311.2524
/// @see RcnnPatchHead(), DetectionRoiLoss(), ComponentDetectionProposals()
inline RcnnTrainingSamples MakeRcnnTrainingSamples(const TensorDetectionScenes &scenes,
                                                   const std::vector<std::int64_t> &rows) {
  const auto &images = scenes.images;
  if (images.dim() != 4 || images.shape()[0] < 1 || images.shape()[1] < 1 ||
      images.shape()[2] < 1 || images.shape()[3] != 1 ||
      (images.dtype() != nn::DType::Float32 && images.dtype() != nn::DType::Float64) ||
      std::int64_t(scenes.boxes.size()) != images.shape()[0] ||
      scenes.labels.size() != scenes.boxes.size())
    throw std::invalid_argument("invalid R-CNN scene data");
  nn::NoGradGuard guard;
  const auto type = images.dtype();
  std::vector<nn::Tensor> boxes, labels, targets, masks;
  std::vector<std::int64_t> indices;
  for (auto row : rows) {
    if (row < 0 || row >= images.shape()[0])
      throw std::invalid_argument("R-CNN scene row outside source");
    const auto &truth = scenes.boxes[row];
    const auto &classes = scenes.labels[row];
    if (truth.dim() != 2 || truth.shape()[1] != 4 || truth.dtype() != type ||
        classes.dtype() != nn::DType::Int64 ||
        classes.shape() != nn::Shape{truth.shape()[0]} ||
        !nn::all(nn::logical_and(nn::greater_equal(classes, 0.), nn::less(classes, 10.)))
             .item<bool>())
      throw std::invalid_argument("invalid R-CNN scene annotations");
    SquareDetectionBoxes(truth, 1.);
    if (nn::any(nn::less_equal(nn::slice_axis(truth, 1, 2, 4) - nn::slice_axis(truth, 1, 0, 2),
                               0.))
            .item<bool>())
      throw std::invalid_argument("R-CNN truth boxes must have positive extents");
    const auto proposals =
        ComponentDetectionProposals(nn::slice_axis(images, 0, row, row + 1)
                                        .reshape({images.shape()[1], images.shape()[2], 1}),
                                    {48, 64, 96, 128, 160, 200, 230}, 3);
    const auto candidate = nn::concat({proposals, truth}, 0);
    const auto K = candidate.shape()[0];
    if (!K)
      continue;
    nn::Tensor label, target, mask;
    if (truth.shape()[0]) {
      const auto overlap = nn::box_iou(candidate, truth), best = nn::max(overlap, {1}),
                 assigned = nn::argmax(overlap, 1);
      label = nn::where(nn::greater_equal(best, .5), nn::gather(classes, assigned),
                        nn::full({}, 10., nn::DType::Int64));
      mask = nn::greater_equal(best, .6).to(type);
      target = nn::where(mask.reshape({K, 1}),
                         nn::encode_boxes(candidate, nn::gather(truth, assigned)),
                         nn::zeros({}, type));
    } else {
      label = nn::full({K}, 10., nn::DType::Int64);
      mask = nn::zeros({K}, type);
      target = nn::zeros({K, 4}, type);
    }
    boxes.push_back(candidate);
    labels.push_back(label);
    targets.push_back(target);
    masks.push_back(mask);
    for (std::int64_t i = 0; i < K; ++i)
      indices.push_back(row);
  }
  RcnnTrainingSamples out;
  out.boxes = boxes.empty() ? nn::zeros({0, 4}, type) : nn::concat(boxes, 0);
  out.labels = labels.empty() ? nn::zeros({0}, nn::DType::Int64) : nn::concat(labels, 0);
  out.targets = targets.empty() ? nn::zeros({0, 4}, type) : nn::concat(targets, 0);
  out.mask = masks.empty() ? nn::zeros({0}, type) : nn::concat(masks, 0);
  out.indices = nn::from_vector(indices);
  out.images =
      CropDetectionPatches(images, out.indices, PadDetectionBoxes(out.boxes, .07), 28);
  return out;
}
/// @brief Internal discrete-index and scene-contract helpers for two-stage detection data.
namespace two_stage_detection_detail {
/// @brief Return increasing indices of true entries in a one-dimensional mask.
/// @param mask One-dimensional Bool tensor; callers provide a validated shape.
/// @return Detached Int64 indices; empty masks return an empty vector.
inline nn::Tensor Indices(const nn::Tensor &mask) {
  const auto count = nn::count_nonzero(mask).item<std::int64_t>();
  if (!count)
    return nn::zeros({0}, nn::DType::Int64);
  return nn::slice_axis(nn::argsort(mask.to(nn::DType::Int64), true), 0, 0, count);
}
/// @brief Validate scene dimensions, selected rows and aligned positive-area annotations.
/// @param scenes Floating normalized single-channel scene data.
/// @param rows Selected source-scene IDs; repeats are permitted.
/// @param background Positive final-class index, greater than every selected truth label.
/// @throws std::invalid_argument For malformed dimensions, rows, boxes or labels.
inline void ValidateScenes(const TensorDetectionScenes &scenes,
                           const std::vector<std::int64_t> &rows, int background) {
  const auto &images = scenes.images;
  if (images.dim() != 4 || images.shape()[0] < 1 || images.shape()[1] < 1 ||
      images.shape()[2] < 1 || images.shape()[3] != 1 ||
      (images.dtype() != nn::DType::Float32 && images.dtype() != nn::DType::Float64) ||
      std::int64_t(scenes.boxes.size()) != images.shape()[0] ||
      scenes.labels.size() != scenes.boxes.size() || background < 1)
    throw std::invalid_argument("invalid two-stage scene dimensions");
  for (auto row : rows) {
    if (row < 0 || row >= images.shape()[0])
      throw std::invalid_argument("two-stage row outside scene data");
    const auto &boxes = scenes.boxes[row];
    const auto &labels = scenes.labels[row];
    if (boxes.dim() != 2 || boxes.shape()[1] != 4 || boxes.dtype() != images.dtype() ||
        labels.dtype() != nn::DType::Int64 || labels.shape() != nn::Shape{boxes.shape()[0]} ||
        !nn::all(nn::logical_and(nn::greater_equal(labels, 0.),
                                 nn::less(labels, double(background))))
             .item<bool>())
      throw std::invalid_argument("invalid two-stage annotations");
    SquareDetectionBoxes(boxes, 1.);
    if (nn::any(nn::less_equal(nn::slice_axis(boxes, 1, 2, 4) - nn::slice_axis(boxes, 1, 0, 2),
                               0.))
            .item<bool>())
      throw std::invalid_argument("two-stage boxes require positive extents");
  }
}
} // namespace two_stage_detection_detail
/// @brief Own normalized RoIs and detached classification/regression targets for a scene minibatch.
///
/// All fields share leading sampled-region count K. Image indices are positions
/// in the selected feature minibatch, not original scene IDs. Tensor handles own
/// storage. There are no fitted parameters or optimizer state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 1;
/// cfg.max_objects = 1;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// n::Generator rng(7);
/// const auto batch = n::models::SampleDetectionRois(scenes, {0}, {scenes.boxes[0]}, rng);
/// @endcode
/// @par References
/// - Girshick (2015), Fast R-CNN, https://arxiv.org/abs/1504.08083
/// @see SampleDetectionRois(), DetectionRoiHead(), DetectionRoiLoss()
struct TensorRoiBatch {
  nn::Tensor rois;    ///< Floating [K,4] normalized corner-aligned feature-grid boxes.
  nn::Tensor pixels;  ///< Floating [K,4] source pixel-edge boxes.
  nn::Tensor targets; ///< Floating [K,4] encoded offsets for positives, zero for background.
  nn::Tensor indices; ///< Int64 [K] selected feature minibatch row positions.
  nn::Tensor labels;  ///< Int64 [K] foreground label or final background index.
  nn::Tensor mask;    ///< Floating [K] regression mask, one for IoU>=.5.
};
/// @brief Sample up to one quarter foreground RoIs and fill with near then far background.
///
/// Appends ground-truth boxes and per_image random boxes (height/width10..40)
/// to candidates. Maximum IoU chooses the first target on ties. Foreground is
/// IoU>=.5, near background[.1,.5), then far background below.1. Each group is
/// shuffled; at most floor(per_image/4) foreground entries are retained before
/// background fills remaining capacity. Foreground scarcity is not oversampled.
/// Tensor RNG differs from legacy std RNG. Random windows may cross boundaries
/// on canvases smaller than40 pixels. Normalized RoIs use source stride-centre
/// mapping. No fitting or image mutation. Empty rows return empty fields.
///
/// @param scenes Valid floating grayscale scene data with positive-area boxes and digit labels.
/// @param rows Source scene IDs whose feature maps form the same-order minibatch.
/// @param candidates One floating [Pi,4] positive-area candidate tensor per selected scene, in image dtype.
/// @param rng Mutable tensor generator for random boxes and within-category shuffles; advances.
/// @param per_image Positive requested maximum regions per image, default64.
/// @param background Positive final class index, larger than all truth labels, default10.
/// @param stride Positive feature stride, default4.
/// @param feature_height Feature grid height at least2, default16.
/// @param feature_width Feature grid width at least2, default16.
/// @return Detached normalized/pixel RoIs, minibatch indices, labels and foreground box targets/masks.
/// @throws std::invalid_argument For invalid scene/candidate geometry, counts, dtype, labels or feature settings.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 1;
/// cfg.max_objects = 1;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// n::Generator rng(7);
/// const auto batch = n::models::SampleDetectionRois(scenes, {0}, {scenes.boxes[0]}, rng);
/// @endcode
/// @par References
/// - Girshick (2015), Fast R-CNN, https://arxiv.org/abs/1504.08083
/// @see DetectionFeatureBoxes(), DetectionRoiLoss()
inline TensorRoiBatch SampleDetectionRois(const TensorDetectionScenes &scenes,
                                          const std::vector<std::int64_t> &rows,
                                          const std::vector<nn::Tensor> &candidates,
                                          nn::Generator &rng, int per_image = 64,
                                          int background = 10, int stride = 4,
                                          int feature_height = 16, int feature_width = 16) {
  two_stage_detection_detail::ValidateScenes(scenes, rows, background);
  if (candidates.size() != rows.size() || per_image < 1 || stride < 1 || feature_height < 2 ||
      feature_width < 2)
    throw std::invalid_argument("invalid RoI sampler configuration");
  nn::NoGradGuard guard;
  const auto type = scenes.images.dtype();
  std::vector<nn::Tensor> pixels, rois, labels, targets, masks, indices;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto row = rows[i];
    const auto &candidate = candidates[i];
    const auto &truth = scenes.boxes[row];
    if (candidate.dim() != 2 || candidate.shape()[1] != 4 || candidate.dtype() != type)
      throw std::invalid_argument("invalid candidate shape/dtype");
    SquareDetectionBoxes(candidate, 1.);
    if (nn::any(nn::less_equal(nn::slice_axis(candidate, 1, 2, 4) -
                                   nn::slice_axis(candidate, 1, 0, 2),
                               0.))
            .item<bool>())
      throw std::invalid_argument("candidate boxes require positive extents");
    const auto u = nn::uniform({per_image, 4}, 0., 1., rng, type);
    // Immutable per-image uniform-draw coordinate accessor.
    const auto coordinate = [&](int c) {
      return nn::slice_axis(u, 1, c, c + 1);
    };
    const auto h = 10. + 30. * coordinate(0), w = 10. + 30. * coordinate(1),
               y = coordinate(2) * (double(scenes.images.shape()[1]) - h),
               x = coordinate(3) * (double(scenes.images.shape()[2]) - w);
    const auto all = nn::concat({candidate, truth, nn::concat({y, x, y + h, x + w}, 1)}, 0);
    const auto N = all.shape()[0];
    nn::Tensor best, assigned;
    if (truth.shape()[0]) {
      const auto overlap = nn::box_iou(all, truth);
      best = nn::max(overlap, {1});
      assigned = nn::argmax(overlap, 1);
    } else {
      best = nn::zeros({N}, type);
      assigned = nn::zeros({N}, nn::DType::Int64);
    }
    auto fg = nn::shuffle(two_stage_detection_detail::Indices(nn::greater_equal(best, .5)),
                          rng),
         near = nn::shuffle(two_stage_detection_detail::Indices(nn::logical_and(
                                nn::greater_equal(best, .1), nn::less(best, .5))),
                            rng),
         far = nn::shuffle(two_stage_detection_detail::Indices(nn::less(best, .1)), rng);
    fg = nn::slice_axis(fg, 0, 0, std::min<std::int64_t>(fg.numel(), per_image / 4));
    near = nn::slice_axis(near, 0, 0,
                          std::min<std::int64_t>(near.numel(), per_image - fg.numel()));
    far = nn::slice_axis(
        far, 0, 0, std::min<std::int64_t>(far.numel(), per_image - fg.numel() - near.numel()));
    const auto chosen = nn::concat({fg, near, far}, 0), selected = nn::gather(all, chosen);
    const auto K = chosen.numel();
    if (!K)
      continue;
    const auto positive = nn::greater_equal(nn::gather(best, chosen), .5);
    nn::Tensor label, target;
    if (truth.shape()[0]) {
      const auto index = nn::gather(assigned, chosen);
      label = nn::where(positive, nn::gather(scenes.labels[row], index),
                        nn::full({}, double(background), nn::DType::Int64));
      target =
          nn::where(positive.reshape({K, 1}),
                    nn::encode_boxes(selected, nn::gather(truth, index)), nn::zeros({}, type));
    } else {
      label = nn::full({K}, double(background), nn::DType::Int64);
      target = nn::zeros({K, 4}, type);
    }
    pixels.push_back(selected);
    rois.push_back(DetectionFeatureBoxes(selected, stride, feature_height, feature_width));
    labels.push_back(label);
    targets.push_back(target);
    masks.push_back(positive.to(type));
    indices.push_back(nn::full({K}, double(i), nn::DType::Int64));
  }
  TensorRoiBatch out;
  out.pixels = pixels.empty() ? nn::zeros({0, 4}, type) : nn::concat(pixels, 0);
  out.rois = rois.empty() ? nn::zeros({0, 4}, type) : nn::concat(rois, 0);
  out.labels = labels.empty() ? nn::zeros({0}, nn::DType::Int64) : nn::concat(labels, 0);
  out.targets = targets.empty() ? nn::zeros({0, 4}, type) : nn::concat(targets, 0);
  out.mask = masks.empty() ? nn::zeros({0}, type) : nn::concat(masks, 0);
  out.indices = indices.empty() ? nn::zeros({0}, nn::DType::Int64) : nn::concat(indices, 0);
  return out;
}
/// @brief Own normalized and pixel RoIs with feature-minibatch indices.
///
/// Tensor handles own their storage and preserve box-coordinate graphs until
/// core crop_and_resize intentionally treats normalized boxes as constants. No
/// learned parameters or optimization state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto rois = n::models::PackDetectionRois({n::tensor({8., 12., 40., 36.}, {1, 4})});
/// @endcode
/// @par References
/// - Girshick (2015), Fast R-CNN, https://arxiv.org/abs/1504.08083
/// @see PackDetectionRois(), DetectionRoiHead()
struct DetectionRoiInputs {
  nn::Tensor rois;    ///< Floating [K,4] normalized corner-aligned feature coordinates.
  nn::Tensor pixels;  ///< Floating [K,4] original unmodified pixel-edge boxes.
  nn::Tensor indices; ///< Int64 [K] feature minibatch row indices.
};
/// @brief Pack per-image pixel proposals into shared-head tensor inputs.
///
/// Order is image position then original candidate order. Zero-area boxes are
/// allowed for pooling. No numerical model computation, mutation, fitting or
/// random state; work O(sum(Ki)). Coordinate transformations use tensor ops.
///
/// @param candidates Nonempty vector of floating [Ki,4] finite ordered boxes, one per feature minibatch row, all same dtype.
/// @param stride Positive integer feature stride, default4.
/// @param feature_height Feature grid height at least2, default16.
/// @param feature_width Feature grid width at least2, default16.
/// @return Concatenated normalized/pixel boxes and repeated minibatch indices; all-empty boxes yield K=0.
/// @throws std::invalid_argument For empty image list, invalid box shape/dtype/values or feature geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto rois = n::models::PackDetectionRois({n::tensor({8., 12., 40., 36.}, {1, 4})});
/// @endcode
/// @par References
/// - Girshick (2015), Fast R-CNN, https://arxiv.org/abs/1504.08083
/// @see DetectionFeatureBoxes(), DecodeDetectionRoiOutputs()
inline DetectionRoiInputs PackDetectionRois(const std::vector<nn::Tensor> &candidates,
                                            int stride = 4, int feature_height = 16,
                                            int feature_width = 16) {
  if (candidates.empty() || stride < 1 || feature_height < 2 || feature_width < 2)
    throw std::invalid_argument("invalid proposal packing configuration");
  const auto type = candidates[0].dtype();
  std::vector<nn::Tensor> pixels, indices;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto &box = candidates[i];
    if (box.dim() != 2 || box.shape()[1] != 4 || box.dtype() != type)
      throw std::invalid_argument("inconsistent proposal shapes/dtypes");
    SquareDetectionBoxes(box, 1.);
    pixels.push_back(box);
    indices.push_back(nn::full({box.shape()[0]}, double(i), nn::DType::Int64));
  }
  DetectionRoiInputs out;
  out.pixels = nn::concat(pixels, 0);
  out.rois = DetectionFeatureBoxes(out.pixels, stride, feature_height, feature_width);
  out.indices = nn::concat(indices, 0);
  return out;
}
/// @brief Decode every foreground class and suppress duplicate tensor detections.
///
/// Background is final class C-1 and is excluded. Candidate order is region
/// then foreground class; input order breaks tied scores. Optional raw mode
/// retains this order and skips NMS/caps. Decoded log scales
/// clip above4; boxes are not clipped to image edges, matching the source.
/// No fitting or graph is retained. Worst-case NMS is quadratic per image/class.
/// All decoding, broadcasting and score selection are tensor operations.
///
/// @param head Consistent probabilities[K,C] and deltas[K,C-1,4], floating with C at least2; logits are not used.
/// @param pixels Floating [K,4] original proposal boxes, same dtype as probabilities.
/// @param images Int64 [K] original dataset image IDs, not feature-minibatch positions.
/// @param regression Decode learned class-specific offsets when true (default); false reports unchanged proposals.
/// @param iou Finite class-wise NMS overlap threshold in[0,1], default.3.
/// @param score_threshold Finite strict class-score cutoff, default.05.
/// @param max_per_image Nonnegative per-image detection budget, default100; ignored when suppress=false.
/// @param suppress True(default) applies NMS/caps;false returns eligible candidates in region/class order.
/// @return Detached boxes/scores/labels/image IDs after per-image/class suppression, preserving all eligible foreground classes.
/// @throws std::invalid_argument For inconsistent input shapes/dtypes, malformed candidate values, settings or IDs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto head = n::models::RcnnPatchHead(p, n::ones({1, 28, 28, 1}));
/// const auto boxes = n::tensor({0., 0., 28., 28.}, {1, 4});
/// const auto d = n::models::DecodeDetectionRoiOutputs(
///     head, boxes, n::from_vector(std::vector<std::int64_t>{0}));
/// @endcode
/// @par References
/// - Girshick (2015), Fast R-CNN, https://arxiv.org/abs/1504.08083
/// @see SuppressDetectionCandidates(), DetectionRoiHead()
inline TensorDetections
DecodeDetectionRoiOutputs(const DetectionRoiOutputs &head, const nn::Tensor &pixels,
                          const nn::Tensor &images, bool regression = true, double iou = .3,
                          double score_threshold = .05, int max_per_image = 100,
                          bool suppress = true) {
  if (head.probabilities.dim() != 2 || head.probabilities.shape()[1] < 2)
    throw std::invalid_argument("invalid RoI probabilities");
  const auto K = head.probabilities.shape()[0], C = head.probabilities.shape()[1] - 1;
  const auto type = head.probabilities.dtype();
  if ((type != nn::DType::Float32 && type != nn::DType::Float64) ||
      head.deltas.shape() != nn::Shape{K, C, 4} || head.deltas.dtype() != type ||
      pixels.shape() != nn::Shape{K, 4} || pixels.dtype() != type ||
      images.shape() != nn::Shape{K} || images.dtype() != nn::DType::Int64)
    throw std::invalid_argument("inconsistent RoI decode shapes/dtypes");
  nn::NoGradGuard guard;
  const auto scores = nn::slice_axis(head.probabilities, 1, 0, C).reshape({K * C});
  const auto boxes = (regression ? nn::decode_boxes(pixels.reshape({K, 1, 4}), head.deltas)
                                 : pixels.reshape({K, 1, 4}) * nn::ones({K, C, 1}, type))
                         .reshape({K * C, 4});
  const auto labels = (nn::arange(0, double(C), 1., nn::DType::Int64).reshape({1, C}) *
                       nn::ones({K, C}, nn::DType::Int64))
                          .reshape({K * C});
  const auto ids =
      (images.reshape({K, 1}) * nn::ones({K, C}, nn::DType::Int64)).reshape({K * C});
  if (!suppress) {
    if (!std::isfinite(iou) || iou < 0 || iou > 1 || !std::isfinite(score_threshold) ||
        max_per_image < 0 ||
        !nn::all(nn::less(nn::abs(boxes), std::numeric_limits<double>::infinity()))
             .item<bool>() ||
        !nn::all(nn::less(nn::abs(scores), std::numeric_limits<double>::infinity()))
             .item<bool>() ||
        !nn::all(nn::greater_equal(ids, 0.)).item<bool>() ||
        nn::any(nn::less(nn::slice_axis(boxes, 1, 2, 4), nn::slice_axis(boxes, 1, 0, 2)))
            .item<bool>())
      throw std::invalid_argument("invalid raw RoI candidates/settings");
    const auto selected =
        two_stage_detection_detail::Indices(nn::greater(scores, score_threshold));
    return {nn::gather(boxes, selected), nn::gather(scores, selected),
            nn::gather(labels, selected), nn::gather(ids, selected)};
  }
  return SuppressDetectionCandidates(boxes, scores, labels, ids, iou, score_threshold,
                                     max_per_image);
}
/// @brief Own detached objectness and box targets for a selected scene minibatch.
///
/// Every entry is a floating tensor in the anchor dtype. Labels are binary,
/// weights and positives are0/1 masks. No learned parameters or optimizer state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 1;
/// cfg.max_objects = 1;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// n::Generator rng(7);
/// const auto targets = n::models::MakeDetectionRpnTargets(
///     n::anchor_grid(16, 16, 4, {12, 20, 30}, {.5, 1, 2}), scenes, {0}, rng);
/// @endcode
/// @par References
/// - Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, https://arxiv.org/abs/1506.01497
/// @see MakeDetectionRpnTargets(), RpnLoss()
struct DetectionRpnTargets {
  nn::Tensor labels;    ///< [N,A] one for sampled positive anchors, zero elsewhere.
  nn::Tensor weights;   ///< [N,A] sampled classification mask for positives and negatives.
  nn::Tensor targets;   ///< [N,A,4] encoded positive box targets, zero elsewhere.
  nn::Tensor positives; ///< [N,A] sampled positive-anchor regression mask.
};
/// @brief Match and sample positive/negative anchors for the RPN multitask objective.
///
/// Uses tensor argmax matching with positive cutoff.7, negative cutoff.3 and
/// forced best-anchor claims for every overlapping target. Positives cap at
/// floor(per_image/2); negatives fill the remaining budget. Ignored anchors
/// receive no classification or regression weight. Empty truth gives only
/// negative samples; empty rows return correctly shaped zero tensors. Tensor
/// RNG sequence differs from legacy std shuffles. No fitting or input mutation.
/// Work includes O(N*A*G) overlap/matching plus index shuffles.
///
/// @param anchors Float32/64 [A,4] finite positive-area boxes, A positive, same dtype as scene images.
/// @param scenes Valid floating tensor scenes with positive-area annotated boxes.
/// @param rows Selected source-scene IDs in RPN feature-minibatch order.
/// @param rng Mutable tensor generator for positive/negative index shuffles.
/// @param per_image Positive maximum classification anchors per image, default128.
/// @return Detached labels/weights/targets/positive masks with leading axes [rows.size(),A].
/// @throws std::invalid_argument For invalid anchor/scene shape/dtype/extent, selected rows or sampling count.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 1;
/// cfg.max_objects = 1;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// n::Generator rng(7);
/// const auto targets = n::models::MakeDetectionRpnTargets(
///     n::anchor_grid(16, 16, 4, {12, 20, 30}, {.5, 1, 2}), scenes, {0}, rng);
/// @endcode
/// @par References
/// - Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, https://arxiv.org/abs/1506.01497
/// @see nn::match_boxes(), RpnLoss()
inline DetectionRpnTargets MakeDetectionRpnTargets(const nn::Tensor &anchors,
                                                   const TensorDetectionScenes &scenes,
                                                   const std::vector<std::int64_t> &rows,
                                                   nn::Generator &rng, int per_image = 128) {
  two_stage_detection_detail::ValidateScenes(scenes, rows, std::numeric_limits<int>::max());
  if (anchors.dim() != 2 || anchors.shape()[0] < 1 || anchors.shape()[1] != 4 ||
      anchors.dtype() != scenes.images.dtype() || per_image < 1)
    throw std::invalid_argument("invalid RPN anchor targets configuration");
  SquareDetectionBoxes(anchors, 1.);
  if (nn::any(nn::less_equal(
                  nn::slice_axis(anchors, 1, 2, 4) - nn::slice_axis(anchors, 1, 0, 2), 0.))
          .item<bool>())
    throw std::invalid_argument("RPN anchors require positive extents");
  nn::NoGradGuard guard;
  const auto A = anchors.shape()[0];
  const auto type = anchors.dtype();
  std::vector<nn::Tensor> labels, weights, targets, positives;
  for (auto row : rows) {
    const auto &truth = scenes.boxes[row];
    const auto match = nn::match_boxes(nn::box_iou(anchors, truth), .7, .3, true);
    auto pos = nn::shuffle(two_stage_detection_detail::Indices(nn::greater_equal(match, 0.)),
                           rng),
         neg = nn::shuffle(two_stage_detection_detail::Indices(nn::equal(match, -1.)), rng);
    pos = nn::slice_axis(pos, 0, 0, std::min<std::int64_t>(pos.numel(), per_image / 2));
    neg = nn::slice_axis(neg, 0, 0,
                         std::min<std::int64_t>(neg.numel(), per_image - pos.numel()));
    const auto positive = nn::sum(nn::one_hot(pos, A, 1, 0, type), {0}),
               negative = nn::sum(nn::one_hot(neg, A, 1, 0, type), {0});
    nn::Tensor target = nn::zeros({A, 4}, type);
    if (truth.shape()[0]) {
      const auto safe = nn::maximum(match, nn::zeros({}, nn::DType::Int64));
      target =
          nn::where(positive.reshape({A, 1}),
                    nn::encode_boxes(anchors, nn::gather(truth, safe)), nn::zeros({}, type));
    }
    labels.push_back(positive);
    positives.push_back(positive);
    weights.push_back(positive + negative);
    targets.push_back(target);
  }
  const auto N = std::int64_t(rows.size());
  return {labels.empty() ? nn::zeros({N, A}, type) : nn::stack(labels, 0),
          weights.empty() ? nn::zeros({N, A}, type) : nn::stack(weights, 0),
          targets.empty() ? nn::zeros({N, A, 4}, type) : nn::stack(targets, 0),
          positives.empty() ? nn::zeros({N, A}, type) : nn::stack(positives, 0)};
}
/// @brief Decode, clip, size-filter and suppress one image's scored RPN anchors.
///
/// Top scores sort descending with input-index tie breaking, an explicit stable
/// choice where the source partial_sort did not promise ties. Decoding clamps
/// log scales above4. Boxes clip to[0,height]x[0,width], then any side below4
/// is dropped. NMS uses strict overlap greater than.7; scores have no lower
/// cutoff. Empty anchors or zero budgets give empty output. Proposal selection
/// is detached: approximate joint training does not differentiate through it.
/// All coordinate/score arithmetic runs through tensors; no fitting.
///
/// @param scores Float32/64 [A] finite objectness scores.
/// @param deltas Floating [A,4] finite box offsets, matching scores dtype.
/// @param anchors Floating [A,4] finite ordered boxes in the same dtype.
/// @param height Positive image height for clipping.
/// @param width Positive image width for clipping.
/// @param before Nonnegative pre-NMS top-score budget, default300; capped at A.
/// @param after Nonnegative final proposal budget, default100.
/// @return Detached [P,4] proposal boxes after clipping, minimum side4 and IoU.7 NMS.
/// @throws std::invalid_argument For malformed shapes/dtypes, nonfinite values, reversed boxes, dimensions or budgets.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto anchors = n::anchor_grid(16, 16, 4, {12, 20, 30}, {.5, 1, 2});
/// const auto proposals =
///     n::models::DetectionRpnProposals(n::ones({2304}), n::zeros({2304, 4}), anchors, 64, 64);
/// @endcode
/// @par References
/// - Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, https://arxiv.org/abs/1506.01497
/// @see ComputeDetectionProposals(), BuildRpn()
inline nn::Tensor DetectionRpnProposals(const nn::Tensor &scores, const nn::Tensor &deltas,
                                        const nn::Tensor &anchors, int height, int width,
                                        int before = 300, int after = 100) {
  if (scores.dim() != 1 ||
      (scores.dtype() != nn::DType::Float32 && scores.dtype() != nn::DType::Float64) ||
      deltas.shape() != nn::Shape{scores.numel(), 4} || anchors.shape() != deltas.shape() ||
      deltas.dtype() != scores.dtype() || anchors.dtype() != scores.dtype() || height < 1 ||
      width < 1 || before < 0 || after < 0)
    throw std::invalid_argument("invalid RPN proposal inputs/settings");
  for (const auto &x : {scores, deltas, anchors})
    if (!nn::all(nn::less(nn::abs(x), std::numeric_limits<double>::infinity())).item<bool>())
      throw std::invalid_argument("RPN proposals require finite values");
  SquareDetectionBoxes(anchors, 1.);
  nn::NoGradGuard guard;
  const auto count = std::min<std::int64_t>(before, scores.numel());
  if (!count || !after)
    return nn::zeros({0, 4}, scores.dtype());
  const auto order = nn::slice_axis(nn::argsort(scores, true), 0, 0, count);
  const auto boxes = nn::clip_boxes(
      nn::decode_boxes(nn::gather(anchors, order), nn::gather(deltas, order)), height, width);
  const auto extents = nn::slice_axis(boxes, 1, 2, 4) - nn::slice_axis(boxes, 1, 0, 2);
  const auto valid =
      two_stage_detection_detail::Indices(nn::all(nn::greater_equal(extents, 4.), {1}));
  const auto filtered = nn::gather(boxes, valid),
             ranked = nn::gather(nn::gather(scores, order), valid);
  const auto kept = nn::non_max_suppression(filtered, ranked, after, .7).indices;
  return nn::gather(filtered, kept);
}
/// @brief Build detached proposal boxes for every image of one shared RPN forward pass.
///
/// Delegates each image to DetectionRpnProposals, using the same unmodified
/// RPN predictions that can also supply the differentiable training loss.
/// No extra backbone pass, optimizer or parameter mutation occurs.
///
/// @param rpn Consistent objectness[N,A] and deltas[N,A,4] tensors; logits are not used.
/// @param anchors Floating [A,4] anchors matching the RPN dtype and anchor order.
/// @param height Positive source-image height.
/// @param width Positive source-image width.
/// @param before Nonnegative top-score budget per image, default300.
/// @param after Nonnegative final proposal cap per image, default100.
/// @return N detached [Pi,4] tensors in RPN batch order.
/// @throws std::invalid_argument For inconsistent RPN shapes or invalid proposal arguments.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto r = n::models::BuildRpn(p, {{n::ones({1, 16, 16, 64}), 64, 16, 9}});
/// const auto boxes = n::models::ComputeDetectionProposals(
///     r, n::anchor_grid(16, 16, 4, {12, 20, 30}, {.5, 1, 2}), 64, 64);
/// @endcode
/// @par References
/// - Ren et al. (2015), Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, https://arxiv.org/abs/1506.01497
/// @see DetectionRpnProposals(), MakeDetectionRpnTargets()
inline std::vector<nn::Tensor> ComputeDetectionProposals(const Rpn &rpn,
                                                         const nn::Tensor &anchors, int height,
                                                         int width, int before = 300,
                                                         int after = 100) {
  if (rpn.objectness.dim() != 2 || rpn.objectness.shape()[0] < 1 ||
      rpn.deltas.shape() != nn::Shape{rpn.objectness.shape()[0], rpn.objectness.shape()[1], 4})
    throw std::invalid_argument("invalid batched RPN proposal shapes");
  std::vector<nn::Tensor> out;
  for (std::int64_t i = 0; i < rpn.objectness.shape()[0]; ++i)
    out.push_back(DetectionRpnProposals(
        nn::slice_axis(rpn.objectness, 0, i, i + 1).reshape({rpn.objectness.shape()[1]}),
        nn::slice_axis(rpn.deltas, 0, i, i + 1).reshape({rpn.objectness.shape()[1], 4}),
        anchors, height, width, before, after));
  return out;
}
} // namespace datamunge::nn::models
