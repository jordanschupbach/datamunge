#pragma once
/// @file feature_pyramid_models.hpp
/// @brief Tensor top-down pyramids, source multiscale RoI routing and compact FPN detector forwards.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/two_stage_detection_models.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Retain unsmoothed top-down maps and separately smoothed pyramid outputs.
///
/// Tensor handles own storage and graphs; learned leaves belong to the store.
/// Top-down recursion uses inner maps, never the smoothed outputs. No fitting.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto f = n::models::FeaturePyramid(
///     p, "fpn", {n::ones({1, 7, 9, 4}), n::ones({1, 4, 5, 8}), n::ones({1, 2, 3, 16})}, 8);
/// @endcode
/// @par References
/// - Lin et al. (2017), Feature Pyramid Networks for Object Detection, https://arxiv.org/abs/1612.03144; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py
/// @see FeaturePyramid()
struct FeaturePyramidOutputs {
  std::vector<nn::Tensor>
      inner; ///< One projected/top-down map per input, increasing stride order.
  std::vector<nn::Tensor>
      levels; ///< Independent 3x3-smoothed maps, plus optional final stride-two1x1 pool.
};
/// @brief Fuse lateral projections with unsmoothed nearest-neighbor top-down maps.
///
/// Every lateral is a biased linear1x1 He projection. Resize the next inner
/// map to the lateral's actual H/W with nearest neighbor (no corner alignment),
/// add, then apply an independent biased linear3x3 He convolution. No activation
/// or batch normalization: negative values remain negative. Odd rectangular
/// grids are valid. Optional coarser output samples pixels0,2,... of the last
/// smoothed map. No fitting, normalization or moving state. Cost is lateral1x1
/// plus smoothing3x3 convolution at each level; retain leaves until backward.
///
/// @param p Mutable Float32/64 learned store; lazily creates named lateral/smoothing weights and biases.
/// @param name Parameter prefix; level i uses /level<i+2>/{lateral,smooth}/{W,b}.
/// @param inputs Nonempty NHWC feature list in increasing stride order, same positive batch/dtype; H/W nonincreasing and channels positive.
/// @param width Positive output channel count, default256.
/// @param extra_level Append a1x1 stride-two VALID max pool of the last smoothed map when true (default).
/// @return Unsmoothed inner maps and smoothed levels preserving each input H/W, plus an optional coarser output.
/// @throws std::invalid_argument For invalid feature geometry/dtype/width or incompatible store leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto f = n::models::FeaturePyramid(
///     p, "fpn", {n::ones({1, 7, 9, 4}), n::ones({1, 4, 5, 8}), n::ones({1, 2, 3, 16})}, 8);
/// @endcode
/// @par References
/// - Lin et al. (2017), Feature Pyramid Networks for Object Detection, https://arxiv.org/abs/1612.03144; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py
/// @see PyramidDetectionRoiPool()
inline FeaturePyramidOutputs FeaturePyramid(nn::ParameterStore &p, const std::string &name,
                                            const std::vector<nn::Tensor> &inputs,
                                            std::int64_t width = 256,
                                            bool extra_level = true) {
  if (inputs.empty() || width < 1 ||
      (p.dtype() != nn::DType::Float32 && p.dtype() != nn::DType::Float64))
    throw std::invalid_argument("FPN requires floating nonempty inputs and positive width");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto &x = inputs[i];
    if (x.dim() != 4 || x.shape()[0] < 1 || x.shape()[1] < 1 || x.shape()[2] < 1 ||
        x.shape()[3] < 1 || x.dtype() != p.dtype() || x.shape()[0] != inputs[0].shape()[0] ||
        (i &&
         (x.shape()[1] > inputs[i - 1].shape()[1] || x.shape()[2] > inputs[i - 1].shape()[2])))
      throw std::invalid_argument("invalid FPN input ordering/shapes/dtypes");
  }
  FeaturePyramidOutputs out;
  out.inner.resize(inputs.size());
  out.levels.resize(inputs.size());
  nn::ConvLayerOptions options;
  options.init = nn::init::he_uniform();
  for (std::size_t reverse = inputs.size(); reverse > 0; --reverse) {
    const auto i = reverse - 1;
    const auto prefix = name + "/level" + std::to_string(i + 2);
    auto h = nn::conv2d(p, prefix + "/lateral", inputs[i], width, 1, options);
    if (i + 1 < inputs.size())
      h = h + nn::resize_nearest(out.inner[i + 1], {h.shape()[1], h.shape()[2]});
    out.inner[i] = h;
    out.levels[i] = nn::conv2d(p, prefix + "/smooth", h, width, 3, options);
  }
  if (extra_level)
    out.levels.push_back(nn::max_pool2d(out.levels.back(), 1, 2, nn::Padding::Valid));
  return out;
}
/// @brief Describe one feature level of a P2–P5 RoI pyramid.
///
/// An owning tensor handle plus integer feature-stride metadata. Four levels
/// correspond to P2/P3/P4/P5 in supplied order; no learned state or fitting.
///
/// @par Example
/// @code
/// dm::nn::models::DetectionPyramidLevel level{dm::nn::ones({1, 16, 16, 64}), 4};
/// @endcode
/// @par References
/// - Lin et al. (2017), Feature Pyramid Networks for Object Detection, https://arxiv.org/abs/1612.03144; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py
/// @see PyramidDetectionRoiPool()
struct DetectionPyramidLevel {
  nn::Tensor
      features; ///< Floating [N,H,W,C] maps; H/W at least2, shared N/C/dtype across levels.
  int stride;   ///< Positive source-image pixel stride; levels must have increasing strides.
};
/// @brief Assign each pixel box to a clamped P2–P5 level using its geometric-mean side.
///
/// Routing arithmetic follows boxes dtype; the source Float32 path is preserved
/// for Float32 boxes. Near level boundaries Float64 can differ. Assignment is
/// discrete and has no gradients or fitting. O(number of boxes) work.
///
/// @param boxes Float32/64 [...,4] finite ordered pixel boxes; zero area is allowed.
/// @param canonical Finite positive reference side at level4, default224; use32 for64-pixel scenes.
/// @return Detached Int64 [...] levels clamp(floor(4+log2(sqrt(max(area,1e-6))/canonical)),2,5).
/// @throws std::invalid_argument For malformed/nonfinite/reversed boxes or invalid canonical side.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto levels = n::models::PyramidDetectionLevels(
///     n::tensor({0., 0., 112., 112., 0., 0., 224., 224.}, {2, 4}), 224.); // [3,4]
/// @endcode
/// @par References
/// - Lin et al. (2017), Feature Pyramid Networks for Object Detection, https://arxiv.org/abs/1612.03144; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py
/// @see PyramidDetectionRoiPool()
inline nn::Tensor PyramidDetectionLevels(const nn::Tensor &boxes, double canonical = 224.) {
  if (!std::isfinite(canonical) || canonical <= 0)
    throw std::invalid_argument("invalid canonical RoI side");
  SquareDetectionBoxes(boxes, 1.);
  nn::NoGradGuard guard;
  const auto type = boxes.dtype();
  const auto side = nn::sqrt(nn::maximum(nn::box_area(boxes), nn::full({}, 1e-6, type)));
  const auto level = nn::floor(4. + nn::log(side / canonical) / std::log(2.));
  return nn::minimum(nn::maximum(level, nn::full({}, 2., type)), nn::full({}, 5., type))
      .to(nn::DType::Int64);
}
/// @brief Pool each pixel box only on its assigned pyramid level and restore input order.
///
/// Selects levels by PyramidDetectionLevels. Each selected group maps pixel
/// boxes to its stride grid, samples14x14 using corner-aligned crop_and_resize,
/// then max-pools2x2 to7x7. Tensor gather restores original RoI order. This is
/// the source bilinear-pooling adaptation, not ROIAlign. Gradients reach only
/// the assigned feature maps, never box coordinates or discrete assignments.
/// No fitting or mutation; sampling work O(K*14²*C), independent of four-level
/// masking of unused crops.
///
/// @param levels Exactly four floating feature/stride levels in P2–P5 order, increasing positive strides, same N/C/dtype; each H/W at least2.
/// @param boxes Float32/64 [K,4] finite ordered pixel boxes; normalized sampling can extend outside features.
/// @param indices Int32/64 [K] valid feature minibatch row indices.
/// @param canonical Finite positive side assigned to level4, default224.
/// @return Feature-dtype [K,7,7,C] pooled maps in original box order; empty boxes give zero-length output.
/// @throws std::invalid_argument For invalid level/box/index geometry, dtypes, values or canonical side.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::vector<n::models::DetectionPyramidLevel> levels;
/// for (int k = 2; k <= 5; ++k)
///   levels.push_back({n::full({1, 2048 / (1 << k), 2048 / (1 << k), 1}, double(k)), 1 << k});
/// const auto y = n::models::PyramidDetectionRoiPool(
///     levels, n::tensor({64., 64., 288., 288.}, {1, 4}),
///     n::from_vector(std::vector<std::int64_t>{0}), 224.); // all4
/// @endcode
/// @par References
/// - Lin et al. (2017), Feature Pyramid Networks for Object Detection, https://arxiv.org/abs/1612.03144; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py
/// @see PooledDetectionRoiHead(), DetectionFeatureBoxes()
inline nn::Tensor PyramidDetectionRoiPool(const std::vector<DetectionPyramidLevel> &levels,
                                          const nn::Tensor &boxes, const nn::Tensor &indices,
                                          double canonical = 224.) {
  if (levels.size() != 4 || boxes.dim() != 2 || boxes.shape()[1] != 4 ||
      indices.shape() != nn::Shape{boxes.shape()[0]} ||
      (indices.dtype() != nn::DType::Int32 && indices.dtype() != nn::DType::Int64))
    throw std::invalid_argument("invalid pyramid RoI inputs");
  for (std::size_t i = 0; i < levels.size(); ++i) {
    const auto &x = levels[i].features;
    if (x.dim() != 4 || x.shape()[0] < 1 || x.shape()[1] < 2 || x.shape()[2] < 2 ||
        x.shape()[3] < 1 ||
        (x.dtype() != nn::DType::Float32 && x.dtype() != nn::DType::Float64) ||
        x.dtype() != levels[0].features.dtype() ||
        x.shape()[0] != levels[0].features.shape()[0] ||
        x.shape()[3] != levels[0].features.shape()[3] || levels[i].stride < 1 ||
        (i && levels[i].stride <= levels[i - 1].stride))
      throw std::invalid_argument("invalid pyramid RoI levels");
  }
  if (!nn::all(nn::logical_and(nn::greater_equal(indices, 0.),
                               nn::less(indices, double(levels[0].features.shape()[0]))))
           .item<bool>())
    throw std::invalid_argument("pyramid RoI index outside batch");
  const auto assigned = PyramidDetectionLevels(boxes, canonical);
  std::vector<nn::Tensor> parts, order;
  for (int i = 0; i < 4; ++i) {
    const auto selected =
        two_stage_detection_detail::Indices(nn::equal(assigned, double(i + 2)));
    if (!selected.numel())
      continue;
    const auto &f = levels[i].features;
    const auto rois = DetectionFeatureBoxes(nn::gather(boxes, selected), levels[i].stride,
                                            int(f.shape()[1]), int(f.shape()[2]));
    parts.push_back(
        nn::max_pool2d(nn::crop_and_resize(f, rois, nn::gather(indices, selected), {14, 14}),
                       2, 2, nn::Padding::Valid));
    order.push_back(selected);
  }
  if (parts.empty())
    return nn::zeros({0, 7, 7, levels[0].features.shape()[3]}, levels[0].features.dtype());
  return nn::gather(nn::concat(parts, 0), nn::argsort(nn::concat(order, 0), false));
}
/// @brief Own source compact FPN detector maps and one shared multi-level RPN result.
///
/// Tensor handles retain backbone/pyramid/RPN graphs; leaves remain owned by
/// the learned store. P2..P5 feed region pooling; P2..P6 feed one shared RPN.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto f = n::models::FpnDigitDetectorForward(p, n::zeros({1, 64, 64, 1}));
/// @endcode
/// @par References
/// - Lin et al. (2017), Feature Pyramid Networks for Object Detection, https://arxiv.org/abs/1612.03144; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py
/// @see FpnDigitDetectorForward(), FpnDigitAnchors()
struct FpnDigitDetectorFeatures {
  FeaturePyramidOutputs pyramid; ///< Four inner maps and five64-channel output maps, P2..P6.
  std::vector<DetectionPyramidLevel> roi_levels; ///< P2..P5 maps with strides4/8/16/32.
  Rpn rpn; ///< Shared128-hidden RPN, three anchors/cell,1023 total anchors.
};
/// @brief Build the source fully learned compact64-pixel FPN detector backbone and RPN.
///
/// The stride-four source backbone is extended by3x3 stride-two ReLU layers
/// C3/C4/C5 of widths128/256/256. A64-channel linear FPN adds pooledP6; one
/// 128-channel RPN head is shared across all five levels with three ratios
/// per cell. This is a source teaching architecture, not a pretrained ResNet.
/// It has1292911 learned scalars before the shared256-unit box head,2174882
/// including that head. No normalization, dropout, moving state or fitting.
/// Keep learned leaves unchanged until backward.
///
/// @param p Mutable Float32/64 learned store, lazy backbone/fpn/rpn leaves.
/// @param images Floating [N,64,64,1], N positive, same dtype as p; caller provides normalization.
/// @return P2..P6 maps, four region-pooling levels and the1023-anchor shared RPN graph.
/// @throws std::invalid_argument For invalid image shape/dtype or incompatible learned leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto f = n::models::FpnDigitDetectorForward(p, n::zeros({1, 64, 64, 1}));
/// @endcode
/// @par References
/// - Lin et al. (2017), Feature Pyramid Networks for Object Detection, https://arxiv.org/abs/1612.03144; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py
/// @see FpnDigitAnchors(), PyramidDetectionRoiPool(), PooledDetectionRoiHead()
inline FpnDigitDetectorFeatures FpnDigitDetectorForward(nn::ParameterStore &p,
                                                        const nn::Tensor &images) {
  if (images.dim() != 4 || images.shape()[0] < 1 || images.shape()[1] != 64 ||
      images.shape()[2] != 64 || images.shape()[3] != 1)
    throw std::invalid_argument("FPN digit detector requires [N,64,64,1]");
  std::vector<nn::Tensor> inputs;
  auto h = StrideFourDetectionBackbone(p, images);
  inputs.push_back(h);
  nn::ConvLayerOptions conv;
  conv.init = nn::init::he_uniform();
  conv.activation = nn::Activation::Relu;
  conv.stride = 2;
  for (int stage = 3; stage <= 5; ++stage) {
    h = nn::conv2d(p, "backbone/c" + std::to_string(stage), h, stage == 3 ? 128 : 256, 3,
                   conv);
    inputs.push_back(h);
  }
  FpnDigitDetectorFeatures out;
  out.pyramid = FeaturePyramid(p, "fpn", inputs, 64, true);
  std::vector<RpnLevel> rpn_levels;
  for (int k = 2; k <= 6; ++k) {
    const int stride = 1 << k, cells = 64 / stride;
    rpn_levels.push_back({out.pyramid.levels[k - 2], 64, cells, 3});
    if (k <= 5)
      out.roi_levels.push_back({out.pyramid.levels[k - 2], stride});
  }
  out.rpn = BuildRpn(p, rpn_levels);
  return out;
}
/// @brief Return the1023 source64-pixel FPN anchors in concatenated P2–P6 order.
///
/// Pyramid strides4/8/16/32/64 use one side8/16/32/64/128 respectively
/// and ratios height/width .5/1/2. Each grid orders row, column, ratio.
/// Anchors can extend beyond the64-pixel canvas; proposals clip them later.
/// No random state, fitting or mutable cache.
///
/// @return Fresh detached Float32 [1023,4] pixel boxes; convert to model dtype if necessary.
/// @par Example
/// @code
/// const auto anchors = dm::nn::models::FpnDigitAnchors(); // [1023,4]
/// @endcode
/// @par References
/// - Lin et al. (2017), Feature Pyramid Networks for Object Detection, https://arxiv.org/abs/1612.03144; TorchVision reference, https://github.com/pytorch/vision/blob/main/torchvision/ops/feature_pyramid_network.py
/// @see FpnDigitDetectorForward(), DetectionRpnProposals()
inline nn::Tensor FpnDigitAnchors() {
  nn::NoGradGuard guard;
  std::vector<nn::Tensor> levels;
  for (int k = 2; k <= 6; ++k)
    levels.push_back(
        nn::anchor_grid(64 / (1 << k), 64 / (1 << k), 1 << k, {double(2 << k)}, {.5, 1., 2.}));
  return nn::concat(levels, 0);
}
} // namespace datamunge::nn::models
