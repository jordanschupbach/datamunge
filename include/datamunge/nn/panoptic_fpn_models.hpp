#pragma once
/// @file panoptic_fpn_models.hpp
/// @brief Shared panoptic ResNet/FPN,group-normalized semantic paths and level-routed aligned instance features.
#include "datamunge/nn/feature_pyramid_models.hpp"
#include "datamunge/nn/mask_rcnn_models.hpp"
#include "datamunge/nn/resnet_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Configure full panoptic ResNet50/FPN/GN semantic features or a same-branch compact graph.
///
/// No optimizer,pretrained weights or hidden fit. Input side is a positive
/// multiple of32,at least64;semantic group count divides semantic_width.
///
/// @throws std::invalid_argument For invalid configuration,tensor geometry/dtype,indices or stored parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::PanopticConfig cfg;
/// cfg.size = 64;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.semantic_width = 16;
/// cfg.groups = 8;
/// cfg.semantic_classes = 3;
/// cfg.grayscale = true;
/// const auto net = n::models::PanopticBackbone(p, state, n::zeros({2, 64, 64, 1}), cfg);
/// @endcode
/// @par References
/// - Kirillov,Girshick,He and Dollár(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446; aligned2x2 region sampling follows the source Mask R-CNN teaching convention.
/// @see PanopticBackbone()
struct PanopticConfig {
  int size = 256;            ///< Square input side,at least64 and divisible by32.
  int depth = 50;            ///< Supported canonical ResNet depth.
  int divisor = 1;           ///< Exact positive divisor of64 for backbone channel widths.
  int width = 256;           ///< Positive FPN channel width.
  int semantic_width = 128;  ///< Positive semantic path width,divisible by groups.
  int groups = 32;           ///< Positive group-normalization group count.
  int semantic_classes = 54; ///< At least two semantic channels,including all-things category.
  bool grayscale = false;    ///< True tiles one input channel to RGB before ResNet.
};
/// @brief Own five shared pyramid maps and input-aligned semantic logits.
///
/// Returned tensors retain caller-store graphs. No independent fitted state.
///
/// @throws std::invalid_argument For invalid configuration,tensor geometry/dtype,indices or stored parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::PanopticConfig cfg;
/// cfg.size = 64;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.semantic_width = 16;
/// cfg.groups = 8;
/// cfg.semantic_classes = 3;
/// cfg.grayscale = true;
/// const auto net = n::models::PanopticBackbone(p, state, n::zeros({2, 64, 64, 1}), cfg);
/// @endcode
/// @par References
/// - Kirillov,Girshick,He and Dollár(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446; aligned2x2 region sampling follows the source Mask R-CNN teaching convention.
/// @see PanopticBackbone()
struct PanopticFeatures {
  std::vector<nn::Tensor>
      pyramid;         ///< P2 through P6 high-to-low maps,shared instance/RPN features.
  nn::Tensor semantic; ///< Floating[N,size,size,semantic_classes] linear semantic logits.
};
/// @brief Build shared ResNet/FPN and summed group-normalized semantic paths.
///
/// FPN projects/sums unsmoothed nearest top-down features and independently
/// smooths each level. P2/P3/P4/P5 semantic paths use1/1/2/3 biased-free3x3,
/// GN/ReLU stages,with bilinear upsampling after each lower-grid stage. Paths
/// sum at stride4,linear1x1 class logits resize to image dimensions. No hidden
/// fit. Warm instance/RPN heads too before optimizer;checkpoint learned/BN stores
/// together. Group normalization does not depend on minibatch statistics.
///
/// @param p Same-dtype floating learned store,creates/reuses all backbone/FPN/semantic leaves.
/// @param state Same-dtype ResNet BN moving state;semantic GN has no moving state.
/// @param image Positive[N,cfg.size,cfg.size,1 or3] floating input.
/// @param cfg Valid architecture settings,default fullResNet50/FPN256/GN32 widths128.
/// @param training False(default) ResNet moving BN;true uses batch moments.
/// @param update_stats True(default) permits training-mode BN moving updates.
/// @return Five pyramid maps and image-aligned semantic logits.
/// @throws std::invalid_argument For invalid configuration,tensor geometry/dtype,indices or stored parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::PanopticConfig cfg;
/// cfg.size = 64;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.semantic_width = 16;
/// cfg.groups = 8;
/// cfg.semantic_classes = 3;
/// cfg.grayscale = true;
/// const auto net = n::models::PanopticBackbone(p, state, n::zeros({2, 64, 64, 1}), cfg);
/// @endcode
/// @par References
/// - Kirillov,Girshick,He and Dollár(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446; aligned2x2 region sampling follows the source Mask R-CNN teaching convention.
/// @see PanopticRoiAlign(), MaskRcnnHead()
inline PanopticFeatures PanopticBackbone(nn::ParameterStore &p, nn::ParameterStore &state,
                                         const nn::Tensor &image,
                                         const PanopticConfig &cfg = {}, bool training = false,
                                         bool update_stats = true) {
  if (cfg.size < 64 || cfg.size % 32 || cfg.width < 1 || cfg.semantic_width < 1 ||
      cfg.groups < 1 || cfg.semantic_width % cfg.groups || cfg.semantic_classes < 2)
    throw std::invalid_argument("invalid Panoptic FPN configuration");
  if (image.dim() != 4 || image.shape()[0] < 1 || image.shape()[1] != cfg.size ||
      image.shape()[2] != cfg.size || image.shape()[3] != (cfg.grayscale ? 1 : 3) ||
      image.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("panoptic image/store mismatch");
  ResNetConfig rc;
  rc.depth = cfg.depth;
  rc.width_divisor = cfg.divisor;
  rc.include_classifier = false;
  rc.stride_on_3x3 = true;
  rc.update_stats = update_stats;
  const auto backbone =
      ResNet(p, state, cfg.grayscale ? nn::tile(image, {1, 1, 1, 3}) : image, rc, training);
  PanopticFeatures out;
  out.pyramid = FeaturePyramid(p, "fpn",
                               {backbone.stages[0], backbone.stages[1], backbone.stages[2],
                                backbone.stages[3]},
                               cfg.width, true)
                    .levels;
  nn::ConvLayerOptions conv;
  conv.init = nn::init::he_uniform();
  conv.bias = false;
  nn::Tensor sum;
  for (int level = 0; level < 4; ++level) {
    nn::Tensor h = out.pyramid[std::size_t(level)];
    for (int stage = 0; stage < std::max(1, level); ++stage) {
      const auto name =
          "semantic/level" + std::to_string(level) + "/stage" + std::to_string(stage);
      h = nn::relu(nn::group_norm(
          p, name + "/gn", nn::conv2d(p, name, h, cfg.semantic_width, 3, conv), cfg.groups));
      if (level > 0) {
        const auto &target = out.pyramid[std::size_t(level - stage - 1)];
        h = nn::resize_bilinear(h, {target.shape()[1], target.shape()[2]});
      }
    }
    sum = level == 0 ? h : sum + h;
  }
  nn::ConvLayerOptions score;
  score.init = nn::init::he_uniform();
  out.semantic = nn::conv2d(p, "semantic/score", sum, cfg.semantic_classes, 1, score);
  out.semantic = nn::resize_bilinear(out.semantic, {image.shape()[1], image.shape()[2]});
  return out;
}

/// @brief Select P2..P5 levels from continuous pixel-box area using the FPN canonical-side formula.
///
/// Level=floor(4+log2(sqrt(max(area,1e-6))/canonical)),clamped2..5.
/// No learned state,fitting or coordinate gradient.
///
/// @param boxes Floating[K,4] finite ordered pixel boxes.
/// @param canonical Finite positive reference side,default224.
/// @return Detached Int64[K] levels2..5;Float32 arithmetic matches source level boundaries.
/// @throws std::invalid_argument For invalid configuration,tensor geometry/dtype,indices or stored parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto levels =
///     n::models::PanopticLevels(n::tensor({0., 0., 8., 8., 0., 0., 64., 64.}, {2, 4}), 32);
/// @endcode
/// @par References
/// - Kirillov,Girshick,He and Dollár(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446; aligned2x2 region sampling follows the source Mask R-CNN teaching convention.
/// @see PyramidDetectionLevels(), PanopticRoiAlign()
inline nn::Tensor PanopticLevels(const nn::Tensor &boxes, double canonical = 224) {
  return PyramidDetectionLevels(boxes.to(nn::DType::Float32), canonical);
}
/// @brief Route each region to its FPN level,apply aligned sampling there and restore input order.
///
/// Only selected image/level routes receive feature gradients. Each level
/// uses MaskRoiAlign regular2x2 samples/bin at strides4/8/16/32. Coordinates
/// and routing are detached;host loops carry level IDs only. Empty K returns
/// zero-sized output connected to every input level. No hidden fitting.
///
/// @param features At least four common-batch/channel floating P2..P5 NHWC maps.
/// @param boxes Same-dtype[K,4] positive finite pixel boxes.
/// @param indices Int64[K] valid feature-batch positions.
/// @param image_size Positive image side at least64,divisible by32.
/// @param bins Positive output bins per spatial axis.
/// @param canonical Finite positive reference side,default224;compact source uses32.
/// @return Graph-connected[K,bins,bins,C] aligned maps in original region order.
/// @throws std::invalid_argument For invalid configuration,tensor geometry/dtype,indices or stored parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::PanopticConfig cfg;
/// cfg.size = 64;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.semantic_width = 16;
/// cfg.groups = 8;
/// cfg.semantic_classes = 3;
/// cfg.grayscale = true;
/// const auto net = n::models::PanopticBackbone(p, state, n::zeros({2, 64, 64, 1}), cfg);
/// const auto aligned =
///     n::models::PanopticRoiAlign(net.pyramid, n::tensor({0., 0., 8., 8.}, {1, 4}),
///                                 n::zeros({1}, n::DType::Int64), 64, 7, 32);
/// @endcode
/// @par References
/// - Kirillov,Girshick,He and Dollár(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446; aligned2x2 region sampling follows the source Mask R-CNN teaching convention.
/// @see MaskRoiAlign(), PanopticLevels()
inline nn::Tensor PanopticRoiAlign(const std::vector<nn::Tensor> &features,
                                   const nn::Tensor &boxes, const nn::Tensor &indices,
                                   int image_size, int bins, double canonical = 224) {
  if (features.size() < 4 || image_size < 64 || image_size % 32 || bins < 1 ||
      boxes.dim() != 2 || boxes.shape()[1] != 4 || features[0].dim() != 4 ||
      boxes.dtype() != features[0].dtype() || indices.shape() != nn::Shape{boxes.shape()[0]} ||
      indices.dtype() != nn::DType::Int64)
    throw std::invalid_argument("invalid panoptic aligned feature inputs");
  for (int i = 0; i < 4; ++i)
    if (features[i].shape() != nn::Shape{features[0].shape()[0], image_size / (4 << i),
                                         image_size / (4 << i), features[0].shape()[3]} ||
        features[i].dtype() != boxes.dtype())
      throw std::invalid_argument("panoptic pyramid geometry mismatch");
  const auto levels = PanopticLevels(boxes, canonical);
  std::vector<nn::Tensor> parts, order;
  for (int i = 0; i < 4; ++i) {
    nn::Tensor selected;
    {
      nn::NoGradGuard guard;
      selected = two_stage_detection_detail::Indices(nn::equal(levels, double(i + 2)));
    }
    if (!selected.numel())
      continue;
    parts.push_back(MaskRoiAlign(features[i], nn::gather(boxes, selected),
                                 nn::gather(indices, selected), 4 << i, bins));
    order.push_back(selected);
  }
  if (parts.empty()) {
    auto out = nn::zeros({0, bins, bins, features[0].shape()[3]}, boxes.dtype());
    for (int i = 0; i < 4; ++i)
      out = out + nn::sum(features[i]) * 0.;
    return out;
  }
  return nn::gather(nn::concat(parts, 0), nn::argsort(nn::concat(order, 0), false));
}
/// @brief Build a shared three-anchor RPN over all five panoptic feature levels.
///
/// No optimizer,dropout or state. Every level predicts three ratios per cell.
/// Warm all heads before optimizer construction.
///
/// @param p Same-dtype learned store;creates/reuses rpn/* leaves.
/// @param features Valid common-N/channel square P2..P6 maps.
/// @param hidden Positive shared3x3 hidden width,default256;compact source uses32.
/// @return Graph-connected RPN logits/probabilities/deltas in pyramid/cell/anchor order.
/// @throws std::invalid_argument For invalid geometry,dtypes,settings or image IDs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::PanopticConfig c;
/// c.size = 64;
/// c.depth = 18;
/// c.divisor = 8;
/// c.width = 32;
/// c.semantic_width = 16;
/// c.groups = 8;
/// c.semantic_classes = 3;
/// c.grayscale = true;
/// const auto features = n::models::PanopticBackbone(p, state, n::zeros({1, 64, 64, 1}), c);
/// const auto rpn = n::models::BuildPanopticRpn(p, features, 32);
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446; compact64px instance-head widths from the source teaching example.
/// @see BuildRpn(), DecodeCompactPanopticInstances()
inline Rpn BuildPanopticRpn(nn::ParameterStore &p, const PanopticFeatures &features,
                            int hidden = 256) {
  if (features.pyramid.size() != 5 || hidden < 1)
    throw std::invalid_argument("five panoptic RPN levels required");
  std::vector<RpnLevel> levels;
  for (const auto &feature : features.pyramid) {
    if (feature.dim() != 4 || feature.shape()[0] < 1 || feature.shape()[1] < 1 ||
        feature.shape()[2] != feature.shape()[1] || feature.shape()[3] < 1 ||
        feature.dtype() != p.dtype() || feature.shape()[0] != features.pyramid[0].shape()[0] ||
        feature.shape()[3] != features.pyramid[0].shape()[3])
      throw std::invalid_argument("invalid panoptic RPN feature");
    levels.push_back({feature, int(feature.shape()[3]), int(feature.shape()[1]), 3});
  }
  return BuildRpn(p, levels, hidden);
}
/// @brief Build the compact panoptic1023-anchor source prior bank.
///
/// P2..P6 sides16/8/4/2/1,strides4/8/16/32/64,scales8/16/24/40/56,
/// ratios.5/1/2. No fitting,random state or input mutation.
///
/// @return Float32[1023,4] pixel boxes,level/cell/ratio order.
/// @par Example
/// @code
/// const auto anchors = dm::nn::models::CompactPanopticAnchors();
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446; priors are the source64px teaching adaptation.
/// @see BuildPanopticRpn()
inline nn::Tensor CompactPanopticAnchors() {
  const std::array<double, 5> scales = {8, 16, 24, 40, 56};
  std::vector<nn::Tensor> levels;
  for (int i = 0; i < 5; ++i)
    levels.push_back(nn::anchor_grid(16 >> i, 16 >> i, 4 << i, {scales[i]}, {.5, 1, 2}));
  return nn::concat(levels, 0);
}
/// @brief Decode aligned compact panoptic instance boxes and class-specific mask supports from shared features.
///
/// RPN proposals top300/NMS.7/cap100 feed scale-routed seven-bin aligned
/// box features with canonical side32 and hidden128/11 classes. Class NMS.3
/// and strict score>.05 cap100 boxes/image. Four16-channel mask convolutions
/// on14-bin aligned crops predict ten28px sigmoid masks,pasted at decoded
/// boxes with inclusive threshold.5. Reuses the supplied shared feature forward;
/// no BN updates,hidden optimizer or coordinate gradients. Source priors use
/// sides8/16/24/40/56,not FpnDigitAnchors defaults8/16/32/64/128.
///
/// @param p Learned compact store;no fitting or mutation of existing leaves.
/// @param features Compact64px gray-derived FPN32/three semantic-logit outputs.
/// @param rpn Same-forward hidden32 RPN over1023 anchors.
/// @param anchors Same-dtype[1023,4] source priors for P2..P6.
/// @param rows Unique nonnegative original scene IDs in feature-batch order.
/// @return Detached scored dataset-ID boxes and Bool[D,64,64] instance masks.
/// @throws std::invalid_argument For invalid geometry,dtypes,settings or image IDs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::PanopticConfig c;
/// c.size = 64;
/// c.depth = 18;
/// c.divisor = 8;
/// c.width = 32;
/// c.semantic_width = 16;
/// c.groups = 8;
/// c.semantic_classes = 3;
/// c.grayscale = true;
/// const auto features = n::models::PanopticBackbone(p, state, n::zeros({1, 64, 64, 1}), c);
/// const auto rpn = n::models::BuildPanopticRpn(p, features, 32);
/// const auto masks = n::models::DecodeCompactPanopticInstances(
///     p, features, rpn, n::models::CompactPanopticAnchors(), {0});
/// @endcode
/// @par References
/// - Kirillov et al.(2019),Panoptic Feature Pyramid Networks, https://arxiv.org/abs/1901.02446; compact64px instance-head widths from the source teaching example.
/// @see PanopticRoiAlign(), PasteInstanceMask()
inline TensorMaskDetections
DecodeCompactPanopticInstances(nn::ParameterStore &p, const PanopticFeatures &features,
                               const Rpn &rpn, const nn::Tensor &anchors,
                               const std::vector<std::int64_t> &rows) {
  if (rows.empty() || features.pyramid.size() != 5 ||
      features.pyramid[0].shape() != nn::Shape{std::int64_t(rows.size()), 16, 16, 32} ||
      anchors.shape() != nn::Shape{1023, 4} || anchors.dtype() != p.dtype())
    throw std::invalid_argument("invalid compact panoptic inference geometry");
  auto unique = rows;
  std::sort(unique.begin(), unique.end());
  if (unique[0] < 0 || std::adjacent_find(unique.begin(), unique.end()) != unique.end())
    throw std::invalid_argument("panoptic inference rows must be distinct/nonnegative");
  nn::NoGradGuard guard;
  const auto proposals = ComputeDetectionProposals(rpn, anchors, 64, 64);
  const auto batch = PackDetectionRois(proposals);
  if (!batch.indices.numel())
    return {{nn::zeros({0, 4}, p.dtype()), nn::zeros({0}, p.dtype()),
             nn::zeros({0}, nn::DType::Int64), nn::zeros({0}, nn::DType::Int64)},
            nn::zeros({0, 64, 64}, nn::DType::Bool)};
  const auto head = PooledDetectionRoiHead(
      p, PanopticRoiAlign(features.pyramid, batch.pixels, batch.indices, 64, 7, 32), 11, 128);
  const auto objects = DecodeDetectionRoiOutputs(
      head, batch.pixels, nn::gather(nn::from_vector(rows), batch.indices));
  if (!objects.scores.numel())
    return {objects, nn::zeros({0, 64, 64}, nn::DType::Bool)};
  std::vector<std::int64_t> local;
  for (const auto image : objects.images.to_vector<std::int64_t>())
    local.push_back(std::find(rows.begin(), rows.end(), image) - rows.begin());
  const auto probability = nn::sigmoid(MaskRcnnHead(
      p, PanopticRoiAlign(features.pyramid, objects.boxes, nn::from_vector(local), 64, 14, 32),
      16, 10));
  std::vector<nn::Tensor> masks;
  for (std::int64_t k = 0; k < objects.scores.numel(); ++k)
    masks.push_back(PasteInstanceMask(
                        nn::slice_axis(probability, 0, k, k + 1).reshape({28, 28, 10}),
                        int(nn::slice_axis(objects.labels, 0, k, k + 1).item<std::int64_t>()),
                        nn::slice_axis(objects.boxes, 0, k, k + 1).reshape({4}), 64, 64)
                        .reshape({1, 64, 64}));
  return {objects, nn::concat(masks, 0)};
}
} // namespace datamunge::nn::models
