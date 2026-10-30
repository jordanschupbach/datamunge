#pragma once
/// @file yolact_models.hpp
/// @brief YOLACT shared prototypes,coefficient prediction,tensor target/loss construction and Fast NMS.
#include "datamunge/nn/instance_scene_models.hpp"
#include "datamunge/nn/resnet_models.hpp"
#include "datamunge/nn/two_stage_detection_models.hpp"
#include <map>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Configure full ResNet101 YOLACT or an explicitly compact same-branch graph.
///
/// No pretrained weights or fitting. All five prediction scales share leaves;
/// size controls spatial geometry without changing parameter count.
///
/// @throws std::invalid_argument For invalid settings,tensor geometry/dtypes,IDs or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::YolactConfig cfg;
/// cfg.size = 64;
/// cfg.classes = 10;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.prototypes = 8;
/// cfg.grayscale = true;
/// const auto net = n::models::Yolact(p, state, n::zeros({2, 64, 64, 1}), cfg);
/// @endcode
/// @par References
/// - Bolya,Zhou,Xiao and Lee(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689; authors implementation https://github.com/dbolya/yolact/blob/master/yolact.py. Background-last,true aspect-ratio anchors,discrete crop area and32 sampled masks are teaching adaptations.
/// @see Yolact()
struct YolactConfig {
  int size = 550;      ///< Square input side,at least32.
  int classes = 80;    ///< Positive foreground classes,background final probability channel.
  int depth = 101;     ///< Supported canonical ResNet depth.
  int divisor = 1;     ///< Positive exact divisor of64 for backbone channels.
  int width = 256;     ///< Positive shared FPN/prediction/protonet channels.
  int prototypes = 32; ///< Positive number of shared ReLU prototype maps.
  bool grayscale = false; ///< True requires one input channel,tiled to RGB before ResNet.
};
/// @brief Own all YOLACT branch outputs with common anchor ordering and prototype geometry.
///
/// Graph tensors borrow caller stores until backward;no optimizer or checkpoint
/// is owned by this result.
///
/// @throws std::invalid_argument For invalid settings,tensor geometry/dtypes,IDs or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::YolactConfig cfg;
/// cfg.size = 64;
/// cfg.classes = 10;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.prototypes = 8;
/// cfg.grayscale = true;
/// const auto net = n::models::Yolact(p, state, n::zeros({2, 64, 64, 1}), cfg);
/// @endcode
/// @par References
/// - Bolya,Zhou,Xiao and Lee(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689; authors implementation https://github.com/dbolya/yolact/blob/master/yolact.py. Background-last,true aspect-ratio anchors,discrete crop area and32 sampled masks are teaching adaptations.
/// @see Yolact()
struct YolactOutputs {
  nn::Tensor logits;        ///< Floating[N,A,C+1] class logits,background last.
  nn::Tensor probabilities; ///< Softmax class probabilities matching logits.
  nn::Tensor deltas;        ///< Floating[N,A,4] default-variance box offsets.
  nn::Tensor coefficients;  ///< Floating[N,A,K] tanh mask coefficients.
  nn::Tensor prototypes;    ///< Floating[N,H,H,K] ReLU shared prototypes,H=2*cells[0].
  nn::Tensor semantic;      ///< Floating[N,cells[0],cells[0],C] auxiliary sigmoid logits.
  std::vector<int> cells;   ///< Five ceil(size/stride) grid sides for strides8,16,32,64,128.
  int anchors = 0;          ///< Sum of3*cells² over the five levels.
  int proto_side = 0;       ///< Twice first prediction grid side.
};
/// @brief Calculate five ceil-divided YOLACT prediction grid sides.
///
/// Metadata only;550 gives69/35/18/9/5 and19248 three-ratio anchors.
///
/// @param size Positive square image side.
/// @return Five integers for strides8 through128.
/// @throws std::invalid_argument For invalid settings,tensor geometry/dtypes,IDs or incompatible stored leaves.
/// @par Example
/// @code
/// const auto cells = dm::nn::models::YolactCells(550);
/// @endcode
/// @par References
/// - Bolya,Zhou,Xiao and Lee(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689; authors implementation https://github.com/dbolya/yolact/blob/master/yolact.py. Background-last,true aspect-ratio anchors,discrete crop area and32 sampled masks are teaching adaptations.
/// @see YolactAnchors()
inline std::vector<int> YolactCells(int size) {
  if (size < 1 || size > std::numeric_limits<int>::max() - 127)
    throw std::invalid_argument("invalid YOLACT input side");
  std::vector<int> cells;
  for (int i = 0; i < 5; ++i)
    cells.push_back((size + (8 << i) - 1) / (8 << i));
  return cells;
}

/// @brief Build ResNet/FPN shared prediction heads,protonet and semantic auxiliary logits.
///
/// ResNet3x3-stride convention,C3/C4/C5 laterals and bilinear unsmoothed
/// top-down maps feed independent3x3/ReLU smoothing. P6/P7 use padded stride2
/// linear convolutions. Shared3x3 prediction feature and box/class/coefficient
/// leaves repeat across scales. Protonet has three3x3/ReLU,2x bilinear resize,
/// one3x3/ReLU and1x1/ReLU. No hidden fitting. Warm every branch before
/// optimizer construction;restore learned and BN stores together. Full graph
/// parameter census may execute at64px while550px geometry is checked separately.
///
/// @param p Learned floating store,creates/reuses all backbone/FPN/head leaves.
/// @param state Same-dtype BN moving state,updated only when training and update_stats.
/// @param images Same-dtype positive[N,cfg.size,cfg.size,1 or3] images.
/// @param cfg Valid architecture metadata,default full550px/ResNet101/FPN256/32 prototypes.
/// @param training False(default) uses moving BN;true uses current batch moments.
/// @param update_stats True(default) permits moving-stat updates;false supports training-mode diagnostics.
/// @return Five-scale logits/deltas/tanh coefficients,ReLU prototypes and semantic logits.
/// @throws std::invalid_argument For invalid settings,tensor geometry/dtypes,IDs or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::YolactConfig cfg;
/// cfg.size = 64;
/// cfg.classes = 10;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.prototypes = 8;
/// cfg.grayscale = true;
/// const auto net = n::models::Yolact(p, state, n::zeros({2, 64, 64, 1}), cfg);
/// @endcode
/// @par References
/// - Bolya,Zhou,Xiao and Lee(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689; authors implementation https://github.com/dbolya/yolact/blob/master/yolact.py. Background-last,true aspect-ratio anchors,discrete crop area and32 sampled masks are teaching adaptations.
/// @see BuildTensorYolactLoss(), YolactCells()
inline YolactOutputs Yolact(nn::ParameterStore &p, nn::ParameterStore &state,
                            const nn::Tensor &images, const YolactConfig &cfg = {},
                            bool training = false, bool update_stats = true) {
  if (cfg.size < 32 || cfg.classes < 1 || cfg.width < 1 || cfg.prototypes < 1 ||
      cfg.divisor < 1 || 64 % cfg.divisor)
    throw std::invalid_argument("invalid YOLACT configuration");
  if (images.dim() != 4 || images.shape()[0] < 1 || images.shape()[1] != cfg.size ||
      images.shape()[2] != cfg.size || images.shape()[3] != (cfg.grayscale ? 1 : 3) ||
      images.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("YOLACT images/store mismatch");
  YolactOutputs out;
  ResNetConfig rc;
  rc.depth = cfg.depth;
  rc.width_divisor = cfg.divisor;
  rc.include_classifier = false;
  rc.stride_on_3x3 = true;
  rc.update_stats = update_stats;
  const auto backbone =
      ResNet(p, state, cfg.grayscale ? nn::tile(images, {1, 1, 1, 3}) : images, rc, training);
  const std::int64_t w = cfg.width;
  std::vector<nn::Tensor> inner(3), features(3);
  nn::ConvLayerOptions linear;
  linear.init = nn::init::he_uniform();
  auto relu = linear;
  relu.activation = nn::Activation::Relu;
  for (int i = 2; i >= 0; --i) {
    auto x = nn::conv2d(p, "fpn/lateral" + std::to_string(i),
                        backbone.stages[std::size_t(i + 1)], w, 1, linear);
    if (i < 2)
      x = x + nn::resize_bilinear(inner[std::size_t(i + 1)], {x.shape()[1], x.shape()[2]});
    inner[std::size_t(i)] = x;
    features[std::size_t(i)] = nn::conv2d(p, "fpn/smooth" + std::to_string(i), x, w, 3, relu);
  }
  auto down = linear;
  down.stride = 2;
  down.padding = nn::Padding::Valid;
  for (int i = 0; i < 2; ++i)
    features.push_back(nn::conv2d(p, "fpn/down" + std::to_string(i),
                                  nn::pad(features.back(), {{0, 0}, {1, 1}, {1, 1}, {0, 0}}),
                                  w, 3, down));
  out.cells = YolactCells(cfg.size);
  std::vector<nn::Tensor> logits, deltas, coefficients;
  for (int i = 0; i < 5; ++i) {
    const int cells = out.cells[std::size_t(i)], count = 3 * cells * cells;
    if (features[std::size_t(i)].shape()[1] != cells ||
        features[std::size_t(i)].shape()[2] != cells)
      throw std::invalid_argument(
          "YOLACT feature grid does not match the configured input size");
    out.anchors += count;
    const auto x = nn::conv2d(p, "prediction/shared", features[std::size_t(i)], w, 3, relu);
    logits.push_back(
        nn::reshape(nn::conv2d(p, "prediction/classes", x, 3 * (cfg.classes + 1), 3, linear),
                    {-1, count, cfg.classes + 1}));
    deltas.push_back(
        nn::reshape(nn::conv2d(p, "prediction/boxes", x, 12, 3, linear), {-1, count, 4}));
    coefficients.push_back(nn::reshape(
        nn::tanh(nn::conv2d(p, "prediction/coefficients", x, 3 * cfg.prototypes, 3, linear)),
        {-1, count, cfg.prototypes}));
  }
  out.logits = nn::concat(logits, 1);
  out.deltas = nn::concat(deltas, 1);
  out.coefficients = nn::concat(coefficients, 1);
  out.probabilities = nn::softmax(out.logits, -1);
  nn::Tensor proto = features[0];
  for (int i = 0; i < 3; ++i)
    proto = nn::conv2d(p, "proto/conv" + std::to_string(i), proto, w, 3, relu);
  out.proto_side = 2 * out.cells[0];
  proto = nn::resize_bilinear(proto, {out.proto_side, out.proto_side});
  proto = nn::conv2d(p, "proto/after_resize", proto, w, 3, relu);
  out.prototypes = nn::conv2d(p, "proto/output", proto, cfg.prototypes, 1, relu);
  out.semantic = nn::conv2d(p, "semantic", features[0], cfg.classes, 1, linear);
  return out;
}

/// @brief Assemble mask logits by a signed linear combination of shared prototype maps.
///
/// No sigmoid,cropping,fitting or mutation. Empty M is allowed. WorkO(MHWK).
///
/// @param prototypes Floating[M,H,W,K] with positive spatial/prototype axes.
/// @param coefficients Same-dtype[M,K] instance coefficients.
/// @return Floating[M,H,W] logits with both prototype and coefficient gradients.
/// @throws std::invalid_argument For invalid settings,tensor geometry/dtypes,IDs or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y =
///     n::models::LinearInstanceMasks(n::ones({1, 2, 3, 2}), n::tensor({.5, -.25}, {1, 2}));
/// @endcode
/// @par References
/// - Bolya,Zhou,Xiao and Lee(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689; authors implementation https://github.com/dbolya/yolact/blob/master/yolact.py. Background-last,true aspect-ratio anchors,discrete crop area and32 sampled masks are teaching adaptations.
/// @see CroppedInstanceMaskLoss()
inline nn::Tensor LinearInstanceMasks(const nn::Tensor &prototypes,
                                      const nn::Tensor &coefficients) {
  if (prototypes.dim() != 4 || prototypes.shape()[1] < 1 || prototypes.shape()[2] < 1 ||
      prototypes.shape()[3] < 1 ||
      coefficients.shape() != nn::Shape{prototypes.shape()[0], prototypes.shape()[3]} ||
      coefficients.dtype() != prototypes.dtype() ||
      (prototypes.dtype() != nn::DType::Float32 && prototypes.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid prototype combination");
  return nn::sum(
      prototypes * coefficients.reshape({prototypes.shape()[0], 1, 1, prototypes.shape()[3]}),
      {3});
}
/// @brief Average per-instance sigmoid BCE normalized by each discrete crop area.
///
/// BCE uses stable softplus. Zero-area crops have exactly zero loss/logit
/// gradients. No hidden optimization or mutation.
///
/// @param logits Floating[M,H,W] mask logits,positive H/W.
/// @param targets Same-dtype/shape target probabilities in[0,1],detached.
/// @param crop Same-dtype/shape finite nonnegative mask weights,detached.
/// @return Scalar mean of sum(crop*BCE)/max(sum(crop),1);empty M gives differentiable zero.
/// @throws std::invalid_argument For invalid settings,tensor geometry/dtypes,IDs or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::CroppedInstanceMaskLoss(n::zeros({1, 2, 3}), n::ones({1, 2, 3}),
///                                                      n::ones({1, 2, 3}));
/// @endcode
/// @par References
/// - Bolya,Zhou,Xiao and Lee(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689; authors implementation https://github.com/dbolya/yolact/blob/master/yolact.py. Background-last,true aspect-ratio anchors,discrete crop area and32 sampled masks are teaching adaptations.
/// @see LinearInstanceMasks(), BuildTensorYolactLoss()
inline nn::Tensor CroppedInstanceMaskLoss(const nn::Tensor &logits, const nn::Tensor &targets,
                                          const nn::Tensor &crop) {
  if (logits.dim() != 3 || logits.shape()[1] < 1 || logits.shape()[2] < 1 ||
      targets.shape() != logits.shape() || crop.shape() != logits.shape() ||
      targets.dtype() != logits.dtype() || crop.dtype() != logits.dtype() ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      !nn::all(nn::logical_and(nn::greater_equal(targets, 0.), nn::less_equal(targets, 1.)))
           .item<bool>() ||
      !nn::all(
           nn::logical_and(nn::greater_equal(crop, 0.),
                           nn::less(nn::abs(crop), std::numeric_limits<double>::infinity())))
           .item<bool>())
    throw std::invalid_argument("invalid cropped mask loss");
  if (!logits.shape()[0])
    return nn::sum(logits) * 0.;
  return nn::mean(
      nn::sum((nn::softplus(logits) - logits * targets.detach()) * crop.detach(), {1, 2}) /
      nn::maximum(nn::sum(crop.detach(), {1, 2}), nn::ones({}, logits.dtype())));
}
/// @brief Match YOLACT anchors with distinct forced SSD positives and an ignored overlap band.
///
/// MatchSsd uses positive threshold.5 and distinct forced greedy pairs.
/// Remaining unmatched anchors whose best truth IoU>=.4 become ignored.
/// Empty truth makes all anchors background;no gradient or fitting.
///
/// @param anchors Floating[A,4] positive boxes.
/// @param truth Same-dtype[G,4] positive truth boxes.
/// @return Detached Int64[A]:truth index for positives,-2 ignored,-1 background.
/// @throws std::invalid_argument For invalid settings,tensor geometry/dtypes,IDs or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto match = n::models::YolactMatching(n::tensor({0., 0., 10., 10.}, {1, 4}),
///                                              n::tensor({0., 0., 10., 10.}, {1, 4}));
/// @endcode
/// @par References
/// - Bolya,Zhou,Xiao and Lee(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689; authors implementation https://github.com/dbolya/yolact/blob/master/yolact.py. Background-last,true aspect-ratio anchors,discrete crop area and32 sampled masks are teaching adaptations.
/// @see MatchSsd(), MakeTensorYolactTargets()
inline nn::Tensor YolactMatching(const nn::Tensor &anchors, const nn::Tensor &truth) {
  if (anchors.dim() != 2 || anchors.shape()[1] != 4 || truth.dim() != 2 ||
      truth.shape()[1] != 4 || anchors.dtype() != truth.dtype())
    throw std::invalid_argument("invalid YOLACT match boxes");
  for (const auto &boxes : {anchors, truth})
    if ((boxes.dtype() != nn::DType::Float32 && boxes.dtype() != nn::DType::Float64) ||
        !nn::all(nn::less(nn::abs(boxes), std::numeric_limits<double>::infinity()))
             .item<bool>() ||
        !nn::all(nn::greater(nn::box_area(boxes), 0.)).item<bool>())
      throw std::invalid_argument("YOLACT matching requires finite positive boxes");
  nn::NoGradGuard guard;
  const auto overlaps = nn::box_iou(anchors, truth);
  const auto match = MatchSsd(overlaps);
  if (!truth.shape()[0])
    return match;
  return nn::where(
      nn::logical_and(nn::less(match, 0.), nn::greater_equal(nn::max(overlaps, {1}), .4)),
      nn::full(match.shape(), -2., nn::DType::Int64), match);
}
/// @brief Build three true-aspect-ratio anchors for each of five YOLACT feature grids.
///
/// Ratios1,.5,2;level cell spacing is size/cells. The true aspect-ratio
/// convention intentionally differs from the original release legacy square
/// anchor bug. No gradients,fitting or store mutation.
///
/// @param net Outputs whose cells contain five positive grid sides.
/// @param size Positive source square image side.
/// @param compact True selects sides8/16/24/40/56;false(default) selects24/48/96/192/384.
/// @return Float32[A,4] pixel boxes in level/cell/ratio order.
/// @throws std::invalid_argument For invalid settings,tensor geometry/dtypes,IDs or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::YolactConfig cfg;
/// cfg.size = 64;
/// cfg.classes = 10;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.prototypes = 8;
/// cfg.grayscale = true;
/// const auto net = n::models::Yolact(p, state, n::zeros({2, 64, 64, 1}), cfg);
/// const auto anchors = n::models::YolactAnchors(net, 64, true);
/// @endcode
/// @par References
/// - Bolya,Zhou,Xiao and Lee(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689; authors implementation https://github.com/dbolya/yolact/blob/master/yolact.py. Background-last,true aspect-ratio anchors,discrete crop area and32 sampled masks are teaching adaptations.
/// @see YolactCells(), YolactMatching()
inline nn::Tensor YolactAnchors(const YolactOutputs &net, int size, bool compact = false) {
  if (size < 1 || net.cells.size() != 5)
    throw std::invalid_argument("invalid YOLACT anchor metadata");
  const std::vector<double> scales = compact ? std::vector<double>{8, 16, 24, 40, 56}
                                             : std::vector<double>{24, 48, 96, 192, 384};
  std::vector<nn::Tensor> levels;
  for (int i = 0; i < 5; ++i) {
    if (net.cells[i] < 1)
      throw std::invalid_argument("invalid YOLACT grid side");
    levels.push_back(nn::anchor_grid(net.cells[i], net.cells[i], double(size) / net.cells[i],
                                     {scales[i]}, {1, .5, 2}));
  }
  return nn::concat(levels, 0);
}
/// @brief Own detached YOLACT class/box,selected-mask and semantic-union target tensors.
///
/// No host numerical targets or learned state. Labels infer batch/anchor axes;
/// mask and semantic tensors carry their own spatial shapes.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtypes,indices,target values or incompatible model outputs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::YolactConfig cfg;
/// cfg.size = 64;
/// cfg.classes = 10;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.prototypes = 8;
/// cfg.grayscale = true;
/// const auto net = n::models::Yolact(p, state, n::zeros({1, 64, 64, 1}), cfg);
/// n::models::DetectionSceneConfig scene;
/// scene.scenes = 1;
/// const auto data = n::models::MakeTensorInstanceScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), scene);
/// n::Generator rng(7);
/// const auto target = n::models::MakeTensorYolactTargets(
///     net, n::models::YolactAnchors(net, 64, true), data, {0}, 10, rng);
/// @endcode
/// @par References
/// - Bolya et al.(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689. Background-last,32 sampled positive masks,pixel-centre targets and discrete crop-area normalization follow the source teaching adaptation.
/// @see MakeTensorYolactTargets(), BuildTensorYolactLoss()
struct TensorYolactTargets {
  nn::Tensor labels;       ///< Int64[B,A],foreground0..C-1,background C,ignored-1.
  nn::Tensor deltas;       ///< Floating[B,A,4] encoded box offsets,zero except positives.
  nn::Tensor pairs;        ///< Int64[M,2] selected positive(feature batch position,anchor ID).
  nn::Tensor mask_targets; ///< Floating[M,H,H] nearest-sampled individual supports.
  nn::Tensor crop;         ///< Floating[M,H,H] pixel-centre half-open box indicator.
  nn::Tensor semantic; ///< Floating[B,G,G,C] class-specific union at first prediction grid G.
};
/// @brief Own YOLACT joint loss,mask component and detached mined-negative indicator.
///
/// All scalar objectives retain model gradients. Negative selection is discrete;
/// no optimizer or state mutation occurs.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtypes,indices,target values or incompatible model outputs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::YolactConfig cfg;
/// cfg.size = 64;
/// cfg.classes = 10;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.prototypes = 8;
/// cfg.grayscale = true;
/// const auto net = n::models::Yolact(p, state, n::zeros({1, 64, 64, 1}), cfg);
/// n::models::DetectionSceneConfig scene;
/// scene.scenes = 1;
/// const auto data = n::models::MakeTensorInstanceScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), scene);
/// n::Generator rng(7);
/// const auto target = n::models::MakeTensorYolactTargets(
///     net, n::models::YolactAnchors(net, 64, true), data, {0}, 10, rng);
/// const auto loss = n::models::BuildTensorYolactLoss(net, target);
/// @endcode
/// @par References
/// - Bolya et al.(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689. Background-last,32 sampled positive masks,pixel-centre targets and discrete crop-area normalization follow the source teaching adaptation.
/// @see BuildTensorYolactLoss()
struct TensorYolactLoss {
  nn::Tensor loss;      ///< Detection+6.125*mask BCE+semantic BCE scalar.
  nn::Tensor mask_loss; ///< Unweighted crop-area-normalized sigmoid mask BCE scalar.
  nn::Tensor negatives; ///< Detached Bool[B,A] selected3:1 hard background anchors.
};
/// @brief Form all YOLACT training objectives with explicit tensor hard-negative mining.
///
/// Per-image backgrounds rank detached CE descending,stable lower-index ties.
/// Select min(3*positives,backgrounds);ignored anchors never classify/regress.
/// Detection is summed CE+1.5*unitSmoothL1 divided by total positives floored1.
/// Mask BCE averages selected supports after discrete crop-area normalization,
/// weighted6.125. Semantic sigmoid BCE averages all image/grid/class positions.
/// No positives leaves only semantic BCE;empty mask term remains graph-connected
/// to prototype/coefficient branches. No fitting or source mutation.
///
/// @param net Consistent logits[B,A,C+1],box/coefficient/prototype/semantic outputs.
/// @param target Same-dtype detached class/box/mask/semantic tensors aligned to net.
/// @return Joint/mask scalar objectives plus Bool[B,A] mined-negative mask.
/// @throws std::invalid_argument For invalid settings,geometry,dtypes,indices,target values or incompatible model outputs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::YolactConfig cfg;
/// cfg.size = 64;
/// cfg.classes = 10;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.prototypes = 8;
/// cfg.grayscale = true;
/// const auto net = n::models::Yolact(p, state, n::zeros({1, 64, 64, 1}), cfg);
/// n::models::DetectionSceneConfig scene;
/// scene.scenes = 1;
/// const auto data = n::models::MakeTensorInstanceScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), scene);
/// n::Generator rng(7);
/// const auto target = n::models::MakeTensorYolactTargets(
///     net, n::models::YolactAnchors(net, 64, true), data, {0}, 10, rng);
/// const auto loss = n::models::BuildTensorYolactLoss(net, target);
/// @endcode
/// @par References
/// - Bolya et al.(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689. Background-last,32 sampled positive masks,pixel-centre targets and discrete crop-area normalization follow the source teaching adaptation.
/// @see LinearInstanceMasks(), CroppedInstanceMaskLoss()
inline TensorYolactLoss BuildTensorYolactLoss(const YolactOutputs &net,
                                              const TensorYolactTargets &target) {
  if (net.logits.dim() != 3 || net.logits.shape()[0] < 1 || net.logits.shape()[1] < 1 ||
      net.logits.shape()[2] < 2)
    throw std::invalid_argument("invalid YOLACT loss logits");
  const auto B = net.logits.shape()[0], A = net.logits.shape()[1],
             C = net.logits.shape()[2] - 1,
             K = net.coefficients.dim() == 3 ? net.coefficients.shape()[2] : 0,
             H = net.prototypes.dim() == 4 ? net.prototypes.shape()[1] : 0,
             G = net.semantic.dim() == 4 ? net.semantic.shape()[1] : 0,
             M = target.pairs.dim() == 2 ? target.pairs.shape()[0] : -1;
  const auto type = net.logits.dtype();
  if ((type != nn::DType::Float32 && type != nn::DType::Float64) || K < 1 || H < 1 || G < 1 ||
      M < 0 || target.labels.shape() != nn::Shape{B, A} ||
      target.labels.dtype() != nn::DType::Int64 ||
      target.deltas.shape() != nn::Shape{B, A, 4} ||
      net.deltas.shape() != target.deltas.shape() ||
      net.coefficients.shape() != nn::Shape{B, A, K} ||
      net.prototypes.shape() != nn::Shape{B, H, H, K} ||
      net.semantic.shape() != nn::Shape{B, G, G, C} ||
      target.semantic.shape() != net.semantic.shape() ||
      target.pairs.shape() != nn::Shape{M, 2} || target.pairs.dtype() != nn::DType::Int64 ||
      target.mask_targets.shape() != nn::Shape{M, H, H} ||
      target.crop.shape() != target.mask_targets.shape())
    throw std::invalid_argument("inconsistent YOLACT loss tensors");
  for (const auto &t : {target.deltas, net.deltas, net.coefficients, net.prototypes,
                        net.semantic, target.semantic, target.mask_targets, target.crop})
    if (t.dtype() != type)
      throw std::invalid_argument("YOLACT loss dtype mismatch");
  if (!nn::all(nn::logical_and(nn::greater_equal(target.labels, -1.),
                               nn::less_equal(target.labels, double(C))))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(target.semantic, 0.),
                               nn::less_equal(target.semantic, 1.)))
           .item<bool>())
    throw std::invalid_argument("invalid YOLACT labels/semantic targets");
  const auto positive = nn::logical_and(nn::greater_equal(target.labels, 0.),
                                        nn::less(target.labels, double(C)));
  const auto onehot =
      nn::one_hot(nn::maximum(target.labels, nn::zeros({}, nn::DType::Int64)), C + 1)
          .to(type) *
      nn::greater_equal(target.labels, 0.).to(type).reshape({B, A, 1});
  const auto ce = -nn::sum(onehot * nn::log_softmax(net.logits, -1), {2});
  std::vector<nn::Tensor> negative_parts;
  {
    nn::NoGradGuard guard;
    for (std::int64_t row = 0; row < B; ++row) {
      const auto background =
          nn::equal(nn::slice_axis(target.labels, 0, row, row + 1).reshape({A}), double(C));
      const auto pos = nn::count_nonzero(nn::slice_axis(positive, 0, row, row + 1))
                           .item<std::int64_t>(),
                 neg = nn::count_nonzero(background).item<std::int64_t>(),
                 wanted = std::min(3 * pos, neg);
      const auto scores =
          nn::where(background, nn::slice_axis(ce.detach(), 0, row, row + 1).reshape({A}),
                    nn::full({A}, -std::numeric_limits<double>::infinity(), type));
      const auto order = nn::argsort(scores, true), ranks = nn::argsort(order, false);
      negative_parts.push_back(
          nn::logical_and(background, nn::less(ranks, double(wanted))).reshape({1, A}));
    }
  }
  TensorYolactLoss out;
  out.negatives = nn::concat(negative_parts, 0);
  const auto count = nn::maximum(nn::count_nonzero(positive).to(type), nn::ones({}, type));
  const auto classification = nn::sum(ce * (positive.to(type) + out.negatives.to(type)));
  const auto regression = nn::sum(
      nn::sum(DetectorSmoothL1(net.deltas - target.deltas.detach()), {2}) * positive.to(type));
  out.mask_loss = (nn::sum(net.prototypes) + nn::sum(net.coefficients)) * 0.;
  if (M) {
    const auto image = nn::slice_axis(target.pairs, 1, 0, 1).reshape({M}),
               anchor = nn::slice_axis(target.pairs, 1, 1, 2).reshape({M});
    if (!nn::all(
             nn::logical_and(
                 nn::logical_and(nn::greater_equal(image, 0.), nn::less(image, double(B))),
                 nn::logical_and(nn::greater_equal(anchor, 0.), nn::less(anchor, double(A)))))
             .item<bool>())
      throw std::invalid_argument("invalid selected YOLACT mask indices");
    out.mask_loss = CroppedInstanceMaskLoss(
        LinearInstanceMasks(nn::gather(net.prototypes, image),
                            nn::gather_nd(net.coefficients, target.pairs)),
        target.mask_targets, target.crop);
  }
  const auto semantic =
      nn::mean(nn::softplus(net.semantic) - net.semantic * target.semantic.detach());
  out.loss = (classification + 1.5 * regression) / count + 6.125 * out.mask_loss + semantic;
  return out;
}
/// @brief Construct matched boxes,random positive masks and class unions for a scene minibatch.
///
/// Distinct forced SSD matching uses positive.5/ignore.4 bands. Each selected
/// object support samples source pixel centres nearest on prototype grid. Crop
/// is its half-open truth box indicator at those centres,no one-pixel expansion.
/// Class-specific semantic target is the union of sampled supports on P3 grid.
/// All geometry/encoding/sampling/union computations use tensors. Host loops
/// carry only discrete row/positive/object IDs. No fitting or source mutation.
///
/// @param net Model outputs supplying anchor/prototype/semantic geometry.
/// @param anchors Floating[A,4] positive prior boxes matching scene/model dtype.
/// @param scenes Valid instance scenes with aligned Bool supports and foreground labels.
/// @param rows Nonempty scene IDs in feature-minibatch order.
/// @param classes Positive foreground class count matching net logits/semantic channels.
/// @param rng Mutable tensor shuffle generator;draws differ from the old std::mt19937 shuffle.
/// @return Detached targets with at most32 randomly selected positive masks per image.
/// @throws std::invalid_argument For invalid settings,geometry,dtypes,indices,target values or incompatible model outputs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::YolactConfig cfg;
/// cfg.size = 64;
/// cfg.classes = 10;
/// cfg.depth = 18;
/// cfg.divisor = 8;
/// cfg.width = 32;
/// cfg.prototypes = 8;
/// cfg.grayscale = true;
/// const auto net = n::models::Yolact(p, state, n::zeros({1, 64, 64, 1}), cfg);
/// n::models::DetectionSceneConfig scene;
/// scene.scenes = 1;
/// const auto data = n::models::MakeTensorInstanceScenes(n::ones({1, 28, 28, 1}),
///                                                       n::zeros({1}, n::DType::Int64), scene);
/// n::Generator rng(7);
/// const auto target = n::models::MakeTensorYolactTargets(
///     net, n::models::YolactAnchors(net, 64, true), data, {0}, 10, rng);
/// @endcode
/// @par References
/// - Bolya et al.(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689. Background-last,32 sampled positive masks,pixel-centre targets and discrete crop-area normalization follow the source teaching adaptation.
/// @see YolactMatching(), BuildTensorYolactLoss()
inline TensorYolactTargets MakeTensorYolactTargets(const YolactOutputs &net,
                                                   const nn::Tensor &anchors,
                                                   const TensorInstanceScenes &scenes,
                                                   const std::vector<std::int64_t> &rows,
                                                   int classes, nn::Generator &rng) {
  if (rows.empty() || classes < 1 || net.logits.dim() != 3 ||
      net.logits.shape()[2] != classes + 1 ||
      anchors.shape() != nn::Shape{net.logits.shape()[1], 4} ||
      anchors.dtype() != net.logits.dtype() || scenes.images.dim() != 4 ||
      scenes.images.dtype() != anchors.dtype() || scenes.images.shape()[1] < 1 ||
      scenes.images.shape()[2] < 1 ||
      scenes.boxes.size() != std::size_t(scenes.images.shape()[0]) ||
      scenes.labels.size() != scenes.boxes.size() ||
      scenes.masks.size() != scenes.boxes.size() || net.proto_side < 1 || net.cells.empty() ||
      net.cells[0] < 1)
    throw std::invalid_argument("invalid YOLACT target inputs");
  const auto A = anchors.shape()[0], H = std::int64_t(net.proto_side),
             G = std::int64_t(net.cells[0]), height = scenes.images.shape()[1],
             width = scenes.images.shape()[2];
  const auto type = anchors.dtype();
  nn::NoGradGuard guard;
  std::vector<nn::Tensor> labels, deltas, mask_targets, crops, semantic;
  std::vector<std::int64_t> pairs;
  for (std::size_t image = 0; image < rows.size(); ++image) {
    const auto row = rows[image];
    if (row < 0 || row >= scenes.images.shape()[0] || scenes.boxes[row].dim() != 2 ||
        scenes.boxes[row].shape()[1] != 4)
      throw std::invalid_argument("invalid YOLACT scene row");
    const auto T = scenes.boxes[row].shape()[0];
    if (scenes.labels[row].shape() != nn::Shape{T} ||
        scenes.labels[row].dtype() != nn::DType::Int64 ||
        scenes.masks[row].shape() != nn::Shape{T, height, width} ||
        scenes.masks[row].dtype() != nn::DType::Bool ||
        !nn::all(nn::logical_and(nn::greater_equal(scenes.labels[row], 0.),
                                 nn::less(scenes.labels[row], double(classes))))
             .item<bool>())
      throw std::invalid_argument("unaligned YOLACT instance annotations");
    const auto match = YolactMatching(anchors, scenes.boxes[row]);
    const auto positive = nn::greater_equal(match, 0.);
    auto rowlabels = nn::full({A}, double(classes), nn::DType::Int64),
         rowdeltas = nn::zeros({A, 4}, type);
    if (T) {
      const auto index = nn::maximum(match, nn::zeros({}, nn::DType::Int64));
      rowlabels = nn::where(positive, nn::gather(scenes.labels[row], index), rowlabels);
      rowdeltas = nn::encode_boxes(anchors, nn::gather(scenes.boxes[row], index)) *
                  positive.to(type).reshape({A, 1});
    }
    rowlabels =
        nn::where(nn::equal(match, -2.), nn::full({A}, -1., nn::DType::Int64), rowlabels);
    labels.push_back(rowlabels.reshape({1, A}));
    deltas.push_back(rowdeltas.reshape({1, A, 4}));
    auto union_grid = nn::zeros({G, G, classes}, type);
    const auto gy = nn::floor((nn::arange(0, double(G)).to(type) + .5) * double(height) /
                              double(G))
                        .to(nn::DType::Int64)
                        .reshape({G, 1}),
               gx = nn::floor((nn::arange(0, double(G)).to(type) + .5) * double(width) /
                              double(G))
                        .to(nn::DType::Int64)
                        .reshape({1, G});
    const auto semantic_index =
        (gy * nn::full({}, double(width), nn::DType::Int64) + gx).reshape({G * G});
    for (std::int64_t truth = 0; truth < T; ++truth) {
      const auto support =
          nn::slice_axis(scenes.masks[row], 0, truth, truth + 1).reshape({height * width});
      const auto mask = nn::gather(support, semantic_index).to(type).reshape({G, G, 1});
      const auto label = nn::slice_axis(scenes.labels[row], 0, truth, truth + 1);
      union_grid = nn::maximum(
          union_grid, mask * nn::one_hot(label, classes).to(type).reshape({1, 1, classes}));
    }
    semantic.push_back(union_grid.reshape({1, G, G, classes}));
    auto selected = nn::shuffle(two_stage_detection_detail::Indices(positive), rng);
    selected = nn::slice_axis(selected, 0, 0, std::min<std::int64_t>(32, selected.numel()));
    const auto py = (nn::arange(0, double(H)).to(type) + .5) * double(height) / double(H),
               px = (nn::arange(0, double(H)).to(type) + .5) * double(width) / double(H);
    const auto support_index = (nn::floor(py).to(nn::DType::Int64).reshape({H, 1}) *
                                    nn::full({}, double(width), nn::DType::Int64) +
                                nn::floor(px).to(nn::DType::Int64).reshape({1, H}))
                                   .reshape({H * H});
    for (const auto anchor : selected.to_vector<std::int64_t>()) {
      const auto target = nn::slice_axis(match, 0, anchor, anchor + 1);
      const auto support = nn::gather(scenes.masks[row], target).reshape({height * width});
      mask_targets.push_back(nn::gather(support, support_index).to(type).reshape({1, H, H}));
      const auto box = nn::unstack(nn::gather(scenes.boxes[row], target).reshape({4}), 0);
      const auto crop =
          nn::logical_and(nn::logical_and(nn::greater_equal(py.reshape({H, 1}), box[0]),
                                          nn::less(py.reshape({H, 1}), box[2])),
                          nn::logical_and(nn::greater_equal(px.reshape({1, H}), box[1]),
                                          nn::less(px.reshape({1, H}), box[3])));
      crops.push_back(crop.to(type).reshape({1, H, H}));
      pairs.push_back(image);
      pairs.push_back(anchor);
    }
  }
  const auto M = std::int64_t(pairs.size() / 2);
  return {nn::concat(labels, 0),
          nn::concat(deltas, 0),
          nn::from_vector(pairs, {M, 2}),
          M ? nn::concat(mask_targets, 0) : nn::zeros({0, H, H}, type),
          M ? nn::concat(crops, 0) : nn::zeros({0, H, H}, type),
          nn::concat(semantic, 0)};
}
/// @brief Select Fast NMS candidates with upper-triangular all-predecessor overlap suppression.
///
/// Every earlier,higher-scored candidate may suppress a later candidate,even
/// if that earlier candidate was itself suppressed. This differs from greedy
/// NMS on an overlap chain. Groups isolate images/classes. Stable input ties
/// apply within a class;final tied class groups retain increasing class order.
/// Host maps hold integer grouping metadata only;score sorting,all-pairs IoU,
/// triangular masking and selection use tensors. WorkO(sum(group_size²))
/// after top_k. No fitting or input mutation.
///
/// @param detections Finite floating boxes[D,4]/scores[D],nonnegative Int64 labels/images.
/// @param threshold Finite overlap cutoff in[0,1],default.5;strictly greater overlap suppresses.
/// @param top_k Nonnegative per-image/class descending-score budget,default200.
/// @param limit Nonnegative final per-image detection budget,default100.
/// @return Detached Int64 selected input indices,image order then descending score.
/// @throws std::invalid_argument For invalid settings,shapes,dtypes,nonfinite predictions or out-of-range IDs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetections d{
///     n::tensor({0., 0., 10., 10., 0., 4., 10., 14., 0., 8., 10., 18.}, {3, 4}),
///     n::tensor({.9, .8, .7}), n::zeros({3}, n::DType::Int64), n::zeros({3}, n::DType::Int64)};
/// const auto kept = n::models::TensorFastNms(d, .3);
/// @endcode
/// @par References
/// - Bolya et al.(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689; teaching pixel-centre crops and mask pasting follow examples/nn/common/yolact.hpp.
/// @see SuppressDetectionCandidates(), DecodeTensorYolact()
inline nn::Tensor TensorFastNms(const TensorDetections &detections, double threshold = .5,
                                int top_k = 200, int limit = 100) {
  const auto &d = detections;
  const auto D = d.scores.numel();
  if (!std::isfinite(threshold) || threshold < 0 || threshold > 1 || top_k < 0 || limit < 0 ||
      d.boxes.shape() != nn::Shape{D, 4} || d.scores.shape() != nn::Shape{D} ||
      d.labels.shape() != nn::Shape{D} || d.images.shape() != nn::Shape{D} ||
      (d.scores.dtype() != nn::DType::Float32 && d.scores.dtype() != nn::DType::Float64) ||
      d.boxes.dtype() != d.scores.dtype() || d.labels.dtype() != nn::DType::Int64 ||
      d.images.dtype() != nn::DType::Int64 ||
      !nn::all(nn::greater_equal(d.labels, 0.)).item<bool>() ||
      !nn::all(nn::greater_equal(d.images, 0.)).item<bool>() ||
      !nn::all(nn::less(nn::abs(d.boxes), std::numeric_limits<double>::infinity()))
           .item<bool>() ||
      !nn::all(nn::less(nn::abs(d.scores), std::numeric_limits<double>::infinity()))
           .item<bool>() ||
      nn::any(nn::less(nn::slice_axis(d.boxes, 1, 2, 4), nn::slice_axis(d.boxes, 1, 0, 2)))
          .item<bool>())
    throw std::invalid_argument("invalid Fast NMS tensors/settings");
  nn::NoGradGuard guard;
  std::map<std::pair<std::int64_t, std::int64_t>, std::vector<std::int64_t>> groups;
  const auto images = d.images.to_vector<std::int64_t>(),
             labels = d.labels.to_vector<std::int64_t>();
  for (std::int64_t i = 0; i < D; ++i)
    groups[{images[i], labels[i]}].push_back(i);
  std::map<std::int64_t, std::vector<nn::Tensor>> parts;
  for (const auto &[key, ids] : groups) {
    const auto original = nn::from_vector(ids);
    const auto order = nn::slice_axis(nn::argsort(nn::gather(d.scores, original), true), 0, 0,
                                      std::min<std::int64_t>(ids.size(), top_k));
    const auto selected = nn::gather(original, order);
    const auto K = selected.numel();
    if (!K)
      continue;
    const auto boxes = nn::gather(d.boxes, selected), overlap = nn::box_iou(boxes, boxes);
    const auto index = nn::arange(0, double(K), 1., nn::DType::Int64);
    const auto upper = nn::where(nn::less(index.reshape({K, 1}), index.reshape({1, K})),
                                 overlap, nn::zeros({}, overlap.dtype()));
    const auto keep =
        two_stage_detection_detail::Indices(nn::less_equal(nn::max(upper, {0}), threshold));
    parts[key.first].push_back(nn::gather(selected, keep));
  }
  std::vector<nn::Tensor> result;
  for (const auto &[image, lists] : parts) {
    const auto ids = nn::concat(lists, 0);
    const auto order = nn::slice_axis(nn::argsort(nn::gather(d.scores, ids), true), 0, 0,
                                      std::min<std::int64_t>(ids.numel(), limit));
    result.push_back(nn::gather(ids, order));
  }
  return result.empty() ? nn::zeros({0}, nn::DType::Int64) : nn::concat(result, 0);
}
/// @brief Decode YOLACT class boxes,Fast NMS and cropped shared-prototype instance masks.
///
/// Background last is excluded. Decode default-variance offsets/log scales
/// clipped4,clip boxes to canvas,discard zero area,retain class score>.05.
/// Fast NMS.5/top200 per class/cap100 per image selects candidates. Combine
/// matching coefficients with same-image prototypes,clip logits[-80,80],
/// sigmoid,crop on prototype pixel centres,no one-pixel box expansion,then
/// bilinearly paste full-canvas probabilities and threshold>=.5. Inference
/// products form in model dtype then sum in Float64,matching the old source
/// accumulation policy. No learned updates,fitting or retained gradients.
///
/// @param net Consistent finite YOLACT probability/delta/coefficient/prototype outputs for rows.
/// @param anchors Same-dtype[A,4] prior boxes in network anchor order.
/// @param rows Nonnegative unique original dataset image IDs in feature-batch order.
/// @param height Positive source scene height.
/// @param width Positive source scene width.
/// @return Detached box detections and Bool[D,height,width] pasted selected masks.
/// @throws std::invalid_argument For invalid settings,shapes,dtypes,nonfinite predictions or out-of-range IDs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::YolactConfig c;
/// c.size = 64;
/// c.classes = 10;
/// c.depth = 18;
/// c.divisor = 8;
/// c.width = 32;
/// c.prototypes = 8;
/// c.grayscale = true;
/// const auto net = n::models::Yolact(p, state, n::zeros({1, 64, 64, 1}), c);
/// const auto masks =
///     n::models::DecodeTensorYolact(net, n::models::YolactAnchors(net, 64, true), {0}, 64, 64);
/// @endcode
/// @par References
/// - Bolya et al.(2019), YOLACT: Real-time Instance Segmentation, https://arxiv.org/abs/1904.02689; teaching pixel-centre crops and mask pasting follow examples/nn/common/yolact.hpp.
/// @see TensorFastNms(), LinearInstanceMasks(), PasteInstanceMask()
inline TensorMaskDetections DecodeTensorYolact(const YolactOutputs &net,
                                               const nn::Tensor &anchors,
                                               const std::vector<std::int64_t> &rows,
                                               int height, int width) {
  if (rows.empty() || height < 1 || width < 1 || net.probabilities.dim() != 3 ||
      net.probabilities.shape()[0] != std::int64_t(rows.size()) ||
      net.probabilities.shape()[2] < 2)
    throw std::invalid_argument("invalid YOLACT decode metadata");
  const auto B = net.probabilities.shape()[0], A = net.probabilities.shape()[1],
             C = net.probabilities.shape()[2] - 1,
             K = net.coefficients.dim() == 3 ? net.coefficients.shape()[2] : 0,
             H = std::int64_t(net.proto_side);
  const auto type = net.probabilities.dtype();
  if (A < 1 || K < 1 || H < 1 || anchors.shape() != nn::Shape{A, 4} ||
      net.deltas.shape() != nn::Shape{B, A, 4} ||
      net.coefficients.shape() != nn::Shape{B, A, K} ||
      net.prototypes.shape() != nn::Shape{B, H, H, K} ||
      (type != nn::DType::Float32 && type != nn::DType::Float64))
    throw std::invalid_argument("inconsistent YOLACT decode tensors");
  for (const auto &x : {anchors, net.deltas, net.coefficients, net.prototypes})
    if (x.dtype() != type ||
        !nn::all(nn::less(nn::abs(x), std::numeric_limits<double>::infinity())).item<bool>())
      throw std::invalid_argument("invalid YOLACT decode values/dtypes");
  if (!nn::all(nn::logical_and(nn::greater_equal(net.probabilities, 0.),
                               nn::less_equal(net.probabilities, 1.)))
           .item<bool>())
    throw std::invalid_argument("invalid YOLACT probabilities");
  auto unique = rows;
  std::sort(unique.begin(), unique.end());
  if (unique[0] < 0 || std::adjacent_find(unique.begin(), unique.end()) != unique.end())
    throw std::invalid_argument("YOLACT rows must be distinct nonnegative IDs");
  nn::NoGradGuard guard;
  const auto decoded =
      nn::clip_boxes(nn::decode_boxes(anchors.reshape({1, A, 4}), net.deltas), height, width);
  const auto boxes = (decoded.reshape({B, A, 1, 4}) * nn::ones({B, A, C, 1}, type))
                         .reshape({B * A * C, 4}),
             scores = nn::slice_axis(net.probabilities, 2, 0, C).reshape({B * A * C});
  const auto labels = (nn::arange(0, double(C), 1., nn::DType::Int64).reshape({1, 1, C}) *
                       nn::ones({B, A, C}, nn::DType::Int64))
                          .reshape({B * A * C}),
             images = (nn::from_vector(rows).reshape({B, 1, 1}) *
                       nn::ones({B, A, C}, nn::DType::Int64))
                          .reshape({B * A * C}),
             source = (nn::arange(0, double(B * A), 1., nn::DType::Int64).reshape({B, A, 1}) *
                       nn::ones({B, A, C}, nn::DType::Int64))
                          .reshape({B * A * C});
  const auto eligible = two_stage_detection_detail::Indices(
      nn::logical_and(nn::greater(scores, .05), nn::greater(nn::box_area(boxes), 0.)));
  const TensorDetections candidates{nn::gather(boxes, eligible), nn::gather(scores, eligible),
                                    nn::gather(labels, eligible),
                                    nn::gather(images, eligible)};
  const auto selected = TensorFastNms(candidates);
  const TensorDetections objects{
      nn::gather(candidates.boxes, selected), nn::gather(candidates.scores, selected),
      nn::gather(candidates.labels, selected), nn::gather(candidates.images, selected)};
  const auto sources = nn::gather(nn::gather(source, eligible), selected);
  std::vector<nn::Tensor> masks;
  const auto py = (nn::arange(0, double(H)).to(type) + .5) * double(height) / double(H),
             px = (nn::arange(0, double(H)).to(type) + .5) * double(width) / double(H);
  for (std::int64_t d = 0; d < selected.numel(); ++d) {
    const auto flat = nn::slice_axis(sources, 0, d, d + 1).item<std::int64_t>(),
               image = flat / A, anchor = flat % A;
    const auto proto = nn::slice_axis(net.prototypes, 0, image, image + 1).reshape({H, H, K}),
               coefficient =
                   nn::slice_axis(nn::slice_axis(net.coefficients, 0, image, image + 1), 1,
                                  anchor, anchor + 1)
                       .reshape({1, 1, K});
    const auto logits = nn::sum((proto * coefficient).to(nn::DType::Float64), {2});
    const auto box = nn::unstack(nn::slice_axis(objects.boxes, 0, d, d + 1).reshape({4}), 0);
    const auto inside =
        nn::logical_and(nn::logical_and(nn::greater_equal(py.reshape({H, 1}), box[0]),
                                        nn::less(py.reshape({H, 1}), box[2])),
                        nn::logical_and(nn::greater_equal(px.reshape({1, H}), box[1]),
                                        nn::less(px.reshape({1, H}), box[3])));
    const auto probabilities =
        nn::sigmoid(nn::clip(logits, -80., 80.)).to(type) * inside.to(type);
    masks.push_back(
        PasteInstanceMask(
            probabilities.reshape({H, H, 1}), 0,
            nn::from_vector(std::vector<double>{0., 0., double(height), double(width)})
                .to(type),
            height, width)
            .reshape({1, height, width}));
  }
  return {objects, masks.empty() ? nn::zeros({0, height, width}, nn::DType::Bool)
                                 : nn::concat(masks, 0)};
}
} // namespace datamunge::nn::models
