#pragma once
/// @file retina_detection_models.hpp
/// @brief Tensor RetinaNet shared towers/pyramid, focal objectives, anchors, targets and decoding.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/feature_pyramid_models.hpp"
#include "datamunge/nn/resnet_models.hpp"
#include "datamunge/nn/ssd_models.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Apply a source RetinaNet3x3 convolution with normal-.01 kernels and optional ReLU.
///
/// Initialization is normal standard deviation.01, distinct from the He
/// backbone/FPN. No normalization, moving state, input mutation or fitting.
/// Keep learned leaves unchanged until backward.
///
/// @param p Mutable Float32/64 learned store; creates <name>/W and /b.
/// @param name Shared parameter prefix; repeated compatible calls reuse leaves.
/// @param x Floating positive NHWC tensor in p dtype.
/// @param out Positive output channel count.
/// @param relu Apply ReLU when true (default), otherwise linear.
/// @param initial_bias Finite bias initializer, default0; reused leaves keep their existing values.
/// @return Differentiable SAME-stride-one3x3 biased convolution, optionally rectified.
/// @throws std::invalid_argument For invalid geometry/dtype/width/bias or incompatible existing leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::RetinaConv(p, "c", n::ones({1, 8, 8, 16}), 32);
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see RetinaNet()
inline nn::Tensor RetinaConv(nn::ParameterStore &p, const std::string &name,
                             const nn::Tensor &x, std::int64_t out, bool relu = true,
                             double initial_bias = 0) {
  if (x.dim() != 4 || x.shape()[0] < 1 || x.shape()[1] < 1 || x.shape()[2] < 1 ||
      x.shape()[3] < 1 || x.dtype() != p.dtype() ||
      (p.dtype() != nn::DType::Float32 && p.dtype() != nn::DType::Float64) || out < 1 ||
      !std::isfinite(initial_bias))
    throw std::invalid_argument("invalid Retina convolution inputs/settings");
  const std::int64_t in = x.shape()[3];
  const auto w = p.get(name + "/W", {3, 3, in, out}, nn::init::normal(.01));
  const auto y = nn::bias_add(nn::conv2d(x, w),
                              p.get(name + "/b", {out}, nn::init::constant(initial_bias)));
  return relu ? nn::relu(y) : y;
}

/// @brief Zero-pad the two NHWC spatial axes symmetrically.
///
/// No parameter/state mutation or fitting. This helper keeps explicit stride-two
/// VALID3x3 padding consistent across P6/P7 branches.
///
/// @param x Rank-four NHWC tensor, any dtype; empty dimensions are allowed by core padding.
/// @param pad Nonnegative number of rows/columns on each spatial edge.
/// @return Tensor with2*pad added to H/W, preserving dtype and floating gradients.
/// @throws std::invalid_argument For invalid rank/pad or overflowing padded dimensions.
/// @par Example
/// @code
/// const auto y = dm::nn::models::RetinaPadSpatial(dm::nn::ones({1, 3, 5, 4}), 1);
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see RetinaPyramid()
inline nn::Tensor RetinaPadSpatial(const nn::Tensor &x, std::int64_t pad) {
  if (x.dim() != 4 || pad < 0)
    throw std::invalid_argument("invalid spatial pad");
  return nn::pad(x, {{0, 0}, {pad, pad}, {pad, pad}, {0, 0}});
}
/// @brief Retain dense sigmoid detector outputs shared by RetinaNet and EfficientDet.
///
/// All classes are foreground sigmoid channels; there is no background score
/// column. Tensor handles retain graphs, while learned leaves/model state remain
/// owned by their parameter stores. Anchor order follows level, row, column, prior.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RetinaNetConfig cfg;
/// cfg.classes = 10;
/// cfg.image_size = 64;
/// cfg.backbone_depth = 18;
/// cfg.width_divisor = 8;
/// cfg.pyramid_width = 32;
/// cfg.grayscale = true;
/// const auto net = n::models::RetinaNet(p, state, n::zeros({1, 64, 64, 1}), cfg);
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see RetinaNet(), RetinaLoss()
struct RetinaNetOutputs {
  std::vector<nn::Tensor> features; ///< Five floating P3–P7 NHWC feature maps.
  std::vector<int> cells;           ///< Actual square grid sides, fine to coarse.
  nn::Tensor logits;                ///< [N,A,C] independent foreground logits.
  nn::Tensor probabilities;         ///< [N,A,C] independent sigmoid probabilities.
  nn::Tensor deltas;                ///< [N,A,4] class-agnostic unit-variance box offsets.
  int anchors = 0;                  ///< Sum of nine anchors per cell across five levels.
};
/// @brief Configure source RetinaNet geometry, shared widths and backbone state updates.
///
/// Full defaults use ResNet50,256-channel pyramid/towers and80 foreground
/// classes. No pretrained weights are implicit. This is metadata, not a fit.
///
/// @par Example
/// @code
/// dm::nn::models::RetinaNetConfig cfg;
/// cfg.image_size = 128;
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see RetinaNet()
struct RetinaNetConfig {
  int classes = 80; ///< Positive number of foreground classes; no explicit background class.
  int image_size = 640;    ///< Square input side, at least32 and divisible by32.
  int backbone_depth = 50; ///< Supported canonical ResNet depth, including18 and50.
  int width_divisor = 1;   ///< Positive divisor of64 for all ResNet widths.
  int pyramid_width = 256; ///< Positive shared FPN/tower width.
  bool grayscale =
      false; ///< True accepts one channel and tiles it to RGB before the backbone.
  bool update_stats = true; ///< Update backbone BN moving state during training when true.
};
/// @brief Build source ResNet C3–C5, linear FPN P3–P5 and C5-derived P6/P7.
///
/// P6 is a padded stride-two3x3 convolution on original C5, not on P5.
/// P7 uses ReLU(P6) and another padded stride-two3x3. Both use He weights;
/// P3–P5 use linear lateral/smoothing FPN. Feature-only ResNet creates no
/// disconnected classifier leaves. No optimizer runs here. Shared stores are
/// not safe for concurrent mutation; keep learned leaf values until backward.
///
/// @param p Mutable learned store; creates canonical feature-only ResNet and fpn leaves.
/// @param state Mutable backbone BN moving-state store, same dtype as p.
/// @param image Floating NHWC RGB or grayscale image with positive batch and H/W at least32, matching p dtype.
/// @param depth Supported ResNet depth.
/// @param width_divisor Positive divisor of64 for backbone widths.
/// @param width Positive pyramid output width.
/// @param grayscale Accept one channel and tile to RGB when true; otherwise require RGB.
/// @param training Use batch statistics when true; false uses moving state.
/// @param update_stats Mutate BN moving state only when training and true.
/// @return Five differentiable feature maps in P3–P7 order, all with requested width.
/// @throws std::invalid_argument For invalid image/settings/dtypes or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto maps =
///     n::models::RetinaPyramid(p, state, n::ones({1, 64, 64, 1}), 18, 8, 32, true, false, true);
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see RetinaNet(), FeaturePyramid()
inline std::vector<nn::Tensor> RetinaPyramid(nn::ParameterStore &p, nn::ParameterStore &state,
                                             const nn::Tensor &image, int depth,
                                             int width_divisor, std::int64_t width,
                                             bool grayscale, bool training,
                                             bool update_stats) {
  if (image.dim() != 4 || image.shape()[0] < 1 || image.shape()[1] < 32 ||
      image.shape()[2] < 32 || image.shape()[3] != (grayscale ? 1 : 3) ||
      image.dtype() != p.dtype() || state.dtype() != p.dtype() || width < 1)
    throw std::invalid_argument("invalid Retina pyramid image/configuration");
  ResNetConfig backbone;
  backbone.depth = depth;
  backbone.width_divisor = width_divisor;
  backbone.include_classifier = false;
  backbone.stride_on_3x3 = true;
  backbone.update_stats = update_stats;
  const auto net =
      ResNet(p, state, grayscale ? nn::tile(image, {1, 1, 1, 3}) : image, backbone, training);
  auto features =
      FeaturePyramid(p, "fpn", {net.stages[1], net.stages[2], net.stages[3]}, width, false)
          .levels;
  nn::ConvLayerOptions down;
  down.stride = 2;
  down.padding = nn::Padding::Valid;
  down.init = nn::init::he_uniform();
  features.push_back(
      nn::conv2d(p, "fpn/p6", RetinaPadSpatial(net.stages[3], 1), width, 3, down));
  features.push_back(
      nn::conv2d(p, "fpn/p7", RetinaPadSpatial(nn::relu(features.back()), 1), width, 3, down));
  return features;
}

/// @brief Build source RetinaNet with shared four-convolution class/regression towers.
///
/// All five levels share each four-layer3x3/ReLU subnet and its predictor.
/// Subnet kernels use normal stddev.01; classification bias is logit(.01).
/// Full ResNet50/256/80-class model has37968692 learned scalars and53120
/// BN-state scalars independent of image side. At640 pixels there are76725
/// anchors; the128-pixel full-weight gradient probe has3069. No pretraining
/// or optimizer is hidden. Keep learned leaves unchanged until backward.
///
/// @param p Mutable Float32/64 learned store; warm every leaf before optimizer construction.
/// @param state Mutable same-dtype BN moving-state store; checkpoint with p.
/// @param image Floating [N,cfg.image_size,cfg.image_size,1 or3] according to grayscale flag.
/// @param cfg Valid class count, geometry, backbone widths and state-update policy.
/// @param training True fits through batch statistics and optional moving updates; false(default) is inference.
/// @return P3–P7 maps, nine-anchor-per-cell sigmoid scores and unit-variance box offsets.
/// @throws std::invalid_argument For invalid cfg/input/dtypes or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::RetinaNetConfig cfg;
/// cfg.classes = 10;
/// cfg.image_size = 64;
/// cfg.backbone_depth = 18;
/// cfg.width_divisor = 8;
/// cfg.pyramid_width = 32;
/// cfg.grayscale = true;
/// const auto net = n::models::RetinaNet(p, state, n::zeros({1, 64, 64, 1}), cfg);
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see RetinaAnchors(), RetinaLoss(), DecodeRetinaDetections()
inline RetinaNetOutputs RetinaNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                  const nn::Tensor &image, const RetinaNetConfig &cfg = {},
                                  bool training = false) {
  if (cfg.classes < 1 || cfg.classes > std::numeric_limits<int>::max() / 9 ||
      cfg.image_size < 32 || cfg.image_size % 32 || cfg.pyramid_width < 1 ||
      cfg.width_divisor < 1 || 64 % cfg.width_divisor || image.dim() != 4 ||
      image.shape()[0] < 1 || image.shape()[1] != cfg.image_size ||
      image.shape()[2] != cfg.image_size || image.shape()[3] != (cfg.grayscale ? 1 : 3) ||
      image.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("invalid RetinaNet configuration");
  RetinaNetOutputs result;
  result.features =
      RetinaPyramid(p, state, image, cfg.backbone_depth, cfg.width_divisor, cfg.pyramid_width,
                    cfg.grayscale, training, cfg.update_stats);
  std::vector<nn::Tensor> logits, deltas;
  for (int level = 0; level < 5; ++level) {
    const int stride = 8 << level, cells = (cfg.image_size + stride - 1) / stride,
              count = cells * cells * 9;
    if (result.features[std::size_t(level)].shape()[1] != cells)
      throw std::invalid_argument("RetinaNet image_size does not match the input images");
    result.cells.push_back(cells);
    result.anchors += count;
    nn::Tensor classification = result.features[std::size_t(level)],
               regression = classification;
    for (int layer = 0; layer < 4; ++layer) {
      classification = RetinaConv(p, "classification/conv" + std::to_string(layer + 1),
                                  classification, cfg.pyramid_width);
      regression = RetinaConv(p, "regression/conv" + std::to_string(layer + 1), regression,
                              cfg.pyramid_width);
    }
    logits.push_back(nn::reshape(RetinaConv(p, "classification/output", classification,
                                            9 * cfg.classes, false, std::log(.01 / .99)),
                                 {-1, count, cfg.classes}));
    deltas.push_back(nn::reshape(RetinaConv(p, "regression/output", regression, 36, false),
                                 {-1, count, 4}));
  }
  result.logits = nn::concat(logits, 1);
  result.deltas = nn::concat(deltas, 1);
  result.probabilities = nn::sigmoid(result.logits);
  return result;
}

/// @brief Evaluate elementwise binary sigmoid focal loss with log-space modulation.
///
/// Modulation avoids explicitly forming powers of1-p. Very confident correct
/// class losses/gradients can still round to zero through floating BCE cancellation
/// or exponential underflow. No mining, reduction, mutation or fitting; logits
/// retain gradients through the modulator as well as BCE.
///
/// @param logits Float32/64 logits, arbitrary shape, finite numerical values are caller requirement.
/// @param target Binary0/1 tensor, same dtype, broadcastable to logits.
/// @param alpha Finite positive-class weight in[0,1], default.25; negatives use1-alpha.
/// @param gamma Finite nonnegative focusing exponent, default2; zero gives weighted BCE.
/// @return Broadcast-shaped alpha_t*exp(-gamma*softplus((2y-1)*logit))*(softplus(logit)-logit*y).
/// @throws std::invalid_argument For invalid dtype/targets, nonbroadcastable shapes or invalid alpha/gamma.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss =
///     n::models::SigmoidFocalLoss(n::tensor({-2., 0., 2.}), n::tensor({0., 1., 1.}), .25, 2.);
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see RetinaLoss()
inline nn::Tensor SigmoidFocalLoss(const nn::Tensor &logits, const nn::Tensor &target,
                                   double alpha = .25, double gamma = 2) {
  if (!(alpha >= 0 && alpha <= 1) || !(gamma >= 0) || !std::isfinite(gamma))
    throw std::invalid_argument("invalid focal-loss parameters");
  if ((logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      target.dtype() != logits.dtype() ||
      !nn::all(nn::logical_or(nn::equal(target, 0.), nn::equal(target, 1.))).item<bool>())
    throw std::invalid_argument("binary same-dtype focal targets required");
  const auto at = alpha * target + (1 - alpha) * (1.0 - target);
  const auto bce = nn::softplus(logits) - logits * target;
  const auto sign = target * 2.0 - 1.0;
  const auto modulator = nn::exp(-gamma * nn::softplus(sign * logits));
  return at * modulator * bce;
}

/// @brief Own dense sigmoid labels, masks and unit-variance regression targets.
///
/// Detached builder outputs use floating scene dtype. Background has all-zero
/// class targets; ignored anchors have zero classification weight. No fitted state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::RetinaTargets t{n::zeros({1, 3, 2}), n::zeros({1, 3}), n::ones({1, 3}),
///                            n::zeros({1, 3, 4})};
/// const auto loss = n::models::RetinaLoss(n::zeros({1, 3, 2}), n::zeros({1, 3, 4}), t);
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see MakeRetinaTargets(), RetinaLoss()
struct RetinaTargets {
  nn::Tensor classes;  ///< [N,A,C] binary foreground class targets, all-zero background.
  nn::Tensor positive; ///< [N,A] floating positive/regression mask.
  nn::Tensor weights;  ///< [N,A] classification mask, zero for ignored anchors.
  nn::Tensor boxes;    ///< [N,A,4] unit-variance encoded offsets, zero for nonpositives.
};
/// @brief Sum masked sigmoid focal loss and beta-one-ninth positive box Smooth-L1.
///
/// Smooth-L1 uses .5*d²/beta below abs(d)<beta, abs(d)-.5*beta otherwise,
/// with beta=1/9. No-positive batches still learn from weighted negative focal
/// loss, unlike SSD mining. All-ignored classification weights suppress focal
/// contributions. No mutation, optimizer or hidden target generation.
///
/// @param logits Float32/64 [N,A,C], N/C positive.
/// @param deltas Floating [N,A,4] unit-variance predictions, same dtype.
/// @param t Same-dtype consistent targets/masks; finite binary class targets and nonnegative finite masks are caller requirements.
/// @return Scalar focal plus box loss divided by max(total positive mask sum,1).
/// @throws std::invalid_argument For malformed shapes/dtypes or invalid focal class targets.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::RetinaTargets t{n::zeros({1, 3, 2}), n::zeros({1, 3}), n::ones({1, 3}),
///                            n::zeros({1, 3, 4})};
/// const auto loss = n::models::RetinaLoss(n::zeros({1, 3, 2}), n::zeros({1, 3, 4}), t);
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see SigmoidFocalLoss(), MakeRetinaTargets()
inline nn::Tensor RetinaLoss(const nn::Tensor &logits, const nn::Tensor &deltas,
                             const RetinaTargets &t) {
  if (logits.dim() != 3 || logits.shape()[0] < 1 || logits.shape()[2] < 1 ||
      deltas.shape() != nn::Shape{logits.shape()[0], logits.shape()[1], 4} ||
      deltas.dtype() != logits.dtype() || t.classes.shape() != logits.shape() ||
      t.positive.shape() != nn::Shape{logits.shape()[0], logits.shape()[1]} ||
      t.weights.shape() != t.positive.shape() || t.boxes.shape() != deltas.shape())
    throw std::invalid_argument("invalid Retina loss shapes/dtypes");
  for (const auto &x : {t.classes, t.positive, t.weights, t.boxes})
    if (x.dtype() != logits.dtype())
      throw std::invalid_argument("Retina target dtype mismatch");
  const auto focal = nn::sum(nn::sum(SigmoidFocalLoss(logits, t.classes), {2}) * t.weights);
  const auto difference = deltas - t.boxes;
  const auto distance = nn::abs(difference);
  const double beta = 1. / 9;
  const auto smooth = nn::where(nn::less(distance, beta), nn::square(difference) * (.5 / beta),
                                distance - .5 * beta);
  const auto regression = nn::sum(nn::sum(smooth, {2}) * t.positive);
  return (focal + regression) / nn::maximum(nn::sum(t.positive), nn::ones({}, logits.dtype()));
}

/// @brief Construct nine RetinaNet pixel anchors at every cell of five P3–P7 grids.
///
/// Strides8/16/32/64/128, doubling base side per level, scales1/cuberoot(2)/
/// cuberoot(4) and ratios.5/1/2 follow the source. Shapes are not clipped.
/// All scale, cube-root and coordinate arithmetic uses tensors; no fitting.
///
/// @param cells Exactly five positive square feature sides.
/// @param first_size Finite positive base side at P3, default32; compact128 uses16.
/// @param dtype Float32 by default or Float64 for geometry arithmetic.
/// @return Detached [9*sum(cells²),4] pixel boxes ordered level,row,column,scale,ratio(height/width).
/// @throws std::invalid_argument For invalid grid metadata, base side or dtype.
/// @par Example
/// @code
/// const auto a = dm::nn::models::RetinaAnchors({16, 8, 4, 2, 1}, 16.); // [3069,4]
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see MakeRetinaTargets(), RetinaNet()
inline nn::Tensor RetinaAnchors(const std::vector<int> &cells, double first_size = 32.,
                                nn::DType dtype = nn::DType::Float32) {
  if (cells.size() != 5 || !std::isfinite(first_size) || first_size <= 0 ||
      (dtype != nn::DType::Float32 && dtype != nn::DType::Float64))
    throw std::invalid_argument("invalid Retina anchor metadata");
  nn::NoGradGuard guard;
  std::vector<nn::Tensor> all;
  const auto scales =
                 nn::pow(nn::from_vector(std::vector<double>{1, 2, 4}).to(dtype), 1. / 3.),
             ratios = nn::from_vector(std::vector<double>{.5, 1, 2}).to(dtype);
  for (int l = 0; l < 5; ++l) {
    if (cells[l] < 1)
      throw std::invalid_argument("Retina grids must be positive");
    const auto C = std::int64_t(cells[l]);
    const auto size =
                   nn::full({}, first_size, dtype) * double(1 << l) * scales.reshape({3, 1}),
               root = nn::sqrt(ratios).reshape({1, 3});
    const auto h = (size * root).reshape({1, 1, 9}), w = (size / root).reshape({1, 1, 9});
    const auto position = (nn::arange(0, double(C)).to(dtype) + .5) * double(8 << l);
    const auto yy = position.reshape({C, 1, 1}) * nn::ones({C, C, 9}, dtype),
               xx = position.reshape({1, C, 1}) * nn::ones({C, C, 9}, dtype);
    all.push_back(nn::stack({yy - h * .5, xx - w * .5, yy + h * .5, xx + w * .5}, 3)
                      .reshape({C * C * 9, 4}));
  }
  return nn::concat(all, 0);
}
/// @brief Assign sigmoid classes and unit box targets with SSD forced matching and an ignored IoU band.
///
/// Distinct greedy SSD matching first forces one anchor per target, then adds
/// IoU>=.5 matches. Forced matches remain positive regardless of overlap.
/// Unmatched anchors whose maximum IoU>=.4 are ignored; others are weighted
/// background. Thus the usual ignored band is[.4,.5). Encoding uses variances
/// 1/1/1/1. Empty truth gives all weighted negatives. No fitting or random draws.
///
/// @param anchors Floating [A,4] finite positive-area pixel anchors in scene dtype, at least one per selected truth.
/// @param scenes Valid floating grayscale scene metadata; pixel values are not read.
/// @param rows Source scene IDs in prediction order.
/// @param classes Positive number of foreground classes; every selected label is smaller.
/// @return Detached class/positive/weight/offset tensors aligned with batch and anchor order.
/// @throws std::invalid_argument For invalid geometry/dtypes/rows/classes or insufficient anchors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetectionSceneConfig cfg;
/// cfg.scenes = 1;
/// cfg.max_objects = 1;
/// const auto scenes = n::models::MakeTensorDetectionScenes(n::ones({1, 28, 28, 1}),
///                                                          n::zeros({1}, n::DType::Int64), cfg);
/// const auto targets = n::models::MakeRetinaTargets(
///     n::models::RetinaAnchors({8, 4, 2, 1, 1}, 8.), scenes, {0}, 10);
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see RetinaLoss(), MatchSsd()
inline RetinaTargets MakeRetinaTargets(const nn::Tensor &anchors,
                                       const TensorDetectionScenes &scenes,
                                       const std::vector<std::int64_t> &rows, int classes) {
  two_stage_detection_detail::ValidateScenes(scenes, rows, classes);
  if (anchors.dim() != 2 || anchors.shape()[1] != 4 ||
      anchors.dtype() != scenes.images.dtype())
    throw std::invalid_argument("invalid Retina target anchors");
  SquareDetectionBoxes(anchors, 1.);
  if (nn::any(nn::less_equal(
                  nn::slice_axis(anchors, 1, 2, 4) - nn::slice_axis(anchors, 1, 0, 2), 0.))
          .item<bool>())
    throw std::invalid_argument("Retina anchors need positive extents");
  nn::NoGradGuard guard;
  const auto A = anchors.shape()[0], N = std::int64_t(rows.size());
  const auto type = anchors.dtype();
  std::vector<nn::Tensor> labels, positive, weights, boxes;
  for (auto row : rows) {
    const auto &truth = scenes.boxes[row];
    nn::Tensor hot, mask, weight, target;
    if (truth.shape()[0]) {
      const auto overlap = nn::box_iou(anchors, truth), match = MatchSsd(overlap),
                 best = nn::max(overlap, {1});
      const auto pos = nn::greater_equal(match, 0.);
      const auto safe = nn::maximum(match, nn::zeros({}, nn::DType::Int64));
      mask = pos.to(type);
      hot = nn::one_hot(nn::gather(scenes.labels[row], safe), classes, 1, 0, type) *
            mask.reshape({A, 1});
      weight = nn::logical_or(pos, nn::less(best, .4)).to(type);
      target = nn::where(pos.reshape({A, 1}),
                         nn::encode_boxes(anchors, nn::gather(truth, safe), {1, 1, 1, 1}),
                         nn::zeros({}, type));
    } else {
      hot = nn::zeros({A, classes}, type);
      mask = nn::zeros({A}, type);
      weight = nn::ones({A}, type);
      target = nn::zeros({A, 4}, type);
    }
    labels.push_back(hot);
    positive.push_back(mask);
    weights.push_back(weight);
    boxes.push_back(target);
  }
  return {labels.empty() ? nn::zeros({N, A, classes}, type) : nn::stack(labels, 0),
          positive.empty() ? nn::zeros({N, A}, type) : nn::stack(positive, 0),
          weights.empty() ? nn::zeros({N, A}, type) : nn::stack(weights, 0),
          boxes.empty() ? nn::zeros({N, A, 4}, type) : nn::stack(boxes, 0)};
}
/// @brief Decode independent foreground sigmoid scores with unit box coding and optional final NMS.
///
/// Composes the tensor SSD candidate decoder with unit-variance offset conversion
/// and an unscored dummy final column; no foreground probabilities are normalized
/// or changed. Every class is independent. No fitting or model/state mutation.
///
/// @param probabilities Floating [N,A,C] independent foreground scores, all C channels considered.
/// @param deltas Floating [N,A,4] unit-variance class-agnostic offsets, same dtype.
/// @param anchors Floating [A,4] finite ordered pixel boxes in the same dtype.
/// @param images Int64 [N] nonnegative dataset image IDs.
/// @param height Positive image height for clipping.
/// @param width Positive image width for clipping.
/// @param suppress True(default) applies IoU.5 class-wise NMS and100/image; false returns pre-NMS top200/class candidates.
/// @return Detached clipped positive-area tensor detections, strict score cutoff.01 and source higher-anchor-index score ties.
/// @throws std::invalid_argument For inconsistent shapes/dtypes or invalid candidate values/IDs/image dimensions.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::DecodeRetinaDetections(n::full({1, 1, 2}, .1), n::zeros({1, 1, 4}),
///                                                  n::tensor({0., 0., 10., 10.}, {1, 4}),
///                                                  n::zeros({1}, n::DType::Int64), 64, 64);
/// @endcode
/// @par References
/// - Lin et al. (2017), Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002
/// @see RetinaNet(), DecodeSsdDetections()
inline TensorDetections DecodeRetinaDetections(const nn::Tensor &probabilities,
                                               const nn::Tensor &deltas,
                                               const nn::Tensor &anchors,
                                               const nn::Tensor &images, int height, int width,
                                               bool suppress = true) {
  if (probabilities.dim() != 3)
    throw std::invalid_argument("Retina scores require [N,A,C]");
  nn::NoGradGuard guard;
  const auto N = probabilities.shape()[0], A = probabilities.shape()[1];
  const auto type = probabilities.dtype();
  const auto dummy = nn::zeros({N, A, 1}, type);
  const auto variances = nn::from_vector(std::vector<double>{.1, .1, .2, .2}).to(type);
  return DecodeSsdDetections(nn::concat({probabilities, dummy}, 2), deltas / variances,
                             anchors, images, height, width, 200, .01, .5, 100, suppress);
}
} // namespace datamunge::nn::models
