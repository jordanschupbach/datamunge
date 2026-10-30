#pragma once
/// @file yolo_v2_models.hpp
/// @brief Tensor YOLO v2/9000 Darknet passthrough, IoU dimension clustering, targets and joint hierarchy objectives.
#include "datamunge/nn/retina_detection_models.hpp"
#include "datamunge/nn/word_tree_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Configure source Darknet19 YOLO v2 and optional9418-node YOLO9000 head.
///
/// Full RGB defaults use20 classes/five anchors;9418 classes/three anchors
/// build the YOLO9000 head. Compact grayscale removes the first two pools.
/// No pretrained ImageNet weights or fitting are implicit.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::YoloV2Config c;
/// c.classes = 10;
/// c.compact = true;
/// c.width_divisor = 8;
/// const auto net = n::models::YoloV2(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see YoloV2()
struct YoloV2Config {
  int classes = 20;         ///< Positive flat class or WordTree node count.
  int anchors = 5;          ///< Positive number of dimension priors per cell.
  int width_divisor = 1;    ///< Positive divisor of32 for convolution widths.
  bool compact = false;     ///< True accepts grayscale and uses stride8 instead of32.
  double momentum = .9;     ///< BN moving momentum in[0,1), epsilon1e-5.
  bool update_stats = true; ///< Update all BN moving state only in training when true.
};
/// @brief Retain YOLO v2 fine route, reorganized passthrough, fusion map and raw anchors.
///
/// Tensor graphs borrow learned leaves from p. BN state remains owned by the
/// separate state store; no optimizer or target creation occurs.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::YoloV2Config c;
/// c.classes = 10;
/// c.compact = true;
/// c.width_divisor = 8;
/// const auto net = n::models::YoloV2(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see YoloV2()
struct YoloV2Outputs {
  nn::Tensor fine; ///< Fine NHWC backbone route at stride16 full/4 compact.
  nn::Tensor
      passthrough;     ///< Reduced route after2x2 space-to-depth, same grid as final backbone.
  nn::Tensor features; ///< Fused NHWC detection feature map,1024/width_divisor channels.
  nn::Tensor raw;      ///< [N,G,G,A,5+C] dy,dx,logh,logw,objectness and class logits.
};
/// @brief Apply source symmetric-padded convolution, scaled BN and leaky-ReLU slope.1.
///
/// He kernel initialization and symmetric floor(kernel/2) padding match source.
/// No fitting or optimizer; keep learned values unchanged until backward.
///
/// @param p Mutable same-dtype learned kernel/BN affine store.
/// @param state Mutable same-dtype BN moving-state store.
/// @param name Compatible shared convolution prefix.
/// @param x Positive Float32/64 NHWC features matching p dtype.
/// @param out Positive output width.
/// @param kernel Positive odd square kernel.
/// @param training Use batch rather than moving BN statistics when true.
/// @param momentum Moving-state momentum in[0,1); epsilon is1e-5.
/// @param update_stats Update moving state only when training and true.
/// @param stride Positive common spatial stride, default1.
/// @return Differentiable NHWC features; no convolution bias before BN.
/// @throws std::invalid_argument For invalid geometry/dtypes/norm settings or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y =
///     n::models::DarknetConv(p, state, "d", n::ones({2, 8, 8, 4}), 8, 3, true, .9, true);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see YoloV2(), YoloV3()
inline nn::Tensor DarknetConv(nn::ParameterStore &p, nn::ParameterStore &state,
                              const std::string &name, const nn::Tensor &x, std::int64_t out,
                              std::int64_t kernel, bool training, double momentum,
                              bool update_stats, std::int64_t stride = 1) {
  ResNetNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.momentum = momentum;
  norm.epsilon = 1e-5;
  norm.update_stats = update_stats;
  return nn::leaky_relu(ResNetConv(p, state, name, x, out, kernel, norm, stride, false), .1);
}

/// @brief Build Darknet19 feature convolutions, space-to-depth passthrough and YOLO anchor head.
///
/// All18 Darknet feature convolutions remain, followed by two extras, reduced
/// fine passthrough, fusion and biased linear prediction. BN has scale/beta,
/// momentum.9/eps1e-5 by default. Full20-class/five-prior count50655389, full
/// 9418-node/three-prior count79502989; both have20672 state scalars. Compact
/// stride8 supports64/80/96-pixel multiscale fitting. No pretrained weights or
/// optimizer are hidden; keep leaves unchanged until backward.
///
/// @param p Mutable Float32/64 learned store, warm every leaf before optimizer construction.
/// @param state Mutable same-dtype BN state store; checkpoint with p.
/// @param image Positive floating square NHWC: RGB multiple32 (>=32), compact gray multiple8 (>=8).
/// @param cfg Valid class/anchor/width/normalization metadata.
/// @param training False(default) uses moving BN; true uses batch stats and optional updates.
/// @return Fine/reorganized/fused maps and raw [N,G,G,A,5+C] predictions.
/// @throws std::invalid_argument For invalid cfg/input/dtype/geometry or incompatible existing leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::YoloV2Config c;
/// c.classes = 10;
/// c.compact = true;
/// c.width_divisor = 8;
/// const auto net = n::models::YoloV2(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see MakeYoloV2Targets(), YoloV2Loss(), YoloV2Probabilities()
inline YoloV2Outputs YoloV2(nn::ParameterStore &p, nn::ParameterStore &state,
                            const nn::Tensor &image, const YoloV2Config &cfg = {},
                            bool training = false) {
  if (cfg.classes < 1 || cfg.anchors < 1 || cfg.width_divisor < 1 || 32 % cfg.width_divisor ||
      !(cfg.momentum >= 0 && cfg.momentum < 1) ||
      cfg.classes > std::numeric_limits<int>::max() - 5 ||
      cfg.anchors > std::numeric_limits<int>::max() / (cfg.classes + 5) || image.dim() != 4 ||
      image.shape()[0] < 1 || image.shape()[1] < (cfg.compact ? 8 : 32) ||
      image.shape()[1] % (cfg.compact ? 8 : 32) || image.shape()[2] != image.shape()[1] ||
      image.shape()[3] != (cfg.compact ? 1 : 3) || image.dtype() != p.dtype() ||
      state.dtype() != p.dtype())
    throw std::invalid_argument("invalid YOLO v2 configuration");
  const std::vector<std::vector<std::pair<int, int>>> stages = {
      {{32, 3}},
      {{64, 3}},
      {{128, 3}, {64, 1}, {128, 3}},
      {{256, 3}, {128, 1}, {256, 3}},
      {{512, 3}, {256, 1}, {512, 3}, {256, 1}, {512, 3}},
      {{1024, 3}, {512, 1}, {1024, 3}, {512, 1}, {1024, 3}}};
  YoloV2Outputs result;
  nn::Tensor h = image;
  std::int64_t in = 0;
  int index = 0;
  /// @brief Borrow cfg/p/state to apply shared source norm settings during this forward.
  auto conv = [&](const std::string &name, const nn::Tensor &x, std::int64_t out,
                  std::int64_t kernel) {
    return DarknetConv(p, state, name, x, out, kernel, training, cfg.momentum,
                       cfg.update_stats);
  };
  for (std::size_t stage = 0; stage < stages.size(); ++stage) {
    for (const auto &[channels, kernel] : stages[stage]) {
      in = channels / cfg.width_divisor;
      h = conv("backbone/conv" + std::to_string(++index), h, in, kernel);
    }
    if (stage == 4)
      result.fine = h;
    if (stage < 5 && (!cfg.compact || stage >= 2))
      h = nn::max_pool2d(h, 2, 2, nn::Padding::Valid);
  }
  h = conv("extra1", h, in, 3);
  h = conv("extra2", h, in, 3);
  const auto route = conv("route", result.fine, 64 / cfg.width_divisor, 1);
  result.passthrough = nn::space_to_depth(route, 2);
  h = conv("fusion", nn::concat({h, result.passthrough}, 3), in, 3);
  result.features = h;
  nn::ConvLayerOptions linear;
  linear.init = nn::init::he_uniform();
  const auto prediction =
      nn::conv2d(p, "prediction", h, cfg.anchors * (5 + cfg.classes), 1, linear);
  const auto &s = prediction.shape();
  result.raw = nn::reshape(prediction, {s[0], s[1], s[2], cfg.anchors, 5 + cfg.classes});
  return result;
}

/// @brief Compute shape-only IoU of positive height/width dimension pairs.
///
/// Shapes align their centres; units cancel and no position enters IoU.
/// Numerical domain validity is a caller requirement. Gradients are retained.
/// No fitting or mutation.
///
/// @param a Float32/64 [...,2] finite positive height,width dimensions.
/// @param b Same-dtype broadcastable positive dimension pairs.
/// @return Broadcast [...] min heights*min widths divided by shape union.
/// @throws std::invalid_argument For invalid rank/last dimension/dtype or incompatible broadcasts.
/// @par Example
/// @code
/// const auto iou =
///     dm::nn::models::YoloDimensionIoU(dm::nn::tensor({2., 4.}), dm::nn::tensor({4., 2.}));
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see ClusterYoloDimensions()
inline nn::Tensor YoloDimensionIoU(const nn::Tensor &a, const nn::Tensor &b) {
  if (a.dim() < 1 || b.dim() < 1 || a.shape().back() != 2 || b.shape().back() != 2 ||
      a.dtype() != b.dtype() ||
      (a.dtype() != nn::DType::Float32 && a.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid dimension IoU shapes/dtypes");
  const auto aa = nn::unstack(a, -1), bb = nn::unstack(b, -1);
  const auto intersection = nn::minimum(aa[0], bb[0]) * nn::minimum(aa[1], bb[1]);
  return intersection / (aa[0] * aa[1] + bb[0] * bb[1] - intersection);
}
/// @brief Cluster positive dimension samples with source IoU-distance assignments and arithmetic-mean updates.
///
/// Farthest-point seeding starts at sample0 and maximizes nearest1-IoU distance,
/// first sample on ties. Up to100 iterations choose highest-IoU centre, first
/// on ties; means accumulate in Double and cast back to sample dtype. Empty
/// clusters retain centres. Stops at max absolute centre change<1e-5. All
/// arithmetic/reductions/ranking use tensors; only cluster/iteration metadata
/// loops are host. This source preprocessing algorithm has no gradient optimizer.
/// No model weights/state mutation or randomness.
///
/// @param samples Float32/64 [M,2] finite positive height,width samples from fitting data only.
/// @param count Positive cluster count no larger than M.
/// @return Detached [count,2] centres sorted by area, stable first-cluster ties.
/// @throws std::invalid_argument For invalid count/shape/dtype or nonfinite/nonpositive samples.
/// @par Example
/// @code
/// const auto c = dm::nn::models::ClusterYoloDimensions(
///     dm::nn::tensor({1., 1., 1.1, 1.2, 4., 4., 4.2, 3.9}, {4, 2}), 2);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see YoloDimensionIoU(), MakeYoloV2Targets()
inline nn::Tensor ClusterYoloDimensions(const nn::Tensor &samples, int count) {
  if (samples.dim() != 2 || samples.shape()[1] != 2 || count < 1 ||
      samples.shape()[0] < count ||
      (samples.dtype() != nn::DType::Float32 && samples.dtype() != nn::DType::Float64) ||
      !nn::all(nn::logical_and(
                   nn::greater(samples, 0.),
                   nn::less(nn::abs(samples), std::numeric_limits<double>::infinity())))
           .item<bool>())
    throw std::invalid_argument("invalid dimension clustering samples/count");
  nn::NoGradGuard guard;
  const auto M = samples.shape()[0];
  const auto type = samples.dtype();
  auto centres = nn::slice_axis(samples, 0, 0, 1);
  for (int k = 1; k < count; ++k) {
    const auto distance =
        1. -
        nn::max(YoloDimensionIoU(samples.reshape({M, 1, 2}), centres.reshape({1, k, 2})), {1});
    const auto next = nn::argmax(distance).reshape({1});
    centres = nn::concat({centres, nn::gather(samples, next)}, 0);
  }
  for (int iteration = 0; iteration < 100; ++iteration) {
    const auto labels = nn::argmax(
        YoloDimensionIoU(samples.reshape({M, 1, 2}), centres.reshape({1, count, 2})), 1);
    const auto hot = nn::one_hot(labels, count, 1, 0, nn::DType::Float64),
               used = nn::sum(hot, {0}).reshape({count, 1});
    const auto mean = nn::matmul(hot, samples.to(nn::DType::Float64), true, false) /
                      nn::maximum(used, nn::ones({}, nn::DType::Float64));
    const auto next = nn::where(nn::greater(used, 0.), mean, centres.to(nn::DType::Float64))
                          .to(type),
               change = nn::max(nn::abs(next - centres));
    centres = next;
    if (nn::less(change, 1e-5).item<bool>())
      break;
  }
  const auto parts = nn::unstack(centres, 1);
  return nn::gather(centres, nn::argsort(parts[0] * parts[1]));
}
/// @brief Decode source sigmoid centres and exponential prior-relative sizes before clipping.
///
/// Centre=(grid+sigmoid(raw centre))/G*image_side; size=prior*exp(clip(raw size,
/// -10,10))/G*image_side. No model mutation, target matching or fitting.
///
/// @param raw Floating [N,G,G,A,5+C] anchor outputs with positive dimensions and C>=1.
/// @param priors Same-dtype [A,2] positive finite height,width in feature-grid units.
/// @param height Positive pixel image height.
/// @param width Positive pixel image width.
/// @return Differentiable [N,G,G,A,4] unclipped pixel corner boxes.
/// @throws std::invalid_argument For invalid shapes/dtypes/dimensions or nonpositive/nonfinite priors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto b =
///     n::models::DecodeYoloV2Boxes(n::zeros({1, 2, 2, 3, 15}), n::ones({3, 2}), 64, 64);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see MakeYoloV2Targets(), DecodeYoloV2TensorDetections()
inline nn::Tensor DecodeYoloV2Boxes(const nn::Tensor &raw, const nn::Tensor &priors,
                                    int height, int width) {
  if (raw.dim() != 5 || raw.shape()[0] < 1 || raw.shape()[1] < 1 ||
      raw.shape()[2] != raw.shape()[1] || raw.shape()[3] < 1 || raw.shape()[4] < 6 ||
      priors.shape() != nn::Shape{raw.shape()[3], 2} || priors.dtype() != raw.dtype() ||
      height < 1 || width < 1 ||
      !nn::all(
           nn::logical_and(nn::greater(priors, 0.),
                           nn::less(nn::abs(priors), std::numeric_limits<double>::infinity())))
           .item<bool>())
    throw std::invalid_argument("invalid YOLO anchor geometry/priors");
  const auto G = raw.shape()[1], A = raw.shape()[3];
  const auto type = raw.dtype();
  const auto d = nn::unstack(nn::slice_axis(raw, 4, 0, 4), 4), p = nn::unstack(priors, 1);
  const auto row = nn::arange(0, double(G)).to(type).reshape({1, G, 1, 1}),
             col = nn::arange(0, double(G)).to(type).reshape({1, 1, G, 1}),
             y = (row + nn::sigmoid(d[0])) * (double(height) / double(G)),
             x = (col + nn::sigmoid(d[1])) * (double(width) / double(G)),
             h = p[0].reshape({1, 1, 1, A}) * nn::exp(nn::clip(d[2], -10., 10.)) *
                 (double(height) / double(G)),
             w = p[1].reshape({1, 1, 1, A}) * nn::exp(nn::clip(d[3], -10., 10.)) *
                 (double(width) / double(G));
  return nn::stack({y - h * .5, x - w * .5, y + h * .5, x + w * .5}, 4);
}
/// @brief Own detached YOLO v2 anchor geometry, confidence, masks and labels.
///
/// Targets align with raw prediction batch/grid/prior order. No learned
/// parameters, moving state or fitting. Confidence targets are current IoUs.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::tensor({8., 8., 24., 24.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// const auto t =
///     n::models::MakeYoloV2Targets(n::zeros({1, 8, 8, 5, 15}), n::ones({5, 2}), data, {0});
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see MakeYoloV2Targets(), YoloV2Loss()
struct YoloV2Targets {
  nn::Tensor geometry;   ///< [N,G,G,A,4] dy,dx,log(h/prior_h),log(w/prior_w).
  nn::Tensor positive;   ///< [N,G,G,A] floating retained-target0/1 mask.
  nn::Tensor confidence; ///< [N,G,G,A] detached current matched IoU, zero elsewhere.
  nn::Tensor weights;    ///< [N,G,G,A]1 for positives,.5 backgrounds,0 ignored backgrounds.
  nn::Tensor labels;     ///< Int64 [N,G,G,A] class or ancestor IDs, zero for background.
  std::int64_t collisions = 0; ///< Truth count discarded by same-cell/same-prior collisions.
};
/// @brief Assign each target to its best shape prior and centre cell, with detached prediction-IoU confidence.
///
/// Every decoded unmatched prediction whose maximum truth IoU>.6 is ignored.
/// Positive slots override ignore weights to1; other backgrounds weight.5.
/// Target prior/cell ties choose first, and collisions keep larger grid-unit area.
/// Confidence is the current unclipped prediction IoU with its matched truth.
/// No fitting, random draws or host numerical box arithmetic.
///
/// @param raw Floating [N,G,G,A,5+C] predictions, detached internally.
/// @param priors Same-dtype finite positive [A,2] grid-unit dimensions.
/// @param scenes Valid same-dtype grayscale scene metadata, in-image truth boxes; pixels are not read.
/// @param rows Source IDs matching raw prediction batch order.
/// @param coarse False(default) uses truth classes; true maps digit labels to parity ancestor0/1.
/// @return Detached geometry/masks/confidence/labels with largest-area/first-tie collision handling.
/// @throws std::invalid_argument For invalid geometry/dtypes/priors/rows/labels or out-of-image boxes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::tensor({8., 8., 24., 24.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// const auto t =
///     n::models::MakeYoloV2Targets(n::zeros({1, 8, 8, 5, 15}), n::ones({5, 2}), data, {0});
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see YoloV2Loss(), DecodeYoloV2Boxes()
inline YoloV2Targets MakeYoloV2Targets(const nn::Tensor &raw, const nn::Tensor &priors,
                                       const TensorDetectionScenes &scenes,
                                       const std::vector<std::int64_t> &rows,
                                       bool coarse = false) {
  if (raw.dim() != 5 || raw.shape()[0] != std::int64_t(rows.size()) ||
      raw.dtype() != scenes.images.dtype())
    throw std::invalid_argument("invalid YOLO target batch/dtype");
  two_stage_detection_detail::ValidateScenes(
      scenes, rows, coarse ? std::numeric_limits<int>::max() : int(raw.shape()[4] - 5));
  nn::NoGradGuard guard;
  const auto decoded = DecodeYoloV2Boxes(raw, priors, int(scenes.images.shape()[1]),
                                         int(scenes.images.shape()[2]));
  const auto N = raw.shape()[0], G = raw.shape()[1], A = raw.shape()[3], P = G * G * A;
  const auto type = raw.dtype();
  const double H = double(scenes.images.shape()[1]), W = double(scenes.images.shape()[2]);
  std::vector<nn::Tensor> geometry, positive, confidence, weights, labels;
  std::int64_t collisions = 0;
  for (std::int64_t i = 0; i < N; ++i) {
    const auto row = rows[i], K = scenes.boxes[row].shape()[0];
    auto target = nn::zeros({P, 4}, type), mask = nn::zeros({P}, type),
         conf = nn::zeros({P}, type), weight = nn::full({P}, .5, type),
         label = nn::zeros({P}, nn::DType::Int64);
    if (K) {
      const auto b = nn::unstack(scenes.boxes[row], 1);
      if (nn::any(nn::logical_or(nn::logical_or(nn::less(b[0], 0.), nn::less(b[1], 0.)),
                                 nn::logical_or(nn::greater(b[2], H), nn::greater(b[3], W))))
              .item<bool>())
        throw std::invalid_argument("YOLO target outside canvas");
      const auto overlap =
          nn::box_iou(nn::slice_axis(decoded, 0, i, i + 1).reshape({P, 4}), scenes.boxes[row]);
      weight = nn::where(nn::greater(nn::max(overlap, {1}), .6), nn::zeros({}, type),
                         nn::full({}, .5, type));
      const auto y = (b[0] + b[2]) * (.5 / H) * double(G),
                 x = (b[1] + b[3]) * (.5 / W) * double(G), h = (b[2] - b[0]) * (double(G) / H),
                 w = (b[3] - b[1]) * (double(G) / W), dims = nn::stack({h, w}, 1);
      const auto prior =
          nn::argmax(YoloDimensionIoU(dims.reshape({K, 1, 2}), priors.reshape({1, A, 2})), 1);
      const auto r = nn::clip(nn::floor(y), 0., double(G - 1)).to(nn::DType::Int64),
                 c = nn::clip(nn::floor(x), 0., double(G - 1)).to(nn::DType::Int64),
                 slot = (r * nn::full({}, double(G), nn::DType::Int64) + c) *
                            nn::full({}, double(A), nn::DType::Int64) +
                        prior;
      const auto same =
                     nn::equal(nn::arange(0, double(P), 1, nn::DType::Int64).reshape({P, 1}),
                               slot.reshape({1, K})),
                 occupied = nn::any(same, {1}),
                 best = nn::argmax(
                     nn::where(same, (h * w).reshape({1, K}),
                               nn::full({}, -std::numeric_limits<double>::infinity(), type)),
                     1);
      mask = occupied.to(type);
      collisions += K - nn::count_nonzero(occupied).item<std::int64_t>();
      const auto chosen_priors = nn::unstack(nn::gather(priors, prior), 1);
      const auto encoded =
          nn::stack({y - r.to(type), x - c.to(type), nn::log(h / chosen_priors[0]),
                     nn::log(w / chosen_priors[1])},
                    1);
      target = nn::gather(encoded, best) * mask.reshape({P, 1});
      label = nn::where(occupied, nn::gather(scenes.labels[row], best),
                        nn::zeros({}, nn::DType::Int64));
      if (coarse)
        label = label - nn::floor(label.to(type) / 2.).to(nn::DType::Int64) *
                            nn::full({}, 2., nn::DType::Int64);
      conf = nn::sum(overlap * nn::one_hot(best, K, 1, 0, type), {1}) * mask;
      weight = nn::where(occupied, nn::ones({}, type), weight);
    }
    geometry.push_back(target.reshape({G, G, A, 4}));
    positive.push_back(mask.reshape({G, G, A}));
    confidence.push_back(conf.reshape({G, G, A}));
    weights.push_back(weight.reshape({G, G, A}));
    labels.push_back(label.reshape({G, G, A}));
  }
  return {nn::stack(geometry, 0), nn::stack(positive, 0), nn::stack(confidence, 0),
          nn::stack(weights, 0),  nn::stack(labels, 0),   collisions};
}
/// @brief Compute YOLO v2 flat class softmax or WordTree marginal class probabilities.
///
/// WordTree marginals include ancestor factors; they do not sum to one over
/// all nodes. Flat softmax sums to one across classes per prediction. No fitting.
///
/// @param raw Floating [N,G,G,A,5+C] outputs.
/// @param classes Positive C matching raw last dimension minus5.
/// @param tree Optional valid C-node hierarchy, null(default) uses flat softmax.
/// @return Same-dtype differentiable [N,G,G,A,C] probabilities.
/// @throws std::invalid_argument For invalid shape/dtype/class count or hierarchy mismatch.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto p = n::models::YoloV2Probabilities(n::zeros({1, 8, 8, 5, 15}), 10);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see YoloV2Loss(), WordTreeProbabilities()
inline nn::Tensor YoloV2Probabilities(const nn::Tensor &raw, int classes,
                                      const WordTree *tree = nullptr) {
  if (raw.dim() != 5 || classes < 1 || raw.shape()[4] != classes + 5 ||
      (tree && tree->nodes() != classes))
    throw std::invalid_argument("invalid YOLO class probabilities");
  const auto &s = raw.shape();
  const auto logits = nn::reshape(
      nn::slice(raw, {0, 0, 0, 0, 5}, {s[0], s[1], s[2], s[3], classes}), {-1, classes});
  const auto p =
      tree ? WordTreeProbabilities(logits, *tree).marginal : nn::softmax(logits, -1);
  return nn::reshape(p, {s[0], s[1], s[2], s[3], classes});
}

/// @brief Configure detection and optional classification-only WordTree objective weights.
///
/// Pure objective metadata and labels; no optimizer, fitted parameters or state.
/// Classification-only supervision requires a valid WordTree.
///
/// @par Example
/// @code
/// dm::nn::models::YoloV2Weights w;
/// w.detection = 0;
/// w.classification = 1;
/// w.classification_labels = dm::nn::zeros({2}, dm::nn::DType::Int64);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see YoloV2Loss()
struct YoloV2Weights {
  double detection = 1;      ///< Finite nonnegative detection-objective multiplier.
  double classification = 0; ///< Finite nonnegative classification-only multiplier.
  nn::Tensor
      classification_labels; ///< Int64 [N] hierarchy IDs when classification weight is nonzero; otherwise may be undefined.
};
/// @brief Sum positive-normalized YOLO v2 SSE/class losses and optional classification-only hierarchy supervision.
///
/// Detection uses centre sigmoid/log-size SSE weight5, sigmoid confidence SSE
/// with target weights, and positive flat CE or ancestor-path NLL, divided by
/// positives floored at1. Classification-only images choose the detached
/// maximum(objectness*marginal(label)) prediction, first on ties, and supervise
/// its ancestor path plus square(ReLU(.3-objectness)). No box targets/gradients
/// for that branch. No optimizer, fitting or BN-state mutation is hidden.
///
/// @param raw Float32/64 [N,G,G,A,5+C] outputs.
/// @param t Consistent same-dtype detached geometry/positive/confidence/weights and Int64 labels.
/// @param classes Positive C matching output channels minus5.
/// @param w Finite nonnegative multipliers and optional one hierarchy label per image.
/// @param tree Optional C-node hierarchy, required for nonzero classification-only weight.
/// @return Differentiable scalar weighted detection and optional classification-only objectives.
/// @throws std::invalid_argument For invalid shapes/dtypes/classes/labels, hierarchy mismatch or objective weights.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::YoloV2Targets t{n::zeros({1, 1, 1, 1, 4}),
///                            n::zeros({1, 1, 1, 1}),
///                            n::zeros({1, 1, 1, 1}),
///                            n::full({1, 1, 1, 1}, .5),
///                            n::zeros({1, 1, 1, 1}, n::DType::Int64),
///                            0};
/// const auto l = n::models::YoloV2Loss(n::zeros({1, 1, 1, 1, 7}), t, 2, {});
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see MakeYoloV2Targets(), WordTreeLoss()
inline nn::Tensor YoloV2Loss(const nn::Tensor &raw, const YoloV2Targets &t, int classes,
                             const YoloV2Weights &w, const WordTree *tree = nullptr) {
  if (raw.dim() != 5 || raw.shape()[0] < 1 || raw.shape()[1] < 1 || raw.shape()[2] < 1 ||
      raw.shape()[3] < 1 || classes < 1 || raw.shape()[4] != classes + 5 ||
      (tree && tree->nodes() != classes) || !(w.detection >= 0) || !(w.classification >= 0) ||
      !std::isfinite(w.detection + w.classification) || (w.classification != 0 && !tree))
    throw std::invalid_argument("invalid YOLO objective settings");
  const auto N = raw.shape()[0], H = raw.shape()[1], W = raw.shape()[2], A = raw.shape()[3],
             P = H * W * A;
  const auto type = raw.dtype();
  const nn::Shape shape = {N, H, W, A};
  if (t.geometry.shape() != nn::Shape{N, H, W, A, 4} || t.positive.shape() != shape ||
      t.confidence.shape() != shape || t.weights.shape() != shape ||
      t.labels.shape() != shape || t.labels.dtype() != nn::DType::Int64 ||
      !nn::all(nn::logical_and(nn::greater_equal(t.labels, 0.),
                               nn::less(t.labels, double(classes))))
           .item<bool>())
    throw std::invalid_argument("invalid YOLO objective targets");
  for (const auto &x : {t.geometry, t.positive, t.confidence, t.weights})
    if (x.dtype() != type)
      throw std::invalid_argument("YOLO objective dtype mismatch");
  const auto encoded = nn::concat(
                 {nn::sigmoid(nn::slice_axis(raw, 4, 0, 2)), nn::slice_axis(raw, 4, 2, 4)}, 4),
             object = nn::sigmoid(nn::slice_axis(raw, 4, 4, 5)).reshape(shape),
             logits = nn::slice_axis(raw, 4, 5, 5 + classes).reshape({N * P, classes});
  const auto class_loss =
      tree ? WordTreeLoss(WordTreeProbabilities(logits, *tree), t.labels.reshape({N * P}),
                          *tree)
           : -nn::sum(nn::log_softmax(logits, -1) *
                          nn::one_hot(t.labels.reshape({N * P}), classes, 1, 0, type),
                      {1});
  const auto coordinates =
                 nn::sum(nn::sum(nn::square(encoded - t.geometry), {4}) * t.positive),
             confidence = nn::sum(nn::square(object - t.confidence) * t.weights),
             classification = nn::sum(class_loss * t.positive.reshape({N * P}));
  auto loss = (5. * coordinates + confidence + classification) /
              nn::maximum(nn::sum(t.positive), nn::ones({}, type)) * w.detection;
  if (tree && w.classification != 0) {
    if (w.classification_labels.shape() != nn::Shape{N} ||
        w.classification_labels.dtype() != nn::DType::Int64 ||
        !nn::all(nn::logical_and(nn::greater_equal(w.classification_labels, 0.),
                                 nn::less(w.classification_labels, double(classes))))
             .item<bool>())
      throw std::invalid_argument("invalid classification-only hierarchy labels");
    nn::Tensor selected;
    {
      nn::NoGradGuard guard;
      const auto marginal = WordTreeProbabilities(logits.detach(), *tree)
                                .marginal.reshape({N, P, classes}),
                 label = nn::one_hot(w.classification_labels, classes, 1, 0, type)
                             .reshape({N, 1, classes});
      const auto score = nn::sum(marginal * label, {2}) * object.detach().reshape({N, P});
      selected = nn::arange(0, double(N), 1, nn::DType::Int64) *
                     nn::full({}, double(P), nn::DType::Int64) +
                 nn::argmax(score, 1);
    }
    const auto selected_class =
                   WordTreeLoss(WordTreeProbabilities(nn::gather(logits, selected), *tree),
                                w.classification_labels, *tree),
               weak = nn::square(nn::relu(.3 - nn::gather(object.reshape({N * P}), selected)));
    loss = loss + nn::mean(selected_class + weak) * w.classification;
  }
  return loss;
}
/// @brief Decode YOLO v2 probability/objectness candidates with optional hierarchy-leaf mapping and NMS.
///
/// Hierarchy digit mapping is explicit and emits fine leaves rather than roots.
/// All classes are considered independently. Candidate input order is image,
/// row,column,anchor,class. No fitting or model/state mutation.
///
/// @param raw Floating [N,G,G,A,5+C] outputs.
/// @param priors Same-dtype positive [A,2] grid-unit dimensions.
/// @param probabilities Same-dtype [N,G,G,A,C] flat or marginal probabilities.
/// @param images Int64 [N] dataset IDs.
/// @param height Positive clipping height.
/// @param width Positive clipping width.
/// @param digit_leaves False(default) emits all C classes; true requires C>=12 and maps nodes2–11 to digits0–9.
/// @param suppress True(default) applies class-wise IoU.45 NMS and100/image; false returns raw candidates.
/// @return Detached clipped positive-area detections with strict sigmoid(objectness)*probability>.01.
/// @throws std::invalid_argument For invalid geometry/dtypes/IDs/probability values or malformed final boxes/scores.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto raw = n::zeros({1, 2, 2, 3, 15});
/// const auto d = n::models::DecodeYoloV2TensorDetections(raw, n::ones({3, 2}),
///                                                        n::models::YoloV2Probabilities(raw, 10),
///                                                        n::zeros({1}, n::DType::Int64), 64, 64);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2017), YOLO9000: Better, Faster, Stronger, https://arxiv.org/abs/1612.08242
/// @see YoloV2Probabilities(), DecodeYoloV2Boxes()
inline TensorDetections DecodeYoloV2TensorDetections(const nn::Tensor &raw,
                                                     const nn::Tensor &priors,
                                                     const nn::Tensor &probabilities,
                                                     const nn::Tensor &images, int height,
                                                     int width, bool digit_leaves = false,
                                                     bool suppress = true) {
  const auto decoded = DecodeYoloV2Boxes(raw, priors, height, width);
  const auto N = raw.shape()[0], G = raw.shape()[1], A = raw.shape()[3],
             C = raw.shape()[4] - 5, P = G * G * A;
  const auto type = raw.dtype();
  if (probabilities.shape() != nn::Shape{N, G, G, A, C} || probabilities.dtype() != type ||
      images.shape() != nn::Shape{N} || images.dtype() != nn::DType::Int64 ||
      (digit_leaves && C < 12) ||
      !nn::all(nn::logical_and(nn::greater_equal(probabilities, 0.),
                               nn::less_equal(probabilities, 1.)))
           .item<bool>())
    throw std::invalid_argument("invalid YOLO decode probabilities/images");
  nn::NoGradGuard guard;
  const auto cls = digit_leaves ? nn::slice_axis(probabilities, 4, 2, 12) : probabilities;
  const auto K = cls.shape()[4];
  const auto score = (cls * nn::sigmoid(nn::slice_axis(raw, 4, 4, 5))).reshape({N * P, K}),
             boxes = nn::clip_boxes(decoded.reshape({N * P, 4}), height, width);
  const auto valid = nn::logical_and(nn::greater(score, .01),
                                     nn::greater(nn::box_area(boxes), 0.).reshape({N * P, 1})),
             index = two_stage_detection_detail::Indices(valid.reshape({N * P * K})),
             box_index = nn::floor(index.to(type) / double(K)).to(nn::DType::Int64),
             label = index - box_index * nn::full({}, double(K), nn::DType::Int64),
             ids = (images.reshape({N, 1}) * nn::ones({N, P}, nn::DType::Int64))
                       .reshape({N * P});
  TensorDetections out{nn::gather(boxes, box_index),
                       nn::gather(score.reshape({N * P * K}), index), label,
                       nn::gather(ids, box_index)};
  if (!suppress) {
    DetectionEvaluationRecords(out);
    return out;
  }
  return SuppressDetectionCandidates(out.boxes, out.scores, out.labels, out.images, .45, 0.,
                                     100);
}
} // namespace datamunge::nn::models
