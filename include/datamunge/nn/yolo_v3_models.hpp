#pragma once
/// @file yolo_v3_models.hpp
/// @brief Tensor Darknet53 three-scale YOLO v3, global-prior matching and independent logistic heads.
#include "datamunge/nn/yolo_v2_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Evaluate stable elementwise logistic cross-entropy with optional soft targets.
///
/// Derivative at zero is sigmoid(0)-target, without absolute-value branch
/// ambiguity. Confident correct extreme values may round to zero. No mutation,
/// optimizer or hidden fitting.
///
/// @param logits Float32/64 finite logits, arbitrary shape.
/// @param target Same-dtype broadcastable finite probabilities in[0,1].
/// @return Broadcast softplus(logits)-logits*target, with gradients through both arguments.
/// @throws std::invalid_argument For invalid dtype/target range or incompatible broadcasts.
/// @par Example
/// @code
/// const auto loss =
///     dm::nn::models::YoloLogisticCrossEntropy(dm::nn::zeros({3}), dm::nn::tensor({0., .5, 1.}));
/// @endcode
/// @par References
/// - Redmon and Farhadi (2018), YOLOv3: An Incremental Improvement, https://arxiv.org/abs/1804.02767; authors model https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
/// @see YoloV3Loss()
inline nn::Tensor YoloLogisticCrossEntropy(const nn::Tensor &logits,
                                           const nn::Tensor &target) {
  if ((logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      target.dtype() != logits.dtype() ||
      !nn::all(nn::logical_and(nn::greater_equal(target, 0.), nn::less_equal(target, 1.)))
           .item<bool>())
    throw std::invalid_argument("invalid logistic target/dtype");
  return nn::softplus(logits) - logits * target;
}
/// @brief Configure source Darknet53 residual backbone and three top-down YOLO v3 heads.
///
/// Default80-class RGB model uses full widths. Width division/grayscale adapt
/// the unchanged topology for digit scenes. No pretrained weights or fitting.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::YoloV3Config c;
/// c.classes = 10;
/// c.width_divisor = 8;
/// c.grayscale = true;
/// const auto net = n::models::YoloV3(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2018), YOLOv3: An Incremental Improvement, https://arxiv.org/abs/1804.02767; authors model https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
/// @see YoloV3()
struct YoloV3Config {
  int classes = 80;         ///< Positive independent sigmoid class count per anchor.
  int width_divisor = 1;    ///< Positive divisor of32 for backbone/head widths.
  bool grayscale = false;   ///< Accept one-channel inputs when true, RGB otherwise.
  double momentum = .9;     ///< BN moving momentum[0,1), epsilon1e-5.
  bool update_stats = true; ///< Mutate BN moving state only during training when true.
};
/// @brief Retain YOLO v3 fine-to-coarse backbone features and coarse-to-fine prediction routes.
///
/// Tensor graphs borrow learned leaves; moving state stays in its separate
/// store. Class logits are independent sigmoid channels, not a softmax.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::YoloV3Config c;
/// c.classes = 10;
/// c.width_divisor = 8;
/// c.grayscale = true;
/// const auto net = n::models::YoloV3(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2018), YOLOv3: An Incremental Improvement, https://arxiv.org/abs/1804.02767; authors model https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
/// @see YoloV3(), YoloV3Loss()
struct YoloV3Outputs {
  std::array<nn::Tensor, 3>
      backbone; ///< Stride8/16/32 NHWC residual endpoints, fine to coarse.
  std::array<nn::Tensor, 3>
      routes; ///< Five-convolution routes at strides32/16/8, coarse to fine.
  std::array<nn::Tensor, 3>
      raw; ///< [N,G,G,3,5+C] dy,dx,logh,logw,objectness,class logits at strides32/16/8.
};
/// @brief Build source Darknet53 residual stages and three top-down anchor prediction heads.
///
/// Stage residual counts are1/2/8/8/4; downsampling uses symmetric-padded
/// stride-two3x3. Each route has five alternating1x1/3x3 layers, then3x3 and
/// linear1x1 prediction. Reduced earlier routes nearest-upsample and concatenate
/// with finer endpoints. Feature convolutions use BN/leaky-ReLU.1. Full80/20
/// class counts61949149/61626049, state52608. No pretraining or optimizer is
/// hidden. Keep learned leaves unchanged before backward.
///
/// @param p Mutable Float32/64 learned store; warm every leaf before optimizer construction.
/// @param state Mutable same-dtype BN moving-state store; checkpoint with p.
/// @param image Positive square floating NHWC, side>=32 divisible by32,1/3 channels per cfg.
/// @param cfg Valid class/width/channel/BN settings.
/// @param training False(default) uses moving BN state; true uses batch stats and optional updates.
/// @return Residual endpoints, route maps and three coarse-to-fine raw prediction tensors.
/// @throws std::invalid_argument For invalid cfg/image/dtypes or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::YoloV3Config c;
/// c.classes = 10;
/// c.width_divisor = 8;
/// c.grayscale = true;
/// const auto net = n::models::YoloV3(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2018), YOLOv3: An Incremental Improvement, https://arxiv.org/abs/1804.02767; authors model https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
/// @see MakeYoloV3Targets(), DecodeYoloV3Detections()
inline YoloV3Outputs YoloV3(nn::ParameterStore &p, nn::ParameterStore &state,
                            const nn::Tensor &image, const YoloV3Config &cfg = {},
                            bool training = false) {
  if (cfg.classes < 1 || cfg.width_divisor < 1 || 32 % cfg.width_divisor ||
      !(cfg.momentum >= 0 && cfg.momentum < 1) ||
      cfg.classes > (std::numeric_limits<int>::max() / 3) - 5 || image.dim() != 4 ||
      image.shape()[0] < 1 || image.shape()[1] < 32 || image.shape()[1] % 32 ||
      image.shape()[2] != image.shape()[1] || image.shape()[3] != (cfg.grayscale ? 1 : 3) ||
      image.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("invalid YOLO v3 configuration");
  YoloV3Outputs result;
  /// @brief Apply borrowed common BN/activation settings only during this forward.
  auto conv = [&](const std::string &name, const nn::Tensor &x, std::int64_t out,
                  std::int64_t kernel, std::int64_t stride = 1) {
    return DarknetConv(p, state, name, x, out, kernel, training, cfg.momentum,
                       cfg.update_stats, stride);
  };
  std::int64_t in = 32 / cfg.width_divisor;
  auto h = conv("backbone/stem", image, in, 3);
  const int units[] = {1, 2, 8, 8, 4};
  for (int stage = 0; stage < 5; ++stage) {
    const std::int64_t out = (64 << stage) / cfg.width_divisor;
    const std::string prefix = "backbone/stage" + std::to_string(stage + 1);
    h = conv(prefix + "/down", h, out, 3, 2);
    in = out;
    for (int unit = 0; unit < units[stage]; ++unit) {
      const auto name = prefix + "/unit" + std::to_string(unit + 1);
      const auto branch = conv(name + "/reduce", h, in / 2, 1);
      h = h + conv(name + "/expand", branch, in, 3);
    }
    if (stage >= 2)
      result.backbone[std::size_t(stage - 2)] = h;
  }
  for (int level = 0; level < 3; ++level) {
    const std::int64_t width = (512 >> level) / cfg.width_divisor;
    const auto prefix = "head" + std::to_string(level + 1);
    if (level > 0) {
      const auto reduced =
          conv(prefix + "/route_reduce", result.routes[std::size_t(level - 1)], width, 1);
      const auto &skip = result.backbone[std::size_t(2 - level)];
      h = nn::concat({nn::resize_nearest(reduced, {skip.shape()[1], skip.shape()[2]}), skip},
                     3);
    }
    for (int layer = 0; layer < 5; ++layer)
      h = conv(prefix + "/conv" + std::to_string(layer + 1), h, layer % 2 ? 2 * width : width,
               layer % 2 ? 3 : 1);
    result.routes[std::size_t(level)] = h;
    const auto features = conv(prefix + "/detect", h, 2 * width, 3);
    nn::ConvLayerOptions linear;
    linear.init = nn::init::he_uniform();
    const auto raw =
        nn::conv2d(p, prefix + "/prediction", features, 3 * (cfg.classes + 5), 1, linear);
    const auto &s = raw.shape();
    result.raw[std::size_t(level)] = nn::reshape(raw, {s[0], s[1], s[2], 3, cfg.classes + 5});
  }
  return result;
}

/// @brief Own detached geometry, classes and loss masks for one YOLO v3 scale.
///
/// No learned parameters or fitted state. All fields align with that raw
/// head's batch/grid/three-prior order.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::zeros({0, 4})};
/// data.labels = {n::zeros({0}, n::DType::Int64)};
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 15}), n::zeros({1, 4, 4, 3, 15}),
///                                 n::zeros({1, 8, 8, 3, 15})};
/// const auto t = n::models::MakeYoloV3Targets(raw, n::ones({9, 2}), data, {0});
/// @endcode
/// @par References
/// - Redmon and Farhadi (2018), YOLOv3: An Incremental Improvement, https://arxiv.org/abs/1804.02767; authors model https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
/// @see MakeYoloV3Targets(), YoloV3Loss()
struct YoloV3LevelTargets {
  nn::Tensor geometry; ///< [N,G,G,3,4] sigmoid-centre offsets and log prior-size ratios.
  nn::Tensor classes;  ///< [N,G,G,3,C] binary independent class targets.
  nn::Tensor positive; ///< [N,G,G,3] retained positive0/1 mask.
  nn::Tensor
      objectness_weight; ///< Same shape,1 unless an unassigned prediction overlaps truth IoU>.5.
  nn::Tensor geometry_weight; ///< Same shape,positive*(2-relative_target_area).
  nn::Tensor
      areas; ///< Same shape,retained target area at current input-pixel resolution,0 background.
};
/// @brief Own three coarse-to-fine YOLO v3 target levels and discarded collision count.
///
/// Each truth chooses exactly one global prior across all three scales.
/// No fitting, random draws or model/state mutation.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::zeros({0, 4})};
/// data.labels = {n::zeros({0}, n::DType::Int64)};
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 15}), n::zeros({1, 4, 4, 3, 15}),
///                                 n::zeros({1, 8, 8, 3, 15})};
/// const auto t = n::models::MakeYoloV3Targets(raw, n::ones({9, 2}), data, {0});
/// @endcode
/// @par References
/// - Redmon and Farhadi (2018), YOLOv3: An Incremental Improvement, https://arxiv.org/abs/1804.02767; authors model https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
/// @see MakeYoloV3Targets(), YoloV3Loss()
struct YoloV3Targets {
  std::array<YoloV3LevelTargets, 3> levels; ///< Coarse stride32, middle16 and fine8 targets.
  std::int64_t collisions =
      0; ///< Discarded same-cell/same-prior truth count across all three levels.
};
/// @brief Sum YOLO v3 logistic/objectness/class and area-weighted coordinate losses across three heads.
///
/// Coordinates use.5*sum((sigmoid centres/raw log sizes-target)²)*geometry_weight.
/// Objectness BCE uses positive targets and detached ignore weights. Independent
/// class BCE sums all classes at positive anchors, allowing multiple labels in
/// explicit fixtures. Background-only batches still learn objectness. No fitting,
/// optimizer or state updates occur here.
///
/// @param raw Three Float32/64 [N,G,G,3,5+C] raw tensors; loss fixtures may use equal grids.
/// @param t Same-dtype matching geometry/classes/positive/objectness/geometry-weight fields; areas is diagnostic only.
/// @param classes Positive C consistent with every raw class axis.
/// @return Scalar sum of all head terms divided by max(total batch positives over heads,1).
/// @throws std::invalid_argument For invalid shapes/dtypes/classes or logistic target range.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::zeros({0, 4})};
/// data.labels = {n::zeros({0}, n::DType::Int64)};
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 15}), n::zeros({1, 4, 4, 3, 15}),
///                                 n::zeros({1, 8, 8, 3, 15})};
/// const auto t = n::models::MakeYoloV3Targets(raw, n::ones({9, 2}), data, {0});
/// const auto loss = n::models::YoloV3Loss(raw, t, 10);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2018), YOLOv3: An Incremental Improvement, https://arxiv.org/abs/1804.02767; authors model https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
/// @see MakeYoloV3Targets(), YoloLogisticCrossEntropy()
inline nn::Tensor YoloV3Loss(const std::array<nn::Tensor, 3> &raw, const YoloV3Targets &t,
                             int classes) {
  nn::Tensor total, positives;
  for (std::size_t level = 0; level < 3; ++level) {
    const auto &s = raw[level].shape();
    const auto &g = t.levels[level];
    if (s.size() != 5 || s[0] < 1 || s[1] < 1 || s[2] < 1 || s[3] != 3 || classes < 1 ||
        s[4] != classes + 5 || raw[level].dtype() != raw[0].dtype() ||
        g.geometry.shape() != nn::Shape{s[0], s[1], s[2], 3, 4} ||
        g.classes.shape() != nn::Shape{s[0], s[1], s[2], 3, classes} ||
        g.positive.shape() != nn::Shape{s[0], s[1], s[2], 3} ||
        g.objectness_weight.shape() != g.positive.shape() ||
        g.geometry_weight.shape() != g.positive.shape())
      throw std::invalid_argument("invalid YOLO v3 loss geometry");
    for (const auto &x :
         {g.geometry, g.classes, g.positive, g.objectness_weight, g.geometry_weight})
      if (x.dtype() != raw[0].dtype())
        throw std::invalid_argument("YOLO v3 target dtype mismatch");
    const nn::Shape grid = {s[0], s[1], s[2], 3};
    /// @brief Slice this head while borrowing raw/s only during the loss call.
    auto part = [&](std::int64_t begin, std::int64_t count) {
      return nn::slice(raw[level], {0, 0, 0, 0, begin}, {s[0], s[1], s[2], 3, count});
    };
    const auto geometry = nn::concat({nn::sigmoid(part(0, 2)), part(2, 2)}, 4);
    const auto positive = g.positive;
    const auto coordinates =
        nn::sum(nn::sum(nn::square(geometry - g.geometry), {4}) * g.geometry_weight) * .5;
    const auto confidence =
        nn::sum(YoloLogisticCrossEntropy(nn::squeeze(part(4, 1), {4}), positive) *
                g.objectness_weight);
    const auto classification = nn::sum(
        nn::sum(YoloLogisticCrossEntropy(part(5, classes), g.classes), {4}) * positive);
    const auto level_total = coordinates + confidence + classification;
    total = level ? total + level_total : level_total;
    positives = level ? positives + nn::sum(positive) : nn::sum(positive);
  }
  return total / nn::maximum(positives, nn::ones({}, raw[0].dtype()));
}

/// @brief Assign each truth to one global prior and its scale/cell using tensor geometry.
///
/// Current input resolution is coarse_grid*32. Truth dimensions normalize from
/// scene pixels to that resolution before global highest-shape-IoU selection,
/// first prior on ties. Largest area/first truth wins slot collisions. Unassigned
/// predictions whose current decoded IoU with any truth>.5 are ignored for
/// objectness; positives restore weight1. Prior blocks map largest three to
/// coarse, middle three to middle, smallest three to fine. No fitting or RNG.
///
/// @param raw Three same-dtype [N,G,G,3,5+C] detached predictions; grids double coarse to fine.
/// @param priors Same-dtype [9,2] positive input-pixel height,width priors, ordered increasing area.
/// @param scenes Valid same-dtype grayscale scene metadata with in-image positive boxes and Int64 labels<C.
/// @param rows Dataset IDs matching prediction batch order.
/// @return Detached three-scale targets, ignore masks, relative-area geometry weights and collision count.
/// @throws std::invalid_argument For invalid shapes/dtypes/priors/rows/labels or out-of-canvas boxes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::zeros({0, 4})};
/// data.labels = {n::zeros({0}, n::DType::Int64)};
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 15}), n::zeros({1, 4, 4, 3, 15}),
///                                 n::zeros({1, 8, 8, 3, 15})};
/// const auto t = n::models::MakeYoloV3Targets(raw, n::ones({9, 2}), data, {0});
/// @endcode
/// @par References
/// - Redmon and Farhadi (2018), YOLOv3: An Incremental Improvement, https://arxiv.org/abs/1804.02767; authors model https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
/// @see YoloV3Loss(), ClusterYoloDimensions()
inline YoloV3Targets MakeYoloV3Targets(const std::array<nn::Tensor, 3> &raw,
                                       const nn::Tensor &priors,
                                       const TensorDetectionScenes &scenes,
                                       const std::vector<std::int64_t> &rows) {
  if (raw[0].dim() != 5 || priors.shape() != nn::Shape{9, 2} ||
      priors.dtype() != raw[0].dtype() || raw[0].dtype() != scenes.images.dtype() ||
      !nn::all(
           nn::logical_and(nn::greater(priors, 0.),
                           nn::less(nn::abs(priors), std::numeric_limits<double>::infinity())))
           .item<bool>())
    throw std::invalid_argument("invalid YOLO v3 priors/dtypes");
  const auto N = std::int64_t(rows.size()), C = raw[0].shape()[4] - 5,
             R = raw[0].shape()[1] * 32;
  two_stage_detection_detail::ValidateScenes(scenes, rows, int(C));
  for (int l = 0; l < 3; ++l)
    if (raw[l].dim() != 5 || raw[l].shape()[0] != N || N < 1 || raw[l].shape()[1] < 1 ||
        raw[l].shape()[1] != raw[0].shape()[1] * (1 << l) ||
        raw[l].shape()[2] != raw[l].shape()[1] || raw[l].shape()[3] != 3 ||
        raw[l].shape()[4] != C + 5 || raw[l].dtype() != raw[0].dtype())
      throw std::invalid_argument("invalid YOLO v3 head grids");
  nn::NoGradGuard guard;
  const auto type = raw[0].dtype();
  const double H = double(scenes.images.shape()[1]), W = double(scenes.images.shape()[2]);
  YoloV3Targets out;
  for (int l = 0; l < 3; ++l) {
    const auto G = raw[l].shape()[1], P = G * G * 3;
    const auto selected_priors =
                   nn::slice_axis(priors, 0, 6 - 3 * l, 9 - 3 * l) / double(32 >> l),
               decoded = DecodeYoloV2Boxes(raw[l], selected_priors, int(H), int(W));
    std::vector<nn::Tensor> geometry, classes, positive, objectness_weight, geometry_weight,
        areas;
    for (std::int64_t i = 0; i < N; ++i) {
      const auto row = rows[i], K = scenes.boxes[row].shape()[0];
      auto target = nn::zeros({P, 4}, type), cls = nn::zeros({P, C}, type),
           mask = nn::zeros({P}, type), obj_weight = nn::ones({P}, type),
           geo_weight = nn::zeros({P}, type), area_out = nn::zeros({P}, type);
      if (K) {
        const auto b = nn::unstack(scenes.boxes[row], 1);
        if (nn::any(nn::logical_or(nn::logical_or(nn::less(b[0], 0.), nn::less(b[1], 0.)),
                                   nn::logical_or(nn::greater(b[2], H), nn::greater(b[3], W))))
                .item<bool>())
          throw std::invalid_argument("YOLO v3 truth outside canvas");
        const auto size = nn::stack(
                       {(b[2] - b[0]) * (double(R) / H), (b[3] - b[1]) * (double(R) / W)}, 1),
                   best_prior = nn::argmax(
                       YoloDimensionIoU(size.reshape({K, 1, 2}), priors.reshape({1, 9, 2})),
                       1),
                   group = nn::floor(best_prior.to(type) / 3.).to(nn::DType::Int64),
                   level = nn::full({}, 2., nn::DType::Int64) - group,
                   anchor = best_prior - group * nn::full({}, 3., nn::DType::Int64),
                   eligible = nn::equal(level, double(l));
        const auto y = (b[0] + b[2]) * (.5 / H) * double(G),
                   x = (b[1] + b[3]) * (.5 / W) * double(G),
                   r = nn::clip(nn::floor(y), 0., double(G - 1)).to(nn::DType::Int64),
                   c = nn::clip(nn::floor(x), 0., double(G - 1)).to(nn::DType::Int64),
                   slot = (r * nn::full({}, double(G), nn::DType::Int64) + c) *
                              nn::full({}, 3., nn::DType::Int64) +
                          anchor;
        const auto dims = nn::unstack(size, 1);
        const auto area = dims[0] * dims[1];
        const auto same = nn::logical_and(
                       nn::equal(nn::arange(0, double(P), 1, nn::DType::Int64).reshape({P, 1}),
                                 slot.reshape({1, K})),
                       eligible.reshape({1, K})),
                   occupied = nn::any(same, {1}),
                   chosen = nn::argmax(
                       nn::where(same, area.reshape({1, K}),
                                 nn::full({}, -std::numeric_limits<double>::infinity(), type)),
                       1);
        mask = occupied.to(type);
        out.collisions += nn::count_nonzero(eligible).item<std::int64_t>() -
                          nn::count_nonzero(occupied).item<std::int64_t>();
        const auto p = nn::unstack(nn::gather(priors, best_prior), 1);
        const auto encoded = nn::stack(
            {y - r.to(type), x - c.to(type), nn::log(dims[0] / p[0]), nn::log(dims[1] / p[1])},
            1);
        target = nn::gather(encoded, chosen) * mask.reshape({P, 1});
        cls = nn::one_hot(nn::gather(scenes.labels[row], chosen), C, 1, 0, type) *
              mask.reshape({P, 1});
        area_out = nn::gather(area, chosen) * mask;
        geo_weight = (2. - nn::gather(area, chosen) / double(R * R)) * mask;
        const auto overlap = nn::box_iou(nn::slice_axis(decoded, 0, i, i + 1).reshape({P, 4}),
                                         scenes.boxes[row]);
        obj_weight =
            nn::where(nn::logical_or(occupied, nn::less_equal(nn::max(overlap, {1}), .5)),
                      nn::ones({}, type), nn::zeros({}, type));
      }
      geometry.push_back(target.reshape({G, G, 3, 4}));
      classes.push_back(cls.reshape({G, G, 3, C}));
      positive.push_back(mask.reshape({G, G, 3}));
      objectness_weight.push_back(obj_weight.reshape({G, G, 3}));
      geometry_weight.push_back(geo_weight.reshape({G, G, 3}));
      areas.push_back(area_out.reshape({G, G, 3}));
    }
    out.levels[l] = {nn::stack(geometry, 0),        nn::stack(classes, 0),
                     nn::stack(positive, 0),        nn::stack(objectness_weight, 0),
                     nn::stack(geometry_weight, 0), nn::stack(areas, 0)};
  }
  return out;
}
/// @brief Decode one YOLO v3 head to raw foreground sigmoid/objectness candidates before NMS.
///
/// Selects largest/middle/smallest prior block, divides by level stride and
/// composes tensor YOLO v2 box geometry with independent class sigmoid scores.
/// No NMS, fitting, random draws or model/state mutation.
///
/// @param raw Float32/64 [N,G,G,3,5+C] head predictions.
/// @param level Integer0(coarse),1(middle),2(fine).
/// @param priors Same-dtype [9,2] positive input-pixel dimensions ordered by increasing area.
/// @param images Int64 [N] dataset IDs.
/// @param height Positive original clipping height.
/// @param width Positive original clipping width.
/// @return Detached raw detections for this level, strict score>.01, clipped positive-area boxes.
/// @throws std::invalid_argument For invalid level/prior/head geometry/dtype/IDs or candidate values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d =
///     n::models::DecodeYoloV3LevelDetections(n::zeros({1, 2, 2, 3, 15}), 0, n::full({9, 2}, 16.),
///                                            n::zeros({1}, n::DType::Int64), 64, 64);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2018), YOLOv3: An Incremental Improvement, https://arxiv.org/abs/1804.02767; authors model https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
/// @see DecodeYoloV3Detections()
inline TensorDetections DecodeYoloV3LevelDetections(const nn::Tensor &raw, int level,
                                                    const nn::Tensor &priors,
                                                    const nn::Tensor &images, int height,
                                                    int width) {
  if (level < 0 || level > 2 || priors.shape() != nn::Shape{9, 2} || raw.dim() != 5 ||
      raw.shape()[4] < 6)
    throw std::invalid_argument("invalid YOLO v3 decode level/priors");
  const auto selected =
      nn::slice_axis(priors, 0, 6 - 3 * level, 9 - 3 * level) / double(32 >> level);
  return DecodeYoloV2TensorDetections(raw, selected,
                                      nn::sigmoid(nn::slice_axis(raw, 4, 5, raw.shape()[4])),
                                      images, height, width, false, false);
}
/// @brief Decode all YOLO v3 levels and optionally suppress across their combined candidates.
///
/// All class channels and all three levels contribute before global per-image
/// class suppression. No fitting, random draws or parameter/state mutation.
///
/// @param raw Three coarse-to-fine raw heads.
/// @param priors Positive same-dtype [9,2] input-pixel dimensions, sorted by area.
/// @param images Int64 [N] dataset IDs.
/// @param height Positive clipping height.
/// @param width Positive clipping width.
/// @param suppress True(default) applies class-wise IoU.45 NMS and100/image across all heads; false returns raw concatenation.
/// @return Detached tensor detections after independent sigmoid class/objectness scoring.
/// @throws std::invalid_argument For inconsistent head/prior/image/candidate shapes, dtypes or values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 15}), n::zeros({1, 4, 4, 3, 15}),
///                                 n::zeros({1, 8, 8, 3, 15})};
/// const auto d = n::models::DecodeYoloV3Detections(raw, n::full({9, 2}, 16.),
///                                                  n::zeros({1}, n::DType::Int64), 64, 64);
/// @endcode
/// @par References
/// - Redmon and Farhadi (2018), YOLOv3: An Incremental Improvement, https://arxiv.org/abs/1804.02767; authors model https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
/// @see YoloV3(), DecodeYoloV3LevelDetections()
inline TensorDetections DecodeYoloV3Detections(const std::array<nn::Tensor, 3> &raw,
                                               const nn::Tensor &priors,
                                               const nn::Tensor &images, int height, int width,
                                               bool suppress = true) {
  if (raw[0].dim() != 5)
    throw std::invalid_argument("invalid YOLO v3 decoder heads");
  for (int l = 0; l < 3; ++l)
    if (raw[l].dim() != 5 || raw[l].shape()[1] != raw[0].shape()[1] * (1 << l) ||
        raw[l].shape()[4] != raw[0].shape()[4])
      throw std::invalid_argument("inconsistent YOLO v3 decoder heads");
  std::vector<nn::Tensor> boxes, scores, labels, ids;
  for (int l = 0; l < 3; ++l) {
    const auto d = DecodeYoloV3LevelDetections(raw[l], l, priors, images, height, width);
    boxes.push_back(d.boxes);
    scores.push_back(d.scores);
    labels.push_back(d.labels);
    ids.push_back(d.images);
  }
  TensorDetections out{nn::concat(boxes, 0), nn::concat(scores, 0), nn::concat(labels, 0),
                       nn::concat(ids, 0)};
  if (!suppress)
    return out;
  return SuppressDetectionCandidates(out.boxes, out.scores, out.labels, out.images, .45, 0.,
                                     100);
}
} // namespace datamunge::nn::models
