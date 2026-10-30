#pragma once
/// @file yolo_v5_models.hpp
/// @brief Tensor YOLO v5-style CSP/PAN/SPPF, CIoU and ratio/neighbour assignment.
#include "datamunge/nn/detection_mosaic.hpp"
#include "datamunge/nn/yolo_v3_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own IoU, distance-IoU, complete-IoU and detached aspect coefficient tensors.
///
/// Fields share broadcast leading dimensions; alpha is detached, other fields preserve geometry gradients.
///
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto b = n::tensor({.5, .5, .3, .2}, {1, 4});
/// const auto q = n::models::CompleteBoxQuality(b, b);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see YoloV5()
struct BoxQuality {
  nn::Tensor iou;   ///< Intersection over union, capped at1.
  nn::Tensor diou;  ///< IoU minus normalized centre distance.
  nn::Tensor ciou;  ///< DIoU minus detached-alpha aspect penalty.
  nn::Tensor alpha; ///< Detached v/(1-IoU+v+epsilon).
};
/// @brief Compute differentiable complete-IoU for centre-format boxes.
///
/// Sizes clamp at epsilon; disjoint CIoU can be negative and supplies centre gradients.
/// No fitting or mutation. Finite differences of ciou must freeze alpha to test the
/// intentional stopped derivative.
///
/// @param prediction Float32/64 [...,4] cy,cx,height,width.
/// @param target Same-dtype broadcastable [...,4] reference boxes.
/// @param epsilon Finite positive size/union floor and denominator addition, default1e-7.
/// @return Broadcast geometry tensors, with alpha detached.
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto b = n::tensor({.5, .5, .3, .2}, {1, 4});
/// const auto q = n::models::CompleteBoxQuality(b, b);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see YoloV5Loss()
inline BoxQuality CompleteBoxQuality(const nn::Tensor &prediction, const nn::Tensor &target,
                                     double epsilon = 1e-7) {
  if (prediction.dim() < 1 || target.dim() < 1 || prediction.shape().back() != 4 ||
      target.shape().back() != 4 || prediction.dtype() != target.dtype() ||
      (prediction.dtype() != nn::DType::Float32 && prediction.dtype() != nn::DType::Float64) ||
      !(epsilon > 0) || !std::isfinite(epsilon))
    throw std::invalid_argument("invalid CIoU geometry/dtype/epsilon");
  const auto p = nn::unstack(prediction, -1), t = nn::unstack(target, -1);
  const auto e = nn::full({}, epsilon, prediction.dtype());
  const auto ph = nn::maximum(p[2], e), pw = nn::maximum(p[3], e), th = nn::maximum(t[2], e),
             tw = nn::maximum(t[3], e);
  const auto py0 = p[0] - ph * .5, px0 = p[1] - pw * .5, py1 = p[0] + ph * .5,
             px1 = p[1] + pw * .5;
  const auto ty0 = t[0] - th * .5, tx0 = t[1] - tw * .5, ty1 = t[0] + th * .5,
             tx1 = t[1] + tw * .5;
  const auto zero = nn::zeros({}, prediction.dtype());
  const auto ih = nn::maximum(nn::minimum(py1, ty1) - nn::maximum(py0, ty0), zero);
  const auto iw = nn::maximum(nn::minimum(px1, tx1) - nn::maximum(px0, tx0), zero);
  const auto intersection = ih * iw;
  BoxQuality q;
  q.iou = nn::minimum(nn::ones({}, prediction.dtype()),
                      intersection / nn::maximum(ph * pw + th * tw - intersection, e));
  const auto enclosing_h = nn::maximum(py1, ty1) - nn::minimum(py0, ty0);
  const auto enclosing_w = nn::maximum(px1, tx1) - nn::minimum(px0, tx0);
  const auto diagonal = nn::square(enclosing_h) + nn::square(enclosing_w) + epsilon;
  const auto distance = nn::square(p[0] - t[0]) + nn::square(p[1] - t[1]);
  q.diou = q.iou - distance / diagonal;
  const double pi = std::acos(-1.);
  const auto v = nn::square(nn::atan(tw / th) - nn::atan(pw / ph)) * (4 / (pi * pi));
  q.alpha = (v / ((1.0 - q.iou) + (v + epsilon))).detach();
  q.ciou = q.diou - q.alpha * v;
  return q;
}

/// @brief Configure YOLO v5s v6.0-style CSP backbone and PAN heads.
///
/// Width divisor reduces the source32/64/128/256/512 channels. No pretrained weights.
/// Reference resolution affects only initial objectness bias, not allowed forward sizes.
///
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(1), state(1);
/// n::models::YoloV5Config cfg;
/// cfg.classes = 10;
/// cfg.width_divisor = 4;
/// cfg.grayscale = true;
/// cfg.reference_size = 64;
/// const auto net = n::models::YoloV5(p, state, n::zeros({1, 64, 64, 1}), cfg);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see YoloV5()
struct YoloV5Config {
  int classes = 80;            ///< Positive foreground class count.
  int width_divisor = 1;       ///< Positive divisor of16, default1.
  int reference_size = 640;    ///< Positive multiple of32 used only for initial biases.
  bool grayscale = false;      ///< False(default) expects RGB, true expects one channel.
  double expected_objects = 8; ///< Finite positive initial object count per reference image.
  bool update_stats = true;    ///< Update BN moving moments during training when true.
};
/// @brief Own fine-to-coarse backbone and coarse-to-fine PAN/raw outputs.
///
/// Raw heads pack[dy,dx,h,w,objectness,class logits] for three priors. No optimizer hidden.
///
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(1), state(1);
/// n::models::YoloV5Config cfg;
/// cfg.classes = 10;
/// cfg.width_divisor = 4;
/// cfg.grayscale = true;
/// cfg.reference_size = 64;
/// const auto net = n::models::YoloV5(p, state, n::zeros({1, 64, 64, 1}), cfg);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see YoloV5()
struct YoloV5Outputs {
  std::array<nn::Tensor, 3> backbone; ///< Fine/middle/coarse backbone maps at strides8/16/32.
  std::array<nn::Tensor, 3> pyramid;  ///< Coarse/middle/fine PAN maps at strides32/16/8.
  std::array<nn::Tensor, 3> raw;      ///< [N,G,G,3,5+C] logits, coarse to fine.
};
/// @brief Borrow parameter stores and normalization settings during a CSP forward.
///
/// Stores must outlive the context and backward; no ownership transfer.
/// Do not mutate learned leaves between forward and backward.
///
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(1), state(1);
/// n::models::CspContext c{p, state};
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see YoloV5()
struct CspContext {
  nn::ParameterStore
      &parameters; ///< Borrowed learned store, lazily creates kernels and BN affine leaves.
  nn::ParameterStore &state; ///< Borrowed BN moving-mean/variance store.
  bool training = false;     ///< Use batch statistics when true, otherwise moving moments.
  bool update_stats = true;  ///< Permit moving-moment updates during training.
};
/// @brief Apply He convolution, batch normalization and SiLU.
///
/// BN uses epsilon.001, momentum.97 and learned scale. No fitting loop.
///
/// @param c Borrowed stores/settings.
/// @param name Unique parameter prefix.
/// @param x Same-dtype positive floating NHWC input.
/// @param out Positive output channels.
/// @param kernel Positive kernel size.
/// @param stride Positive stride, default1.
/// @param padding Explicit nonnegative padding, or default-1 means kernel/2.
/// @return Normalized SiLU feature tensor.
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(1), state(1);
/// n::models::CspContext c{p, state};
/// const auto y = n::models::CspConv(c, "unit", n::ones({1, 4, 4, 2}), 4, 3);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see CspBlock()
inline nn::Tensor CspConv(CspContext &c, const std::string &name, const nn::Tensor &x,
                          std::int64_t out, std::int64_t kernel, std::int64_t stride = 1,
                          std::int64_t padding = -1) {
  ResNetNormConfig norm;
  norm.training = c.training;
  norm.scale = true;
  norm.epsilon = .001;
  norm.momentum = .97;
  norm.update_stats = c.update_stats;
  const auto y = ResNetNormalizedConv(c.parameters, c.state, name, x, out, kernel, norm,
                                      stride, padding < 0 ? kernel / 2 : padding, false);
  return y * nn::sigmoid(y);
}
/// @brief Own transformed, bypass, concatenation and fused CSP feature tensors.
///
/// No fitting or independent state; returned tensors retain the forward graph.
///
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(1), state(1);
/// n::models::CspContext c{p, state};
/// const auto b = n::models::CspBlock(c, "csp", n::ones({1, 4, 4, 2}), 4, 1);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see YoloV5()
struct CspOutputs {
  nn::Tensor transformed; ///< Repeated bottleneck branch.
  nn::Tensor bypass;      ///< Direct projected branch.
  nn::Tensor joined;      ///< Channel concatenation of both branches.
  nn::Tensor output;      ///< Fused output after1x1 convolution/BN/SiLU.
};
/// @brief Split, transform, concatenate and fuse a C3 CSP block.
///
/// Backbone units use identity shortcuts; PAN neck units disable them.
///
/// @param c Borrowed stores/settings.
/// @param name Unique parameter prefix.
/// @param x Positive floating NHWC features.
/// @param out Positive even output channels>=2.
/// @param repeats Positive bottleneck count.
/// @param shortcut True(default) adds identities inside transformed branch.
/// @return Both branches, concatenation and fused output.
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(1), state(1);
/// n::models::CspContext c{p, state};
/// const auto b = n::models::CspBlock(c, "csp", n::ones({1, 4, 4, 2}), 4, 1);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see CspConv()
inline CspOutputs CspBlock(CspContext &c, const std::string &name, const nn::Tensor &x,
                           std::int64_t out, int repeats, bool shortcut = true) {
  if (out < 2 || out % 2 || repeats < 1)
    throw std::invalid_argument("invalid CSP block dimensions");
  const std::int64_t hidden = out / 2;
  CspOutputs result;
  auto h = CspConv(c, name + "/left", x, hidden, 1);
  result.bypass = CspConv(c, name + "/right", x, hidden, 1);
  for (int i = 0; i < repeats; ++i) {
    const auto prefix = name + "/unit" + std::to_string(i + 1);
    auto branch = CspConv(c, prefix + "/reduce", h, hidden, 1);
    branch = CspConv(c, prefix + "/expand", branch, hidden, 3);
    h = shortcut ? h + branch : branch;
  }
  result.transformed = h;
  result.joined = nn::concat({h, result.bypass}, 3);
  result.output = CspConv(c, name + "/fuse", result.joined, out, 1);
  return result;
}
/// @brief Fuse serial5x5 max-pooling at three receptive scales.
///
/// A half-width projection feeds serial5x5 pooling, equivalent to parallel
/// 5/9/13 windows including negative-valued borders. Concatenation includes input.
///
/// @param c Borrowed stores/settings.
/// @param name Unique prefix.
/// @param x Positive NHWC features.
/// @param channels Positive even output channels>=2.
/// @return Same-spatial-size SPPF feature tensor.
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(1), state(1);
/// n::models::CspContext c{p, state};
/// const auto y = n::models::SpatialPyramidFast(c, "pool", n::ones({1, 4, 4, 2}), 4);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see CspConv()
inline nn::Tensor SpatialPyramidFast(CspContext &c, const std::string &name,
                                     const nn::Tensor &x, std::int64_t channels) {
  if (channels < 2 || channels % 2)
    throw std::invalid_argument("SPPF requires an even channel count");
  const auto h = CspConv(c, name + "/reduce", x, channels / 2, 1);
  const auto first = nn::max_pool2d(h, 5, 1, nn::Padding::Same);
  const auto second = nn::max_pool2d(first, 5, 1, nn::Padding::Same);
  const auto third = nn::max_pool2d(second, 5, 1, nn::Padding::Same);
  return CspConv(c, name + "/fuse", nn::concat({h, first, second, third}, 3), channels, 1);
}

/// @brief Build YOLO v5s v6.0-style CSP/SPPF/PAN and three detection heads.
///
/// C3 repeats1/2/3/1 and a6x6 stride2 pad2 stem preserve source depth.
/// BN epsilon.001/momentum.97 and SiLU throughout. Head priors initialize via
/// tensor logarithms; no pretrained weights or optimizer. Warm all leaves before
/// creating an optimizer, and checkpoint learned/BN stores together.
///
/// @param p Learned same-dtype store; creates every leaf lazily.
/// @param state Same-dtype BN moving state.
/// @param image Positive floating square NHWC, side>=32 divisible by32, channels per cfg.
/// @param cfg Architecture, bias and normalization settings.
/// @param training False(default) uses moving moments; true uses batch statistics.
/// @return Backbone, PAN and coarse-to-fine raw outputs.
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(1), state(1);
/// n::models::YoloV5Config cfg;
/// cfg.classes = 10;
/// cfg.width_divisor = 4;
/// cfg.grayscale = true;
/// cfg.reference_size = 64;
/// const auto net = n::models::YoloV5(p, state, n::zeros({1, 64, 64, 1}), cfg);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see MakeYoloV5Targets(), YoloV5Loss()
inline YoloV5Outputs YoloV5(nn::ParameterStore &p, nn::ParameterStore &state,
                            const nn::Tensor &image, const YoloV5Config &cfg = {},
                            bool training = false) {
  if (cfg.classes < 1 || cfg.width_divisor < 1 || 16 % cfg.width_divisor ||
      cfg.reference_size < 32 || cfg.reference_size % 32 || !(cfg.expected_objects > 0) ||
      !std::isfinite(cfg.expected_objects) ||
      cfg.classes > std::numeric_limits<int>::max() / 3 - 5 || image.dim() != 4 ||
      image.shape()[0] < 1 || image.shape()[1] < 32 || image.shape()[1] % 32 ||
      image.shape()[2] != image.shape()[1] || image.shape()[3] != (cfg.grayscale ? 1 : 3) ||
      image.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("invalid YOLOv5-style configuration");
  YoloV5Outputs result;
  CspContext c{p, state, training, cfg.update_stats};
  /// @brief Scale source channel counts, borrowing cfg only during forward.
  const auto width = [&](int n) {
    return std::int64_t(n / cfg.width_divisor);
  };
  auto h = CspConv(c, "stem", image, width(32), 6, 2, 2);
  const int repeats[] = {1, 2, 3, 1};
  for (int stage = 0; stage < 4; ++stage) {
    const auto name = "backbone/stage" + std::to_string(stage + 1);
    const auto out = width(64 << stage);
    h = CspConv(c, name + "/down", h, out, 3, 2);
    h = CspBlock(c, name + "/c3", h, out, repeats[stage]).output;
    if (stage >= 1)
      result.backbone[std::size_t(stage - 1)] = h;
  }
  h = SpatialPyramidFast(c, "sppf", h, width(512));
  const auto top_route = CspConv(c, "neck/top_route", h, width(256), 1);
  /// @brief Resize to the explicitly borrowed target feature shape.
  auto upsample = [](const nn::Tensor &x, const nn::Tensor &like) {
    return nn::resize_nearest(x, {like.shape()[1], like.shape()[2]});
  };
  auto middle =
      CspBlock(c, "neck/top_down_middle",
               nn::concat({upsample(top_route, result.backbone[1]), result.backbone[1]}, 3),
               width(256), 1, false)
          .output;
  const auto middle_route = CspConv(c, "neck/middle_route", middle, width(128), 1);
  const auto fine =
      CspBlock(c, "neck/top_down_fine",
               nn::concat({upsample(middle_route, result.backbone[0]), result.backbone[0]}, 3),
               width(128), 1, false)
          .output;
  auto down = CspConv(c, "neck/fine_down", fine, width(128), 3, 2);
  middle = CspBlock(c, "neck/bottom_up_middle", nn::concat({down, middle_route}, 3),
                    width(256), 1, false)
               .output;
  down = CspConv(c, "neck/middle_down", middle, width(256), 3, 2);
  const auto coarse = CspBlock(c, "neck/bottom_up_coarse", nn::concat({down, top_route}, 3),
                               width(512), 1, false)
                          .output;
  result.pyramid = {coarse, middle, fine};
  nn::ConvLayerOptions linear;
  linear.init = nn::init::he_uniform();
  for (int level = 0; level < 3; ++level) {
    const auto name = "prediction" + std::to_string(level + 1);
    if (!p.contains(name + "/b")) {
      const auto type = p.dtype();
      const auto index = nn::arange(0, double(cfg.classes + 5)).to(type);
      const auto cells = nn::full({}, double(cfg.reference_size) / double(32 >> level), type);
      const auto object_bias =
          nn::log(nn::full({}, cfg.expected_objects, type) / (cells * cells));
      const auto class_bias =
          cfg.classes == 1 ? nn::zeros({}, type)
                           : nn::log(nn::full({}, .6, type) / (double(cfg.classes) - .999999));
      const auto bias =
          nn::where(nn::equal(index, 4.), object_bias,
                    nn::where(nn::greater_equal(index, 5.), class_bias, nn::zeros({}, type)));
      p.set(name + "/b", nn::tile(bias, {3}));
    }
    const auto logits = nn::conv2d(p, name, result.pyramid[std::size_t(level)],
                                   3 * (cfg.classes + 5), 1, linear);
    const auto &s = logits.shape();
    result.raw[std::size_t(level)] =
        nn::reshape(logits, {s[0], s[1], s[2], 3, cfg.classes + 5});
  }
  return result;
}

/// @brief Decode one raw YOLO v5 head to normalized centre-format boxes.
///
/// Centre=(cell+2*sigmoid(raw)-.5)/G; size=(2*sigmoid(raw))²*prior/(stride*G).
/// No clipping, suppression, fitting or mutation.
///
/// @param raw Float32/64 [N,G,G,3,5+C], N/G/C positive and square grid.
/// @param priors Same-dtype [9,2] finite positive input-pixel height,width in ascending area order.
/// @param level Zero(coarse),one(middle) or two(fine).
/// @return [N,G,G,3,4] normalized cy,cx,h,w with geometry gradients.
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 7}), n::zeros({1, 4, 4, 3, 7}),
///                                 n::zeros({1, 8, 8, 3, 7})};
/// const auto priors = n::full({9, 2}, 8.);
/// const auto boxes = n::models::DecodeYoloV5Boxes(raw[0], priors, 0);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see DecodeYoloV5Detections()
inline nn::Tensor DecodeYoloV5Boxes(const nn::Tensor &raw, const nn::Tensor &priors,
                                    int level) {
  if (raw.dim() != 5 || raw.shape()[0] < 1 || raw.shape()[1] < 1 ||
      raw.shape()[2] != raw.shape()[1] || raw.shape()[3] != 3 || raw.shape()[4] < 6 ||
      level < 0 || level > 2 || priors.shape() != nn::Shape{9, 2} ||
      priors.dtype() != raw.dtype() ||
      (raw.dtype() != nn::DType::Float32 && raw.dtype() != nn::DType::Float64) ||
      !nn::all(
           nn::logical_and(nn::greater(priors, 0.),
                           nn::less(nn::abs(priors), std::numeric_limits<double>::infinity())))
           .item<bool>())
    throw std::invalid_argument("invalid YOLO v5 decode geometry");
  const auto G = raw.shape()[1];
  const auto p = nn::unstack(nn::slice_axis(priors, 0, 6 - 3 * level, 9 - 3 * level), 1),
             v = nn::unstack(nn::slice_axis(raw, 4, 0, 4), 4);
  const auto index = nn::arange(0, double(G)).to(raw.dtype());
  return nn::stack({(index.reshape({1, G, 1, 1}) + 2. * nn::sigmoid(v[0]) - .5) / double(G),
                    (index.reshape({1, 1, G, 1}) + 2. * nn::sigmoid(v[1]) - .5) / double(G),
                    nn::square(2. * nn::sigmoid(v[2])) * p[0].reshape({1, 1, 1, 3}) /
                        double((32 >> level) * G),
                    nn::square(2. * nn::sigmoid(v[3])) * p[1].reshape({1, 1, 1, 3}) /
                        double((32 >> level) * G)},
                   4);
}
/// @brief Own geometry, classes, positive masks and matching quality for one YOLO v5 level.
///
/// Targets are detached; quality is diagnostic shape IoU, not prediction IoU.
///
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 7}), n::zeros({1, 4, 4, 3, 7}),
///                                 n::zeros({1, 8, 8, 3, 7})};
/// const auto priors = n::full({9, 2}, 8.);
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::tensor({22., 22., 30., 30.}, {1, 4})};
/// data.labels = {n::ones({1}, n::DType::Int64)};
/// const auto targets = n::models::MakeYoloV5Targets(raw, priors, data, {0});
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see YoloV5()
struct YoloV5LevelTargets {
  nn::Tensor geometry; ///< [N,G,G,3,4] normalized cy,cx,height,width, zero background.
  nn::Tensor classes;  ///< [N,G,G,3,C] one-hot foreground targets, zero background.
  nn::Tensor positive; ///< [N,G,G,3] binary positive mask.
  nn::Tensor quality;  ///< Same shape, retained shape IoU or-1 background.
};
/// @brief Own three YOLO v5 target levels and collision diagnostics.
///
/// A truth may occupy multiple priors and axial neighbour cells; no diagonal copies.
///
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 7}), n::zeros({1, 4, 4, 3, 7}),
///                                 n::zeros({1, 8, 8, 3, 7})};
/// const auto priors = n::full({9, 2}, 8.);
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::tensor({22., 22., 30., 30.}, {1, 4})};
/// data.labels = {n::ones({1}, n::DType::Int64)};
/// const auto targets = n::models::MakeYoloV5Targets(raw, priors, data, {0});
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see YoloV5()
struct YoloV5Targets {
  std::array<YoloV5LevelTargets, 3> levels; ///< Coarse/middle/fine tensor targets.
  std::int64_t collisions =
      0; ///< Every eligible candidate beyond the first per slot, including replacements.
};
/// @brief Compute compact-adaptation CIoU, class BCE and balanced objectness BCE.
///
/// Count sums positives across heads and floors at1. Objectness target is
/// positive*max(current IoU,0), detached; head means balance.4/1/4 coarse to fine.
/// Box loss is1-CIoU with detached alpha. These5/5/1 weights are the compact
/// source adaptation, not official YOLO training hyperparameters. Empty scenes
/// still learn objectness. No optimizer or model mutation.
///
/// @param raw Three same-dtype [N,G,G,3,5+C] heads; equal grids allowed for fixtures.
/// @param t Matching same-dtype geometry/classes/positive fields; quality diagnostic only.
/// @param classes Positive foreground C.
/// @param priors Same-dtype positive [9,2] pixel dimensions.
/// @return Scalar5*box/positive_count+5*class/(positive_count*C)+balanced objectness.
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 7}), n::zeros({1, 4, 4, 3, 7}),
///                                 n::zeros({1, 8, 8, 3, 7})};
/// const auto priors = n::full({9, 2}, 8.);
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::tensor({22., 22., 30., 30.}, {1, 4})};
/// data.labels = {n::ones({1}, n::DType::Int64)};
/// const auto targets = n::models::MakeYoloV5Targets(raw, priors, data, {0});
/// const auto loss = n::models::YoloV5Loss(raw, targets, 2, priors);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see CompleteBoxQuality(), MakeYoloV5Targets()
inline nn::Tensor YoloV5Loss(const std::array<nn::Tensor, 3> &raw, const YoloV5Targets &t,
                             int classes, const nn::Tensor &priors) {
  if (priors.shape() != nn::Shape{9, 2} || classes < 1)
    throw std::invalid_argument("YOLOv5-style loss requires nine priors and positive classes");
  const double balance[] = {.4, 1, 4};
  nn::Tensor boxes, labels, counts, objectness;
  for (std::size_t level = 0; level < 3; ++level) {
    const auto &s = raw[level].shape();
    const auto &g = t.levels[level];
    if (s.size() != 5 || s[0] < 1 || s[1] < 1 || s[2] != s[1] || s[3] != 3 ||
        s[4] != classes + 5 || raw[level].dtype() != raw[0].dtype() ||
        g.geometry.shape() != nn::Shape{s[0], s[1], s[2], 3, 4} ||
        g.classes.shape() != nn::Shape{s[0], s[1], s[2], 3, classes} ||
        g.positive.shape() != nn::Shape{s[0], s[1], s[2], 3})
      throw std::invalid_argument("invalid YOLO v5 loss targets");
    for (const auto &x : {g.geometry, g.classes, g.positive})
      if (x.dtype() != raw[0].dtype())
        throw std::invalid_argument("YOLO v5 target dtype mismatch");
    const nn::Shape grid = {s[0], s[1], s[2], 3};
    const auto positive = g.positive;
    const auto quality =
        CompleteBoxQuality(DecodeYoloV5Boxes(raw[level], priors, int(level)), g.geometry);
    const auto box = nn::sum((1.0 - quality.ciou) * positive);
    const auto predicted_objectness =
        nn::squeeze(nn::slice(raw[level], {0, 0, 0, 0, 4}, {s[0], s[1], s[2], 3, 1}), {4});
    const auto confidence =
        positive * nn::maximum(quality.iou, nn::zeros({}, raw[0].dtype())).detach();
    const auto object =
        nn::mean(YoloLogisticCrossEntropy(predicted_objectness, confidence)) * balance[level];
    const auto logits = nn::slice(raw[level], {0, 0, 0, 0, 5}, {s[0], s[1], s[2], 3, classes});
    const auto label =
        nn::sum(nn::sum(YoloLogisticCrossEntropy(logits, g.classes), {4}) * positive);
    boxes = level ? boxes + box : box;
    labels = level ? labels + label : label;
    counts = level ? counts + nn::sum(positive) : nn::sum(positive);
    objectness = level ? objectness + object : object;
  }
  const auto count = nn::maximum(counts, nn::ones({}, raw[0].dtype()));
  return boxes / count * 5.0 + labels / (count * double(classes)) * 5.0 + objectness;
}

/// @brief Assign ratio-matched priors to base and eligible axial neighbour cells using tensors.
///
/// A prior matches iff every dimension ratio<4; if none, best shape IoU
/// (first ties) is fallback. Each matched prior assigns base plus one vertical
/// and one horizontal neighbour when fractional centre is below/above.5.
/// Exact.5 copies neither axis. Slot conflicts keep greater shape IoU, first
/// truth on ties. Current input pixels=coarse_grid*32. No fitting or random draws.
///
/// @param raw Three same-dtype [N,G,G,3,5+C] heads with grids doubling coarse to fine.
/// @param priors Same-dtype positive finite [9,2] input-pixel height,width priors.
/// @param scenes Valid same-dtype grayscale scene metadata with in-canvas positive boxes and Int64 labels<C.
/// @param rows Dataset IDs matching batch order.
/// @return Detached target levels and discarded/replaced candidate collision count.
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 7}), n::zeros({1, 4, 4, 3, 7}),
///                                 n::zeros({1, 8, 8, 3, 7})};
/// const auto priors = n::full({9, 2}, 8.);
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::tensor({22., 22., 30., 30.}, {1, 4})};
/// data.labels = {n::ones({1}, n::DType::Int64)};
/// const auto targets = n::models::MakeYoloV5Targets(raw, priors, data, {0});
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see YoloV5Loss()
inline YoloV5Targets MakeYoloV5Targets(const std::array<nn::Tensor, 3> &raw,
                                       const nn::Tensor &priors,
                                       const TensorDetectionScenes &scenes,
                                       const std::vector<std::int64_t> &rows) {
  if (raw[0].dim() != 5)
    throw std::invalid_argument("invalid YOLO v5 targets");
  DecodeYoloV5Boxes(raw[0], priors, 0);
  const auto N = std::int64_t(rows.size()), C = raw[0].shape()[4] - 5,
             R = raw[0].shape()[1] * 32;
  two_stage_detection_detail::ValidateScenes(scenes, rows, int(C));
  if (N < 1 || raw[0].dtype() != scenes.images.dtype())
    throw std::invalid_argument("invalid YOLO v5 target batch/dtype");
  for (int l = 0; l < 3; ++l)
    if (raw[l].dim() != 5 ||
        raw[l].shape() != nn::Shape{N, raw[0].shape()[1] * (1 << l),
                                    raw[0].shape()[1] * (1 << l), 3, C + 5} ||
        raw[l].dtype() != raw[0].dtype())
      throw std::invalid_argument("invalid YOLO v5 grids");
  nn::NoGradGuard guard;
  const auto type = raw[0].dtype();
  const double H = double(scenes.images.shape()[1]), W = double(scenes.images.shape()[2]);
  YoloV5Targets out;
  for (int l = 0; l < 3; ++l) {
    const auto G = raw[l].shape()[1], P = G * G * 3;
    std::vector<nn::Tensor> geometries, classes, positives, qualities;
    for (auto row : rows) {
      const auto K = scenes.boxes[row].shape()[0];
      auto geometry = nn::zeros({P, 4}, type), cls = nn::zeros({P, C}, type),
           positive = nn::zeros({P}, type), quality = nn::full({P}, -1., type);
      if (K) {
        const auto b = nn::unstack(scenes.boxes[row], 1);
        if (nn::any(nn::logical_or(nn::logical_or(nn::less(b[0], 0.), nn::less(b[1], 0.)),
                                   nn::logical_or(nn::greater(b[2], H), nn::greater(b[3], W))))
                .item<bool>())
          throw std::invalid_argument("truth outside YOLO v5 canvas");
        const auto normalized = nn::stack({(b[0] + b[2]) * (.5 / H), (b[1] + b[3]) * (.5 / W),
                                           (b[2] - b[0]) / H, (b[3] - b[1]) / W},
                                          1),
                   size = nn::stack(
                       {(b[2] - b[0]) * (double(R) / H), (b[3] - b[1]) * (double(R) / W)}, 1),
                   ratio = size.reshape({K, 1, 2}) / priors.reshape({1, 9, 2}),
                   iou = YoloDimensionIoU(size.reshape({K, 1, 2}), priors.reshape({1, 9, 2}));
        const auto matched = nn::less(nn::max(nn::maximum(ratio, 1. / ratio), {2}), 4.),
                   fallback = nn::one_hot(nn::argmax(iou, 1), 9, 1, 0, type);
        const auto eligible_priors = nn::logical_or(
            matched, nn::logical_and(nn::logical_not(nn::any(matched, {1})).reshape({K, 1}),
                                     nn::greater(fallback, 0.)));
        const auto selected = nn::slice_axis(eligible_priors, 1, 6 - 3 * l, 9 - 3 * l),
                   selected_quality = nn::slice_axis(iou, 1, 6 - 3 * l, 9 - 3 * l);
        const auto y = (b[0] + b[2]) * (.5 / H) * double(G),
                   x = (b[1] + b[3]) * (.5 / W) * double(G),
                   r = nn::clip(nn::floor(y), 0., double(G - 1)),
                   c = nn::clip(nn::floor(x), 0., double(G - 1));
        const auto slot = nn::arange(0, double(P)).to(type), cell = nn::floor(slot / 3.),
                   sr = nn::floor(cell / double(G)).reshape({P, 1}),
                   sc = (cell - nn::floor(cell / double(G)) * double(G)).reshape({P, 1}),
                   anchor = (slot - cell * 3.).to(nn::DType::Int64);
        const auto rr = r.reshape({1, K}), cc = c.reshape({1, K}),
                   fy = (y - r).reshape({1, K}), fx = (x - c).reshape({1, K});
        const auto same_r = nn::equal(sr, rr), same_c = nn::equal(sc, cc);
        const auto vertical = nn::logical_or(
                       nn::logical_and(nn::less(fy, .5), nn::equal(sr, rr - 1.)),
                       nn::logical_and(nn::greater(fy, .5), nn::equal(sr, rr + 1.))),
                   horizontal = nn::logical_or(
                       nn::logical_and(nn::less(fx, .5), nn::equal(sc, cc - 1.)),
                       nn::logical_and(nn::greater(fx, .5), nn::equal(sc, cc + 1.)));
        const auto cells = nn::logical_or(nn::logical_and(same_r, same_c),
                                          nn::logical_or(nn::logical_and(vertical, same_c),
                                                         nn::logical_and(horizontal, same_r))),
                   eligible = nn::logical_and(
                       cells, nn::transpose(nn::gather(selected, anchor, 1), {1, 0})),
                   q = nn::transpose(nn::gather(selected_quality, anchor, 1), {1, 0}),
                   occupied = nn::any(eligible, {1}),
                   winner = nn::argmax(
                       nn::where(eligible, q,
                                 nn::full({}, -std::numeric_limits<double>::infinity(), type)),
                       1);
        positive = occupied.to(type);
        out.collisions +=
            nn::sum(nn::maximum(nn::sum(eligible.to(type), {1}) - 1., nn::zeros({}, type)))
                .item<std::int64_t>();
        geometry = nn::gather(normalized, winner) * positive.reshape({P, 1});
        cls = nn::one_hot(nn::gather(scenes.labels[row], winner), C, 1, 0, type) *
              positive.reshape({P, 1});
        quality =
            nn::where(occupied, nn::max(nn::where(eligible, q, nn::full({}, -1., type)), {1}),
                      nn::full({}, -1., type));
      }
      geometries.push_back(geometry.reshape({G, G, 3, 4}));
      classes.push_back(cls.reshape({G, G, 3, C}));
      positives.push_back(positive.reshape({G, G, 3}));
      qualities.push_back(quality.reshape({G, G, 3}));
    }
    out.levels[l] = {nn::stack(geometries, 0), nn::stack(classes, 0), nn::stack(positives, 0),
                     nn::stack(qualities, 0)};
  }
  return out;
}
/// @brief Decode one YOLO v5 head to raw independent-class candidates before suppression.
///
/// Uses YOLO v5 geometry, independent sigmoid classes times objectness, strict
/// score>.01 and positive-area clipping. No NMS or fitting.
///
/// @param raw Floating [N,G,G,3,5+C] head.
/// @param level Zero/coarse,one/middle,two/fine.
/// @param priors Same-dtype [9,2] positive pixel priors.
/// @param images Int64[N] dataset image IDs.
/// @param height Positive clipping height.
/// @param width Positive clipping width.
/// @return Detached pixel boxes, scores, labels and dataset IDs.
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 7}), n::zeros({1, 4, 4, 3, 7}),
///                                 n::zeros({1, 8, 8, 3, 7})};
/// const auto priors = n::full({9, 2}, 8.);
/// const auto d = n::models::DecodeYoloV5LevelDetections(raw[0], 0, priors,
///                                                       n::zeros({1}, n::DType::Int64), 64, 64);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see DecodeYoloV5Detections()
inline TensorDetections DecodeYoloV5LevelDetections(const nn::Tensor &raw, int level,
                                                    const nn::Tensor &priors,
                                                    const nn::Tensor &images, int height,
                                                    int width) {
  if (height < 1 || width < 1)
    throw std::invalid_argument("invalid YOLO v5 image size");
  const auto normalized = DecodeYoloV5Boxes(raw, priors, level);
  const auto scale = nn::from_vector(std::vector<double>{double(height), double(width),
                                                         double(height), double(width)})
                         .to(raw.dtype());
  const auto decoded = nn::boxes_from_center(normalized) * scale,
             probabilities = nn::sigmoid(nn::slice_axis(raw, 4, 5, raw.shape()[4]));
  const bool digit_leaves = false, suppress = false;
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
} /// @brief Decode all three YOLO v5 heads with optional combined per-class suppression.
///
/// False returns all raw candidates, preserving head/anchor/class input order.
/// No optimizer, random draws or parameter/state mutation.
///
/// @param raw Three coarse-to-fine floating heads with doubling grids.
/// @param priors Same-dtype [9,2] positive pixel priors.
/// @param images Int64[N] dataset IDs.
/// @param height Positive original height.
/// @param width Positive original width.
/// @param suppress True(default) applies class IoU.45 NMS and cap100/image after combining heads.
/// @return Detached detections from all independent foreground classes.
/// @throws std::invalid_argument For invalid shapes, dtypes or configuration.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::array<n::Tensor, 3> raw = {n::zeros({1, 2, 2, 3, 7}), n::zeros({1, 4, 4, 3, 7}),
///                                 n::zeros({1, 8, 8, 3, 7})};
/// const auto priors = n::full({9, 2}, 8.);
/// const auto d =
///     n::models::DecodeYoloV5Detections(raw, priors, n::zeros({1}, n::DType::Int64), 64, 64);
/// @endcode
/// @par References
/// - Ultralytics YOLOv5 v6.0 architecture https://github.com/ultralytics/yolov5/blob/v6.0/models/yolov5s.yaml and loss https://github.com/ultralytics/yolov5/blob/v6.0/utils/loss.py; Z. Zheng et al.(2020), Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression, AAAI, https://arxiv.org/abs/1911.08287. Compact loss weights and fixed-centre mosaic are source adaptations.
/// @see YoloV5()
inline TensorDetections DecodeYoloV5Detections(const std::array<nn::Tensor, 3> &raw,
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
    const auto d = DecodeYoloV5LevelDetections(raw[l], l, priors, images, height, width);
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
