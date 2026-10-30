#pragma once
/// @file center_detection_models.hpp
/// @brief Tensor CenterNet encoder/deconvolution heads, Gaussian targets and peak decoding.
#include "datamunge/nn/retina_detection_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Apply a normal-.0013x3/ReLU then linear1x1 CenterNet prediction head.
///
/// No normalization, state updates or optimizer. Kernel stddev.001 differs from
/// He-initialized backbone/deconvolutions. Keep learned leaves until backward.
///
/// @param p Mutable Float32/64 learned store, creates/reuses named leaves.
/// @param name Unique compatible shared prefix.
/// @param x Positive floating NHWC feature map matching p dtype.
/// @param hidden Positive3x3 hidden width.
/// @param out Positive linear output width.
/// @param bias Finite output bias initializer, default0; existing leaves are reused.
/// @return Differentiable NHWC output with out channels, same spatial dimensions.
/// @throws std::invalid_argument For invalid dtype/shape/width/bias or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::CenterHead(p, "h", n::ones({1, 8, 8, 32}), 32, 10, -2.19);
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see CenterNet()
inline nn::Tensor CenterHead(nn::ParameterStore &p, const std::string &name,
                             const nn::Tensor &x, std::int64_t hidden, std::int64_t out,
                             double bias = 0) {
  if (x.dim() != 4 || x.shape()[0] < 1 || x.shape()[1] < 1 || x.shape()[2] < 1 ||
      x.shape()[3] < 1 || x.dtype() != p.dtype() || hidden < 1 || out < 1 ||
      !std::isfinite(bias))
    throw std::invalid_argument("invalid CenterNet head");
  nn::ConvLayerOptions first;
  first.init = nn::init::normal(.001);
  first.activation = nn::Activation::Relu;
  const auto h = nn::conv2d(p, name + "/conv", x, hidden, 3, first);
  const auto w = p.get(name + "/output/W", {1, 1, hidden, out}, nn::init::normal(.001));
  const auto b = p.get(name + "/output/b", {out}, nn::init::constant(bias));
  return nn::bias_add(nn::conv2d(h, w), b);
}
/// @brief Retain sigmoid probabilities at per-class3x3 local maxima.
///
/// SAME max pooling is class independent. Mask equality is nondifferentiable;
/// sigmoid gradients flow at selected maxima. No fitting or mutation.
///
/// @param logits Float32/64 positive NHWC class logits.
/// @return Same-shape probabilities, zero off maxima; equal-valued plateaus retain every tied pixel.
/// @throws std::invalid_argument For invalid rank/geometry or unsupported core dtype.
/// @par Example
/// @code
/// const auto p = dm::nn::models::CenterPeaks(dm::nn::zeros({1, 4, 4, 2}));
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see DecodeCenterDetections()
inline nn::Tensor CenterPeaks(const nn::Tensor &logits) {
  if (logits.dim() != 4 || logits.shape()[0] < 1 || logits.shape()[1] < 1 ||
      logits.shape()[2] < 1 || logits.shape()[3] < 1)
    throw std::invalid_argument("invalid heatmap logits");
  const auto probability = nn::sigmoid(logits);
  return probability *
         nn::cast(nn::equal(probability, nn::max_pool2d(probability, 3, 1, nn::Padding::Same)),
                  probability.dtype());
}

/// @brief Configure CenterNet feature-only ResNet, learned upsampling and three prediction heads.
///
/// Defaults use ResNet18, deconvolution widths256/128/64, head64 and80 classes.
/// No pretrained weights or fit. BN state belongs to a separate store.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::CenterNetConfig c;
/// c.classes = 10;
/// c.width_divisor = 8;
/// c.upsample_widths = {64, 32, 32};
/// c.head_width = 32;
/// c.grayscale = true;
/// const auto net = n::models::CenterNet(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see CenterNet()
struct CenterNetConfig {
  int classes = 80;        ///< Positive foreground heatmap channel count.
  int backbone_depth = 18; ///< Supported feature-only ResNet depth.
  int width_divisor = 1;   ///< Positive divisor of64 for backbone widths.
  int head_width = 64;     ///< Positive hidden width in all three independent heads.
  std::array<int, 3> upsample_widths = {
      256, 128, 64};      ///< Positive widths of three4x4 stride-two transposed convolutions.
  bool grayscale = false; ///< Tile one-channel input to RGB before backbone.
  bool update_stats = true; ///< Update backbone and upsampling BN state in training when true.
};
/// @brief Retain stride-four CenterNet features, heatmap logits, sizes, offsets and local peaks.
///
/// Tensor graphs borrow learned leaves from p; moving state remains in state.
/// Size and offset order is height,width and dy,dx, in stride-four grid units.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::CenterNetConfig c;
/// c.classes = 10;
/// c.width_divisor = 8;
/// c.upsample_widths = {64, 32, 32};
/// c.head_width = 32;
/// c.grayscale = true;
/// const auto net = n::models::CenterNet(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see CenterNet(), CenterLoss()
struct CenterNetOutputs {
  nn::Tensor features; ///< [N,H/4,W/4,last_width] learned upsampled features.
  nn::Tensor heatmap;  ///< [N,H/4,W/4,C] class logits.
  nn::Tensor size;     ///< [N,H/4,W/4,2] unconstrained predicted height,width in grid units.
  nn::Tensor offset;   ///< [N,H/4,W/4,2] predicted dy,dx within grid cells.
  nn::Tensor peaks;    ///< Same-shape heatmap probabilities masked to per-class local maxima.
};
/// @brief Build CenterNet ResNet encoder, three learned deconvolutions and independent heads.
///
/// Each4x4 stride-two transposed convolution has BN/ReLU. Heatmap output bias
/// is-2.19; head kernels use normal stddev.001. Full18/80-class model counts
/// 14046164 learned/10496 state scalars. Compact18/8,64/32/32 upsampling and
/// 32-wide10-class heads count319862/1456. No optimizer or pretraining is hidden;
/// keep learned leaves unchanged until backward, and checkpoint both stores.
///
/// @param p Mutable Float32/64 learned store; warm leaves before optimizer construction.
/// @param state Mutable same-dtype BN state store, checkpoint with p.
/// @param image Floating positive NHWC, H/W>=32 divisible by32,1 or3 channels per cfg.
/// @param cfg Valid class/backbone/upsampling/head widths and state-update policy.
/// @param training False(default) uses moving BN state; true uses batch stats and optional updates.
/// @return Stride-four features, heatmap/size/offset heads and per-class local peaks.
/// @throws std::invalid_argument For invalid input/settings/dtypes or incompatible existing leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::CenterNetConfig c;
/// c.classes = 10;
/// c.width_divisor = 8;
/// c.upsample_widths = {64, 32, 32};
/// c.head_width = 32;
/// c.grayscale = true;
/// const auto net = n::models::CenterNet(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see MakeCenterTargets(), CenterLoss(), DecodeCenterDetections()
inline CenterNetOutputs CenterNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                  const nn::Tensor &image, const CenterNetConfig &cfg = {},
                                  bool training = false) {
  if (cfg.classes < 1 || cfg.head_width < 1 || image.dim() != 4 || image.shape()[0] < 1 ||
      image.shape()[1] < 32 || image.shape()[2] < 32 || image.shape()[1] % 32 ||
      image.shape()[2] % 32 || image.shape()[3] != (cfg.grayscale ? 1 : 3) ||
      image.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("invalid CenterNet class/head widths");
  for (int width : cfg.upsample_widths)
    if (width < 1)
      throw std::invalid_argument("CenterNet upsample widths must be positive");
  ResNetConfig backbone;
  backbone.depth = cfg.backbone_depth;
  backbone.width_divisor = cfg.width_divisor;
  backbone.include_classifier = false;
  backbone.update_stats = cfg.update_stats;
  const auto net = ResNet(p, state, cfg.grayscale ? nn::tile(image, {1, 1, 1, 3}) : image,
                          backbone, training);
  nn::Tensor h = net.stages[3];
  nn::ConvLayerOptions options;
  options.stride = 2;
  options.bias = false;
  options.init = nn::init::he_uniform();
  nn::BatchNormOptions bn;
  bn.momentum = .9;
  bn.eps = 1e-5;
  bn.update_stats = cfg.update_stats;
  for (int i = 0; i < 3; ++i) {
    const auto name = "upsample" + std::to_string(i + 1);
    h = nn::relu(nn::batch_norm(
        p, state, name + "/bn",
        nn::conv2d_transpose(p, name, h, cfg.upsample_widths[std::size_t(i)], 4, options),
        training, bn));
  }
  CenterNetOutputs result;
  result.features = h;
  result.heatmap = CenterHead(p, "heatmap", h, cfg.head_width, cfg.classes, -2.19);
  result.size = CenterHead(p, "size", h, cfg.head_width, 2);
  result.offset = CenterHead(p, "offset", h, cfg.head_width, 2);
  result.peaks = CenterPeaks(result.heatmap);
  return result;
}

/// @brief Evaluate the authors three-quadratic Gaussian radius formula with tensors.
///
/// Preserves the published code denominators, including r2/r3 division by2.
/// Does not reinterpret them as alternative quadratic roots. No mutation or fitting.
///
/// @param height Float32/64 tensor of finite strictly positive object heights in heatmap units.
/// @param width Same-dtype broadcastable finite positive widths.
/// @param overlap Finite desired overlap in(0,1], default.7.
/// @return Broadcast tensor minimum radius from the authors formula, before floor/integer conversion.
/// @throws std::invalid_argument For invalid dtype, dimensions, nonpositive/nonfinite extents or overlap.
/// @par Example
/// @code
/// const auto radius =
///     dm::nn::models::CenterGaussianRadius(dm::nn::full({}, 10., dm::nn::DType::Float64),
///                                          dm::nn::full({}, 10., dm::nn::DType::Float64));
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see MakeCenterTargets()
inline nn::Tensor CenterGaussianRadius(const nn::Tensor &height, const nn::Tensor &width,
                                       double overlap = .7) {
  if ((height.dtype() != nn::DType::Float32 && height.dtype() != nn::DType::Float64) ||
      width.dtype() != height.dtype() || !(overlap > 0 && overlap <= 1) ||
      !nn::all(
           nn::logical_and(nn::greater(height, 0.),
                           nn::less(nn::abs(height), std::numeric_limits<double>::infinity())))
           .item<bool>() ||
      !nn::all(
           nn::logical_and(nn::greater(width, 0.),
                           nn::less(nn::abs(width), std::numeric_limits<double>::infinity())))
           .item<bool>())
    throw std::invalid_argument("invalid CenterNet radius dimensions/overlap");
  const auto b1 = height + width, c1 = height * width * ((1 - overlap) / (1 + overlap)),
             r1 = (b1 + nn::sqrt(nn::square(b1) - 4. * c1)) * .5, b2 = 2. * (height + width),
             c2 = (1 - overlap) * height * width,
             r2 = (b2 + nn::sqrt(nn::square(b2) - 16. * c2)) * .5,
             b3 = -2. * overlap * (height + width), c3 = (overlap - 1) * height * width,
             r3 = (b3 + nn::sqrt(nn::square(b3) - 16. * overlap * c3)) * .5;
  return nn::minimum(nn::minimum(r1, r2), r3);
}
/// @brief Render maximum-combined finite-support Gaussian class heatmaps from tensor centres.
///
/// Same-class overlaps combine by maximum, not sum. Empty K yields zeros.
/// All grid, support, exponent and combination arithmetic is tensor based; no fit.
///
/// @param height Positive heatmap height.
/// @param width Positive heatmap width.
/// @param classes Positive heatmap class count.
/// @param centres Int64 [K,2] row,column within grid.
/// @param labels Int64 [K] class IDs in[0,classes).
/// @param radii Int64 [K] nonnegative support radius.
/// @param dtype Float32(default) or Float64 output.
/// @return Detached [H,W,C] max-combined Gaussians, centre1, sigma=(2*radius+1)/6, clipped at grid edges.
/// @throws std::invalid_argument For invalid geometry/dtypes or out-of-range centres/labels/radii.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto h = n::models::CenterGaussianHeatmap(
///     7, 7, 1, n::from_vector(std::vector<std::int64_t>{3, 3}, {1, 2}),
///     n::zeros({1}, n::DType::Int64), n::full({1}, 2., n::DType::Int64));
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see MakeCenterTargets()
inline nn::Tensor CenterGaussianHeatmap(std::int64_t height, std::int64_t width,
                                        std::int64_t classes, const nn::Tensor &centres,
                                        const nn::Tensor &labels, const nn::Tensor &radii,
                                        nn::DType dtype = nn::DType::Float32) {
  if (height < 1 || width < 1 || classes < 1 || centres.dim() != 2 ||
      centres.shape()[1] != 2 || centres.dtype() != nn::DType::Int64 ||
      labels.shape() != nn::Shape{centres.shape()[0]} || radii.shape() != labels.shape() ||
      labels.dtype() != nn::DType::Int64 || radii.dtype() != nn::DType::Int64 ||
      (dtype != nn::DType::Float32 && dtype != nn::DType::Float64))
    throw std::invalid_argument("invalid Gaussian metadata");
  nn::NoGradGuard guard;
  const auto K = centres.shape()[0];
  if (!K)
    return nn::zeros({height, width, classes}, dtype);
  const auto c = nn::unstack(centres, 1);
  if (nn::any(nn::logical_or(nn::less(c[0], 0.), nn::greater_equal(c[0], double(height))))
          .item<bool>() ||
      nn::any(nn::logical_or(nn::less(c[1], 0.), nn::greater_equal(c[1], double(width))))
          .item<bool>() ||
      nn::any(nn::logical_or(nn::less(labels, 0.), nn::greater_equal(labels, double(classes))))
          .item<bool>() ||
      nn::any(nn::less(radii, 0.)).item<bool>())
    throw std::invalid_argument("invalid Gaussian centre/label/radius");
  const auto dy = nn::arange(0, double(height)).to(dtype).reshape({height, 1, 1}) -
                  c[0].to(dtype).reshape({1, 1, K}),
             dx = nn::arange(0, double(width)).to(dtype).reshape({1, width, 1}) -
                  c[1].to(dtype).reshape({1, 1, K}),
             radius = radii.to(dtype).reshape({1, 1, K}), sigma = (radius * 2. + 1.) / 6.;
  const auto support = nn::logical_and(nn::less_equal(nn::abs(dy), radius),
                                       nn::less_equal(nn::abs(dx), radius)),
             kernel = nn::where(
                 support,
                 nn::exp(-(nn::square(dy) + nn::square(dx)) / (2. * nn::square(sigma))),
                 nn::zeros({}, dtype));
  const auto hot = nn::one_hot(labels, classes, 1, 0, dtype);
  return nn::max(kernel.reshape({height, width, K, 1}) * hot.reshape({1, 1, K, classes}), {2});
}
/// @brief Compute CenterNet Gaussian-shoulder penalty-reduced heatmap focal terms.
///
/// This differs from alpha-balanced RetinaNet focal loss. Gaussian shoulders
/// reduce negative penalties; they are not soft positive labels. No mutation or fit.
///
/// @param logits Float32/64 logits, finite numerical values required.
/// @param target Same-dtype broadcastable Gaussian values in[0,1].
/// @return Broadcast elementwise positive sigmoid(-x)^2*softplus(-x) at target1, else (1-target)^4*sigmoid(x)^2*softplus(x).
/// @throws std::invalid_argument For invalid dtype/target range or nonbroadcastable shape.
/// @par Example
/// @code
/// const auto l =
///     dm::nn::models::CenterHeatmapFocal(dm::nn::zeros({3}), dm::nn::tensor({0., .5, 1.}));
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see CenterLoss()
inline nn::Tensor CenterHeatmapFocal(const nn::Tensor &logits, const nn::Tensor &target) {
  if ((logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      target.dtype() != logits.dtype() ||
      !nn::all(nn::logical_and(nn::greater_equal(target, 0.), nn::less_equal(target, 1.)))
           .item<bool>())
    throw std::invalid_argument("invalid heatmap focal target");
  const auto positive = nn::square(nn::sigmoid(-logits)) * nn::softplus(-logits);
  const auto negative =
      nn::pow(1.0 - target, 4.0) * nn::square(nn::sigmoid(logits)) * nn::softplus(logits);
  return nn::where(nn::equal(target, 1.0), positive, negative);
}

/// @brief Own detached CenterNet Gaussian, size, offset and retained-centre targets.
///
/// Same-cell collisions keep the larger object, first truth on area ties,
/// regardless of class. Only retained objects draw heatmaps. No fitted state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::tensor({4., 8., 24., 28.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// const auto t = n::models::MakeCenterTargets(data, {0}, 2);
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see MakeCenterTargets(), CenterLoss()
struct CenterTargets {
  nn::Tensor heatmap; ///< [N,H/4,W/4,C] max-combined Gaussian targets.
  nn::Tensor size;    ///< [N,H/4,W/4,2] height,width targets in grid units.
  nn::Tensor offset;  ///< [N,H/4,W/4,2] dy,dx relative to retained integer cell.
  nn::Tensor mask;    ///< [N,H/4,W/4,1] floating retained-centre0/1.
  std::int64_t collisions =
      0; ///< Discarded object count, including repeated replacements at a shared centre.
};
/// @brief Sum CenterNet heatmap focal and retained-centre L1 size/offset losses.
///
/// An empty image still learns negative heatmap focal; size/offset contributions
/// and derivatives are zero. No fitting, parameter/state mutation or target creation.
///
/// @param heatmap Floating [N,H,W,C] class logits, positive dimensions.
/// @param size Same-dtype [N,H,W,2] unconstrained height,width predictions.
/// @param offset Same-dtype [N,H,W,2] dy,dx predictions.
/// @param target_heatmap Same-shape/dtype Gaussian targets in[0,1].
/// @param target_size Same-shape/dtype size targets.
/// @param target_offset Same-shape/dtype offset targets.
/// @param mask Same-dtype [N,H,W,1] floating retained-centre0/1; finite values required.
/// @return Scalar (summed focal+.1*masked size L1+masked offset L1)/max(retained centres,1).
/// @throws std::invalid_argument For invalid shapes/dtypes or Gaussian target range.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto l = n::models::CenterLoss(n::zeros({1, 4, 4, 2}), n::zeros({1, 4, 4, 2}),
///                                      n::zeros({1, 4, 4, 2}), n::zeros({1, 4, 4, 2}),
///                                      n::zeros({1, 4, 4, 2}), n::zeros({1, 4, 4, 2}),
///                                      n::zeros({1, 4, 4, 1}));
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see CenterHeatmapFocal(), MakeCenterTargets()
inline nn::Tensor CenterLoss(const nn::Tensor &heatmap, const nn::Tensor &size,
                             const nn::Tensor &offset, const nn::Tensor &target_heatmap,
                             const nn::Tensor &target_size, const nn::Tensor &target_offset,
                             const nn::Tensor &mask) {
  if (heatmap.dim() != 4 || heatmap.shape()[0] < 1 || heatmap.shape()[1] < 1 ||
      heatmap.shape()[2] < 1 || heatmap.shape()[3] < 1 ||
      target_heatmap.shape() != heatmap.shape() ||
      size.shape() !=
          nn::Shape{heatmap.shape()[0], heatmap.shape()[1], heatmap.shape()[2], 2} ||
      offset.shape() != size.shape() || target_size.shape() != size.shape() ||
      target_offset.shape() != size.shape() ||
      mask.shape() != nn::Shape{heatmap.shape()[0], heatmap.shape()[1], heatmap.shape()[2], 1})
    throw std::invalid_argument("invalid CenterNet loss geometry");
  for (const auto &x : {size, offset, target_heatmap, target_size, target_offset, mask})
    if (x.dtype() != heatmap.dtype())
      throw std::invalid_argument("CenterNet loss dtype mismatch");
  const auto count = nn::maximum(nn::sum(mask), nn::ones({}, heatmap.dtype()));
  const auto heat = nn::sum(CenterHeatmapFocal(heatmap, target_heatmap));
  const auto l1s = nn::sum(nn::abs(size - target_size) * mask);
  const auto l1o = nn::sum(nn::abs(offset - target_offset) * mask);
  return (heat + l1s * .1 + l1o) / count;
}
/// @brief Evaluate CenterNet loss from network outputs and tensor target fields.
///
/// No fitting or mutation; delegates to the tensor-field overload.
///
/// @param net Consistent heatmap/size/offset tensors.
/// @param t Matching Gaussian/size/offset/mask tensors.
/// @return Scalar loss identical to tensor-field overload.
/// @throws std::invalid_argument For invalid geometry/dtypes/target range.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::CenterNetOutputs net;
/// net.heatmap = n::zeros({1, 4, 4, 2});
/// net.size = n::zeros({1, 4, 4, 2});
/// net.offset = n::zeros({1, 4, 4, 2});
/// n::models::CenterTargets t{n::zeros({1, 4, 4, 2}), n::zeros({1, 4, 4, 2}),
///                            n::zeros({1, 4, 4, 2}), n::zeros({1, 4, 4, 1}), 0};
/// const auto l = n::models::CenterLoss(net, t);
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see CenterLoss()
inline nn::Tensor CenterLoss(const CenterNetOutputs &net, const CenterTargets &t) {
  return CenterLoss(net.heatmap, net.size, net.offset, t.heatmap, t.size, t.offset, t.mask);
}
/// @brief Create stride-four CenterNet targets with larger-object centre collision handling.
///
/// Centres floor(pixel centre/4) and clamp to grid; offsets retain original
/// centre minus clamped integer cell. Largest area wins a shared cell across
/// classes; first truth wins ties. Only winners draw maximum-combined Gaussians.
/// Radius uses Double tensor evaluation of the authors formula before floor,
/// matching source host-double geometry. No fitting, random draws or host arithmetic.
///
/// @param scenes Valid Float32/64 grayscale scene metadata, H/W divisible by32; pixels are not read.
/// @param rows Scene IDs in prediction order.
/// @param classes Positive foreground count above every selected label.
/// @return Detached Gaussian/size/offset/mask targets and discarded-object collision count.
/// @throws std::invalid_argument For invalid scene dimensions, rows, labels or box geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::tensor({4., 8., 24., 28.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// const auto t = n::models::MakeCenterTargets(data, {0}, 2);
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see CenterGaussianRadius(), CenterGaussianHeatmap()
inline CenterTargets MakeCenterTargets(const TensorDetectionScenes &scenes,
                                       const std::vector<std::int64_t> &rows, int classes) {
  two_stage_detection_detail::ValidateScenes(scenes, rows, classes);
  if (scenes.images.shape()[1] % 32 || scenes.images.shape()[2] % 32)
    throw std::invalid_argument("CenterNet needs dimensions divisible by32");
  nn::NoGradGuard guard;
  const auto N = std::int64_t(rows.size()), H = scenes.images.shape()[1] / 4,
             W = scenes.images.shape()[2] / 4;
  const auto type = scenes.images.dtype();
  std::vector<nn::Tensor> heats, sizes, offsets, masks;
  std::int64_t collisions = 0;
  for (auto row : rows) {
    const auto G = scenes.boxes[row].shape()[0];
    auto heat = nn::zeros({H, W, classes}, type), size = nn::zeros({H, W, 2}, type),
         offset = nn::zeros({H, W, 2}, type), mask = nn::zeros({H, W, 1}, type);
    if (G) {
      const auto b = nn::unstack(scenes.boxes[row], 1);
      const auto cy = (b[0] + b[2]) / 8., cx = (b[1] + b[3]) / 8.;
      const auto r = nn::clip(nn::floor(cy), 0., double(H - 1)).to(nn::DType::Int64),
                 c = nn::clip(nn::floor(cx), 0., double(W - 1)).to(nn::DType::Int64),
                 cell = r * nn::full({}, double(W), nn::DType::Int64) + c,
                 area = (b[2] - b[0]) * (b[3] - b[1]);
      const auto same = nn::equal(
                     nn::arange(0, double(H * W), 1, nn::DType::Int64).reshape({H * W, 1}),
                     cell.reshape({1, G})),
                 occupied = nn::any(same, {1});
      const auto winners = nn::argmax(
                     nn::where(same, area.reshape({1, G}),
                               nn::full({}, -std::numeric_limits<double>::infinity(), type)),
                     1),
                 cells = two_stage_detection_detail::Indices(occupied),
                 chosen = nn::gather(winners, cells);
      const auto K = chosen.shape()[0];
      collisions += G - K;
      const auto heights = nn::gather((b[2] - b[0]) / 4., chosen),
                 widths = nn::gather((b[3] - b[1]) / 4., chosen), rr = nn::gather(r, chosen),
                 cc = nn::gather(c, chosen);
      const auto radius = nn::floor(CenterGaussianRadius(heights.to(nn::DType::Float64),
                                                         widths.to(nn::DType::Float64)))
                              .to(nn::DType::Int64);
      heat = CenterGaussianHeatmap(H, W, classes, nn::stack({rr, cc}, 1),
                                   nn::gather(scenes.labels[row], chosen), radius, type);
      const auto hot = nn::transpose(nn::one_hot(cells, H * W, 1, 0, type));
      size = nn::matmul(hot, nn::stack({heights, widths}, 1)).reshape({H, W, 2});
      offset = nn::matmul(hot, nn::stack({nn::gather(cy, chosen) - rr.to(type),
                                          nn::gather(cx, chosen) - cc.to(type)},
                                         1))
                   .reshape({H, W, 2});
      mask = occupied.to(type).reshape({H, W, 1});
    }
    heats.push_back(heat);
    sizes.push_back(size);
    offsets.push_back(offset);
    masks.push_back(mask);
  }
  return {rows.empty() ? nn::zeros({N, H, W, classes}, type) : nn::stack(heats, 0),
          rows.empty() ? nn::zeros({N, H, W, 2}, type) : nn::stack(sizes, 0),
          rows.empty() ? nn::zeros({N, H, W, 2}, type) : nn::stack(offsets, 0),
          rows.empty() ? nn::zeros({N, H, W, 1}, type) : nn::stack(masks, 0), collisions};
}
/// @brief Decode up to100 classwise local peaks per image without box NMS.
///
/// Takes top100 BEFORE rejecting nonpositive sizes/zero-area boxes, preserving
/// source order. Centres/sizes multiply by stride4. Every tied3x3 maximum can
/// survive; no box suppression or fitting. Only supplied peak values are decoded.
///
/// @param peaks Float32/64 [N,H,W,C] local-maximum probabilities in[0,1].
/// @param size Same-dtype [N,H,W,2] height,width in grid units.
/// @param offset Same-dtype [N,H,W,2] dy,dx grid offsets.
/// @param images Int64 [N] nonnegative dataset IDs.
/// @param height Positive original image clipping height.
/// @param width Positive original image clipping width.
/// @return Detached tensor detections, strict score>.01, higher flat-index ties, clipped positive-area boxes.
/// @throws std::invalid_argument For invalid shapes/dtypes, IDs, dimensions, nonfinite values or probability range.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::DecodeCenterDetections(
///     n::full({1, 1, 1, 2}, .9), n::full({1, 1, 1, 2}, 2.), n::full({1, 1, 1, 2}, .5),
///     n::zeros({1}, n::DType::Int64), 64, 64);
/// @endcode
/// @par References
/// - Zhou et al. (2019), Objects as Points, https://arxiv.org/abs/1904.07850; authors radius/gaussian code, https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
/// @see CenterPeaks(), CenterNet()
inline TensorDetections DecodeCenterDetections(const nn::Tensor &peaks, const nn::Tensor &size,
                                               const nn::Tensor &offset,
                                               const nn::Tensor &images, int height,
                                               int width) {
  if (peaks.dim() != 4 || peaks.shape()[0] < 1 || peaks.shape()[1] < 1 ||
      peaks.shape()[2] < 1 || peaks.shape()[3] < 1 ||
      size.shape() != nn::Shape{peaks.shape()[0], peaks.shape()[1], peaks.shape()[2], 2} ||
      offset.shape() != size.shape() || images.shape() != nn::Shape{peaks.shape()[0]} ||
      images.dtype() != nn::DType::Int64 || size.dtype() != peaks.dtype() ||
      offset.dtype() != peaks.dtype() || height < 1 || width < 1 ||
      !nn::all(nn::logical_and(nn::greater_equal(peaks, 0.), nn::less_equal(peaks, 1.)))
           .item<bool>())
    throw std::invalid_argument("invalid CenterNet decode inputs");
  nn::NoGradGuard guard;
  const auto N = peaks.shape()[0], H = peaks.shape()[1], W = peaks.shape()[2],
             C = peaks.shape()[3];
  const auto type = peaks.dtype();
  std::vector<nn::Tensor> bs, ss, ls, ids;
  for (std::int64_t row = 0; row < N; ++row) {
    const auto prob = nn::slice_axis(peaks, 0, row, row + 1).reshape({H * W * C});
    const auto above = nn::reverse(two_stage_detection_detail::Indices(nn::greater(prob, .01)),
                                   {0}),
               score = nn::gather(prob, above);
    const auto ranked = nn::slice_axis(nn::argsort(score, true), 0, 0,
                                       std::min<std::int64_t>(100, above.shape()[0])),
               flat = nn::gather(above, ranked);
    const auto pixel = nn::floor(flat.to(type) / double(C)).to(nn::DType::Int64),
               label = flat - pixel * nn::full({}, double(C), nn::DType::Int64),
               r = nn::floor(pixel.to(type) / double(W)), c = pixel.to(type) - r * double(W);
    const auto
        sz = nn::unstack(
            nn::gather(nn::slice_axis(size, 0, row, row + 1).reshape({H * W, 2}), pixel), 1),
        off = nn::unstack(
            nn::gather(nn::slice_axis(offset, 0, row, row + 1).reshape({H * W, 2}), pixel), 1);
    const auto y = (r + off[0]) * 4., x = (c + off[1]) * 4., h = sz[0] * 4., w = sz[1] * 4.;
    const auto boxes = nn::clip_boxes(
        nn::stack({y - h * .5, x - w * .5, y + h * .5, x + w * .5}, 1), height, width);
    const auto keep = two_stage_detection_detail::Indices(
        nn::logical_and(nn::logical_and(nn::greater(h, 0.), nn::greater(w, 0.)),
                        nn::greater(nn::box_area(boxes), 0.)));
    bs.push_back(nn::gather(boxes, keep));
    ss.push_back(nn::gather(nn::gather(prob, flat), keep));
    ls.push_back(nn::gather(label, keep));
    ids.push_back(nn::ones({keep.shape()[0]}, nn::DType::Int64) *
                  nn::gather(images, nn::full({}, double(row), nn::DType::Int64)));
  }
  TensorDetections out{nn::concat(bs, 0), nn::concat(ss, 0), nn::concat(ls, 0),
                       nn::concat(ids, 0)};
  DetectionEvaluationRecords(out);
  return out;
}
} // namespace datamunge::nn::models
