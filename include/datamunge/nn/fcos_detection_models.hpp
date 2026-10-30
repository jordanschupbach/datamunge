#pragma once
/// @file fcos_detection_models.hpp
/// @brief Tensor FCOS towers, point geometry, smallest-area targets and centerness inference.
#include "datamunge/nn/retina_detection_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Configure FCOS backbone, shared group-normalized towers and distance units.
///
/// Full defaults use ResNet50 and width256/GN32. Compact source uses ResNet18/8,
/// width32/GN8 and optional stride-scaled distances. No fitting or pretraining.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::FcosConfig c;
/// c.classes = 10;
/// c.image_size = 64;
/// c.backbone_depth = 18;
/// c.width_divisor = 8;
/// c.pyramid_width = 32;
/// c.groups = 8;
/// c.grayscale = true;
/// c.stride_scaled_distances = true;
/// const auto net = n::models::Fcos(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Tian et al. (2019), FCOS: Fully Convolutional One-Stage Object Detection, https://arxiv.org/abs/1904.01355
/// @see Fcos()
struct FcosConfig {
  int classes = 80;        ///< Positive foreground sigmoid class count.
  int image_size = 640;    ///< Square side >=32, divisible by32.
  int backbone_depth = 50; ///< Supported ResNet depth.
  int width_divisor = 1;   ///< Positive divisor of64 for backbone widths.
  int pyramid_width = 256; ///< Positive shared tower/pyramid width.
  int groups = 32;         ///< Positive group-norm group count dividing pyramid width.
  bool grayscale = false;  ///< Tile one-channel input to RGB when true.
  bool stride_scaled_distances = false; ///< Multiply exp regression by level stride when true.
  bool update_stats = true;             ///< Update backbone BN state in training when true.
};
/// @brief Retain five-level FCOS logits, positive distances and centerness logits.
///
/// Graphs borrow learned leaves owned by p. No background class column. Point
/// order is level,row,column; classes are last. No fit or state ownership.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::FcosConfig c;
/// c.classes = 10;
/// c.image_size = 64;
/// c.backbone_depth = 18;
/// c.width_divisor = 8;
/// c.pyramid_width = 32;
/// c.groups = 8;
/// c.grayscale = true;
/// c.stride_scaled_distances = true;
/// const auto net = n::models::Fcos(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Tian et al. (2019), FCOS: Fully Convolutional One-Stage Object Detection, https://arxiv.org/abs/1904.01355
/// @see Fcos(), FcosLoss()
struct FcosOutputs {
  std::vector<nn::Tensor> features; ///< Five P3–P7 NHWC pyramid tensors.
  std::vector<int> cells;           ///< Square grid sides for five levels.
  nn::Tensor logits;                ///< [N,P,C] foreground sigmoid logits.
  nn::Tensor distances;             ///< [N,P,4] positive l,t,r,b pixel distances.
  nn::Tensor centerness;            ///< [N,P] centerness logits from classification tower.
  int points = 0;                   ///< Sum of grid cell counts.
};
/// @brief Build source FCOS with shared four-convolution GN/ReLU towers and five learned scales.
///
/// Centerness comes from the classification tower, following original FCOS.
/// Each level has one learned scalar before exp(min(scale*raw,20)). GN affines
/// are learned but have no moving state; backbone BN does. Full model counts
/// 36426138 learned and53120 state scalars; compact counts314684/1200. No fit
/// or pretraining is hidden. Do not mutate learned leaves before backward.
///
/// @param p Mutable same-dtype learned store; warm all leaves before optimizer creation.
/// @param state Mutable same-dtype backbone BN store; checkpoint with p.
/// @param images Floating [N,S,S,1 or3] matching cfg geometry and p dtype.
/// @param cfg Valid class, backbone, width, group, distance-unit and state-update metadata.
/// @param training False(default) uses moving BN state; true uses batch statistics and optional updates.
/// @return Five pyramid maps, foreground logits, ltrb distances and centerness logits.
/// @throws std::invalid_argument For invalid input/settings/dtypes or incompatible store leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::FcosConfig c;
/// c.classes = 10;
/// c.image_size = 64;
/// c.backbone_depth = 18;
/// c.width_divisor = 8;
/// c.pyramid_width = 32;
/// c.groups = 8;
/// c.grayscale = true;
/// c.stride_scaled_distances = true;
/// const auto net = n::models::Fcos(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Tian et al. (2019), FCOS: Fully Convolutional One-Stage Object Detection, https://arxiv.org/abs/1904.01355
/// @see FcosPoints(), MakeFcosTargets(), FcosLoss()
inline FcosOutputs Fcos(nn::ParameterStore &p, nn::ParameterStore &state,
                        const nn::Tensor &images, const FcosConfig &cfg = {},
                        bool training = false) {
  if (cfg.classes < 1 || cfg.image_size < 32 || cfg.image_size % 32 || cfg.groups < 1 ||
      cfg.pyramid_width < 1 || cfg.pyramid_width % cfg.groups || images.dim() != 4 ||
      images.shape()[0] < 1 || images.shape()[1] != cfg.image_size ||
      images.shape()[2] != cfg.image_size || images.shape()[3] != (cfg.grayscale ? 1 : 3) ||
      images.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("invalid FCOS configuration");
  FcosOutputs result;
  result.features =
      RetinaPyramid(p, state, images, cfg.backbone_depth, cfg.width_divisor, cfg.pyramid_width,
                    cfg.grayscale, training, cfg.update_stats);
  std::vector<nn::Tensor> logits, distances, centerness;
  for (int level = 0; level < 5; ++level) {
    const int stride = 8 << level, cells = (cfg.image_size + stride - 1) / stride,
              count = cells * cells;
    if (result.features[std::size_t(level)].shape()[1] != cells)
      throw std::invalid_argument("FCOS image_size does not match the input images");
    result.cells.push_back(cells);
    result.points += count;
    nn::Tensor classification = result.features[std::size_t(level)],
               regression = classification;
    for (int i = 0; i < 4; ++i) {
      const auto name = std::to_string(i + 1);
      classification =
          nn::relu(nn::group_norm(p, "classification/gn" + name,
                                  RetinaConv(p, "classification/conv" + name, classification,
                                             cfg.pyramid_width, false),
                                  cfg.groups));
      regression = nn::relu(nn::group_norm(
          p, "regression/gn" + name,
          RetinaConv(p, "regression/conv" + name, regression, cfg.pyramid_width, false),
          cfg.groups));
    }
    logits.push_back(nn::reshape(RetinaConv(p, "classification/output", classification,
                                            cfg.classes, false, std::log(.01 / .99)),
                                 {-1, count, cfg.classes}));
    centerness.push_back(
        nn::reshape(RetinaConv(p, "centerness", classification, 1, false), {-1, count}));
    const auto raw = RetinaConv(p, "regression/output", regression, 4, false);
    const auto scale = p.get("scale" + std::to_string(level), {}, nn::init::ones());
    // Limit the exponent only far beyond useful image distances, preventing overflow.
    const auto positive = nn::exp(nn::minimum(raw * scale, nn::full({}, 20., raw.dtype())));
    distances.push_back(nn::reshape(
        positive * (cfg.stride_scaled_distances ? double(stride) : 1.0), {-1, count, 4}));
  }
  result.logits = nn::concat(logits, 1);
  result.distances = nn::concat(distances, 1);
  result.centerness = nn::concat(centerness, 1);
  return result;
}

/// @brief Create stride-centred FCOS point coordinates and inclusive level ranges.
///
/// Strides are8/16/32/64/128. Ranges are[0,64],[64,128],...,[512,+inf],
/// scaled only at their finite endpoints. Points use stride*(index+.5). No fit.
///
/// @param cells Five positive grid sides in P3–P7 order.
/// @param range_scale Finite positive regression-range multiplier, default1; compact128 uses.25.
/// @param dtype Float32(default) or Float64 coordinate arithmetic.
/// @return Detached [P,5] rows y,x,stride,minimum,maximum in level,row,column order.
/// @throws std::invalid_argument For invalid grid metadata, scale or dtype.
/// @par Example
/// @code
/// const auto points = dm::nn::models::FcosPoints({16, 8, 4, 2, 1}, .25);
/// @endcode
/// @par References
/// - Tian et al. (2019), FCOS: Fully Convolutional One-Stage Object Detection, https://arxiv.org/abs/1904.01355
/// @see MakeFcosTargets()
inline nn::Tensor FcosPoints(const std::vector<int> &cells, double range_scale = 1.,
                             nn::DType dtype = nn::DType::Float32) {
  if (cells.size() != 5 || !(range_scale > 0) || !std::isfinite(range_scale) ||
      (dtype != nn::DType::Float32 && dtype != nn::DType::Float64))
    throw std::invalid_argument("invalid FCOS point metadata");
  nn::NoGradGuard guard;
  std::vector<nn::Tensor> result;
  for (int l = 0; l < 5; ++l) {
    if (cells[l] < 1)
      throw std::invalid_argument("FCOS grid side must be positive");
    const auto C = std::int64_t(cells[l]);
    const double stride = 8 << l;
    const auto pos = (nn::arange(0, double(C)).to(dtype) + .5) * stride,
               ones = nn::ones({C, C}, dtype);
    result.push_back(nn::stack({pos.reshape({C, 1}) * ones, pos.reshape({1, C}) * ones,
                                ones * stride, ones * (l ? double(32 << l) * range_scale : 0.),
                                nn::full({C, C},
                                         l < 4 ? double(64 << l) * range_scale
                                               : std::numeric_limits<double>::infinity(),
                                         dtype)},
                               2)
                         .reshape({C * C, 5}));
  }
  return nn::concat(result, 0);
}
/// @brief Compute zero-safe FCOS centerness from positive l,t,r,b distances.
///
/// Background rows use a safe denominator before masking, avoiding0/0. No mutation or fit.
///
/// @param distances Float32/64 [...,4] finite distances; nonpositive rows return zero.
/// @return Same-dtype [...] sqrt(min(l,r)/max(l,r)*min(t,b)/max(t,b)), gradients retained.
/// @throws std::invalid_argument For invalid rank, last dimension or dtype.
/// @par Example
/// @code
/// const auto c = dm::nn::models::FcosCenterness(dm::nn::tensor({1., 2., 4., 8.}, {1, 4}));
/// @endcode
/// @par References
/// - Tian et al. (2019), FCOS: Fully Convolutional One-Stage Object Detection, https://arxiv.org/abs/1904.01355
/// @see MakeFcosTargets()
inline nn::Tensor FcosCenterness(const nn::Tensor &distances) {
  if (distances.dim() < 1 || distances.shape().back() != 4 ||
      (distances.dtype() != nn::DType::Float32 && distances.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid FCOS distances");
  const auto d = nn::unstack(distances, -1);
  const auto positive = nn::greater(nn::min(distances, {-1}), 0.);
  const auto one = nn::ones({}, distances.dtype());
  const auto lr = nn::minimum(d[0], d[2]) / nn::where(positive, nn::maximum(d[0], d[2]), one),
             tb = nn::minimum(d[1], d[3]) / nn::where(positive, nn::maximum(d[1], d[3]), one);
  return nn::sqrt(nn::where(positive, lr * tb, nn::zeros({}, distances.dtype())));
}
/// @brief Compute source FCOS IoU loss with plus-one area smoothing.
///
/// The additive one is area smoothing, not a small epsilon. Distances share a
/// reference point, so intersection sides sum minima. No mutation or fitting.
///
/// @param prediction Float32/64 [N,P,4] nonnegative l,t,r,b distances; finite values are caller requirement.
/// @param target Same-shape/dtype nonnegative target distances.
/// @return [N,P] -log((intersection+1)/(union+1)), with gradients to both arguments.
/// @throws std::invalid_argument For invalid shape/dtype.
/// @par Example
/// @code
/// const auto loss =
///     dm::nn::models::FcosIouLoss(dm::nn::ones({1, 2, 4}), dm::nn::full({1, 2, 4}, 2.));
/// @endcode
/// @par References
/// - Tian et al. (2019), FCOS: Fully Convolutional One-Stage Object Detection, https://arxiv.org/abs/1904.01355
/// @see FcosLoss()
inline nn::Tensor FcosIouLoss(const nn::Tensor &prediction, const nn::Tensor &target) {
  if (prediction.dim() != 3 || prediction.shape()[2] != 4 ||
      target.shape() != prediction.shape() || target.dtype() != prediction.dtype() ||
      (prediction.dtype() != nn::DType::Float32 && prediction.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid FCOS IoU inputs");
  const auto a = nn::unstack(prediction, 2), b = nn::unstack(target, 2);
  const auto area_a = (a[0] + a[2]) * (a[1] + a[3]);
  const auto area_b = (b[0] + b[2]) * (b[1] + b[3]);
  const auto intersection = (nn::minimum(a[0], b[0]) + nn::minimum(a[2], b[2])) *
                            (nn::minimum(a[1], b[1]) + nn::minimum(a[3], b[3]));
  return -nn::log((intersection + 1.0) / (area_a + area_b - intersection + 1.0));
}
/// @brief Own detached FCOS classes, ltrb targets, positives and centerness.
///
/// No model parameters or moving state. All tensors use the scene dtype and
/// point/batch order. Background classes/distances/centerness are zero.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FcosTargets t{n::zeros({1, 2, 3}), n::ones({1, 2, 4}), n::ones({1, 2}),
///                          n::ones({1, 2})};
/// const auto loss =
///     n::models::FcosLoss(n::zeros({1, 2, 3}), n::ones({1, 2, 4}), n::zeros({1, 2}), t);
/// @endcode
/// @par References
/// - Tian et al. (2019), FCOS: Fully Convolutional One-Stage Object Detection, https://arxiv.org/abs/1904.01355
/// @see MakeFcosTargets(), FcosLoss()
struct FcosTargets {
  nn::Tensor classes;    ///< [N,P,C] binary foreground targets.
  nn::Tensor distances;  ///< [N,P,4] l,t,r,b targets in pixels.
  nn::Tensor positive;   ///< [N,P] floating0/1 mask.
  nn::Tensor centerness; ///< [N,P] target centerness, zero for backgrounds.
};
/// @brief Sum focal classification, positive centerness BCE and centerness-weighted IoU loss.
///
/// Count is batch positives floored at one. Background-only batches retain focal
/// learning; centre/box losses are zero. No optimizer, mutation or hidden fitting.
///
/// @param logits Floating [N,P,C] logits with positive N/C; Float32/64.
/// @param distances Same-dtype [N,P,4] positive predicted ltrb distances.
/// @param centerness Same-dtype [N,P] centerness logits.
/// @param t Consistent same-dtype targets; finite valid binary classes/masks and centerness in[0,1] required.
/// @return Scalar focal/count + centre-BCE/count + weighted-IoU/max(sum weights,1e-6).
/// @throws std::invalid_argument For invalid tensor shapes/dtypes or binary focal targets.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FcosTargets t{n::zeros({1, 2, 3}), n::ones({1, 2, 4}), n::ones({1, 2}),
///                          n::ones({1, 2})};
/// const auto loss =
///     n::models::FcosLoss(n::zeros({1, 2, 3}), n::ones({1, 2, 4}), n::zeros({1, 2}), t);
/// @endcode
/// @par References
/// - Tian et al. (2019), FCOS: Fully Convolutional One-Stage Object Detection, https://arxiv.org/abs/1904.01355
/// @see FcosIouLoss(), SigmoidFocalLoss()
inline nn::Tensor FcosLoss(const nn::Tensor &logits, const nn::Tensor &distances,
                           const nn::Tensor &centerness, const FcosTargets &t) {
  if (logits.dim() != 3 || logits.shape()[0] < 1 || logits.shape()[2] < 1 ||
      distances.shape() != nn::Shape{logits.shape()[0], logits.shape()[1], 4} ||
      centerness.shape() != nn::Shape{logits.shape()[0], logits.shape()[1]} ||
      t.classes.shape() != logits.shape() || t.distances.shape() != distances.shape() ||
      t.positive.shape() != centerness.shape() || t.centerness.shape() != centerness.shape())
    throw std::invalid_argument("invalid FCOS loss geometry");
  for (const auto &x :
       {distances, centerness, t.classes, t.distances, t.positive, t.centerness})
    if (x.dtype() != logits.dtype())
      throw std::invalid_argument("FCOS loss dtype mismatch");
  const auto count = nn::maximum(nn::sum(t.positive), nn::ones({}, logits.dtype()));
  const auto classification = nn::sum(SigmoidFocalLoss(logits, t.classes)) / count;
  const auto center_bce = nn::softplus(centerness) - centerness * t.centerness;
  const auto center = nn::sum(center_bce * t.positive) / count;
  const auto weights = t.centerness * t.positive;
  const auto regression = nn::sum(FcosIouLoss(distances, t.distances) * weights) /
                          nn::maximum(nn::sum(weights), nn::full({}, 1e-6, logits.dtype()));
  return classification + (center + regression);
}
/// @brief Evaluate FCOS loss directly from retained network outputs.
///
/// No fitting or mutation; delegates to documented tensor-field overload.
///
/// @param net Network tensors with consistent shape/dtype.
/// @param t Matching detached targets and masks.
/// @return Scalar objective identical to the tensor-field overload.
/// @throws std::invalid_argument For invalid geometry/dtype/targets.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FcosOutputs net;
/// net.logits = n::zeros({1, 2, 3});
/// net.distances = n::ones({1, 2, 4});
/// net.centerness = n::zeros({1, 2});
/// n::models::FcosTargets t{n::zeros({1, 2, 3}), n::zeros({1, 2, 4}), n::zeros({1, 2}),
///                          n::zeros({1, 2})};
/// const auto loss = n::models::FcosLoss(net, t);
/// @endcode
/// @par References
/// - Tian et al. (2019), FCOS: Fully Convolutional One-Stage Object Detection, https://arxiv.org/abs/1904.01355
/// @see FcosLoss(const nn::Tensor&, const nn::Tensor&, const nn::Tensor&, const FcosTargets&)
inline nn::Tensor FcosLoss(const FcosOutputs &net, const FcosTargets &t) {
  return FcosLoss(net.logits, net.distances, net.centerness, t);
}
/// @brief Assign strictly interior, range-valid FCOS points to the smallest eligible truth.
///
/// Largest ltrb distance must lie within inclusive level min/max. Centre sampling
/// uses strict abs(x-cx),abs(y-cy)<radius*stride. Smallest area wins; exact ties
/// choose first truth. No forced match, fitting, random draws or host box arithmetic.
///
/// @param points Float32/64 [P,5] y,x,stride,min,max with positive stride and valid ranges; +inf max allowed.
/// @param scenes Valid same-dtype grayscale scene metadata; pixels are not read.
/// @param rows Dataset IDs in prediction batch order.
/// @param classes Positive foreground class count, above all selected labels.
/// @param center_radius Finite radius multiplier, default1.5; <=0 disables centre sampling.
/// @return Detached tensor targets with first-truth ties and zero background rows.
/// @throws std::invalid_argument For invalid scene/point shape, dtype, ranges, stride, labels or rows.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::tensor({0., 0., 16., 16.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// const auto t =
///     n::models::MakeFcosTargets(n::models::FcosPoints({8, 4, 2, 1, 1}, .125), data, {0}, 2);
/// @endcode
/// @par References
/// - Tian et al. (2019), FCOS: Fully Convolutional One-Stage Object Detection, https://arxiv.org/abs/1904.01355
/// @see FcosPoints(), FcosLoss()
inline FcosTargets MakeFcosTargets(const nn::Tensor &points,
                                   const TensorDetectionScenes &scenes,
                                   const std::vector<std::int64_t> &rows, int classes,
                                   double center_radius = 1.5) {
  two_stage_detection_detail::ValidateScenes(scenes, rows, classes);
  if (points.dim() != 2 || points.shape()[1] != 5 || points.dtype() != scenes.images.dtype() ||
      !std::isfinite(center_radius))
    throw std::invalid_argument("invalid FCOS target points");
  nn::NoGradGuard guard;
  const auto P = points.shape()[0], N = std::int64_t(rows.size());
  const auto type = points.dtype();
  const auto cols = nn::unstack(points, 1);
  if (nn::any(nn::less_equal(cols[2], 0.)).item<bool>() ||
      nn::any(nn::less(cols[3], 0.)).item<bool>() ||
      nn::any(nn::less(cols[4], cols[3])).item<bool>() ||
      !nn::all(nn::less(nn::abs(nn::slice_axis(points, 1, 0, 4)),
                        std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("invalid FCOS coordinate/range values");
  std::vector<nn::Tensor> classes_out, distances_out, positive_out, centerness_out;
  for (auto row : rows) {
    const auto G = scenes.boxes[row].shape()[0];
    nn::Tensor hot, distance, mask, centre;
    if (G) {
      const auto b = nn::unstack(scenes.boxes[row], 1);
      const auto y = cols[0].reshape({P, 1}), x = cols[1].reshape({P, 1});
      const auto ds = nn::stack({x - b[1].reshape({1, G}), y - b[0].reshape({1, G}),
                                 b[3].reshape({1, G}) - x, b[2].reshape({1, G}) - y},
                                2);
      const auto largest = nn::max(ds, {2});
      auto eligible =
          nn::logical_and(nn::greater(nn::min(ds, {2}), 0.),
                          nn::logical_and(nn::greater_equal(largest, cols[3].reshape({P, 1})),
                                          nn::less_equal(largest, cols[4].reshape({P, 1}))));
      if (center_radius > 0) {
        const auto radius = cols[2].reshape({P, 1}) * center_radius;
        eligible = nn::logical_and(
            eligible,
            nn::logical_and(
                nn::less(nn::abs(x - ((b[1] + b[3]) * .5).reshape({1, G})), radius),
                nn::less(nn::abs(y - ((b[0] + b[2]) * .5).reshape({1, G})), radius)));
      }
      const auto areas = ((b[2] - b[0]) * (b[3] - b[1])).reshape({1, G});
      const auto best =
          nn::argmin(nn::where(eligible, areas,
                               nn::full({}, std::numeric_limits<double>::infinity(), type)),
                     1);
      mask = nn::any(eligible, {1}).to(type);
      const auto chosen = nn::one_hot(best, G, 1, 0, type);
      distance = nn::sum(ds * chosen.reshape({P, G, 1}), {1}) * mask.reshape({P, 1});
      hot = nn::one_hot(nn::gather(scenes.labels[row], best), classes, 1, 0, type) *
            mask.reshape({P, 1});
      centre = FcosCenterness(distance);
    } else {
      hot = nn::zeros({P, classes}, type);
      distance = nn::zeros({P, 4}, type);
      mask = nn::zeros({P}, type);
      centre = nn::zeros({P}, type);
    }
    classes_out.push_back(hot);
    distances_out.push_back(distance);
    positive_out.push_back(mask);
    centerness_out.push_back(centre);
  }
  return {rows.empty() ? nn::zeros({N, P, classes}, type) : nn::stack(classes_out, 0),
          rows.empty() ? nn::zeros({N, P, 4}, type) : nn::stack(distances_out, 0),
          rows.empty() ? nn::zeros({N, P}, type) : nn::stack(positive_out, 0),
          rows.empty() ? nn::zeros({N, P}, type) : nn::stack(centerness_out, 0)};
}
/// @brief Decode anchor-free distances and rank candidates by sqrt(class probability times centerness).
///
/// The .01 threshold applies to class probability before multiplying centerness.
/// Top1000 is global across classes per image; stable point/class input order
/// breaks score ties (source partial_sort left ties unspecified). No fitting.
///
/// @param net Floating consistent [N,P,C] logits,[N,P,4] positive finite distances,[N,P] centerness logits.
/// @param points Same-dtype [P,5] point rows; first two columns supply y,x.
/// @param images Int64 [N] dataset IDs.
/// @param height Positive clipping height.
/// @param width Positive clipping width.
/// @param suppress True(default) applies class-wise IoU.6 NMS and100/image; false returns top1000/image before NMS.
/// @return Detached tensor detections after strict class probability>.01, clipping and positive-area filtering.
/// @throws std::invalid_argument For invalid shapes/dtypes, IDs, dimensions, nonfinite or unordered boxes/scores.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FcosOutputs net;
/// net.logits = n::zeros({1, 1, 2});
/// net.distances = n::full({1, 1, 4}, 4.);
/// net.centerness = n::zeros({1, 1});
/// const auto d = n::models::DecodeFcosDetections(net, n::tensor({8., 8., 8., 0., 64.}, {1, 5}),
///                                                n::zeros({1}, n::DType::Int64), 64, 64);
/// @endcode
/// @par References
/// - Tian et al. (2019), FCOS: Fully Convolutional One-Stage Object Detection, https://arxiv.org/abs/1904.01355
/// @see Fcos(), SuppressDetectionCandidates()
inline TensorDetections DecodeFcosDetections(const FcosOutputs &net, const nn::Tensor &points,
                                             const nn::Tensor &images, int height, int width,
                                             bool suppress = true) {
  const auto &logits = net.logits;
  if (logits.dim() != 3 || logits.shape()[0] < 1 || logits.shape()[2] < 1 ||
      points.shape() != nn::Shape{logits.shape()[1], 5} ||
      net.distances.shape() != nn::Shape{logits.shape()[0], logits.shape()[1], 4} ||
      net.centerness.shape() != nn::Shape{logits.shape()[0], logits.shape()[1]} ||
      images.shape() != nn::Shape{logits.shape()[0]} || images.dtype() != nn::DType::Int64 ||
      height < 1 || width < 1)
    throw std::invalid_argument("invalid FCOS decode geometry");
  for (const auto &t : {points, net.distances, net.centerness})
    if (t.dtype() != logits.dtype())
      throw std::invalid_argument("FCOS decode dtype mismatch");
  nn::NoGradGuard guard;
  const auto N = logits.shape()[0], P = logits.shape()[1], C = logits.shape()[2];
  const auto type = logits.dtype();
  const auto xy = nn::unstack(points, 1);
  std::vector<nn::Tensor> bs, ss, ls, ids;
  for (std::int64_t row = 0; row < N; ++row) {
    const auto prob = nn::sigmoid(nn::slice_axis(logits, 0, row, row + 1)).reshape({P, C}),
               centre = nn::sigmoid(nn::slice_axis(net.centerness, 0, row, row + 1))
                            .reshape({P, 1}),
               score = nn::sqrt(prob * centre).reshape({P * C});
    const auto d =
        nn::unstack(nn::slice_axis(net.distances, 0, row, row + 1).reshape({P, 4}), 1);
    const auto box = nn::clip_boxes(
        nn::stack({xy[0] - d[1], xy[1] - d[0], xy[0] + d[3], xy[1] + d[2]}, 1), height, width);
    const auto positive = nn::greater(nn::box_area(box), 0.);
    const auto eligible =
        nn::logical_and(nn::greater(prob, .01), positive.reshape({P, 1})).reshape({P * C});
    const auto index = two_stage_detection_detail::Indices(eligible);
    const auto selected = nn::gather(score, index);
    const auto order = nn::slice_axis(nn::argsort(selected, true), 0, 0,
                                      std::min<std::int64_t>(1000, index.shape()[0]));
    const auto keep = nn::gather(index, order);
    const auto point_index = nn::floor(keep.to(type) / double(C)).to(nn::DType::Int64),
               label = keep - point_index * nn::full({}, double(C), nn::DType::Int64);
    bs.push_back(nn::gather(box, point_index));
    ss.push_back(nn::gather(score, keep));
    ls.push_back(label);
    ids.push_back(nn::ones({keep.shape()[0]}, nn::DType::Int64) *
                  nn::gather(images, nn::full({}, double(row), nn::DType::Int64)));
  }
  TensorDetections out{nn::concat(bs, 0), nn::concat(ss, 0), nn::concat(ls, 0),
                       nn::concat(ids, 0)};
  if (!suppress)
    return out;
  return SuppressDetectionCandidates(out.boxes, out.scores, out.labels, out.images, .6, 0.,
                                     100);
}
} // namespace datamunge::nn::models
