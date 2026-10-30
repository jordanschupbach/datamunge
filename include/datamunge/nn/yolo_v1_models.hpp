#pragma once
/// @file yolo_v1_models.hpp
/// @brief Tensor YOLO v1 convolution/dense detector, cell/IoU targets and optional prefix pretraining.
#include "datamunge/nn/detection_augmentation.hpp"
#include "datamunge/nn/resnet_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Configure source24-convolution YOLO v1 with paper dense or explicit compact head.
///
/// Full defaults use448RGB, no BN,4096 dense units and20 classes. Compact56gray
/// keeps all24 convolutions but removes initial/22nd stride and fourth max-pool
/// to retain7x7 features. Optional BN is a separate compact experiment.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::YoloV1Config c;
/// c.classes = 10;
/// c.compact = true;
/// c.width_divisor = 8;
/// c.dense_width = 8;
/// c.keep_probability = 1;
/// const auto net = n::models::YoloV1(p, &state, n::ones({1, 56, 56, 1}), c);
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see YoloV1()
struct YoloV1Config {
  int classes = 20;       ///< Positive cell-level class score count.
  int width_divisor = 1;  ///< Positive divisor of64 for convolution widths.
  int dense_width = 4096; ///< Positive dense hidden width.
  bool compact =
      false; ///< True uses56gray geometry and removes source downsampling operations.
  bool batch_norm = false; ///< Optional BN; requires nonnull same-dtype state store.
  double keep_probability =
      .5; ///< Dropout keep probability in(0,1], applied before output during training.
  double momentum = .9;     ///< Optional BN moving-state momentum in[0,1), epsilon1e-5.
  bool update_stats = true; ///< Update optional BN moving state only in training when true.
};
/// @brief Retain YOLO v1 features, all prefix convolution activations and optional dense predictions.
///
/// Tensor graphs borrow learned leaves from p. Boxes use dy,dx,sqrt(h),sqrt(w),
/// confidence, with sizes relative to the entire image. Cell classes are shared
/// by both box predictors. No optimizer or owned moving state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::YoloV1Config c;
/// c.classes = 10;
/// c.compact = true;
/// c.width_divisor = 8;
/// c.dense_width = 8;
/// c.keep_probability = 1;
/// const auto net = n::models::YoloV1(p, &state, n::ones({1, 56, 56, 1}), c);
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see YoloV1(), YoloV1Backbone()
struct YoloV1Outputs {
  nn::Tensor features; ///< Last requested convolution/pooling NHWC map.
  nn::Tensor
      prediction; ///< [N,7,7,C+10] linear outputs after full model; undefined for backbone-only calls.
  std::vector<nn::Tensor>
      convolutions; ///< Activation after each requested convolution, before its subsequent pool.
};
/// @brief Return the source24 convolution output-width/kernel-size pairs.
///
/// Immutable static architecture metadata, not parameter allocation or fitting.
/// Returned reference remains valid for program lifetime and is read-only.
///
/// @return Const reference to24 ordered width/kernel pairs.
/// @par Example
/// @code
/// const auto &layers = dm::nn::models::YoloV1Layers();
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see YoloV1Backbone()
inline const std::vector<std::pair<int, int>> &YoloV1Layers() {
  /// @brief Immutable paper architecture table, shared read-only across calls.
  static const std::vector<std::pair<int, int>> layers = {
      {64, 7},  {192, 3},  {128, 1}, {256, 3},  {256, 1},  {512, 3},  {256, 1},  {512, 3},
      {256, 1}, {512, 3},  {256, 1}, {512, 3},  {256, 1},  {512, 3},  {512, 1},  {1024, 3},
      {512, 1}, {1024, 3}, {512, 1}, {1024, 3}, {1024, 3}, {1024, 3}, {1024, 3}, {1024, 3}};
  return layers;
}

/// @brief Build a parameter-name-preserving prefix of the source YOLO v1 convolution stack.
///
/// Leaky-ReLU slope.1 follows each convolution. Full model has no BN by default.
/// Optional BN scales channels with epsilon1e-5/momentum cfg.momentum. Prefix
/// parameter names equal the corresponding detector names. No fitting; keep
/// learned leaves unchanged until backward. Shared stores cannot mutate concurrently.
///
/// @param p Mutable Float32/64 learned store; creates/reuses named convolution/optional BN leaves.
/// @param state Optional mutable same-dtype BN store, required when cfg.batch_norm.
/// @param image Floating NHWC matching p dtype,448RGB full or56gray compact.
/// @param cfg Valid class/width/head/dropout/BN metadata.
/// @param training False(default) uses moving BN state; true uses batch statistics and optional updates.
/// @param convolutions Prefix length1–24, default24; first20 supports crop classifier pretraining.
/// @return Requested convolution activations and final features, with undefined dense prediction.
/// @throws std::invalid_argument For invalid input/settings/dtypes, missing BN store or incompatible existing leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::YoloV1Config c;
/// c.classes = 10;
/// c.compact = true;
/// c.width_divisor = 8;
/// c.dense_width = 8;
/// c.keep_probability = 1;
/// const auto net = n::models::YoloV1(p, &state, n::ones({1, 56, 56, 1}), c);
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see YoloV1(), YoloV1PretrainingLogits()
inline YoloV1Outputs YoloV1Backbone(nn::ParameterStore &p, nn::ParameterStore *state,
                                    const nn::Tensor &image, const YoloV1Config &cfg = {},
                                    bool training = false, int convolutions = 24) {
  if (cfg.classes < 1 || cfg.width_divisor < 1 || 64 % cfg.width_divisor ||
      cfg.dense_width < 1 || (cfg.batch_norm && !state) ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1) ||
      !(cfg.momentum >= 0 && cfg.momentum < 1) || convolutions < 1 || convolutions > 24 ||
      image.dim() != 4 || image.shape()[0] < 1 ||
      image.shape()[1] != (cfg.compact ? 56 : 448) ||
      image.shape()[2] != (cfg.compact ? 56 : 448) ||
      image.shape()[3] != (cfg.compact ? 1 : 3) || image.dtype() != p.dtype() ||
      (state && state->dtype() != p.dtype()))
    throw std::invalid_argument("invalid YOLO v1 configuration");
  const auto &layers = YoloV1Layers();
  YoloV1Outputs result;
  nn::Tensor h = image;
  ResNetNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.epsilon = 1e-5;
  norm.momentum = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  for (std::size_t i = 0; i < std::size_t(convolutions); ++i) {
    const int out = layers[i].first / cfg.width_divisor, kernel = layers[i].second;
    const int stride = !cfg.compact && (i == 0 || i == 21) ? 2 : 1;
    const auto name = "conv" + std::to_string(i + 1);
    if (cfg.batch_norm) {
      h = ResNetConv(p, *state, name, h, out, kernel, norm, stride, false);
    } else {
      nn::ConvLayerOptions conv;
      conv.init = nn::init::he_uniform();
      conv.padding = nn::Padding::Valid;
      conv.stride = stride;
      const std::int64_t pad = kernel / 2;
      h = nn::conv2d(p, name, pad ? nn::pad(h, {{0, 0}, {pad, pad}, {pad, pad}, {0, 0}}) : h,
                     out, kernel, conv);
    }
    h = nn::leaky_relu(h, .1);
    result.convolutions.push_back(h);
    if (i == 0 || i == 1 || i == 5 || (i == 15 && !cfg.compact))
      h = nn::max_pool2d(h, 2, 2, nn::Padding::Valid);
  }
  result.features = h;
  return result;
}

/// @brief Build source YOLO v1 with24 convolutions, a dense hidden layer and linear grid outputs.
///
/// Paper dense model has271703550 scalars at20 classes. Full448 convolution
/// backbone is executable separately; the huge4096-wide head may be counted from
/// shapes without allocation. Compact10-class count is942604+7253*dense_width
/// plus1648 when BN is enabled. Dropout uses the explicit generator or a
/// persistent thread-local fallback, so repeatable training should pass its RNG.
/// No optimizer or fitting is hidden; keep learned leaves until backward.
///
/// @param p Mutable learned store; warm every leaf before optimizer construction.
/// @param state Optional BN moving-state store; required only with cfg.batch_norm and checkpointed with p.
/// @param image Floating448RGB or compact56gray NHWC matching cfg/p.
/// @param cfg Valid source or explicit compact metadata.
/// @param training False(default) disables dropout/BN updates; true enables configured training behavior.
/// @param dropout Optional borrowed generator; null uses persistent per-thread seed37 fallback.
/// @return Convolutional maps and [N,7,7,C+10] linear class/box predictions.
/// @throws std::invalid_argument For invalid model inputs/settings or incompatible existing leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::YoloV1Config c;
/// c.classes = 10;
/// c.compact = true;
/// c.width_divisor = 8;
/// c.dense_width = 8;
/// c.keep_probability = 1;
/// const auto net = n::models::YoloV1(p, &state, n::ones({1, 56, 56, 1}), c);
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see YoloV1Loss(), DecodeYoloV1Detections()
inline YoloV1Outputs YoloV1(nn::ParameterStore &p, nn::ParameterStore *state,
                            const nn::Tensor &image, const YoloV1Config &cfg = {},
                            bool training = false, nn::Generator *dropout = nullptr) {
  auto result = YoloV1Backbone(p, state, image, cfg, training);
  const std::int64_t in = 1024 / cfg.width_divisor;
  auto h = nn::leaky_relu(
      nn::dense(p, "fc1", nn::reshape(result.features, {-1, 7 * 7 * in}), cfg.dense_width),
      .1);
  if (training && cfg.keep_probability < 1) {
    /// @brief Persistent per-thread fallback RNG; explicit caller RNG gives reproducible run ownership.
    static thread_local nn::Generator fallback(37);
    h = nn::dropout(h, 1 - cfg.keep_probability, true, dropout ? *dropout : fallback);
  }
  result.prediction = nn::reshape(nn::dense(p, "output", h, 7 * 7 * (cfg.classes + 10)),
                                  {-1, 7, 7, cfg.classes + 10});
  return result;
}

/// @brief Own YOLO v1 cell object masks, class IDs and root-size targets.
///
/// Detached tensor data; one retained object per cell, largest normalized area
/// first on ties. No learned model/state or fitting.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 56, 56, 1});
/// data.boxes = {n::tensor({8., 8., 16., 16.}, {1, 4})};
/// data.labels = {n::from_vector(std::vector<std::int64_t>{3})};
/// const auto t = n::models::MakeYoloV1Targets(data, {0}, 10);
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see MakeYoloV1Targets(), YoloV1Loss()
struct YoloV1Targets {
  nn::Tensor object; ///< Floating [N,G,G] retained-object0/1 mask.
  nn::Tensor boxes;  ///< Floating [N,G,G,4] dy,dx,sqrt(height_fraction),sqrt(width_fraction).
  nn::Tensor labels; ///< Int64 [N,G,G] class IDs, zero for empty cells.
  std::int64_t discarded = 0; ///< Number of objects discarded by shared-cell collisions.
};
/// @brief Assign each truth centre to a grid cell and keep the largest colliding object.
///
/// Centres normalize to grid units, clamp integer cells and retain dy/dx
/// offsets. Sizes normalize to image extents before square root. Largest
/// normalized area wins collisions; first truth wins exact ties. No fitting,
/// random draws or host numerical box arithmetic.
///
/// @param scenes Valid same-dtype Float32/64 grayscale scenes, positive in-image truth boxes.
/// @param rows Dataset IDs in output batch order.
/// @param classes Positive class count above selected labels.
/// @param grid Positive square grid side, default7.
/// @return Detached object/class/root-size tensors and collision count.
/// @throws std::invalid_argument For invalid scenes/rows/labels/grid or out-of-image boxes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 56, 56, 1});
/// data.boxes = {n::tensor({8., 8., 16., 16.}, {1, 4})};
/// data.labels = {n::from_vector(std::vector<std::int64_t>{3})};
/// const auto t = n::models::MakeYoloV1Targets(data, {0}, 10);
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see YoloV1Loss()
inline YoloV1Targets MakeYoloV1Targets(const TensorDetectionScenes &scenes,
                                       const std::vector<std::int64_t> &rows, int classes,
                                       int grid = 7) {
  two_stage_detection_detail::ValidateScenes(scenes, rows, classes);
  if (grid < 1)
    throw std::invalid_argument("YOLO grid must be positive");
  nn::NoGradGuard guard;
  const auto N = std::int64_t(rows.size()), G = std::int64_t(grid), P = G * G;
  const double H = double(scenes.images.shape()[1]), W = double(scenes.images.shape()[2]);
  const auto type = scenes.images.dtype();
  std::vector<nn::Tensor> objects, boxes, labels;
  std::int64_t discarded = 0;
  for (auto row : rows) {
    const auto K = scenes.boxes[row].shape()[0];
    auto mask = nn::zeros({G, G}, type), target = nn::zeros({G, G, 4}, type),
         label = nn::zeros({G, G}, nn::DType::Int64);
    if (K) {
      const auto b = nn::unstack(scenes.boxes[row], 1);
      if (nn::any(nn::logical_or(nn::logical_or(nn::less(b[0], 0.), nn::less(b[1], 0.)),
                                 nn::logical_or(nn::greater(b[2], H), nn::greater(b[3], W))))
              .item<bool>())
        throw std::invalid_argument("YOLO truth outside image");
      const auto y = (b[0] + b[2]) * (.5 / H) * double(G),
                 x = (b[1] + b[3]) * (.5 / W) * double(G), h = (b[2] - b[0]) / H,
                 w = (b[3] - b[1]) / W;
      const auto r = nn::clip(nn::floor(y), 0., double(G - 1)).to(nn::DType::Int64),
                 c = nn::clip(nn::floor(x), 0., double(G - 1)).to(nn::DType::Int64),
                 cell = r * nn::full({}, double(G), nn::DType::Int64) + c;
      const auto same =
                     nn::equal(nn::arange(0, double(P), 1, nn::DType::Int64).reshape({P, 1}),
                               cell.reshape({1, K})),
                 occupied = nn::any(same, {1}),
                 best = nn::argmax(
                     nn::where(same, (h * w).reshape({1, K}),
                               nn::full({}, -std::numeric_limits<double>::infinity(), type)),
                     1),
                 used = two_stage_detection_detail::Indices(occupied),
                 chosen = nn::gather(best, used);
      discarded += K - used.shape()[0];
      const auto encoded =
          nn::stack({y - r.to(type), x - c.to(type), nn::sqrt(h), nn::sqrt(w)}, 1);
      const auto hot = nn::transpose(nn::one_hot(used, P, 1, 0, type));
      target = nn::matmul(hot, nn::gather(encoded, chosen)).reshape({G, G, 4});
      label = nn::where(occupied, nn::gather(scenes.labels[row], best),
                        nn::zeros({}, nn::DType::Int64))
                  .reshape({G, G});
      mask = occupied.to(type).reshape({G, G});
    }
    objects.push_back(mask);
    boxes.push_back(target);
    labels.push_back(label);
  }
  return {rows.empty() ? nn::zeros({N, G, G}, type) : nn::stack(objects, 0),
          rows.empty() ? nn::zeros({N, G, G, 4}, type) : nn::stack(boxes, 0),
          rows.empty() ? nn::zeros({N, G, G}, nn::DType::Int64) : nn::stack(labels, 0),
          discarded};
}
/// @brief Retain YOLO v1 scalar squared-error objective and detached predictor responsibility.
///
/// Responsibility chooses the larger current IoU in occupied cells, predictor0
/// on ties. IoU confidence targets and selection masks are detached.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::YoloV1Targets t{n::ones({1, 1, 1}), n::full({1, 1, 1, 4}, .5),
///                            n::ones({1, 1, 1}, n::DType::Int64), 0};
/// const auto out = n::models::YoloV1Loss(n::zeros({1, 1, 1, 12}), t, 2, 1);
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see YoloV1Loss()
struct YoloV1LossOutputs {
  nn::Tensor loss;        ///< Scalar batch mean of per-image summed terms.
  nn::Tensor responsible; ///< Detached floating [N,G,G,2] responsible-predictor mask.
  nn::Tensor iou;         ///< Detached floating [N,G,G,2] current IoU confidence targets.
};
/// @brief Compute source YOLO v1 sum-of-squares objective with tensor IoU responsibility.
///
/// Each occupied cell assigns one of two predictors by current IoU, first on
/// ties. Coordinates weight5, responsible confidence targets detached IoU, all
/// other confidences target0 with weight.5. Shared class squared error applies
/// only in occupied cells. Sum over cells/predictors/classes, then batch mean.
/// Root sizes are direct predictions; squaring decodes them. No optimizer or fit.
///
/// @param prediction Float32/64 [N,G,G,C+10] linear predictions, positive N.
/// @param t Same-dtype object/root-size targets plus Int64 labels in[0,C).
/// @param classes Positive shared cell-level class count.
/// @param grid Positive grid side, default7, matching predictions/targets.
/// @return Scalar loss and detached IoU/responsibility masks.
/// @throws std::invalid_argument For invalid geometry/dtypes/class targets.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::YoloV1Targets t{n::ones({1, 1, 1}), n::full({1, 1, 1, 4}, .5),
///                            n::ones({1, 1, 1}, n::DType::Int64), 0};
/// const auto out = n::models::YoloV1Loss(n::zeros({1, 1, 1, 12}), t, 2, 1);
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see MakeYoloV1Targets(), DecodeYoloV1Detections()
inline YoloV1LossOutputs YoloV1Loss(const nn::Tensor &prediction, const YoloV1Targets &t,
                                    int classes, int grid = 7) {
  if (prediction.dim() != 4 || prediction.shape()[0] < 1 || classes < 1 || grid < 1 ||
      prediction.shape()[1] != grid || prediction.shape()[2] != grid ||
      prediction.shape()[3] != classes + 10 ||
      t.object.shape() != nn::Shape{prediction.shape()[0], grid, grid} ||
      t.boxes.shape() != nn::Shape{prediction.shape()[0], grid, grid, 4} ||
      t.labels.shape() != t.object.shape() || t.object.dtype() != prediction.dtype() ||
      t.boxes.dtype() != prediction.dtype() || t.labels.dtype() != nn::DType::Int64 ||
      !nn::all(nn::logical_and(nn::greater_equal(t.labels, 0.),
                               nn::less(t.labels, double(classes))))
           .item<bool>())
    throw std::invalid_argument("invalid YOLO v1 loss inputs");
  const std::int64_t n = prediction.shape()[0], g = grid;
  const auto object = t.object;
  const auto scores = nn::slice(prediction, {0, 0, 0, 0}, {n, g, g, classes});
  const auto parts = nn::unstack(
      nn::reshape(nn::slice(prediction, {0, 0, 0, classes}, {n, g, g, 10}), {n, g, g, 2, 5}),
      4);
  const auto target = nn::unstack(nn::expand_dims(t.boxes, 3), 4);
  const auto index = nn::arange(0, double(g)).to(prediction.dtype());
  const auto rows = index.reshape({1, g, 1, 1}), cols = index.reshape({1, 1, g, 1});
  const auto py = (rows + parts[0]) / double(grid), px = (cols + parts[1]) / double(grid);
  const auto ty = (rows + target[0]) / double(grid), tx = (cols + target[1]) / double(grid);
  const auto ph = nn::square(parts[2]), pw = nn::square(parts[3]), th = nn::square(target[2]),
             tw = nn::square(target[3]);
  YoloV1LossOutputs out;
  {
    const auto zero = nn::zeros({}, prediction.dtype());
    const auto ih = nn::maximum(nn::minimum(py + ph * .5, ty + th * .5) -
                                    nn::maximum(py - ph * .5, ty - th * .5),
                                zero);
    const auto iw = nn::maximum(nn::minimum(px + pw * .5, tx + tw * .5) -
                                    nn::maximum(px - pw * .5, tx - tw * .5),
                                zero);
    const auto intersection = ih * iw;
    out.iou = (intersection / nn::maximum(ph * pw + th * tw - intersection,
                                          nn::full({}, 1e-12, prediction.dtype())))
                  .detach();
  }
  out.responsible = nn::one_hot(nn::argmax(out.iou, 3), 2, 1, 0, prediction.dtype()) *
                    t.object.reshape({n, g, g, 1});
  nn::Tensor coordinate = nn::square(parts[0] - target[0]);
  for (std::size_t k = 1; k < 4; ++k)
    coordinate = coordinate + nn::square(parts[k] - target[k]);
  const auto object_confidence = out.responsible * nn::square(parts[4] - out.iou);
  const auto background_confidence = (1.0 - out.responsible) * nn::square(parts[4]);
  const auto localization = nn::sum(coordinate * (out.responsible * 5.0) +
                                        (object_confidence + background_confidence * .5),
                                    {1, 2, 3});
  const auto one_hot = nn::one_hot(t.labels, classes, 1, 0, prediction.dtype());
  const auto classification =
      nn::sum(nn::square(scores - one_hot) * nn::expand_dims(object, 3), {1, 2, 3});
  out.loss = nn::mean(localization + classification);
  return out;
}

/// @brief Decode both YOLO v1 box predictors and all cell classes with optional final NMS.
///
/// Root sizes square, centres add grid indices and divide by grid. Class and
/// confidence outputs clip separately to[0,1] before multiplication. Candidate
/// order is image,row,column,predictor,class. No fitted state or random draws.
///
/// @param prediction Floating [N,G,G,C+10] linear predictions.
/// @param images Int64 [N] dataset IDs.
/// @param height Positive clipping height in pixels.
/// @param width Positive clipping width in pixels.
/// @param classes Positive cell class count.
/// @param suppress True(default) applies class-wise IoU.45 NMS and100/image; false returns raw candidates.
/// @return Detached clipped positive-area detections with strict score>.01.
/// @throws std::invalid_argument For invalid geometry/dtypes/IDs, nonfinite or unordered final boxes/scores.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::DecodeYoloV1Detections(n::ones({1, 1, 1, 12}),
///                                                  n::zeros({1}, n::DType::Int64), 56, 56, 2);
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see YoloV1(), YoloV1Loss()
inline TensorDetections DecodeYoloV1Detections(const nn::Tensor &prediction,
                                               const nn::Tensor &images, int height, int width,
                                               int classes, bool suppress = true) {
  if (prediction.dim() != 4 || prediction.shape()[0] < 1 || prediction.shape()[1] < 1 ||
      prediction.shape()[2] != prediction.shape()[1] || classes < 1 ||
      prediction.shape()[3] != classes + 10 ||
      images.shape() != nn::Shape{prediction.shape()[0]} ||
      images.dtype() != nn::DType::Int64 || height < 1 || width < 1)
    throw std::invalid_argument("invalid YOLO v1 decode inputs");
  nn::NoGradGuard guard;
  const auto N = prediction.shape()[0], G = prediction.shape()[1], P = G * G * 2;
  const auto type = prediction.dtype();
  const auto parts = nn::unstack(
      nn::slice_axis(prediction, 3, classes, classes + 10).reshape({N, G, G, 2, 5}), 4);
  const auto row = nn::arange(0, double(G)).to(type).reshape({1, G, 1, 1}),
             col = nn::arange(0, double(G)).to(type).reshape({1, 1, G, 1});
  const auto y = (row + parts[0]) * (double(height) / double(G)),
             x = (col + parts[1]) * (double(width) / double(G)),
             h = nn::square(parts[2]) * double(height),
             w = nn::square(parts[3]) * double(width);
  const auto boxes = nn::clip_boxes(
      nn::stack({y - h * .5, x - w * .5, y + h * .5, x + w * .5}, 4).reshape({N * P, 4}),
      height, width);
  const auto cls = nn::clip(nn::slice_axis(prediction, 3, 0, classes), 0., 1.)
                       .reshape({N, G, G, 1, classes});
  const auto scores =
      (cls * nn::clip(parts[4], 0., 1.).reshape({N, G, G, 2, 1})).reshape({N * P, classes});
  const auto valid = nn::logical_and(nn::greater(scores, .01),
                                     nn::greater(nn::box_area(boxes), 0.).reshape({N * P, 1}));
  const auto index = two_stage_detection_detail::Indices(valid.reshape({N * P * classes})),
             box_index = nn::floor(index.to(type) / double(classes)).to(nn::DType::Int64),
             labels = index - box_index * nn::full({}, double(classes), nn::DType::Int64);
  const auto ids =
      (images.reshape({N, 1}) * nn::ones({N, P}, nn::DType::Int64)).reshape({N * P});
  TensorDetections out{nn::gather(boxes, box_index),
                       nn::gather(scores.reshape({N * P * classes}), index), labels,
                       nn::gather(ids, box_index)};
  if (!suppress) {
    DetectionEvaluationRecords(out);
    return out;
  }
  return SuppressDetectionCandidates(out.boxes, out.scores, out.labels, out.images, .45, 0.,
                                     100);
}
/// @brief Extract source aspect-preserving28-pixel object crops in the centre of56-pixel canvases.
///
/// Square boxes use maximum side and pixel-centre bilinear zero-padded crops,
/// then byte quantization and14-pixel zero borders. Original scenes are not
/// mutated. No randomness, fitted preprocessing or optimizer; empty truth yields
/// empty crop storage. Tensor and source float rounding can differ by one byte.
///
/// @param scenes Valid Float32/64 grayscale scenes with positive in-image boxes and digit labels0–9.
/// @param rows Source scene IDs allowed for this crop split; validation/test exclusion belongs to caller.
/// @return Detached tensor scenes, one56x56 image and fixed[14,14,42,42] box per selected object, with attached label.
/// @throws std::invalid_argument For invalid source geometry/values/rows/labels or boxes outside the canvas.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::ones({1, 56, 56, 1});
/// data.boxes = {n::tensor({8., 8., 24., 32.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// const auto crops = n::models::YoloV1PretrainingCrops(data, {0});
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see YoloV1PretrainingLogits(), TransformTensorDetectionScenes()
inline TensorDetectionScenes YoloV1PretrainingCrops(const TensorDetectionScenes &scenes,
                                                    const std::vector<std::int64_t> &rows) {
  two_stage_detection_detail::ValidateScenes(scenes, rows, 10);
  nn::NoGradGuard guard;
  const auto type = scenes.images.dtype();
  std::vector<nn::Tensor> all_boxes, all_labels;
  std::vector<std::int64_t> image_rows;
  for (auto row : rows) {
    const auto &boxes = scenes.boxes[row];
    const auto b = nn::unstack(boxes, 1);
    if (nn::any(nn::logical_or(
                    nn::logical_or(nn::less(b[0], 0.), nn::less(b[1], 0.)),
                    nn::logical_or(nn::greater(b[2], double(scenes.images.shape()[1])),
                                   nn::greater(b[3], double(scenes.images.shape()[2])))))
            .item<bool>())
      throw std::invalid_argument("pretraining truth outside source image");
    all_boxes.push_back(boxes);
    all_labels.push_back(scenes.labels[row]);
    for (std::int64_t k = 0; k < boxes.shape()[0]; ++k)
      image_rows.push_back(row);
  }
  TensorDetectionScenes out;
  if (image_rows.empty()) {
    out.images = nn::zeros({0, 56, 56, 1}, type);
    return out;
  }
  const auto boxes = nn::concat(all_boxes, 0), labels = nn::concat(all_labels, 0);
  const auto patches = CropDetectionPatches(scenes.images, nn::from_vector(image_rows),
                                            SquareDetectionBoxes(boxes, 1.), 28);
  out.images =
      nn::pad((nn::floor(nn::clip(patches, 0., 1.).to(nn::DType::Float64) * 255. + .5) / 255.)
                  .to(type),
              {{0, 0}, {14, 14}, {14, 14}, {0, 0}});
  const auto K = labels.shape()[0];
  for (std::int64_t i = 0; i < K; ++i) {
    out.boxes.push_back(nn::from_vector(std::vector<double>{14, 14, 42, 42}, {1, 4}).to(type));
    out.labels.push_back(nn::slice_axis(labels, 0, i, i + 1));
  }
  return out;
}
/// @brief Predict digit-crop logits from the first20 source convolutions and a global-average classifier.
///
/// First20 features pool spatially before pretrain_head dense10. Counts352578
/// without BN and353714 with BN. No optimizer or fitting is hidden; caller shows
/// its cross-entropy/L2 objective, loop and validation checkpoint selection.
///
/// @param p Mutable learned prefix/head store; names of conv1–20 match the detector.
/// @param state Mutable optional BN moving-state store matching p dtype.
/// @param x Floating [N,56*56] or [N,56,56,1] crop input normalized to[-1,1].
/// @param cfg Compact56gray configuration with matching width/BN options; dropout is unused.
/// @param training Use batch BN statistics and optional updates when true, inference state when false.
/// @return Differentiable [N,10] digit logits.
/// @throws std::invalid_argument For invalid compact geometry/config/dtypes or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::YoloV1Config c;
/// c.compact = true;
/// c.width_divisor = 8;
/// c.classes = 10;
/// c.dense_width = 8;
/// const auto logits =
///     n::models::YoloV1PretrainingLogits(p, state, n::ones({1, 56 * 56}), c, false);
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see YoloV1PretrainingCrops(), CopyYoloV1Prefix()
inline nn::Tensor YoloV1PretrainingLogits(nn::ParameterStore &p, nn::ParameterStore &state,
                                          const nn::Tensor &x, const YoloV1Config &cfg,
                                          bool training) {
  const auto features =
      YoloV1Backbone(p, &state, nn::reshape(x, {-1, 56, 56, 1}), cfg, training, 20).features;
  return nn::dense(p, "pretrain_head", nn::mean(features, {1, 2}), 10);
}

/// @brief Copy only pretrained conv1–20 learned leaves and optional BN state into a warmed detector.
///
/// Validates all names/shapes/dtypes/finite values before mutation. Existing
/// learned leaves retain identity; last four convolutions/dense heads retain
/// their initialized values. Copies values, not optimizer moments. No fitting.
/// Stores must not mutate concurrently; learned values cannot change before
/// a pending backward. Caller owns source/destination lifetimes.
///
/// @param source Learned prefix/classifier store; all20 convolution prefixes must exist,40 unnormalized or60 BN leaves.
/// @param source_state Corresponding0 or40 prefix BN moving-state tensors.
/// @param destination Warmed compatible full-detector learned store, same dtype.
/// @param destination_state Warmed compatible detector BN moving state, same dtype.
/// @throws std::invalid_argument For incomplete prefix, shape/dtype/nonfinite mismatches or missing destination entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17), pre(17), prestate(17);
/// n::models::YoloV1Config c;
/// c.compact = true;
/// c.classes = 10;
/// c.width_divisor = 8;
/// c.dense_width = 8;
/// c.keep_probability = 1;
/// n::models::YoloV1(p, &state, n::zeros({1, 56, 56, 1}), c);
/// n::models::YoloV1PretrainingLogits(pre, prestate, n::zeros({1, 56 * 56}), c, false);
/// n::models::CopyYoloV1Prefix(pre, prestate, p, state);
/// @endcode
/// @par References
/// - Redmon et al. (2016), You Only Look Once: Unified, Real-Time Object Detection, https://arxiv.org/abs/1506.02640
/// @see YoloV1PretrainingLogits(), YoloV1()
inline void CopyYoloV1Prefix(
    const nn::ParameterStore &source, const nn::ParameterStore &source_state,
    nn::ParameterStore &destination,
    nn::ParameterStore &
        destination_state) { /// @brief Recognize metadata names for the first20 convolution prefixes.
  const auto prefix = [](const std::string &name) {
    for (int i = 1; i <= 20; ++i)
      if (name.rfind("conv" + std::to_string(i) + "/", 0) == 0)
        return true;
    return false;
  };
  std::vector<std::string> learned, state;
  for (const auto &name : source.names())
    if (prefix(name))
      learned.push_back(name);
  for (const auto &name : source_state.names())
    if (prefix(name))
      state.push_back(name);
  if ((learned.size() != 40 && learned.size() != 60) ||
      state.size() != (learned.size() == 60 ? 40 : 0))
    throw std::invalid_argument(
        "incomplete YOLO pretraining prefix"); /// @brief Validate all stored tensor contracts before any destination value changes.
  const auto validate = [](const nn::ParameterStore &from, const nn::ParameterStore &to,
                           const std::vector<std::string> &names) {
    if (from.dtype() != to.dtype())
      throw std::invalid_argument("pretrained prefix dtype mismatch");
    for (const auto &name : names)
      if (!to.contains(name) || to.at(name).shape() != from.at(name).shape() ||
          !nn::all(nn::less(nn::abs(from.at(name)), std::numeric_limits<double>::infinity()))
               .item<bool>())
        throw std::invalid_argument("invalid pretrained prefix destination/value");
  };
  for (int i = 1; i <= 20; ++i) {
    const auto base = "conv" + std::to_string(i);
    const bool bn = learned.size() == 60;
    if (!source.contains(base + "/W") ||
        (bn ? (!source.contains(base + "/bn/gamma") || !source.contains(base + "/bn/beta") ||
               !source_state.contains(base + "/bn/moving_mean") ||
               !source_state.contains(base + "/bn/moving_var"))
            : !source.contains(base + "/b")))
      throw std::invalid_argument("incomplete pretrained convolution names/state");
  }
  validate(source, destination, learned);
  validate(source_state, destination_state, state);
  for (const auto &name : learned)
    destination.set(name, source.at(name));
  for (const auto &name : state)
    destination_state.set(name, source_state.at(name));
}
} // namespace datamunge::nn::models
