#pragma once
/// @file detr_models.hpp
/// @brief Tensor DETR positional attention, independent-layer bipartite targets, GIoU and query decoding.
#include "datamunge/nn/retina_detection_models.hpp"
#include "datamunge/nn/tensor_assignment.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Project the final axis with source DETR Glorot dense weights and bias.
///
/// No activation, moving state or fitting. Keep leaf values until backward.
///
/// @param p Mutable Float32/64 learned store; creates/reuses name/W and name/b.
/// @param name Compatible parameter prefix.
/// @param x Floating [...,in] activation matching p dtype.
/// @param out Positive output width.
/// @return Differentiable [...,out] dense projection.
/// @throws std::invalid_argument For invalid shapes/dtypes/out width or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::DetrLinear(p, "linear", n::ones({1, 4, 8}), 16);
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see DetrAttention()
inline nn::Tensor DetrLinear(nn::ParameterStore &p, const std::string &name,
                             const nn::Tensor &x, std::int64_t out) {
  return nn::dense(p, name, x, out);
}
/// @brief Construct source normalized2D sine/cosine positions using Float64 tensor geometry.
///
/// Positions use(index+1)*2*pi/(axis_size+1e-6), temperature10000 with
/// paired sine/cosine frequencies. Float64 evaluation then cast preserves source
/// host-double position arithmetic; no fitted state or random draws.
///
/// @param height Positive grid height.
/// @param width Positive grid width.
/// @param dimension Positive dimension divisible by4.
/// @param dtype Float32(default) or Float64 returned positions.
/// @return Detached [1,H*W,D] positions; first D/2 row encodings, then column encodings.
/// @throws std::invalid_argument For invalid grid/dimension/dtype.
/// @par Example
/// @code
/// const auto p = dm::nn::models::DetrPositions(2, 3, 8);
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see Detr()
inline nn::Tensor DetrPositions(int height, int width, int dimension,
                                nn::DType dtype = nn::DType::Float32) {
  if (height < 1 || width < 1 || dimension < 4 || dimension % 4 ||
      (dtype != nn::DType::Float32 && dtype != nn::DType::Float64))
    throw std::invalid_argument("invalid DETR positional geometry");
  nn::NoGradGuard guard;
  const auto half = std::int64_t(dimension / 2);
  const auto type = nn::DType::Float64;
  const auto k = nn::arange(0, double(half)).to(type), paired = nn::floor(k / 2.) * 2.;
  const auto scale = nn::pow(nn::full({}, 10000., type), paired / double(half))
                         .reshape({1, 1, half}),
             sine = nn::equal(k - paired, 0.).reshape({1, 1, half});
  const auto y = (nn::arange(0, double(height)).to(type) + 1.).reshape({height, 1, 1}) *
                 (2. * std::acos(-1.) / (height + 1e-6)) / scale,
             x = (nn::arange(0, double(width)).to(type) + 1.).reshape({1, width, 1}) *
                 (2. * std::acos(-1.) / (width + 1e-6)) / scale;
  const auto rows = nn::where(sine, nn::sin(y), nn::cos(y)) *
                    nn::ones({height, width, half}, type),
             cols = nn::where(sine, nn::sin(x), nn::cos(x)) *
                    nn::ones({height, width, half}, type);
  return nn::concat({rows, cols}, 2)
      .reshape({1, std::int64_t(height) * width, dimension})
      .to(dtype);
}
/// @brief Apply source multi-head scaled dot-product attention with independent Q/K/V projections.
///
/// Caller adds position embeddings to Q/K only; values remain contents. No
/// fitting or moving state. Named local projection borrows p only during call.
///
/// @param p Mutable learned store, also supplies dropout RNG.
/// @param name Compatible unique attention prefix.
/// @param query Float32/64 [N,Q,D] matching p dtype.
/// @param key Same-dtype [N,K,D].
/// @param value Same-dtype [N,K,D], unpositioned content.
/// @param dimension Positive D divisible by heads.
/// @param heads Positive number of attention heads.
/// @param dropout Finite probability in[0,1), applied to attention weights only in training.
/// @param training True enables weight dropout; false is deterministic.
/// @return Differentiable [N,Q,D] attention with learned output projection.
/// @throws std::invalid_argument For invalid rank/shape/dtype/head/dropout settings or incompatible leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto x = n::ones({1, 4, 8});
/// const auto y = n::models::DetrAttention(p, "a", x, x, x, 8, 2, 0., false);
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see Detr()
inline nn::Tensor DetrAttention(nn::ParameterStore &p, const std::string &name,
                                const nn::Tensor &query, const nn::Tensor &key,
                                const nn::Tensor &value, int dimension, int heads,
                                double dropout, bool training) {
  if (dimension < 1 || heads < 1 || dimension % heads || !(dropout >= 0 && dropout < 1) ||
      query.dim() != 3 || key.dim() != 3 || value.shape() != key.shape() ||
      query.shape()[0] < 1 || query.shape()[1] < 1 || key.shape()[1] < 1 ||
      key.shape()[0] != query.shape()[0] || query.shape()[2] != dimension ||
      key.shape()[2] != dimension || query.dtype() != p.dtype() || key.dtype() != p.dtype() ||
      value.dtype() != p.dtype())
    throw std::invalid_argument("invalid DETR attention inputs");
  const std::int64_t head = dimension / heads;
  /// Project and split heads; this closure borrows p/name only for the current call.
  auto project = [&](const std::string &part, const nn::Tensor &x) {
    const auto flat = DetrLinear(p, name + "/" + part, x, dimension);
    return nn::transpose(nn::reshape(flat, {x.shape()[0], x.shape()[1], heads, head}),
                         {0, 2, 1, 3});
  };
  const auto q = project("q", query), k = project("k", key), v = project("v", value);
  auto weights =
      nn::softmax(nn::matmul(q, k, false, true) * (1. / std::sqrt(double(head))), -1);
  if (training && dropout > 0)
    weights = nn::dropout(weights, dropout, true, p.generator());
  const auto merged = nn::transpose(nn::matmul(weights, v), {0, 2, 1, 3});
  return DetrLinear(p, name + "/output", nn::reshape(merged, query.shape()), dimension);
}

/// @brief Configure source post-norm DETR backbone, encoder/decoder and learned query set.
///
/// Defaults use ResNet50, six encoder/decoder layers, D256, eight heads,
/// FF2048,100 queries and80 foreground classes. No pretraining/frozen BN.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DetrConfig c;
/// c.classes = 10;
/// c.image_size = 64;
/// c.backbone_depth = 18;
/// c.width_divisor = 8;
/// c.dimension = 64;
/// c.heads = 4;
/// c.feedforward = 128;
/// c.encoder_layers = 2;
/// c.decoder_layers = 2;
/// c.queries = 10;
/// c.dropout = 0;
/// c.grayscale = true;
/// c.small_stem = true;
/// const auto net = n::models::Detr(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see Detr()
struct DetrConfig {
  int classes = 80;        ///< Positive foreground count; no-object is final output class.
  int image_size = 640;    ///< Square side>=32 divisible by32.
  int backbone_depth = 50; ///< Supported feature-only ResNet depth.
  int width_divisor = 1;   ///< Positive divisor of64 for backbone widths.
  int dimension = 256;     ///< Positive model dimension divisible by4 and heads.
  int heads = 8;           ///< Positive attention head count.
  int feedforward = 2048;  ///< Positive hidden FFN dimension.
  int encoder_layers = 6;  ///< Positive encoder block count.
  int decoder_layers = 6;  ///< Positive decoder block count; all outputs supervised.
  int queries = 100;      ///< Positive fixed query count, at least objects per training image.
  double dropout = .1;    ///< Drop probability[0,1), from p generator during training.
  bool grayscale = false; ///< Tile one-channel input to RGB before backbone.
  bool small_stem = false;  ///< Small stem gives final stride8 rather than32.
  bool update_stats = true; ///< Update backbone BN moving state only in training when true.
};
/// @brief Retain DETR auxiliary logits/boxes from every decoder layer and encoder memory.
///
/// Tensor graphs borrow learned leaves. Class and three-layer box heads share
/// across decoder layers. Centres are normalized cy,cx,h,w, not corner boxes.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DetrConfig c;
/// c.classes = 10;
/// c.image_size = 64;
/// c.backbone_depth = 18;
/// c.width_divisor = 8;
/// c.dimension = 64;
/// c.heads = 4;
/// c.feedforward = 128;
/// c.encoder_layers = 2;
/// c.decoder_layers = 2;
/// c.queries = 10;
/// c.dropout = 0;
/// c.grayscale = true;
/// c.small_stem = true;
/// const auto net = n::models::Detr(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see Detr(), DetrLoss()
struct DetrOutputs {
  std::vector<nn::Tensor> logits; ///< One [N,Q,C+1] tensor per decoder layer, no-object last.
  std::vector<nn::Tensor> boxes;  ///< One [N,Q,4] sigmoid cy,cx,h,w tensor per layer.
  nn::Tensor memory;              ///< [N,(S/stride)^2,D] encoder contents.
};
/// @brief Build source DETR ResNet encoder and post-norm transformer set decoder.
///
/// Positions enter attention Q/K only, never V. Decoder contents start at zero;
/// learned query positions enter every layer Q/K. Heads share across layers, with
/// auxiliary supervision at every normalized decoder output. Backbone BN is
/// trainable/moving rather than pretrained frozen BN. Dropout draws from p RNG.
/// Full80-class ResNet50 has41575061 learned/53120 state scalars and424
/// learned tensors; compact18/8,D64,2/2 layers,10 queries/classes has357399/1200.
/// No optimizer, fitting or pretraining is hidden. Keep leaves until backward.
///
/// @param p Mutable Float32/64 learned store and dropout RNG; warm all leaves before optimizer construction.
/// @param state Mutable same-dtype backbone BN moving state, checkpoint with p.
/// @param images Positive floating NHWC with resolved square side and1/3 channels per cfg.
/// @param cfg Valid class/backbone/transformer/query/dropout metadata.
/// @param training False(default) uses moving BN and no dropout; true uses batch stats/optional updates and dropout.
/// @return Per-decoder-layer logits/normalized boxes plus encoder memory.
/// @throws std::invalid_argument For invalid cfg/input/dtypes or incompatible existing leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DetrConfig c;
/// c.classes = 10;
/// c.image_size = 64;
/// c.backbone_depth = 18;
/// c.width_divisor = 8;
/// c.dimension = 64;
/// c.heads = 4;
/// c.feedforward = 128;
/// c.encoder_layers = 2;
/// c.decoder_layers = 2;
/// c.queries = 10;
/// c.dropout = 0;
/// c.grayscale = true;
/// c.small_stem = true;
/// const auto net = n::models::Detr(p, state, n::ones({1, 64, 64, 1}), c);
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see DetrLoss(), MatchDetrTargets(), DecodeDetrDetections()
inline DetrOutputs Detr(nn::ParameterStore &p, nn::ParameterStore &state,
                        const nn::Tensor &images, const DetrConfig &cfg = {},
                        bool training = false) {
  if (cfg.classes < 1 || cfg.image_size < 32 || cfg.image_size % 32 || cfg.dimension < 4 ||
      cfg.dimension % 4 || cfg.heads < 1 || cfg.dimension % cfg.heads ||
      cfg.encoder_layers < 1 || cfg.decoder_layers < 1 || cfg.feedforward < 1 ||
      cfg.queries < 1 || !(cfg.dropout >= 0 && cfg.dropout < 1) || images.dim() != 4 ||
      images.shape()[0] < 1 || images.shape()[1] != cfg.image_size ||
      images.shape()[2] != cfg.image_size || images.shape()[3] != (cfg.grayscale ? 1 : 3) ||
      images.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("invalid DETR configuration");
  ResNetConfig backbone;
  backbone.depth = cfg.backbone_depth;
  backbone.width_divisor = cfg.width_divisor;
  backbone.include_classifier = false;
  backbone.small_stem = cfg.small_stem;
  backbone.stride_on_3x3 = true;
  backbone.update_stats = cfg.update_stats;
  const auto features = ResNet(
      p, state, cfg.grayscale ? nn::tile(images, {1, 1, 1, 3}) : images, backbone, training);
  const int side = cfg.image_size / (cfg.small_stem ? 8 : 32);
  if (features.stages[3].shape()[1] != side)
    throw std::invalid_argument("DETR image_size does not match the input images");
  DetrOutputs result;
  nn::Tensor memory =
      nn::reshape(nn::conv2d(p, "input_projection", features.stages[3], cfg.dimension, 1),
                  {-1, side * side, cfg.dimension});
  const auto position = DetrPositions(side, side, cfg.dimension, p.dtype());
  /// Apply seeded training dropout without extending borrowed-store lifetime.
  auto drop = [&](const nn::Tensor &x) {
    return training && cfg.dropout > 0 ? nn::dropout(x, cfg.dropout, true, p.generator()) : x;
  };
  /// Shared-shape feed-forward projection; borrows p/cfg/drop for this call.
  auto ffn = [&](const std::string &name, const nn::Tensor &x) {
    const auto h = nn::relu(DetrLinear(p, name + "/linear1", x, cfg.feedforward));
    return DetrLinear(p, name + "/linear2", drop(h), cfg.dimension);
  };
  for (int layer = 0; layer < cfg.encoder_layers; ++layer) {
    const auto name = "encoder" + std::to_string(layer);
    const auto positioned = memory + position;
    const auto attention =
        DetrAttention(p, name + "/attention", positioned, positioned, memory, cfg.dimension,
                      cfg.heads, cfg.dropout, training);
    memory = nn::layer_norm(p, name + "/norm1", memory + drop(attention));
    memory = nn::layer_norm(p, name + "/norm2", memory + drop(ffn(name + "/ffn", memory)));
  }
  result.memory = memory;
  const auto query =
      p.get("query_positions", {1, cfg.queries, cfg.dimension}, nn::init::normal(1.0));
  nn::Tensor contents = nn::zeros({images.shape()[0], cfg.queries, cfg.dimension}, p.dtype());
  for (int layer = 0; layer < cfg.decoder_layers; ++layer) {
    const auto name = "decoder" + std::to_string(layer);
    const auto q = contents + query;
    const auto self = DetrAttention(p, name + "/self", q, q, contents, cfg.dimension,
                                    cfg.heads, cfg.dropout, training);
    contents = nn::layer_norm(p, name + "/norm1", contents + drop(self));
    const auto cross = DetrAttention(p, name + "/cross", contents + query, memory + position,
                                     memory, cfg.dimension, cfg.heads, cfg.dropout, training);
    contents = nn::layer_norm(p, name + "/norm2", contents + drop(cross));
    contents =
        nn::layer_norm(p, name + "/norm3", contents + drop(ffn(name + "/ffn", contents)));
    const auto normalized = nn::layer_norm(p, "decoder_norm", contents);
    result.logits.push_back(DetrLinear(p, "class_head", normalized, cfg.classes + 1));
    nn::Tensor box = nn::relu(DetrLinear(p, "box_head/layer1", normalized, cfg.dimension));
    box = nn::relu(DetrLinear(p, "box_head/layer2", box, cfg.dimension));
    result.boxes.push_back(nn::sigmoid(DetrLinear(p, "box_head/layer3", box, 4)));
  }
  return result;
}

/// @brief Compute elementwise generalized IoU for broadcast centre-format boxes.
///
/// Disjoint boxes can yield negative GIoU. Tensor geometry differentiates both
/// arguments; no parameter mutation or fitting. Positive extents avoid degeneracy.
///
/// @param prediction Float32/64 [...,4] cy,cx,h,w with nonnegative extents; finite inputs required.
/// @param target Same-dtype broadcastable centre-format boxes.
/// @return Broadcast [...] intersection/union minus uncovered enclosing fraction, denominator floors1e-7.
/// @throws std::invalid_argument For invalid rank/last dimension/dtype or incompatible broadcasts.
/// @par Example
/// @code
/// const auto g = dm::nn::models::GeneralizedIou(dm::nn::tensor({.5, .5, .2, .2}),
///                                               dm::nn::tensor({.6, .6, .3, .3}));
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see DetrLayerLoss()
inline nn::Tensor GeneralizedIou(const nn::Tensor &prediction, const nn::Tensor &target) {
  if (prediction.dim() < 1 || target.dim() < 1 || prediction.shape().back() != 4 ||
      target.shape().back() != 4 || target.dtype() != prediction.dtype() ||
      (prediction.dtype() != nn::DType::Float32 && prediction.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid GIoU geometry/dtype");
  const auto p = nn::unstack(prediction, -1), t = nn::unstack(target, -1);
  const auto py0 = p[0] - p[2] * .5, px0 = p[1] - p[3] * .5, py1 = p[0] + p[2] * .5,
             px1 = p[1] + p[3] * .5;
  const auto ty0 = t[0] - t[2] * .5, tx0 = t[1] - t[3] * .5, ty1 = t[0] + t[2] * .5,
             tx1 = t[1] + t[3] * .5;
  const auto zero = nn::zeros({}, prediction.dtype());
  const auto ih = nn::maximum(nn::minimum(py1, ty1) - nn::maximum(py0, ty0), zero);
  const auto iw = nn::maximum(nn::minimum(px1, tx1) - nn::maximum(px0, tx0), zero);
  const auto intersection = ih * iw;
  const auto combined = p[2] * p[3] + t[2] * t[3] - intersection;
  const auto enclosing = (nn::maximum(py1, ty1) - nn::minimum(py0, ty0)) *
                         (nn::maximum(px1, tx1) - nn::minimum(px0, tx0));
  const auto eps = nn::full({}, 1e-7, prediction.dtype());
  return intersection / nn::maximum(combined, eps) -
         (enclosing - combined) / nn::maximum(enclosing, eps);
}

/// @brief Own tensor one-hot classes, normalized boxes and matched-query mask for one DETR layer.
///
/// No-object is the last class. Matching outputs are detached; target tensor
/// handles have no learned parameters or moving state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetrTargets t{n::tensor({1., 0., 0., 0., 0., 1.}, {1, 2, 3}),
///                          n::full({1, 2, 4}, .3), n::tensor({1., 0.}, {1, 2})};
/// const auto l = n::models::DetrLayerLoss(n::zeros({1, 2, 3}), n::full({1, 2, 4}, .4), t);
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see MatchDetrTargets(), DetrLayerLoss()
struct DetrTargets {
  nn::Tensor labels;   ///< [N,Q,C+1] one-hot class/no-object target.
  nn::Tensor boxes;    ///< [N,Q,4] normalized cy,cx,h,w, zero at unmatched queries.
  nn::Tensor positive; ///< [N,Q] floating matched-query0/1 mask.
};
/// @brief Evaluate DETR weighted no-object classification and matched L1/GIoU regression.
///
/// Matched class weight1, no-object.1. Regression normalizes batch matched
/// objects; empty targets keep background classification and zero box gradients.
/// No target matching or optimizer is hidden here.
///
/// @param logits Float32/64 [N,Q,C+1], positive N/Q and C>=1.
/// @param boxes Same-dtype [N,Q,4] normalized centre-format predictions.
/// @param t Same-shape/dtype one-hot targets and binary positive mask; finite valid values required.
/// @return Scalar weighted CE/max(sum class weights,1e-7) + sum(pos*(5L1+2(1-GIoU)))/max(sum pos,1).
/// @throws std::invalid_argument For invalid geometry/dtypes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DetrTargets t{n::tensor({1., 0., 0., 0., 0., 1.}, {1, 2, 3}),
///                          n::full({1, 2, 4}, .3), n::tensor({1., 0.}, {1, 2})};
/// const auto l = n::models::DetrLayerLoss(n::zeros({1, 2, 3}), n::full({1, 2, 4}, .4), t);
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see MatchDetrTargets(), DetrLoss()
inline nn::Tensor DetrLayerLoss(const nn::Tensor &logits, const nn::Tensor &boxes,
                                const DetrTargets &t) {
  if (logits.dim() != 3 || logits.shape()[0] < 1 || logits.shape()[1] < 1 ||
      logits.shape()[2] < 2 ||
      boxes.shape() != nn::Shape{logits.shape()[0], logits.shape()[1], 4} ||
      t.labels.shape() != logits.shape() || t.boxes.shape() != boxes.shape() ||
      t.positive.shape() != nn::Shape{logits.shape()[0], logits.shape()[1]})
    throw std::invalid_argument("invalid DETR loss geometry");
  for (const auto &x : {boxes, t.labels, t.boxes, t.positive})
    if (x.dtype() != logits.dtype())
      throw std::invalid_argument("DETR loss dtype mismatch");
  const auto &labels = t.labels;
  const auto &target = t.boxes;
  const auto &positive = t.positive;
  const auto log_prob = nn::log_softmax(logits, -1);
  const auto weight = positive * .9 + .1;
  const auto ce = -nn::sum(labels * log_prob, {2});
  const auto classification =
      nn::sum(ce * weight) / nn::maximum(nn::sum(weight), nn::full({}, 1e-7, logits.dtype()));
  const auto l1 = nn::sum(nn::abs(boxes - target), {2});
  const auto giou = GeneralizedIou(boxes, target);
  const auto regression = nn::sum(positive * (l1 * 5.0 + (1.0 - giou) * 2.0)) /
                          nn::maximum(nn::sum(positive), nn::ones({}, logits.dtype()));
  return classification + regression;
}

/// @brief Match each image truth to distinct queries with detached tensor bipartite costs.
///
/// Cost is -foreground probability +5*centre-format L1 -2*GIoU. Costs and
/// assignment potentials use Float64 tensors; source host mixed Float32 corners
/// can differ by rounding near exact assignment ties. Deterministic first-column
/// ties apply. Q must be at least each truth count. Empty truths are all no-object.
/// No gradients flow through assignments and no fitting occurs.
///
/// @param logits Float32/64 [N,Q,C+1], with C=classes and no-object last.
/// @param predictions Same-dtype [N,Q,4] finite normalized cy,cx,h,w with positive extents.
/// @param scenes Valid same-dtype grayscale scene metadata; pixels are not read.
/// @param rows Dataset IDs in prediction order, N entries.
/// @param classes Positive foreground count above every selected label.
/// @return Detached aligned one-hot/no-object labels, normalized matched boxes and positive mask.
/// @throws std::invalid_argument For invalid geometry/dtypes/rows/labels, nonfinite costs or more truths than queries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::tensor({0., 0., 16., 16.}, {1, 4})};
/// data.labels = {n::zeros({1}, n::DType::Int64)};
/// const auto t =
///     n::models::MatchDetrTargets(n::zeros({1, 3, 3}), n::full({1, 3, 4}, .25), data, {0}, 2);
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see TensorHungarianAssignment(), DetrLayerLoss()
inline DetrTargets MatchDetrTargets(const nn::Tensor &logits, const nn::Tensor &predictions,
                                    const TensorDetectionScenes &scenes,
                                    const std::vector<std::int64_t> &rows, int classes) {
  two_stage_detection_detail::ValidateScenes(scenes, rows, classes);
  if (logits.dim() != 3 || logits.shape()[0] != std::int64_t(rows.size()) ||
      logits.shape()[1] < 1 || logits.shape()[2] != classes + 1 ||
      predictions.shape() != nn::Shape{logits.shape()[0], logits.shape()[1], 4} ||
      logits.dtype() != scenes.images.dtype() || predictions.dtype() != logits.dtype())
    throw std::invalid_argument("invalid DETR matching geometry/dtype");
  nn::NoGradGuard guard;
  const auto N = logits.shape()[0], Q = logits.shape()[1];
  const auto type = logits.dtype();
  std::vector<nn::Tensor> labels, boxes, positives;
  for (std::int64_t i = 0; i < N; ++i) {
    const auto row = rows[i], G = scenes.boxes[row].shape()[0];
    const auto background =
        nn::one_hot(nn::full({Q}, double(classes), nn::DType::Int64), classes + 1, 1, 0, type);
    auto label = background, target = nn::zeros({Q, 4}, type), positive = nn::zeros({Q}, type);
    if (G) {
      if (G > Q)
        throw std::invalid_argument("DETR queries fewer than truths");
      const auto corners = nn::unstack(scenes.boxes[row], 1);
      const double H = double(scenes.images.shape()[1]), W = double(scenes.images.shape()[2]);
      const auto normalized = nn::stack(
          {(corners[0] + corners[2]) / (2. * H), (corners[1] + corners[3]) / (2. * W),
           (corners[2] - corners[0]) / H, (corners[3] - corners[1]) / W},
          1);
      const auto pred = nn::slice_axis(predictions, 0, i, i + 1)
                            .to(nn::DType::Float64)
                            .reshape({1, Q, 4}),
                 truth = normalized.to(nn::DType::Float64).reshape({G, 1, 4}),
                 prob = nn::softmax(nn::slice_axis(logits, 0, i, i + 1).to(nn::DType::Float64),
                                    -1)
                            .reshape({Q, classes + 1});
      const auto class_cost = -nn::transpose(nn::gather(prob, scenes.labels[row], 1));
      const auto cost = class_cost + 5. * nn::sum(nn::abs(pred - truth), {2}) -
                        2. * GeneralizedIou(pred, truth);
      const auto assignment = TensorHungarianAssignment(cost),
                 hot = nn::one_hot(assignment, Q, 1, 0, type);
      positive = nn::sum(hot, {0});
      label = nn::matmul(hot, nn::one_hot(scenes.labels[row], classes + 1, 1, 0, type), true,
                         false) +
              background * (1. - positive).reshape({Q, 1});
      target = nn::matmul(hot, normalized, true, false);
    }
    labels.push_back(label);
    boxes.push_back(target);
    positives.push_back(positive);
  }
  return {rows.empty() ? nn::zeros({N, Q, classes + 1}, type) : nn::stack(labels, 0),
          rows.empty() ? nn::zeros({N, Q, 4}, type) : nn::stack(boxes, 0),
          rows.empty() ? nn::zeros({N, Q}, type) : nn::stack(positives, 0)};
}
/// @brief Sum independently matched auxiliary DETR objectives over every decoder output.
///
/// Every decoder layer receives its own detached matching, following this source
/// port. It does not reuse only the last layer assignment. No optimizer or fit.
///
/// @param net Nonempty same-length vectors of decoder logits and normalized boxes.
/// @param scenes Valid same-dtype scene metadata.
/// @param rows Dataset IDs matching network batch order.
/// @param classes Positive foreground count, excluding final no-object channel.
/// @return Differentiable scalar sum of all per-layer set losses.
/// @throws std::invalid_argument For empty/inconsistent decoder vectors or invalid matching/loss inputs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorDetectionScenes data;
/// data.images = n::zeros({1, 64, 64, 1});
/// data.boxes = {n::zeros({0, 4})};
/// data.labels = {n::zeros({0}, n::DType::Int64)};
/// n::models::DetrOutputs net;
/// net.logits = {n::zeros({1, 3, 3})};
/// net.boxes = {n::full({1, 3, 4}, .25)};
/// const auto loss = n::models::DetrLoss(net, data, {0}, 2);
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see MatchDetrTargets(), DetrLayerLoss()
inline nn::Tensor DetrLoss(const DetrOutputs &net, const TensorDetectionScenes &scenes,
                           const std::vector<std::int64_t> &rows, int classes) {
  if (net.logits.empty() || net.boxes.size() != net.logits.size())
    throw std::invalid_argument("invalid DETR decoder vectors");
  nn::Tensor total;
  for (std::size_t i = 0; i < net.logits.size(); ++i) {
    const auto target = MatchDetrTargets(net.logits[i], net.boxes[i], scenes, rows, classes);
    const auto loss = DetrLayerLoss(net.logits[i], net.boxes[i], target);
    total = i ? total + loss : loss;
  }
  return total;
}
/// @brief Decode one best foreground class per DETR query without anchors or NMS.
///
/// Best class excludes no-object, but its softmax denominator includes no-object.
/// Source retains a foreground score above threshold even if no-object is the
/// overall winner. First foreground class wins ties. No box NMS or100 cap; fixed
/// query count bounds output. No fitting, state mutation or random draws.
///
/// @param logits Float32/64 [N,Q,C+1] finite logits, no-object last.
/// @param boxes Same-dtype [N,Q,4] normalized cy,cx,h,w with positive extents.
/// @param images Int64 [N] dataset IDs.
/// @param height Positive pixel clipping height.
/// @param width Positive pixel clipping width.
/// @return Detached tensor detections in image/query order, inclusive score>=.01, clipped positive-area boxes.
/// @throws std::invalid_argument For invalid shapes/dtypes/IDs/dimensions or nonfinite/unordered result boxes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::DecodeDetrDetections(n::zeros({1, 3, 3}), n::full({1, 3, 4}, .25),
///                                                n::zeros({1}, n::DType::Int64), 64, 64);
/// @endcode
/// @par References
/// - Carion et al. (2020), End-to-End Object Detection with Transformers, https://arxiv.org/abs/2005.12872; authors implementation https://github.com/facebookresearch/detr
/// @see Detr(), DetrLoss()
inline TensorDetections DecodeDetrDetections(const nn::Tensor &logits, const nn::Tensor &boxes,
                                             const nn::Tensor &images, int height, int width) {
  if (logits.dim() != 3 || logits.shape()[0] < 1 || logits.shape()[1] < 1 ||
      logits.shape()[2] < 2 ||
      boxes.shape() != nn::Shape{logits.shape()[0], logits.shape()[1], 4} ||
      boxes.dtype() != logits.dtype() || images.shape() != nn::Shape{logits.shape()[0]} ||
      images.dtype() != nn::DType::Int64 || height < 1 || width < 1)
    throw std::invalid_argument("invalid DETR decode inputs");
  nn::NoGradGuard guard;
  const auto N = logits.shape()[0], Q = logits.shape()[1], C = logits.shape()[2] - 1;
  const auto type = logits.dtype();
  const auto prob = nn::slice_axis(nn::softmax(logits, -1), 2, 0, C),
             labels = nn::argmax(prob, 2), scores = nn::max(prob, {2});
  const auto b = nn::unstack(boxes, 2);
  const auto corners =
      nn::clip_boxes(nn::stack({(b[0] - b[2] * .5) * height, (b[1] - b[3] * .5) * width,
                                (b[0] + b[2] * .5) * height, (b[1] + b[3] * .5) * width},
                               2)
                         .reshape({N * Q, 4}),
                     height, width);
  const auto keep = two_stage_detection_detail::Indices(
      nn::logical_and(nn::greater_equal(scores.reshape({N * Q}), .01),
                      nn::greater(nn::box_area(corners), 0.)));
  const auto ids =
      (images.reshape({N, 1}) * nn::ones({N, Q}, nn::DType::Int64)).reshape({N * Q});
  TensorDetections out{nn::gather(corners, keep), nn::gather(scores.reshape({N * Q}), keep),
                       nn::gather(labels.reshape({N * Q}), keep), nn::gather(ids, keep)};
  DetectionEvaluationRecords(out);
  return out;
}
} // namespace datamunge::nn::models
