#pragma once
/// @file angular_margin_models.hpp
/// @brief Bias-free normalized cosine classification with ArcFace/CosFace target-only training margins.
#include "datamunge/nn/facenet_models.hpp"
#include "datamunge/nn/metric_learning_data.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Choose additive angular or additive cosine training margins.
///
/// Enum values are metadata;no fitting or learned state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::AngularMarginConfig c;
/// const auto head = n::models::NormalizedAngularHead(
///     p, n::ones({2, 3}), n::from_vector(std::vector<std::int64_t>{0, 1}), 3, 2, c);
/// @endcode
/// @par References
/// - Deng et al.(2019),ArcFace:Additive Angular Margin Loss for Deep Face Recognition, https://arxiv.org/abs/1801.07698v3;Wang et al.(2018),CosFace:Large Margin Cosine Loss for Deep Face Recognition, https://arxiv.org/abs/1801.09414. Monotonic clipped tail and compact NN1 training follow explicit source teaching choices.
/// @see ApplyAngularMargin(), NormalizedAngularHead()
enum class AngularMarginKind {
  kArcFace, ///< Replace target cosine by cos(theta+margin),with optional monotonic tail.
  kCosFace  ///< Replace target cosine by cosine-margin.
};
/// @brief Configure the target-only angular margin,scale and clipping policy.
///
/// Defaults ArcFace margin.5,scale64,epsilon1e-6,monotonic tail enabled.
/// No learned parameters,optimizer or RNG. Inference always uses plain cosines.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::AngularMarginConfig c;
/// const auto head = n::models::NormalizedAngularHead(
///     p, n::ones({2, 3}), n::from_vector(std::vector<std::int64_t>{0, 1}), 3, 2, c);
/// @endcode
/// @par References
/// - Deng et al.(2019),ArcFace:Additive Angular Margin Loss for Deep Face Recognition, https://arxiv.org/abs/1801.07698v3;Wang et al.(2018),CosFace:Large Margin Cosine Loss for Deep Face Recognition, https://arxiv.org/abs/1801.09414. Monotonic clipped tail and compact NN1 training follow explicit source teaching choices.
/// @see ApplyAngularMargin(), NormalizedAngularHead()
struct AngularMarginConfig {
  AngularMarginKind kind = AngularMarginKind::kArcFace; ///< Margin family.
  float margin =
      .5f; ///< Finite margin in[0,pi/2);radians for ArcFace,cosine units for CosFace.
  float scale = 64.f;    ///< Finite positive common logit multiplier.
  float epsilon = 1e-6f; ///< Finite clipping/sine floor in(0,.01).
  bool monotonic = true; ///< Extend ArcFace beyond pi-margin with c-margin*sin(margin).

  /// @brief Validate margin family,finite scale,margin and clipping epsilon.
  ///
  /// No mutation,fitting or RNG. Throws before numerical head processing.
  ///
  /// @throws std::invalid_argument For invalid configuration,rank,dtype,dimensions or class labels.
  /// @par Example
  /// @code
  /// dm::nn::models::AngularMarginConfig c;
  /// c.Validate();
  /// @endcode
  /// @par References
  /// - Deng et al.(2019),ArcFace:Additive Angular Margin Loss for Deep Face Recognition, https://arxiv.org/abs/1801.07698v3;Wang et al.(2018),CosFace:Large Margin Cosine Loss for Deep Face Recognition, https://arxiv.org/abs/1801.09414. Monotonic clipped tail and compact NN1 training follow explicit source teaching choices.
  /// @see ApplyAngularMargin(), NormalizedAngularHead()
  void Validate() const {
    if ((kind != AngularMarginKind::kArcFace && kind != AngularMarginKind::kCosFace) ||
        !std::isfinite(margin) || margin < 0 || margin >= 1.57079632679f ||
        !std::isfinite(scale) || scale <= 0 || !std::isfinite(epsilon) || epsilon <= 0 ||
        epsilon >= .01f)
      throw std::invalid_argument("invalid angular margin/scale/clipping epsilon");
  }
};

/// @brief Internal integer class-label shape/range validation.
namespace angular_margin_detail {
/// @brief Require integer labels aligned to a positive class-logit batch.
/// @param logits Floating[B,classes] matrix,B positive.
/// @param labels Int32/64[B] valid class IDs.
/// @param classes Number of columns,at least two.
/// @throws std::invalid_argument For invalid rank,dtype,shape or class values.
inline void Validate(const nn::Tensor &logits, const nn::Tensor &labels, int classes) {
  if (classes < 2 || logits.dim() != 2 || logits.shape()[0] < 1 ||
      logits.shape()[1] != classes ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      labels.shape() != nn::Shape{logits.shape()[0]} ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64) ||
      !nn::all(
           nn::logical_and(nn::greater_equal(labels, 0.), nn::less(labels, double(classes))))
           .item<bool>())
    throw std::invalid_argument("invalid angular logits or class labels");
}
} // namespace angular_margin_detail
/// @brief Apply a target-only ArcFace or CosFace margin then scale every class logit.
///
/// Only the labelled column changes before common scaling. Zero margin
/// returns plain scaled cosines exactly. ArcFace clips c to[-1+epsilon,1-epsilon],
/// uses sqrt(max(1-c*c,epsilon)),and optionally replaces c<=cos(pi-margin)
/// by c-margin*sin(margin). Non-target columns bypass these transforms. No fitting.
///
/// @param cosine Float32/64[B,classes] cosine values,B positive.
/// @param labels Int32/64[B] target class IDs.
/// @param classes At least two class columns.
/// @param cfg Valid margin/scale/epsilon policy.
/// @return Same-type[B,classes] training logits with analytic cosine gradients.
/// @throws std::invalid_argument For invalid configuration,rank,dtype,dimensions or class labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::AngularMarginConfig c;
/// c.scale = 1;
/// const auto logits = n::models::ApplyAngularMargin(
///     n::tensor({.8, .2}, {1, 2}), n::from_vector(std::vector<std::int64_t>{0}), 2, c);
/// @endcode
/// @par References
/// - Deng et al.(2019),ArcFace:Additive Angular Margin Loss for Deep Face Recognition, https://arxiv.org/abs/1801.07698v3;Wang et al.(2018),CosFace:Large Margin Cosine Loss for Deep Face Recognition, https://arxiv.org/abs/1801.09414. Monotonic clipped tail and compact NN1 training follow explicit source teaching choices.
/// @see ApplyAngularMargin(), NormalizedAngularHead()
inline nn::Tensor ApplyAngularMargin(const nn::Tensor &cosine, const nn::Tensor &labels,
                                     int classes, const AngularMarginConfig &cfg) {
  cfg.Validate();
  angular_margin_detail::Validate(cosine, labels, classes);
  if (classes < 2)
    throw std::invalid_argument("at least two angular classes");
  if (cfg.margin == 0)
    return cosine * double(cfg.scale);
  nn::Tensor phi;
  if (cfg.kind == AngularMarginKind::kCosFace) {
    phi = cosine - double(cfg.margin);
  } else {
    const nn::Tensor c =
        nn::clip(cosine, double(-1.f + cfg.epsilon), double(1.f - cfg.epsilon));
    const nn::Tensor sine =
        nn::sqrt(nn::maximum(1.0 - nn::square(c), nn::scalar_like(double(cfg.epsilon), c)));
    phi = c * double(std::cos(cfg.margin)) - sine * double(std::sin(cfg.margin));
    if (cfg.monotonic)
      phi = nn::where(nn::greater(c, double(std::cos(3.14159265358979323846f - cfg.margin))),
                      phi, c - double(std::sin(cfg.margin) * cfg.margin));
  }
  const nn::Tensor target = nn::one_hot(labels, classes, 1.0, 0.0, cosine.dtype());
  return (cosine + target * (phi - cosine)) * double(cfg.scale);
}

/// @brief Own normalized features,class weights,cosines and separate training/inference logits.
///
/// Tensor handles retain parameter/input graphs;inference accepts no target
/// labels. No optimizer or moving state is stored in the result.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::AngularMarginConfig c;
/// const auto head = n::models::NormalizedAngularHead(
///     p, n::ones({2, 3}), n::from_vector(std::vector<std::int64_t>{0, 1}), 3, 2, c);
/// @endcode
/// @par References
/// - Deng et al.(2019),ArcFace:Additive Angular Margin Loss for Deep Face Recognition, https://arxiv.org/abs/1801.07698v3;Wang et al.(2018),CosFace:Large Margin Cosine Loss for Deep Face Recognition, https://arxiv.org/abs/1801.09414. Monotonic clipped tail and compact NN1 training follow explicit source teaching choices.
/// @see ApplyAngularMargin(), NormalizedAngularHead()
struct AngularHead {
  nn::Tensor embedding; ///< Floating[B,dimensions] normalized feature rows.
  nn::Tensor weights;   ///< Floating[dimensions,classes] normalized weight columns.
  nn::Tensor cosine;    ///< Floating[B,classes] clipped cosine matrix in[-1,1].
  nn::Tensor training;  ///< Floating[B,classes] target-margin logits,undefined without labels.
  nn::Tensor inference; ///< Floating[B,classes] plain scaled cosine logits,always defined.
};
/// @brief Normalize feature rows and bias-free classifier columns for cosine classification.
///
/// Column and row squared norms floor at1e-12. Class weights initialize with
/// Glorot uniform and have no bias. Inference is scale*clipped cosine with no
/// label input. Training alone applies a target margin. Positive row/column
/// scaling cancels above the norm floor. Warm this leaf before caller optimization.
///
/// @param p Same-type learned store,creates/reuses angular/W[dimensions,classes].
/// @param raw Float32/64 positive[B,dimensions] raw feature matrix.
/// @param labels Undefined for inference-only,otherwise valid Int32/64[B] target IDs.
/// @param dimensions Positive feature dimension.
/// @param classes Number of classes,at least two.
/// @param cfg Valid margin/scale/epsilon policy.
/// @return Normalized features/weights,cosines,plain inference and optional margin logits.
/// @throws std::invalid_argument For invalid configuration,rank,dtype,dimensions or class labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::AngularMarginConfig c;
/// const auto head = n::models::NormalizedAngularHead(
///     p, n::ones({2, 3}), n::from_vector(std::vector<std::int64_t>{0, 1}), 3, 2, c);
/// @endcode
/// @par References
/// - Deng et al.(2019),ArcFace:Additive Angular Margin Loss for Deep Face Recognition, https://arxiv.org/abs/1801.07698v3;Wang et al.(2018),CosFace:Large Margin Cosine Loss for Deep Face Recognition, https://arxiv.org/abs/1801.09414. Monotonic clipped tail and compact NN1 training follow explicit source teaching choices.
/// @see ApplyAngularMargin(), NormalizedAngularHead()
inline AngularHead NormalizedAngularHead(nn::ParameterStore &p, const nn::Tensor &raw,
                                         const nn::Tensor &labels, int dimensions, int classes,
                                         const AngularMarginConfig &cfg) {
  cfg.Validate();
  if (dimensions < 1 || classes < 2)
    throw std::invalid_argument("angular head dimensions/classes");
  if (raw.dim() != 2 || raw.shape()[0] < 1 || raw.shape()[1] != dimensions ||
      raw.dtype() != p.dtype())
    throw std::invalid_argument("invalid normalized angular feature matrix");
  AngularHead out;
  const nn::Tensor w = p.get("angular/W", {dimensions, classes}, nn::init::glorot_uniform());
  out.embedding = UnitEmbedding(raw);
  out.weights = w * nn::rsqrt(nn::maximum(nn::sum(nn::square(w), {0}, true),
                                          nn::scalar_like(double(1e-12f), w)));
  out.cosine = nn::clip(nn::matmul(out.embedding, out.weights), -1.0, 1.0);
  out.inference = out.cosine * double(cfg.scale);
  if (labels.defined())
    out.training = ApplyAngularMargin(out.cosine, labels, classes, cfg);
  return out;
}

/// @brief Compute mean stable multiclass softmax cross entropy for angular logits.
///
/// No margin transformation inside this loss. Pass training logits while
/// fitting and plain inference logits when selecting/evaluating. No optimizer.
///
/// @param logits Float32/64[B,classes] class logits,B positive.
/// @param labels Int32/64[B] valid class IDs.
/// @param classes Number of classes,at least two.
/// @return Same-dtype scalar negative mean labelled log-softmax,with logit gradients.
/// @throws std::invalid_argument For invalid configuration,rank,dtype,dimensions or class labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::AngularCrossEntropy(
///     n::zeros({2, 3}), n::from_vector(std::vector<std::int64_t>{0, 1}), 3);
/// @endcode
/// @par References
/// - Deng et al.(2019),ArcFace:Additive Angular Margin Loss for Deep Face Recognition, https://arxiv.org/abs/1801.07698v3;Wang et al.(2018),CosFace:Large Margin Cosine Loss for Deep Face Recognition, https://arxiv.org/abs/1801.09414. Monotonic clipped tail and compact NN1 training follow explicit source teaching choices.
/// @see ApplyAngularMargin(), NormalizedAngularHead()
inline nn::Tensor AngularCrossEntropy(const nn::Tensor &logits, const nn::Tensor &labels,
                                      int classes) {
  angular_margin_detail::Validate(logits, labels, classes);
  return -nn::mean(nn::sum(
      nn::log_softmax(logits) * nn::one_hot(labels, classes, 1.0, 0.0, logits.dtype()), {1}));
}

/// @brief Gather integer class targets in an arbitrary requested dataset-row order.
///
/// Only row metadata enters a host vector. Label gathering uses tensors;no
/// class-value remapping,pixel arithmetic,fitting or source mutation.
///
/// @param d Dataset with aligned Int32/64[N] labels.
/// @param rows Valid source row IDs,duplicates/empty allowed.
/// @return Same-label-type[rows.size()] gathered class IDs.
/// @throws std::invalid_argument For invalid configuration,rank,dtype,dimensions or class labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset d{n::zeros({2, 28, 28, 1}), n::from_vector(std::vector<std::int64_t>{0, 1})};
/// const auto y = n::models::AngularLabels(d, {1, 0});
/// @endcode
/// @par References
/// - Deng et al.(2019),ArcFace:Additive Angular Margin Loss for Deep Face Recognition, https://arxiv.org/abs/1801.07698v3;Wang et al.(2018),CosFace:Large Margin Cosine Loss for Deep Face Recognition, https://arxiv.org/abs/1801.09414. Monotonic clipped tail and compact NN1 training follow explicit source teaching choices.
/// @see ApplyAngularMargin(), NormalizedAngularHead()
inline nn::Tensor AngularLabels(const nn::ImageDataset &d,
                                const std::vector<std::size_t> &rows) {
  if (d.images.dim() != 4 || d.labels.shape() != nn::Shape{d.images.shape()[0]} ||
      (d.labels.dtype() != nn::DType::Int32 && d.labels.dtype() != nn::DType::Int64))
    throw std::invalid_argument("invalid angular dataset labels");
  for (const auto row : rows)
    if (row >= std::size_t(d.images.shape()[0]))
      throw std::invalid_argument("angular label row out of bounds");
  std::vector<std::int64_t> ids;
  for (auto r : rows)
    ids.push_back(std::int64_t(r));
  return nn::gather(d.labels, nn::from_vector(ids));
}

/// @brief Count detached first-maximum class predictions as a tensor confusion matrix.
///
/// Argmax selects the first class on ties. Exact integer segment reductions
/// count every row once. No model forward,fitting or input gradients.
///
/// @param logits Float32/64[B,classes] logits,B positive,classes at least two.
/// @param labels Int32/64[B] valid true class IDs.
/// @return Int64[classes,classes] confusion counts,true rows/predicted columns.
/// @throws std::invalid_argument For invalid configuration,rank,dtype,dimensions or class labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto counts = n::models::AngularClassificationConfusion(
///     n::tensor({1., 0., 0., 1.}, {2, 2}), n::from_vector(std::vector<std::int64_t>{0, 1}));
/// @endcode
/// @par References
/// - Deng et al.(2019),ArcFace:Additive Angular Margin Loss for Deep Face Recognition, https://arxiv.org/abs/1801.07698v3;Wang et al.(2018),CosFace:Large Margin Cosine Loss for Deep Face Recognition, https://arxiv.org/abs/1801.09414. Monotonic clipped tail and compact NN1 training follow explicit source teaching choices.
/// @see ApplyAngularMargin(), NormalizedAngularHead()
inline nn::Tensor AngularClassificationConfusion(const nn::Tensor &logits,
                                                 const nn::Tensor &labels) {
  nn::NoGradGuard guard;
  if (logits.dim() != 2)
    throw std::invalid_argument("rank-two class logits required");
  const auto classes = logits.shape()[1];
  angular_margin_detail::Validate(logits, labels, int(classes));
  const auto ids =
      labels.to(nn::DType::Int64) * nn::full({}, double(classes), nn::DType::Int64) +
      nn::argmax(logits, 1);
  return nn::unsorted_segment_sum(nn::ones_like(ids), ids, classes * classes)
      .reshape({classes, classes});
}
} // namespace datamunge::nn::models
