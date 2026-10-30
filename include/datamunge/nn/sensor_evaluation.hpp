#pragma once
/// @file sensor_evaluation.hpp
/// @brief Tensor balanced classification/AUROC and standardized source-reference parity helpers.
#include "datamunge/nn/sensor_data.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Own tensor-reduced sensor classification diagnostics in source-compatible presentation fields.
///
/// Numerical reductions use tensors;Double/vector fields serialize final
/// scalar metrics and discrete class counts for existing C++ examples.
/// Every true class must occur;macro scores weight each source class equally.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto z = n::tensor({2., 0., 0., 2.}, {2, 2});
/// const auto labels = n::tensor({0, 1}).to(n::DType::Int32);
/// const auto s = m::ScoreSensor(z, labels, 2);
/// @endcode
/// @par References
/// - Official ECG200, https://www.timeseriesclassification.com/description.php?Dataset=ECG200;official BasicMotions, https://www.timeseriesclassification.com/description.php?Dataset=BasicMotions. Source indices and equal-class diagnostics are preserved;these are small source-record-held-out teaching comparisons,not patient/participant benchmark claims.
/// @see ScoreSensor()
struct SensorScore {
  double accuracy = 0; ///< Equal-record correct fraction,first class wins logit ties.
  double balanced = 0; ///< Equal-class mean recall,every class must have true support.
  double macro_f1 = 0; ///< Equal-class mean2TP/(true support+predicted support).
  double nll = 0;      ///< Unweighted equal-record stable Double categorical NLL.
  double auc0 =
      0; ///< Class0-versus-class1 pairwise margin AUROC for binary data;unused0 for multiclass.
  std::vector<int> confusion; ///< Row-major Int32 true-row/predicted-column discrete counts.
  std::vector<double> recall; ///< Double per-source-class recalls in original index order.
};
/// @brief Compute tie-aware class0-versus-rest ROC area with pairwise tensor comparisons.
///
/// No threshold fitting or numerical host pair loops. Require at least one
/// class0 and one other-class row. Complexity/storage O(N²),appropriate for
/// small source test pools. Input scores convert once to Double before comparison.
///
/// @param score Finite floating[N] ranking scores,higher means source class0.
/// @param labels Int32/64[N] nonnegative original source class identities;zero is positive.
/// @return Double fraction of positive/negative pairs correctly ranked,ties half credit.
/// @throws std::invalid_argument For invalid vectors,nonfinite scores or missing positive/negative class.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto labels = n::tensor({0, 1, 0, 1}).to(n::DType::Int32);
/// const auto auc =
///     m::TensorSensorAuc0(n::tensor({.8, .8, .2, .1}).to(n::DType::Float64), labels); // .625
/// @endcode
/// @see SensorAuc0(), ScoreSensor()

inline double TensorSensorAuc0(const Tensor &score, const Tensor &labels) {
  nn::NoGradGuard guard;
  if (score.dim() != 1 || score.size(0) < 2 || labels.shape() != score.shape() ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64) ||
      !nn::all(nn::is_finite(score)).item<bool>() ||
      !nn::all(nn::greater_equal(labels, 0.)).item<bool>())
    throw std::invalid_argument(
        "aligned finite class0 ranking scores and integer labels required");
  const auto v = score.to(nn::DType::Float64), positive = nn::equal(labels, 0.),
             pairs = nn::logical_and(nn::expand_dims(positive, 1),
                                     nn::expand_dims(nn::logical_not(positive), 0)),
             count = nn::sum(pairs.to(nn::DType::Float64));
  if (count.item() <= 0)
    throw std::invalid_argument("both class0 and other-class rows required");
  const auto a = nn::expand_dims(v, 1), b = nn::expand_dims(v, 0),
             credit = nn::greater(a, b).to(nn::DType::Float64) +
                      .5 * nn::equal(a, b).to(nn::DType::Float64);
  return (nn::sum(nn::where(pairs, credit, 0.)) / count).item();
}
/// @brief Adapt source-compatible Double ranking values to the canonical tensor AUROC.
///
/// Compatibility ingress only;all comparisons and reductions delegate to tensors.
///
/// @param score Finite numeric ingress vector,one ranking score per source row.
/// @param labels Aligned integer original class identities,zero positive.
/// @return Double pairwise class0 AUROC with ties half credited.
/// @throws std::invalid_argument For invalid scores/labels or missing positive/negative class.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto labels = n::tensor({0, 1, 0, 1}).to(n::DType::Int32);
/// const auto auc = m::SensorAuc0({.8, .8, .2, .1}, labels); // .625
/// @endcode
/// @see TensorSensorAuc0()

inline double SensorAuc0(const std::vector<double> &score, const Tensor &labels) {
  return TensorSensorAuc0(nn::from_vector(score, {std::int64_t(score.size())}), labels);
}
/// @brief Score raw sensor logits with complete tensor confusion,balanced recall,F1 and binary AUROC.
///
/// First class wins equal logits. MacroF1 includes every source class;missing
/// true classes reject rather than silently changing denominator. Stable NLL
/// uses Double logits independently of fitting class weights. Binary class0
/// AUROC ranks Double(logit0)-Double(logit1),with half-credit score ties.
/// Multiclass auc0 remains unused0;callers label it unavailable. No numeric
/// host ranking/metric loops or threshold optimization.
///
/// @param logits Finite floating[N,classes] raw unbounded scores,N>0.
/// @param labels Aligned Int32/64[N] true class indices,every class has support.
/// @param classes Score-column count>=2,original source class order.
/// @return Equal-record accuracy/unweighted NLL,equal-class recall/F1,all counts and binary AUROC.
/// @throws std::invalid_argument For invalid logits,label inventory or absent source class support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto z = n::tensor({2., 0., 0., 2.}, {2, 2});
/// const auto labels = n::tensor({0, 1}).to(n::DType::Int32);
/// const auto s = m::ScoreSensor(z, labels, 2);
/// @endcode
/// @par References
/// - Official ECG200, https://www.timeseriesclassification.com/description.php?Dataset=ECG200;official BasicMotions, https://www.timeseriesclassification.com/description.php?Dataset=BasicMotions. Source indices and equal-class diagnostics are preserved;these are small source-record-held-out teaching comparisons,not patient/participant benchmark claims.
/// @see SensorCE(), TensorSensorAuc0()

inline SensorScore ScoreSensor(const Tensor &logits, const Tensor &labels, int classes) {
  nn::NoGradGuard guard;
  if (logits.dim() != 2 || logits.size(1) != classes || classes < 2)
    throw std::invalid_argument("matching original sensor class columns required");
  const auto base = ScoreSequenceClassifier(logits, labels);
  const auto cm = base.confusion.to(nn::DType::Float64), truth = nn::sum(cm, {1}),
             predicted = nn::sum(cm, {0}),
             identity = nn::one_hot(nn::arange(0, classes, 1, nn::DType::Int64), classes, 1.,
                                    0., nn::DType::Float64),
             tp = nn::sum(cm * identity, {1});
  if (!nn::all(nn::greater(truth, 0.)).item<bool>())
    throw std::invalid_argument("every original sensor test class needs support");
  const auto recall = tp / truth, f1 = 2. * tp / (truth + predicted);
  SensorScore out;
  out.accuracy = base.accuracy.item();
  out.nll = base.nll.item();
  out.balanced = nn::mean(recall).item();
  out.macro_f1 = nn::mean(f1).item();
  out.confusion =
      base.confusion.to(nn::DType::Int32).reshape({classes * classes}).to_vector<int>();
  out.recall = recall.to_vector<double>();
  if (classes == 2) {
    const auto z = logits.to(nn::DType::Float64);
    out.auc0 = TensorSensorAuc0(nn::select(z, 1, 0) - nn::select(z, 1, 1), labels);
  }
  return out;
}
/// @brief Compute standardized retained-observation1NN or a centroid optimization parity oracle.
///
/// No centering is needed because shared channel means cancel in differences.
/// Nearest retains source Float32 subtraction/division before Double squared
/// reduction and first fitting-row distance ties. Centroid sums/means and
/// standardized differences use Double,output roundsFloat32. Reports visibly
/// fit centroid fields with SGD;the closed-form branch verifies parity only.
/// Neither test labels nor held-out population statistics enter either score.
///
/// @param fit Fitting-only Float32[N,T,C] signals with every bounded class present.
/// @param test Float32[M,T,C] observed signals;true labels are ignored.
/// @param stddev Positive finite[C] Float32 fitting population standard deviations.
/// @param classes Number of original source classes>=2.
/// @param centroid False selects parameter-free1NN;true computes closed-form reference oracle.
/// @return Float32[M,classes] onehot nearest-class pseudo-logits or negative standardized centroid MSE.
/// @throws std::invalid_argument For invalid signal geometry,labels,std values or missing fitting classes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SeriesData f{n::concat({n::zeros({1, 8, 1}), n::ones({1, 8, 1})}, 0),
///                 n::tensor({0, 1}).to(n::DType::Int32), n::zeros({2}, n::DType::Int32)};
/// const auto z = m::SensorReference(f, f, n::ones({1}), 2, false);
/// @endcode
/// @par References
/// - Official ECG200, https://www.timeseriesclassification.com/description.php?Dataset=ECG200;official BasicMotions, https://www.timeseriesclassification.com/description.php?Dataset=BasicMotions. Source indices and equal-class diagnostics are preserved;these are small source-record-held-out teaching comparisons,not patient/participant benchmark claims.
/// @see LoadSensor(), ScoreSensor()

inline Tensor SensorReference(const SeriesData &fit, const SeriesData &test,
                              const Tensor &stddev, int classes, bool centroid) {
  nn::NoGradGuard guard;
  if (fit.signals.dim() != 3 || test.signals.dim() != 3 || fit.signals.size(0) < 1 ||
      test.signals.size(0) < 1 || fit.signals.size(1) != test.signals.size(1) ||
      fit.signals.size(2) != test.signals.size(2) ||
      fit.labels.shape() != nn::Shape{fit.signals.size(0)} ||
      stddev.shape() != nn::Shape{fit.signals.size(2)} || classes < 2 ||
      !nn::all(nn::logical_and(nn::is_finite(stddev), nn::greater(stddev, 0.))).item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(fit.labels, 0.),
                               nn::less(fit.labels, double(classes))))
           .item<bool>())
    throw std::invalid_argument(
        "aligned original sensor references and positive fitting std required");
  const auto weights = nn::one_hot(fit.labels, classes, 1., 0., nn::DType::Float64)
                           .reshape({fit.signals.size(0), classes, 1, 1}),
             counts = nn::sum(weights, {0});
  if (!nn::all(nn::greater(counts, 0.)).item<bool>())
    throw std::invalid_argument("every fitting sensor class required");
  if (centroid) {
    const auto centers =
                   nn::sum(
                       weights *
                           nn::expand_dims(
                               fit.signals.to(nn::DType::Float32).to(nn::DType::Float64), 1),
                       {0}) /
                   counts,
               diff = (nn::expand_dims(
                           test.signals.to(nn::DType::Float32).to(nn::DType::Float64), 1) -
                       nn::expand_dims(centers, 0)) /
                      stddev.to(nn::DType::Float32).to(nn::DType::Float64);
    return (-nn::mean(nn::square(diff), {2, 3})).to(nn::DType::Float32);
  }
  const auto diff = ((nn::expand_dims(test.signals.to(nn::DType::Float32), 1) -
                      nn::expand_dims(fit.signals.to(nn::DType::Float32), 0)) /
                     stddev.to(nn::DType::Float32))
                        .to(nn::DType::Float64),
             distance = nn::sum(nn::square(diff), {2, 3}), nearest = nn::argmin(distance, 1);
  return nn::one_hot(nn::gather(fit.labels, nearest, 0), classes);
}
/// @brief Reduce a declared observed-input intervention to mean absolute Double output change.
///
/// Used for per-channel replacement/time-reversal diagnostics;nonzero change
/// shows computation sensitivity,not accuracy or feature importance.
///
/// @param a First numeric output tensor.
/// @param b Matching-shape second numeric output tensor.
/// @return Double mean abs(Double(a)-Double(b)),without fitting or mutation.
/// @throws std::invalid_argument For differing shapes or empty tensors.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto change = m::MeanAbsChange(n::ones({2, 3}), n::zeros({2, 3})); // 1
/// @endcode
/// @see SensorCnn()

inline double MeanAbsChange(const Tensor &a, const Tensor &b) {
  if (a.shape() != b.shape() || a.numel() < 1)
    throw std::invalid_argument("matching nonempty observed outputs required");
  return nn::mean(nn::abs(a.to(nn::DType::Float64) - b.to(nn::DType::Float64))).item();
}
} // namespace datamunge::nn::models
