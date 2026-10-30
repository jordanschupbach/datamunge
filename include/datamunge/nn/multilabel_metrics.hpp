#pragma once
/// @file multilabel_metrics.hpp
/// @brief Tensor grouped-score-tie AP,independent-bit metrics and validation-only global thresholds.
#include "datamunge/nn/multilabel_data.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Internal probability/target matrix validation shared by metrics and previews.
namespace multilabel_metrics_detail {
/// @brief Require aligned nonempty independent probabilities and binary target bits.
/// @param probability Float32/64[N,C] finite[0,1],N/C positive.
/// @param labels Same-shape numeric/bool binary targets.
/// @throws std::invalid_argument For invalid rank,type,shape or values.
inline void Validate(const nn::Tensor &probability, const nn::Tensor &labels) {
  multilabel_data_detail::ValidateLabels(labels);
  if (probability.shape() != labels.shape() || probability.numel() < 1 ||
      (probability.dtype() != nn::DType::Float32 &&
       probability.dtype() != nn::DType::Float64) ||
      !nn::all(nn::logical_and(nn::greater_equal(probability, 0.),
                               nn::less_equal(probability, 1.)))
           .item<bool>())
    throw std::invalid_argument("valid independent probability matrix required");
}
} // namespace multilabel_metrics_detail
/// @brief Compute noninterpolated AP with equal scores grouped at one operating point.
///
/// Stable sorting orders score groups;only group endpoints enter precision.
/// Host iteration serializes endpoint indices computed by tensor equality;
/// all TP,recall,precision and AP arithmetic uses Float64 tensors. Constant
/// scores yield prevalence,independent of row order. O(N log N) time. No
/// optimizer,interpolation,optimistic tie breaking or input gradients.
///
/// @param scores Finite Float32/64[N] scores,any real range,N may be zero.
/// @param labels Same-length numeric/bool binary targets.
/// @return Detached Float64 scalar AP,or-1 when no true positives exist.
/// @throws std::invalid_argument For invalid probability/score shapes,dtypes,nonbinary labels,nonfinite values or threshold.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto ap =
///     n::models::MultilabelAP(n::tensor({.5, .5, .5, .5}), n::tensor({1., 0., 1., 0.}));
/// @endcode
/// @par References
/// - Scikit-learn official average_precision_score definition, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html. Noninterpolated precision weighted by recall increments;equal-score groups are processed together. Global micro-F1 grid/tie policy follows the source teaching protocol.
/// @see MultilabelAP(), ScoreMultilabel(), CalibrateMultilabel()
inline nn::Tensor MultilabelAP(const nn::Tensor &scores, const nn::Tensor &labels) {
  nn::NoGradGuard guard;
  if (scores.dim() != 1 || scores.shape() != labels.shape() ||
      (scores.dtype() != nn::DType::Float32 && scores.dtype() != nn::DType::Float64) ||
      !nn::all(nn::less(nn::abs(scores), std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("finite aligned AP vectors required");
  multilabel_data_detail::ValidateLabels(labels.reshape({labels.numel(), 1}));
  if (scores.numel() == 0)
    return nn::full({}, -1., nn::DType::Float64);
  const auto p = scores.to(nn::DType::Float64), y = labels.to(nn::DType::Float64);
  const auto order = nn::argsort(p, true), sorted = nn::gather(p, order),
             truth = nn::gather(y, order);
  const auto n = p.numel();
  const auto group_end = nn::concat(
      {nn::not_equal(nn::slice_axis(sorted, 0, 0, n - 1), nn::slice_axis(sorted, 0, 1, n)),
       nn::ones({1}, nn::DType::Bool)},
      0);
  const auto ends = group_end.to_vector<bool>();
  std::vector<std::int64_t> indices;
  for (std::size_t i = 0; i < ends.size(); ++i)
    if (ends[i])
      indices.push_back(std::int64_t(i));
  const auto rows = nn::from_vector(indices);
  const auto tp = nn::gather(nn::cumsum(truth, 0), rows);
  const auto before = nn::concat(
      {nn::zeros({1}, nn::DType::Float64), nn::slice_axis(tp, 0, 0, tp.numel() - 1)}, 0);
  const auto positives = nn::sum(y);
  const auto ap = nn::sum((tp - before) / nn::maximum(positives, nn::ones_like(positives)) *
                          tp / (rows.to(nn::DType::Float64) + 1.));
  return nn::where(nn::greater(positives, 0.), ap, nn::full({}, -1., nn::DType::Float64));
}
/// @brief Own detached independent-bit loss,ranking,decision and support metrics.
///
/// Floating metrics are Float64 scalar tensors;class supports and confusion
/// counts are Int64. mAP excludes true-absent classes;macro F1 averages every
/// class,including absent classes with zero F1. No mutable running state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto probability = n::tensor({.9, .2, .3, .1, .8, .4}, {2, 3}),
///            labels = n::tensor({1., 0., 0., 0., 1., 0.}, {2, 3});
/// const auto score = n::models::ScoreMultilabel(probability, labels);
/// @endcode
/// @par References
/// - Scikit-learn official average_precision_score definition, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html. Noninterpolated precision weighted by recall increments;equal-score groups are processed together. Global micro-F1 grid/tie policy follows the source teaching protocol.
/// @see MultilabelAP(), ScoreMultilabel(), CalibrateMultilabel()
struct TensorMultilabelMetrics {
  nn::Tensor nll;      ///< Float64 scalar mean unweighted bit BCE,probability floors1e-12.
  nn::Tensor brier;    ///< Float64 scalar mean squared probability error per bit.
  nn::Tensor map;      ///< Float64 scalar mean grouped-tie AP over positive-support classes.
  nn::Tensor micro_ap; ///< Float64 scalar grouped-tie AP over all bits,zero if no positives.
  nn::Tensor micro_f1; ///< Float64 scalar global2TP/(2TP+FP+FN),zero for zero denominator.
  nn::Tensor
      macro_f1; ///< Float64 scalar mean per-class F1 across all classes,zero-empty convention.
  nn::Tensor hamming;        ///< Float64 scalar fraction of incorrect class bits.
  nn::Tensor exact;          ///< Float64 scalar fraction of images with every bit correct.
  nn::Tensor ap_classes;     ///< Int64 scalar classes with at least one positive target.
  nn::Tensor absent_classes; ///< Int64 scalar classes without a positive target.
  nn::Tensor tp;             ///< Int64 scalar true-positive bit count.
  nn::Tensor fp;             ///< Int64 scalar false-positive bit count.
  nn::Tensor fn;             ///< Int64 scalar false-negative bit count.
  nn::Tensor tn;             ///< Int64 scalar true-negative bit count.
};
/// @brief Score independent category probabilities with grouped ties and explicit absent support.
///
/// Promote probabilities to Float64 before threshold comparisons and loss
/// arithmetic. Floors apply only to log probabilities. AP groups exact ties;
/// classes without positives do not enter mAP,but enter macro F1 with zero.
/// All-negative targets give mAP/microAP/F1 zero;exact match remains one when
/// all decisions are negative. No model forward,optimizer or calibration.
///
/// @param probability Float32/64[N,C] finite probabilities[0,1],N/C positive.
/// @param labels Same-shape numeric/bool binary targets.
/// @param threshold Finite global cutoff,default.5;predict positive at or above it.
/// @return Detached unweighted BCE,Brier,mAP,micro AP/F1,macro F1,Hamming,exact match and counts.
/// @throws std::invalid_argument For invalid probability/score shapes,dtypes,nonbinary labels,nonfinite values or threshold.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto probability = n::tensor({.9, .2, .3, .1, .8, .4}, {2, 3}),
///            labels = n::tensor({1., 0., 0., 0., 1., 0.}, {2, 3});
/// const auto score = n::models::ScoreMultilabel(probability, labels);
/// @endcode
/// @par References
/// - Scikit-learn official average_precision_score definition, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html. Noninterpolated precision weighted by recall increments;equal-score groups are processed together. Global micro-F1 grid/tie policy follows the source teaching protocol.
/// @see MultilabelAP(), ScoreMultilabel(), CalibrateMultilabel()
inline TensorMultilabelMetrics ScoreMultilabel(const nn::Tensor &probability,
                                               const nn::Tensor &labels,
                                               double threshold = .5) {
  nn::NoGradGuard guard;
  multilabel_metrics_detail::Validate(probability, labels);
  if (!std::isfinite(threshold))
    throw std::invalid_argument("finite multilabel threshold required");
  const auto p = probability.to(nn::DType::Float64), y = labels.to(nn::DType::Float64);
  const auto truth = nn::equal(y, 1.), predicted = nn::greater_equal(p, threshold);
  const auto tp_class = nn::count_nonzero(nn::logical_and(predicted, truth), {0}),
             fp_class =
                 nn::count_nonzero(nn::logical_and(predicted, nn::logical_not(truth)), {0}),
             fn_class =
                 nn::count_nonzero(nn::logical_and(nn::logical_not(predicted), truth), {0}),
             tn_class = nn::count_nonzero(
                 nn::logical_and(nn::logical_not(predicted), nn::logical_not(truth)), {0});
  const auto tp = nn::sum(tp_class), fp = nn::sum(fp_class), fn = nn::sum(fn_class),
             tn = nn::sum(tn_class);
  const auto class_denominator =
      (tp_class + tp_class + fp_class + fn_class).to(nn::DType::Float64);
  const auto per_class_f1 = (tp_class + tp_class).to(nn::DType::Float64) /
                            nn::maximum(class_denominator, nn::ones_like(class_denominator));
  const auto denominator = (tp + tp + fp + fn).to(nn::DType::Float64);
  std::vector<nn::Tensor> aps;
  for (std::int64_t c = 0; c < p.shape()[1]; ++c)
    aps.push_back(MultilabelAP(nn::select(p, 1, c), nn::select(y, 1, c)));
  const auto class_ap = nn::stack(aps), present = nn::greater_equal(class_ap, 0.);
  const auto count = nn::count_nonzero(present);
  const auto map = nn::sum(nn::where(present, class_ap, 0.)) /
                   nn::maximum(count.to(nn::DType::Float64), nn::ones({}, nn::DType::Float64));
  const auto micro = nn::maximum(MultilabelAP(p.reshape({-1}), y.reshape({-1})),
                                 nn::zeros({}, nn::DType::Float64));
  const auto floor = nn::full({}, 1e-12, nn::DType::Float64);
  return {-nn::mean(y * nn::log(nn::maximum(p, floor)) +
                    (1. - y) * nn::log(nn::maximum(1. - p, floor))),
          nn::mean(nn::square(p - y)),
          map,
          micro,
          (tp + tp).to(nn::DType::Float64) /
              nn::maximum(denominator, nn::ones_like(denominator)),
          nn::mean(per_class_f1),
          (fp + fn).to(nn::DType::Float64) / double(p.numel()),
          nn::mean(nn::all(nn::equal(predicted, truth), {1}).to(nn::DType::Float64)),
          count,
          nn::from_scalar(p.shape()[1]) - count,
          tp,
          fp,
          fn,
          tn};
}
/// @brief Select one global probability threshold using validation micro F1 only.
///
/// Candidate1.01 predicts no positives. Scan candidates in ascending order;
/// accept scores strictly above best+1e-12 or within1e-12 of the running best,
/// so exact F1 ties choose the highest cutoff. Tensor comparisons retain the
/// source tolerance rule without host score arithmetic. No test labels,model
/// forward or optimizer;use a separate validation subset for this selection.
///
/// @param validation Float32/64[N,C] finite independent validation probabilities.
/// @param truth Same-shape numeric/bool validation targets,N/C positive.
/// @return Detached Float64 scalar selected threshold from.05,.10,...,.95,1.01.
/// @throws std::invalid_argument For invalid probability/score shapes,dtypes,nonbinary labels,nonfinite values or threshold.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto probability = n::tensor({.9, .2, .3, .1, .8, .4}, {2, 3}),
///            labels = n::tensor({1., 0., 0., 0., 1., 0.}, {2, 3});
/// const auto score = n::models::ScoreMultilabel(probability, labels);
/// const auto threshold = n::models::CalibrateMultilabel(probability, labels);
/// @endcode
/// @par References
/// - Scikit-learn official average_precision_score definition, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html. Noninterpolated precision weighted by recall increments;equal-score groups are processed together. Global micro-F1 grid/tie policy follows the source teaching protocol.
/// @see MultilabelAP(), ScoreMultilabel(), CalibrateMultilabel()
inline nn::Tensor CalibrateMultilabel(const nn::Tensor &validation, const nn::Tensor &truth) {
  nn::NoGradGuard guard;
  multilabel_metrics_detail::Validate(validation, truth);
  const auto grid = nn::concat({nn::arange(1, 20, 1, nn::DType::Float64) * .05,
                                nn::full({1}, 1.01, nn::DType::Float64)},
                               0);
  const auto p = validation.to(nn::DType::Float64)
                     .reshape({1, validation.shape()[0], validation.shape()[1]});
  const auto target = nn::equal(truth, 1.).reshape({1, truth.shape()[0], truth.shape()[1]});
  const auto predicted = nn::greater_equal(p, grid.reshape({20, 1, 1}));
  const auto
      tp =
          nn::count_nonzero(nn::logical_and(predicted, target), {1, 2}).to(nn::DType::Float64),
      fp = nn::count_nonzero(nn::logical_and(predicted, nn::logical_not(target)), {1, 2})
               .to(nn::DType::Float64),
      fn = nn::count_nonzero(nn::logical_and(nn::logical_not(predicted), target), {1, 2})
               .to(nn::DType::Float64);
  const auto denominator = 2. * tp + fp + fn;
  const auto scores = 2. * tp / nn::maximum(denominator, nn::ones_like(denominator));
  auto best = nn::full({}, -1., nn::DType::Float64),
       chosen = nn::full({}, 1.01, nn::DType::Float64);
  for (std::int64_t i = 0; i < 20; ++i) {
    const auto score = nn::select(scores, 0, i);
    const auto improve = nn::logical_or(nn::greater(score, best + 1e-12),
                                        nn::less_equal(nn::abs(score - best), 1e-12));
    best = nn::where(improve, score, best);
    chosen = nn::where(improve, nn::select(grid, 0, i), chosen);
  }
  return chosen;
}
} // namespace datamunge::nn::models
