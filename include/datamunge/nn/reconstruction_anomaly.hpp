#pragma once
/// @file reconstruction_anomaly.hpp
/// @brief Detached per-case residual scores, explicit nominal calibration and tied ranking probes.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/latent_visualization.hpp"
#include "datamunge/nn/metrics.hpp"

namespace datamunge::nn::models {
/// @brief Internal finite score-vector ingress for detached anomaly diagnostics.
namespace anomaly_detail {
/// @brief Copy a nonempty finite Float32/64 CPU score vector to stopped Double values.
/// @param scores Finite signed floating CPU[N] vector with N>0, strides allowed.
/// @return Owning logical-order Double values, with no input gradient path.
/// @throws std::invalid_argument For invalid rank, dtype, empty or nonfinite input.
/// @throws std::logic_error For non-CPU storage.
inline std::vector<double> Scores(const Tensor& scores) {
  if (!scores.defined() || scores.dim() != 1)
    throw std::invalid_argument("anomaly diagnostics require score vectors");
  NoGradGuard guard;
  return DoubleSampleRows(scores).to_vector<double>();
}
} // namespace anomaly_detail

/// @brief Compute detached per-case mean squared residuals in original feature order.
///
/// Finite signed Float32/64 CPU inputs[N,...] flatten trailing axes to a common
/// positive width D; both inputs need the same N and D, not identical trailing
/// shapes or dtypes. For each row, widens each value to double before subtracting,
/// squares and accumulates in feature order, then divides by D. No clipping,
/// fitting, reduction across cases or training-loss gradient is hidden. Result
/// owns contiguous Float64[N] storage and stops both paths without changing
/// caller flags. Strides accepted; inputs unchanged. Work/storage O(N*D), with
/// host validation/copies; nonfinite computed residual sums are rejected.
/// @param original Finite floating CPU samples[N,...], rank>=1, positive axes.
/// @param reconstruction Finite floating CPU samples with matching N and flattened D.
/// @return Owning detached Float64[N] per-case nonnegative scores.
/// @throws std::invalid_argument For invalid samples/alignment or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto s=dm::nn::models::ReconstructionScores(
///     dm::nn::tensor({0.,1.,2.,3.},{2,2}),dm::nn::zeros({2,2}));
/// // s contains .5 and6.5; no gradient path is retained.
/// @endcode
/// @see InkScores(), NominalThreshold(), RankAnomalies(), mse()
inline Tensor ReconstructionScores(const Tensor& original, const Tensor& reconstruction) {
  NoGradGuard guard;
  const auto  a = DoubleSampleRows(original), b = DoubleSampleRows(reconstruction);
  if (a.shape() != b.shape())
    throw std::invalid_argument("residual scores need matching sample count and feature width");
  const auto          x = a.to_vector<double>(), y = b.to_vector<double>();
  std::vector<double> out(a.size(0));
  for (std::int64_t i = 0; i < a.size(0); ++i) {
    double total = 0;
    for (std::int64_t j = 0; j < a.size(1); ++j) {
      const double delta = x[i * a.size(1) + j] - y[i * a.size(1) + j];
      total += delta * delta;
    }
    if (!std::isfinite(total))
      throw std::invalid_argument("nonfinite residual score sum");
    out[i] = total / double(a.size(1));
  }
  return from_vector(out, {a.size(0)});
}

/// @brief Compute detached absolute per-case mean-coordinate deviations from an explicit reference.
///
/// Finite signed Float32/64 CPU samples[N,...] use N>0 and positive trailing
/// width D (a vector has D=1). Widens values then sums in feature order in double,
/// divides by D, and takes abs(mean-fit_mean). The caller owns the reference's
/// fitting policy; no reference estimation, model, clipping or labels are used.
/// Owning contiguous Float64[N] output stops input paths without changing flags.
/// Strides accepted; inputs unchanged. Work/storage O(N*D) with host validation.
/// Nonfinite sums/results are rejected. The historical name refers to image
/// ink, but any finite signed sample coordinates use the same explicit statistic.
/// @param images Finite floating CPU samples[N,...], rank>=1, all axes positive.
/// @param fit_mean Finite explicit reference mean, in the same coordinate units.
/// @return Owning detached nonnegative Float64[N] absolute deviations.
/// @throws std::invalid_argument For invalid samples/reference or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto s=dm::nn::models::InkScores(dm::nn::tensor({0.,1.,1.,1.},{2,2}),.5);
/// // s contains0 and.5; the supplied reference is never refitted.
/// @endcode
/// @see ReconstructionScores(), NominalThreshold()
inline Tensor InkScores(const Tensor& images, double fit_mean) {
  if (!std::isfinite(fit_mean))
    throw std::invalid_argument("finite explicit ink reference required");
  NoGradGuard         guard;
  const auto          a = DoubleSampleRows(images);
  const auto          x = a.to_vector<double>();
  std::vector<double> out(a.size(0));
  for (std::int64_t i = 0; i < a.size(0); ++i) {
    double total = 0;
    for (std::int64_t j = 0; j < a.size(1); ++j)
      total += x[i * a.size(1) + j];
    out[i] = std::abs(total / double(a.size(1)) - fit_mean);
    if (!std::isfinite(total) || !std::isfinite(out[i]))
      throw std::invalid_argument("nonfinite ink score computation");
  }
  return from_vector(out, {a.size(0)});
}

/// @brief Select an explicit nearest-rank nominal score quantile without interpolation.
///
/// Sorts a copy of finite signed Float32/64 CPU scores[N], N>0, then selects
/// one-based rank ceil(quantile*N), evaluating the product in double. Boundary
/// rounding is bounded by the last existing rank. Default .95 reproduces the
/// original nominal-validation95th percentile; N=20 and scores0..19 give18.
/// Like the inverse empirical CDF (R quantile type1 with no fuzz), not R's default
/// linear interpolation. No labels, test outcomes, NaN dropping or refit occurs.
/// The caller supplies only the intended calibration pool; this function cannot
/// detect leakage. Detached scalar; input/flags unchanged. O(N log N) work/O(N)
/// storage; synchronize external mutation of inputs.
/// @param scores Nonempty finite Float32/64 CPU vector[N], strides allowed.
/// @param quantile Finite probability in(0,1], default .95; zero is rejected.
/// @return Existing sorted score at the explicit nearest rank.
/// @throws std::invalid_argument For invalid vector/values or probability.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto threshold=dm::nn::models::NominalThreshold(dm::nn::arange(0,20));
/// // threshold==18; an anomaly flag uses score>threshold, not>=.
/// @endcode
/// @par References
/// - R stats quantile, type1 inverse empirical CDF:
///   https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html .
///   This explicit nearest-rank Double rule uses no fuzz and rejects probability0.
/// @see RankAnomalies(), ReconstructionScores()
inline double NominalThreshold(const Tensor& scores, double quantile = .95) {
  if (!std::isfinite(quantile) || quantile <= 0 || quantile > 1)
    throw std::invalid_argument("nominal quantile must lie in(0,1]");
  auto values = anomaly_detail::Scores(scores);
  std::sort(values.begin(), values.end());
  const auto index =
      std::min(std::size_t(std::ceil(quantile * double(values.size())) - 1), values.size() - 1);
  return values[index];
}

/// @brief Own four detached ranking/operating-point diagnostics for explicit score pools.
///
/// Larger scores indicate the supplied anomaly pool. AUROC gives half credit
/// to equal-score normal/anomaly pairs; AP processes equal-score groups at one
/// precision/recall endpoint. FPR/TPR use strictly score>threshold. AP depends
/// on the supplied prevalence; no real-world anomaly or quality guarantee is
/// implied. Plain value fields have no tensor lifetime or gradient dependency.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto r=dm::nn::models::RankAnomalies(dm::nn::tensor({1.,2.},{2}),
///     dm::nn::tensor({2.,3.},{2}),2.);
/// // r.auroc==.875, r.normal_fpr==0, r.anomaly_tpr==.5; AP is approximately5/6.
/// @endcode
/// @see RankAnomalies(), NominalThreshold()
struct ReconstructionRanking {
  double auroc;             ///< Anomaly-versus-normal AUROC in[0,1], score ties half credited.
  double average_precision; ///< Noninterpolated group-tied AP in[0,1], supplied anomalies positive.
  double normal_fpr;  ///< Fraction of supplied nominal scores strictly greater than threshold.
  double anomaly_tpr; ///< Fraction of supplied anomaly scores strictly greater than threshold.
};

/// @brief Rank explicit nominal/anomaly scores with canonical tied AUROC/AP and a strict threshold.
///
/// Requires two nonempty finite signed Float32/64 CPU vectors; lengths/dtypes
/// may differ. Normal=0, anomaly=1 labels are assembled only for existing
/// nn::roc_auc and nn::average_precision. No alternate ranking algorithm is
/// implemented. Constant equal scores give AUROC .5 and AP equal to supplied
/// anomaly prevalence. FPR/TPR use score>threshold, so exact threshold ties
/// are unflagged. No calibration, fitting, weighting or input gradient path;
/// input/flags unchanged. Scalar record owns values independently of inputs.
/// O((N+M)log(N+M)) work/O(N+M) storage, including two canonical sorted sweeps.
/// @param normal_scores Finite Float32/64 CPU nominal vector[N], N>0.
/// @param anomaly_scores Finite Float32/64 CPU anomaly vector[M], M>0.
/// @param threshold Finite explicit operating threshold, usually separately calibrated.
/// @return Owning AUROC, group-tied AP and strict nominal/anomaly flag fractions.
/// @throws std::invalid_argument For invalid vectors/nonfinite scores or threshold.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto r=dm::nn::models::RankAnomalies(dm::nn::tensor({1.,1.},{2}),
///     dm::nn::tensor({1.},{1}),1.);
/// // AUROC=.5, AP=1/3, FPR=0 and TPR=0: equal values are processed as one group.
/// @endcode
/// @par References
/// - Scikit-learn ROC AUC convention,
/// https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html .
/// - Scikit-learn noninterpolated AP,
/// https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html .
///   Recall increments weight precision at each equal-score group's endpoint.
/// @see ReconstructionRanking, NominalThreshold(), roc_auc(), average_precision()
inline ReconstructionRanking
RankAnomalies(const Tensor& normal_scores, const Tensor& anomaly_scores, double threshold) {
  if (!std::isfinite(threshold))
    throw std::invalid_argument("finite anomaly threshold required");
  auto         normal  = anomaly_detail::Scores(normal_scores);
  const auto   anomaly = anomaly_detail::Scores(anomaly_scores);
  const double fp =
      double(std::count_if(
          normal.begin(), normal.end(), [threshold](double x) { return x > threshold; })) /
      double(normal.size());
  const double tp =
      double(std::count_if(
          anomaly.begin(), anomaly.end(), [threshold](double x) { return x > threshold; })) /
      double(anomaly.size());
  const auto count = normal.size();
  normal.insert(normal.end(), anomaly.begin(), anomaly.end());
  std::vector<int> labels(normal.size(), 0);
  std::fill(labels.begin() + count, labels.end(), 1);
  return {nn::roc_auc(labels, normal), nn::average_precision(labels, normal), fp, tp};
}

/// @brief Apply one of three explicit detached28x28 grayscale stress transformations.
///
/// Finite signed Float32 CPU original[N,28,28,1], N>0, may be strided. Kind0
/// replaces the upper-left8x8 block by1; kind1 multiplies every pixel by the
/// Float32 representation of .35; kind2 shifts6 columns right with zero fill.
/// No clipping, random augmentation, labels, learned transform or anomaly
/// inference is hidden. These are paired synthetic stress controls, not a
/// definition of real anomalies. Owning contiguous Float32 output stops input
/// paths without changing flags; input unchanged. Unmodified/shifted values
/// retain their original bits. O(N*784) work/storage, including finite ingress
/// validation through DoubleSampleRows. Different kinds never fit a model.
/// @param original Finite signed Float32 CPU grayscale batch[N,28,28,1], N>0.
/// @param kind Explicit0=white patch,1=darkening,2=zero-filled right shift.
/// @return Owning contiguous detached Float32 batch with unchanged shape.
/// @throws std::invalid_argument For invalid shape/dtype/kind or nonfinite values.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto darker=dm::nn::models::AnomalyCorruption(dm::nn::ones({2,28,28,1}),1);
/// // darker is Float32[2,28,28,1], filled with Float32(.35), without gradients.
/// @endcode
/// @see ReconstructionScores(), InkScores()
inline Tensor AnomalyCorruption(const Tensor& original, int kind) {
  if (kind < 0 || kind > 2 || !original.defined() || original.dtype() != DType::Float32 ||
      original.dim() != 4 || original.size(1) != 28 || original.size(2) != 28 ||
      original.size(3) != 1)
    throw std::invalid_argument("anomaly stress needs Float32[N,28,28,1] and kind0..2");
  NoGradGuard guard;
  (void)DoubleSampleRows(original);
  if (kind == 1)
    return (original * .35).contiguous();
  if (kind == 2)
    return concat(
               {zeros({original.size(0), 28, 6, 1}),
                slice(original, {0, 0, 0, 0}, {original.size(0), 28, 22, 1})},
               2)
        .contiguous();
  const auto row    = less(arange(0, 28).reshape({1, 28, 1, 1}), 8.),
             column = less(arange(0, 28).reshape({1, 1, 28, 1}), 8.);
  return where(logical_and(row, column), 1., original).contiguous();
}
} // namespace datamunge::nn::models
