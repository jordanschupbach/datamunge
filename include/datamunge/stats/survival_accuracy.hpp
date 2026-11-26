#pragma once

/// @file survival_accuracy.hpp
/// @brief Inverse-probability-of-censoring-weighted time-dependent AUC and (integrated) Brier scores.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/survival.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Cumulative/dynamic time-dependent AUC at each evaluation time.
/// @par Example
/// @code
/// const dm::stats::TimeDependentAuc a;
/// std::println("{}", a.auc.empty()); // true
/// @endcode
struct TimeDependentAuc {
    std::vector<double> times;         ///< Evaluation times.
    std::vector<double> auc;           ///< Weighted probability that a case outranks a control (ties count one half).
    std::vector<std::size_t> cases;    ///< Events at or before each time.
    std::vector<std::size_t> controls; ///< Subjects still under observation after each time.
};

/// @brief Time-dependent AUC with inverse-probability-of-censoring weights, as `riskRegression::Score`.
///
/// At time \f$t\f$, cases are subjects with an observed event by \f$t\f$ and controls are subjects
/// still event-free after \f$t\f$. Case \f$i\f$ gets weight \f$1/\hat G(T_i-)\f$ and every control gets
/// \f$1/\hat G(t)\f$, where \f$\hat G\f$ is the Kaplan–Meier estimate of the censoring distribution
/// (with failures at a tied time removed before censorings). The AUC is the weighted fraction of
/// case–control pairs whose risk ranks the case higher, which estimates
/// \f$P(M_i>M_j\mid T_i\le t<T_j)\f$ consistently under random censoring.
/// @param observations Times and event indicators.
/// @param risk One risk value per observation and time (rows: observations, columns: times), higher meaning earlier failure.
/// @param times Evaluation times, increasing and positive.
/// @return The AUC and the case and control counts at each time.
/// @throws std::invalid_argument For mismatched sizes, invalid data or times, a time with no cases or no
/// controls, or a time beyond the support of the censoring distribution.
/// @par Example
/// @code
/// const std::vector<dm::stats::SurvivalObservation> obs{{1, true}, {2, false}, {3, true}, {4, true}, {5, false}};
/// const auto a = dm::stats::time_dependent_auc(obs, dm::linalg::DenseMatrix<double>(5, 1, {5, 1, 1, 2, 1}), {3.5});
/// std::println("{:.3f}", a.auc[0]); // 0.571
/// @endcode
/// @par References
/// - P. J. Heagerty, T. Lumley and M. S. Pepe (2000). "Time-dependent ROC curves for censored
///   survival data and a diagnostic marker." Biometrics 56(2), 337-344. doi:10.1111/j.0006-341X.2000.00337.x
/// - H. Uno, T. Cai, L. Tian and L. J. Wei (2007). "Evaluating prediction rules for t-year survivors
///   with censored regression models." JASA 102(478), 527-537. doi:10.1198/016214507000000149
/// @see brier_score()
[[nodiscard]] TimeDependentAuc time_dependent_auc(const std::vector<SurvivalObservation>& observations,
                                                  const linalg::DenseMatrix<double>& risk,
                                                  const std::vector<double>& times);

/// @brief Brier scores over a time grid and their integral.
/// @par Example
/// @code
/// const dm::stats::BrierScore b;
/// std::println("{}", b.integrated); // 0
/// @endcode
struct BrierScore {
    std::vector<double> times;       ///< Evaluation times.
    std::vector<double> brier;       ///< Weighted mean squared error of the predicted survival.
    std::vector<double> null_brier;  ///< The same for the pooled Kaplan–Meier prediction.
    double integrated{};             ///< Left-step integral of the Brier score over the grid divided by its length.
    double null_integrated{};        ///< The same for the Kaplan–Meier prediction.
    double explained{};              ///< \f$1-\f$ integrated / null integrated (integrated Brier skill).
};

/// @brief Brier scores with inverse-probability-of-censoring weights, as `pec::pec` and `pec::crps`.
///
/// The score at time \f$t\f$ is
/// \f[\mathrm{BS}(t)=\frac1n\sum_i\Big[\frac{\hat S(t\mid x_i)^2\,1(T_i\le t,\delta_i=1)}{\hat G(T_i-)}+\frac{\{1-\hat S(t\mid x_i)\}^2\,1(T_i>t)}{\hat G(t)}\Big],\f]
/// where \f$\hat G\f$ is the Kaplan–Meier censoring distribution. Subjects censored by \f$t\f$ contribute only through the weights. The
/// integrated score treats \f$\mathrm{BS}\f$ as a left-continuous step function on the grid
/// \f$t_0<\dots<t_m\f$ and divides \f$\sum_k \mathrm{BS}(t_k)(t_{k+1}-t_k)\f$ by \f$t_m-t_0\f$. The pooled
/// Kaplan–Meier prediction gives the benchmark that `pec` calls the reference model.
/// @param observations Times and event indicators.
/// @param survival Predicted survival probabilities (rows: observations, columns: times).
/// @param times Evaluation grid, increasing and nonnegative, at least one point.
/// @return Brier scores, the Kaplan–Meier benchmark and their integrals.
/// @throws std::invalid_argument For mismatched sizes, invalid data, probabilities outside [0, 1], or a
/// time beyond the support of the censoring distribution.
/// @par Example
/// @code
/// const std::vector<dm::stats::SurvivalObservation> obs{{1, true}, {2, false}, {3, true}, {4, true}, {5, false}};
/// const auto b = dm::stats::brier_score(obs, dm::linalg::DenseMatrix<double>(5, 1, {.2, .9, .5, .6, .9}), {3.5});
/// std::println("{:.3f}", b.brier[0]); // 0.120
/// @endcode
/// @par References
/// - E. Graf, C. Schmoor, W. Sauerbrei and M. Schumacher (1999). "Assessment and comparison of
///   prognostic classification schemes for survival data." Statistics in Medicine 18(17-18),
///   2529-2545. doi:10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5
/// @see time_dependent_auc()
[[nodiscard]] BrierScore brier_score(const std::vector<SurvivalObservation>& observations,
                                     const linalg::DenseMatrix<double>& survival,
                                     const std::vector<double>& times);

} // namespace datamunge::stats
