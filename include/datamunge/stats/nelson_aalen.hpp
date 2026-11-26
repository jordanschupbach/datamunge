#pragma once

/// @file nelson_aalen.hpp
/// @brief Nelson–Aalen cumulative hazard with variance and confidence intervals, allowing delayed entry.

#include <datamunge/stats/survival.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Handling of tied event times in the hazard increment, as R's `survfit` `ctype`.
/// @par Example
/// @code
/// const auto t = dm::stats::HazardTies::fleming_harrington;
/// std::println("{}", t == dm::stats::HazardTies::nelson_aalen); // false
/// @endcode
enum class HazardTies {
    nelson_aalen,       ///< \f$d/n\f$ with variance \f$d/n^2\f$ (`ctype = 1`).
    fleming_harrington, ///< \f$\sum_{j=0}^{d-1}1/(n-j)\f$ with variance \f$\sum_j1/(n-j)^2\f$ (`ctype = 2`).
};

/// @brief Scale on which pointwise intervals for the cumulative hazard are formed.
/// @par Example
/// @code
/// const auto s = dm::stats::HazardInterval::log;
/// std::println("{}", s == dm::stats::HazardInterval::plain); // false
/// @endcode
enum class HazardInterval {
    log,   ///< \f$H\exp(\pm z\,\mathrm{se}/H)\f$; stays positive and is more accurate in small risk sets.
    plain, ///< \f$H\pm z\,\mathrm{se}\f$, lower end truncated at 0 (R's log-survival interval for \f$e^{-H}\f$).
};

/// @brief Options for nelson_aalen().
/// @par Example
/// @code
/// const dm::stats::NelsonAalenOptions options{.confidence = .9};
/// std::println("{}", options.entry.empty()); // true
/// @endcode
struct NelsonAalenOptions {
    HazardTies ties{HazardTies::nelson_aalen};      ///< Tie handling.
    HazardInterval interval{HazardInterval::log};   ///< Interval scale.
    double confidence{0.95};                        ///< Interval coverage.
    std::vector<double> entry{};                    ///< Delayed-entry times (empty: all enter at 0); at risk on (entry, time].
};

/// @brief Cumulative hazard at one distinct event or censoring time.
/// @par Example
/// @code
/// const dm::stats::NelsonAalenPoint p{};
/// std::println("{}", p.events); // 0
/// @endcode
struct NelsonAalenPoint {
    double time{};               ///< Follow-up time.
    std::size_t at_risk{};       ///< Number at risk just before the time.
    std::size_t events{};        ///< Events at the time.
    std::size_t censored{};      ///< Censorings at the time.
    double increment{};          ///< Hazard increment at the time.
    double cumulative_hazard{};  ///< \f$\hat H(t)\f$.
    double standard_error{};     ///< Square root of the accumulated variance.
    double lower{};              ///< Lower interval end.
    double upper{};              ///< Upper interval end.
    double survival{};           ///< Breslow–Fleming–Harrington survival \f$\exp(-\hat H(t))\f$.
};

/// @brief Nelson–Aalen estimator of the cumulative hazard, as R's `survfit(...)$cumhaz` and `$std.chaz`.
///
/// \f$\hat H(t)=\sum_{t_i\le t}d_i/n_i\f$ over distinct times, where \f$n_i\f$ counts subjects with
/// entry \f$<t_i\le\f$ exit. Censorings at an event time stay in that risk set. Points are reported
/// at every distinct exit time. Intervals are pointwise.
/// @param observations Exit times and event indicators (finite, nonnegative times).
/// @param options Ties, interval type, coverage and optional entry times.
/// @return Points in increasing time.
/// @throws std::invalid_argument For empty input, invalid times, entry times not below exit times,
/// a mismatched entry vector, or confidence outside (0, 1).
/// @par Example
/// @code
/// const auto na = dm::stats::nelson_aalen({{1, true}, {2, true}, {2, false}, {3, true}});
/// std::println("{:.4f}", na.back().cumulative_hazard); // 1.5833
/// @endcode
/// @par References
/// - O. Aalen (1978). "Nonparametric inference for a family of counting processes." Annals of
///   Statistics 6(4), 701-726. doi:10.1214/aos/1176344247
/// - J. P. Klein (1991). "Small sample moments of some estimators of the variance of the
///   Kaplan-Meier and Nelson-Aalen estimators." Scandinavian Journal of Statistics 18(4), 333-340.
/// @see kaplan_meier()
[[nodiscard]] std::vector<NelsonAalenPoint> nelson_aalen(const std::vector<SurvivalObservation>& observations,
                                                         const NelsonAalenOptions& options = {});

} // namespace datamunge::stats
