#pragma once

/// @file rmst.hpp
/// @brief Restricted mean survival time to a fixed horizon and two-group comparisons.

#include <datamunge/stats/survival.hpp>

#include <array>
#include <vector>

namespace datamunge::stats {

/// @brief A point estimate with standard error, interval and two-sided p-value.
/// @par Example
/// @code
/// const dm::stats::RmstContrast c;
/// std::println("{}", c.p_value); // 1
/// @endcode
struct RmstContrast {
    double estimate{};       ///< Estimate on the natural scale.
    double standard_error{}; ///< Standard error (on the log scale for ratios).
    double lower{};          ///< Lower interval end.
    double upper{};          ///< Upper interval end.
    double p_value{1};       ///< Two-sided normal p-value for no difference.
};

/// @brief Restricted mean survival time and restricted mean time lost for one sample.
/// @par Example
/// @code
/// const dm::stats::RmstEstimate e;
/// std::println("{}", e.tau); // 0
/// @endcode
struct RmstEstimate {
    double tau{};              ///< Horizon.
    double rmst{};             ///< \f$\int_0^\tau\hat S(t)\,dt\f$.
    double standard_error{};   ///< Greenwood-type standard error (shared by RMST and RMTL).
    double lower{};            ///< Lower interval end for RMST.
    double upper{};            ///< Upper interval end for RMST.
    double rmtl{};             ///< Restricted mean time lost \f$\tau-\mathrm{RMST}\f$.
};

/// @brief Two-group restricted mean comparison, second group (larger label) versus first.
/// @par Example
/// @code
/// const dm::stats::RmstComparison c;
/// std::println("{}", c.groups.size()); // 2
/// @endcode
struct RmstComparison {
    std::array<int, 2> groups{};          ///< Group labels in increasing order.
    std::array<RmstEstimate, 2> arms{};   ///< Per-group estimates.
    RmstContrast difference;              ///< RMST(second) − RMST(first).
    RmstContrast ratio;                   ///< RMST(second) / RMST(first), interval from the log scale.
    RmstContrast time_lost_ratio;         ///< RMTL(second) / RMTL(first), interval from the log scale.
};

/// @brief Restricted mean survival time up to \f$\tau\f$, as survRM2's `rmst1`.
///
/// Integrates the Kaplan–Meier step function from 0 to \f$\tau\f$. The variance is
/// \f$\sum_{t_i\le\tau}A_i^2\,d_i/\{n_i(n_i-d_i)\}\f$, where \f$A_i=\int_{t_i}^\tau\hat S\f$ (terms with
/// \f$n_i=d_i\f$ are zero), and the interval is Wald on the natural scale.
/// @param observations Right-censored records with finite nonnegative times.
/// @param tau Positive horizon, at most the largest observed time so that \f$\hat S\f$ is defined on \f$[0,\tau]\f$.
/// @param confidence Interval coverage in (0, 1).
/// @return Estimate, standard error, interval and time lost.
/// @throws std::invalid_argument For empty input, invalid times, or tau outside (0, max time].
/// @par Example
/// @code
/// const auto e = dm::stats::rmst({{1, true}, {2, false}, {3, true}, {4, true}}, 3);
/// std::println("{:.2f} {:.2f}", e.rmst, e.rmtl); // 2.50 0.50
/// @endcode
/// @par References
/// - P. Royston and M. K. B. Parmar (2013). "Restricted mean survival time: an alternative to the
///   hazard ratio for the design and analysis of randomized trials with a time-to-event outcome."
///   BMC Medical Research Methodology 13, 152. doi:10.1186/1471-2288-13-152
/// @see rmst_compare(), kaplan_meier()
[[nodiscard]] RmstEstimate rmst(const std::vector<SurvivalObservation>& observations, double tau,
                                double confidence = 0.95);

/// @brief Compares restricted mean survival between two groups, as survRM2's `rmst2` without covariates.
///
/// The difference uses \f$\mathrm{se}=\sqrt{v_1+v_0}\f$. Ratios are formed on the log scale with
/// delta-method standard errors \f$\sqrt{v_1/\mu_1^2+v_0/\mu_0^2}\f$. The horizon must not exceed the
/// largest observed time in either group.
/// @param observations Right-censored records.
/// @param groups Exactly two distinct labels, one per observation.
/// @param tau Horizon.
/// @param confidence Interval coverage.
/// @return Per-group estimates and the difference, RMST ratio and RMTL ratio.
/// @throws std::invalid_argument For mismatched lengths, not exactly two groups, or an invalid tau.
/// @par Example
/// @code
/// const std::vector<dm::stats::SurvivalObservation> obs{{1, true}, {2, true}, {3, false},
///                                                      {2, true}, {4, true}, {5, false}};
/// const auto c = dm::stats::rmst_compare(obs, {0, 0, 0, 1, 1, 1}, 3);
/// std::println("{:.3f}", c.difference.estimate); // 0.667
/// @endcode
/// @par References
/// - H. Uno et al. (2014). "Moving beyond the hazard ratio in quantifying the between-group
///   difference in survival analysis." Journal of Clinical Oncology 32(22), 2380-2385.
///   doi:10.1200/JCO.2013.53.2149
/// @see rmst()
[[nodiscard]] RmstComparison rmst_compare(const std::vector<SurvivalObservation>& observations,
                                          const std::vector<int>& groups, double tau,
                                          double confidence = 0.95);

} // namespace datamunge::stats
