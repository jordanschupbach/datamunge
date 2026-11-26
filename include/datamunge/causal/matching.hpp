#pragma once

/// @file matching.hpp
/// @brief Propensity score matching, covariate balance diagnostics and matched effect estimates.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace datamunge::causal {

/// @brief Order in which treated units choose their matches.
enum class MatchOrder {
    largest,  ///< Descending propensity (MatchIt default for propensity scores): hardest-to-match units first.
    smallest, ///< Ascending propensity.
    data,     ///< Row order of the data.
};

/// @brief Options for nearest-neighbor propensity score matching.
/// @par Example
/// @code
/// const dm::causal::PropensityMatchOptions o{.ratio = 2, .caliper = 0.2};
/// std::println("{} {}", o.ratio, *o.caliper); // 2 0.2
/// @endcode
struct PropensityMatchOptions {
    std::size_t ratio{1};          ///< Controls matched to each treated unit (k:1 matching).
    bool replace{false};           ///< Whether a control may be reused for several treated units.
    std::optional<double> caliper; ///< Maximum distance in standard deviations of the score (all units); none: no caliper.
    bool logit_distance{false};    ///< Match on the logit of the propensity instead of the probability.
    MatchOrder order{MatchOrder::largest}; ///< Matching order of treated units (irrelevant with replacement).
};

/// @brief Balance of each covariate between treated and control units.
///
/// Statistics follow MatchIt's summary for the ATT: standardized mean differences divide by the
/// standard deviation of the treated units in the full sample (\f$\sqrt{p(1-p)}\f$ for 0/1
/// covariates), so before and after values share a denominator.
/// @par Example
/// @code
/// const dm::causal::CovariateBalance b{.standardized_difference = {0.05}};
/// std::println("{}", b.standardized_difference[0]); // 0.05
/// @endcode
struct CovariateBalance {
    std::vector<double> treated_mean;            ///< Weighted treated mean of each covariate.
    std::vector<double> control_mean;            ///< Weighted control mean of each covariate.
    std::vector<double> standardized_difference; ///< (treated - control) mean over the full-sample treated SD.
    std::vector<double> variance_ratio;          ///< Treated over control weighted variance (NaN for 0/1 covariates).
    std::vector<double> ecdf_max;                ///< Largest gap between the weighted empirical CDFs (Kolmogorov-Smirnov distance).
};

/// @brief Result of propensity score matching.
/// @par Example
/// @code
/// const dm::causal::PropensityMatch m{.matched_treated = 3, .matched_controls = 3};
/// std::println("{}", m.matched_treated + m.matched_controls); // 6
/// @endcode
struct PropensityMatch {
    std::vector<double> propensity;                    ///< Estimated (or supplied) treatment probabilities.
    std::vector<std::vector<std::size_t>> matches;     ///< Matched control rows of each treated row, in matching order (empty for controls and unmatched units).
    std::vector<double> weights;                       ///< MatchIt ATT weights: 1 for matched treated, scaled 1/k shares for controls, 0 if unmatched.
    std::vector<std::size_t> subclass;                 ///< Matched set of each unit (treated row index) without replacement; npos when unmatched or with replacement.
    std::size_t matched_treated{};                     ///< Treated units with at least one match.
    std::size_t matched_controls{};                    ///< Distinct controls used.
    double caliper_width{};                            ///< Caliper on the distance scale (infinite without a caliper).
    CovariateBalance before;                           ///< Balance in the unmatched sample.
    CovariateBalance after;                            ///< Balance in the matched, weighted sample.
};

/// @brief Matched estimate of the average treatment effect on the treated.
/// @par Example
/// @code
/// const dm::causal::MatchedEffect e{.estimate = 1.5, .standard_error = 0.5};
/// std::println("{}", e.estimate / e.standard_error); // 3
/// @endcode
struct MatchedEffect {
    double estimate{};       ///< Weighted treated-minus-control mean in the matched sample.
    double standard_error{}; ///< Cluster-robust (by matched set) or, with replacement, HC1 standard error.
    double df{};             ///< Residual degrees of freedom (matched units minus 2).
    double p_value{};        ///< Two-sided t p-value.
    double lower{};          ///< Lower confidence limit.
    double upper{};          ///< Upper confidence limit.
};

/// @brief Fits a logistic propensity model of treatment on the covariates.
/// @param treatment Indicators (0 or 1) with both groups represented.
/// @param covariates Finite covariates in columns, one row per unit.
/// @return Fitted probabilities of treatment.
/// @throws std::invalid_argument For mismatched sizes, invalid indicators or non-finite covariates.
/// @throws std::runtime_error When the logistic fit does not converge or separates the groups.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6, 1, {1, 2, 3, 4, 5, 6});
/// const auto p = dm::causal::propensity_scores({0, 1, 0, 1, 0, 1}, x);
/// std::println("{}", p[0] < p[5]); // true
/// @endcode
/// @see propensity_score_match()
[[nodiscard]] std::vector<double> propensity_scores(const std::vector<int>& treatment,
                                                    const linalg::DenseMatrix<double>& covariates);

/// @brief Weighted balance statistics of covariates between treated and control units.
/// @param treatment Indicators (0 or 1) with both groups represented.
/// @param covariates Finite covariates in columns.
/// @param weights Nonnegative unit weights (empty: all ones). Units with zero weight are excluded.
/// @param reference_sd Denominators of the standardized differences (empty: computed from the
/// treated units with unit weights).
/// @return Means, standardized differences, variance ratios and empirical-CDF distances.
/// @throws std::invalid_argument For mismatched sizes, negative weights or a group with zero total weight.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4, 1, {1, 3, 1, 3});
/// const auto b = dm::causal::covariate_balance({1, 1, 0, 0}, x);
/// std::println("{}", b.standardized_difference[0]); // 0
/// @endcode
/// @see propensity_score_match()
[[nodiscard]] CovariateBalance covariate_balance(const std::vector<int>& treatment,
                                                 const linalg::DenseMatrix<double>& covariates,
                                                 const std::vector<double>& weights = {},
                                                 const std::vector<double>& reference_sd = {});

/// @brief Greedy nearest-neighbor matching on supplied propensity scores.
///
/// Treated units are processed in the chosen order. Without replacement, matching proceeds in
/// `ratio` rounds: in each round every treated unit, in order, takes the closest unused control
/// within the caliper (exact ties are broken by score side and row as in MatchIt). With replacement each treated unit takes its
/// `ratio` closest controls within the caliper. Treated units without any match are discarded,
/// which changes the estimand from the ATT to the effect in the matched treated units.
/// @param treatment Indicators (0 or 1) with both groups represented.
/// @param propensity Treatment probabilities strictly between 0 and 1.
/// @param covariates Covariates for the balance report (may have zero columns).
/// @param options Ratio, replacement, caliper, distance scale and order.
/// @return Matches, weights, matched sets and balance before and after matching.
/// @throws std::invalid_argument For invalid inputs, a zero ratio, a nonpositive caliper or when no
/// treated unit can be matched. Without replacement, treated units left over when controls run out stay unmatched.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4, 0);
/// const auto m = dm::causal::match_on_propensity({1, 0, 0, 1}, {0.6, 0.55, 0.2, 0.3}, x);
/// std::println("{} {}", m.matches[0][0], m.matches[3][0]); // 1 2
/// @endcode
/// @par References
/// - D. E. Ho, K. Imai, G. King and E. A. Stuart (2011). MatchIt: Nonparametric preprocessing for
///   parametric causal inference. Journal of Statistical Software 42(8). doi:10.18637/jss.v042.i08
/// - E. A. Stuart (2010). Matching methods for causal inference: A review and a look forward.
///   Statistical Science 25(1), 1-21. doi:10.1214/09-STS313
/// @see propensity_score_match(), matched_effect()
[[nodiscard]] PropensityMatch match_on_propensity(const std::vector<int>& treatment,
                                                  const std::vector<double>& propensity,
                                                  const linalg::DenseMatrix<double>& covariates,
                                                  const PropensityMatchOptions& options = {});

/// @brief Estimates logistic propensity scores and matches on them.
/// @param treatment Indicators (0 or 1) with both groups represented.
/// @param covariates Finite covariates used for the propensity model and the balance report.
/// @param options Ratio, replacement, caliper, distance scale and order.
/// @return As match_on_propensity().
/// @throws std::invalid_argument As match_on_propensity().
/// @throws std::runtime_error As propensity_scores().
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6, 1, {1, 2, 3, 4, 5, 6});
/// const auto m = dm::causal::propensity_score_match({0, 1, 0, 0, 1, 0}, x);
/// std::println("{}", m.matched_controls); // 2
/// @endcode
/// @see match_on_propensity(), propensity_scores()
[[nodiscard]] PropensityMatch propensity_score_match(const std::vector<int>& treatment,
                                                     const linalg::DenseMatrix<double>& covariates,
                                                     const PropensityMatchOptions& options = {});

/// @brief Effect on the matched treated units with a matched-set cluster-robust standard error.
///
/// Fits the weighted regression of outcome on an intercept and treatment over units with positive
/// weight, as MatchIt recommends. Without replacement the standard error is clustered by matched set
/// (sandwich vcovCL, HC1 with the G/(G-1) adjustment); with replacement it is HC1 robust (vcovHC).
/// @param outcome Finite outcomes for every unit.
/// @param treatment Indicators (0 or 1).
/// @param match Result of matching on the same units.
/// @param confidence Confidence level of the t interval.
/// @return Estimate, standard error, df, p-value and interval.
/// @throws std::invalid_argument For mismatched sizes, non-finite outcomes, a level outside (0, 1) or too few matched units.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4, 0);
/// const auto m = dm::causal::match_on_propensity({1, 0, 0, 1}, {0.6, 0.55, 0.2, 0.3}, x);
/// const auto e = dm::causal::matched_effect({5, 2, 1, 4}, {1, 0, 0, 1}, m);
/// std::println("{}", e.estimate); // 3
/// @endcode
/// @see match_on_propensity()
[[nodiscard]] MatchedEffect matched_effect(const std::vector<double>& outcome, const std::vector<int>& treatment,
                                           const PropensityMatch& match, double confidence = 0.95);

} // namespace datamunge::causal
