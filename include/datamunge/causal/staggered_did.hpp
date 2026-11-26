#pragma once

/// @file staggered_did.hpp
/// @brief Difference in differences with staggered treatment adoption (Callaway and Sant'Anna).

#include <cstddef>
#include <vector>

namespace datamunge::causal {

/// @brief Units used as comparisons for a treated cohort.
enum class DidControlGroup {
    never_treated,   ///< Units never treated during the panel.
    not_yet_treated, ///< Never-treated units and units first treated after both compared periods (plus anticipation).
};

/// @brief Base period of pre-treatment group-time effects.
enum class DidBasePeriod {
    varying,   ///< Pre-period effects compare consecutive periods (t - 1 to t); post effects use g - 1.
    universal, ///< Every effect uses g - 1 - anticipation as the base, so event time -1 is zero by construction.
};

/// @brief Options for staggered_did().
/// @par Example
/// @code
/// const dm::causal::StaggeredDidOptions o{.anticipation = 1};
/// std::println("{}", o.anticipation); // 1
/// @endcode
struct StaggeredDidOptions {
    DidControlGroup control_group{DidControlGroup::never_treated}; ///< Comparison units.
    DidBasePeriod base_period{DidBasePeriod::varying};             ///< Base-period convention.
    std::size_t anticipation{0}; ///< Periods before adoption in which effects may already occur.
    double confidence{0.95};     ///< Level of the pointwise normal confidence intervals.
};

/// @brief Average effect for one adoption cohort in one period, ATT(g, t).
/// @par Example
/// @code
/// const dm::causal::GroupTimeEffect e{.group = 2004, .time = 2006, .att = -0.1};
/// std::println("{}", e.time - e.group); // 2
/// @endcode
struct GroupTimeEffect {
    double group{};          ///< First treated period of the cohort.
    double time{};           ///< Period of the effect.
    double base{};           ///< Base period compared against.
    double att{};            ///< Effect estimate (0 at the universal base period).
    double standard_error{}; ///< Influence-function standard error (NaN at the universal base period).
    double lower{};          ///< Lower pointwise confidence limit.
    double upper{};          ///< Upper pointwise confidence limit.
    std::size_t treated{};   ///< Units in the cohort.
    std::size_t controls{};  ///< Comparison units.
};

/// @brief An aggregated effect (event time, cohort or overall).
/// @par Example
/// @code
/// const dm::causal::AggregatedEffect e{.key = 0, .att = -0.02};
/// std::println("{}", e.key); // 0
/// @endcode
struct AggregatedEffect {
    double key{};            ///< Event time e = t - g, or cohort g, or 0 for an overall summary.
    double att{};            ///< Weighted average of group-time effects.
    double standard_error{}; ///< Standard error including the estimation of the cohort weights.
    double lower{};          ///< Lower pointwise confidence limit.
    double upper{};          ///< Upper pointwise confidence limit.
};

/// @brief Group-time effects and their aggregations.
/// @par Example
/// @code
/// const dm::causal::StaggeredDid r{.units = 500};
/// std::println("{}", r.units); // 500
/// @endcode
struct StaggeredDid {
    std::vector<GroupTimeEffect> group_time;  ///< ATT(g, t) for every cohort and estimable period.
    std::vector<AggregatedEffect> event_study; ///< Dynamic effects by event time t - g (pre and post).
    AggregatedEffect dynamic_overall;          ///< Average of event-study effects over e >= 0.
    std::vector<AggregatedEffect> by_group;    ///< Average post-treatment effect of each cohort.
    AggregatedEffect group_overall;            ///< Cohort averages weighted by cohort size.
    AggregatedEffect simple;                   ///< Post-treatment ATT(g, t) weighted by cohort size.
    std::size_t units{};                       ///< Number of units in the balanced panel.
};

/// @brief Staggered-adoption difference in differences without covariates.
///
/// For each adoption cohort g and period t, estimates
/// \f$ATT(g,t)=E[Y_t-Y_b\mid G=g]-E[Y_t-Y_b\mid C]\f$, with base period b and comparison units C
/// that are never treated or not yet treated. Comparing every cohort only with clean controls avoids
/// the negative weights and forbidden comparisons of two-way fixed-effects regressions under
/// heterogeneous effects. Event-study, cohort and simple aggregations weight the ATT(g, t) by cohort
/// size; their standard errors include the uncertainty in those weights. Inference uses each unit's
/// influence function (the analytical, non-bootstrap version of R did).
/// @param unit Unit identifier of each observation.
/// @param period Time period of each observation (numeric, equally meaningful across units).
/// @param cohort First treated period of the unit, constant within unit; 0 for never treated.
/// @param outcome Finite outcome of each observation.
/// @param options Control group, base period, anticipation and confidence level.
/// @return Group-time effects and aggregations.
/// @throws std::invalid_argument For an unbalanced panel, duplicate unit-periods, a cohort that varies
/// within a unit, non-finite values, no comparison units or an invalid confidence level.
/// @par Example
/// @code
/// const auto r = dm::causal::staggered_did({1, 1, 1, 2, 2, 2, 3, 3, 3}, {1, 2, 3, 1, 2, 3, 1, 2, 3},
///                                          {3, 3, 3, 0, 0, 0, 0, 0, 0}, {1, 2, 5, 1, 2, 3, 0, 1, 2});
/// std::println("{}", r.group_time.back().att); // 2
/// @endcode
/// @par References
/// - B. Callaway and P. H. C. Sant'Anna (2021). Difference-in-differences with multiple time periods.
///   Journal of Econometrics 225(2), 200-230. doi:10.1016/j.jeconom.2020.12.001
/// - A. Goodman-Bacon (2021). Difference-in-differences with variation in treatment timing. Journal of
///   Econometrics 225(2), 254-277. doi:10.1016/j.jeconom.2021.03.014
/// @see difference_in_differences()
[[nodiscard]] StaggeredDid staggered_did(const std::vector<double>& unit, const std::vector<double>& period,
                                         const std::vector<double>& cohort, const std::vector<double>& outcome,
                                         const StaggeredDidOptions& options = {});

} // namespace datamunge::causal
