#pragma once

/// @file dunnett.hpp
/// @brief Dunnett's many-to-one comparisons of treatments with a shared control.

#include <datamunge/stats/hypothesis_test_result.hpp>
#include <datamunge/stats/mvn_probability.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Controls for dunnett_test().
/// @par Example
/// @code
/// const dm::stats::DunnettOptions options{.control = 2};
/// std::println("{}", options.control); // 2
/// @endcode
struct DunnettOptions {
    std::size_t control{0};                      ///< Index of the control group.
    Alternative alternative{Alternative::TwoSided}; ///< Direction of each treatment-minus-control comparison.
    double conf_level{.95};                      ///< Simultaneous coverage of the intervals in (0, 1).

    /// Integration controls for the multivariate t probabilities; the seed is fixed so
    /// that p-values and the critical value are reproducible.
    MvnProbabilityOptions integration{.abs_tol = 1e-5, .max_points = 2000000};
};

/// @brief One treatment-minus-control comparison.
/// @par Example
/// @code
/// const dm::stats::DunnettComparison comparison;
/// std::println("{}", comparison.treatment); // 0
/// @endcode
struct DunnettComparison {
    std::size_t treatment{};  ///< Index of the treatment group.
    double estimate{};        ///< Treatment mean minus control mean.
    double std_error{};       ///< \f$s\sqrt{1/n_j+1/n_0}\f$ with the pooled standard deviation s.
    double statistic{};       ///< \f$t_j=\f$ estimate / std_error.
    double p_value{};         ///< Simultaneous (single-step adjusted) p-value.
    double conf_low{};        ///< Lower simultaneous confidence limit (negative infinity for `Less`).
    double conf_high{};       ///< Upper simultaneous confidence limit (positive infinity for `Greater`).
};

/// @brief Dunnett comparisons with their shared critical value and error variance.
/// @par Example
/// @code
/// const dm::stats::DunnettResult result;
/// std::println("{}", result.comparisons.size()); // 0
/// @endcode
struct DunnettResult {
    std::vector<DunnettComparison> comparisons; ///< One entry per treatment, in group order.
    double critical_value{};  ///< Quantile c of the maximum (absolute) statistic for the intervals.
    double df{};              ///< Error degrees of freedom \f$N-k\f$.
    double pooled_sd{};       ///< Pooled within-group standard deviation s.

    /// Largest integration error estimate among the p-values and the critical-value search.
    double integration_error{};
};

/// @brief Single-step Dunnett comparisons of each treatment with a shared control.
///
/// Assumes independent normal groups with a common variance. The statistics
/// \f$t_j\f$ share the pooled variance and the control mean, so jointly they follow a
/// central multivariate t law with \f$N-k\f$ degrees of freedom and correlations
/// \f$\rho_{jl}=\sqrt{n_jn_l/((n_j+n_0)(n_l+n_0))}\f$. Adjusted p-values are
/// \f$1-P(\max_l|T_l|\le|t_j|)\f$ for two-sided tests and \f$1-P(\max_lT_l\le t_j)\f$ (or the
/// mirror image) for one-sided tests, computed by mvt_rectangle_probability(). The
/// critical value solves the same probability equal to `conf_level` by bisection. This
/// matches `glht(..., mcp(group = "Dunnett"))` in the R multcomp package up to integration error.
/// @param groups At least two groups of finite observations, each nonempty, with \f$N-k\ge1\f$.
/// @param options Control index, alternative, level and integration controls.
/// @return Comparisons, critical value, degrees of freedom and pooled standard deviation.
/// @throws std::invalid_argument For fewer than two groups, an empty group, nonfinite data, no error
/// degrees of freedom, zero pooled variance, an invalid control index or alternative, or a level outside (0, 1).
/// @par Example
/// @code
/// const auto d = dm::stats::dunnett_test({{10, 11, 9, 10}, {12, 13, 12, 14}, {10, 10, 11, 9}});
/// std::println("{:.3f} {:.4f}", d.comparisons[0].statistic, d.comparisons[0].p_value); // 4.491 0.0028
/// @endcode
/// @par References
/// - C. W. Dunnett (1955). "A multiple comparison procedure for comparing several treatments with
///   a control". JASA 50(272), 1096-1121. doi:10.1080/01621459.1955.10501294
/// - T. Hothorn, F. Bretz and P. Westfall (2008). "Simultaneous inference in general parametric
///   models". Biometrical Journal 50(3), 346-363. doi:10.1002/bimj.200810425
/// - https://cran.r-project.org/package=multcomp
/// @see mvt_rectangle_probability()
[[nodiscard]] DunnettResult dunnett_test(const std::vector<std::vector<double>>& groups,
                                         const DunnettOptions& options = {});

} // namespace datamunge::stats
