#pragma once

/// @file generalizability.hpp
/// @brief Generalizability theory for balanced, fully crossed random designs: G-study variance components
/// and D-study reliability coefficients.

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief One variance component of a crossed design (a main effect or interaction).
/// @par Example
/// @code
/// dm::stats::VarianceComponent c{.factors = {0, 1}};
/// std::println("{}", c.factors.size()); // 2
/// @endcode
struct VarianceComponent {
    std::vector<std::size_t> factors; ///< Factors of the effect (0 is the object of measurement, e.g. persons).
    double df{};                      ///< Degrees of freedom, the product of (n_f - 1).
    double sum_of_squares{};          ///< Sum of squares.
    double mean_square{};             ///< Mean square.
    double raw_variance{};            ///< Unconstrained ANOVA (EMS) estimate, possibly negative.
    double variance{};                ///< Estimate truncated at 0 (used in D-studies).
    double standard_error{};          ///< Brennan's (2001) estimated standard error of the raw estimate.
    double percent{};                 ///< Share of the total of truncated variances, in percent.
};

/// @brief Result of g_study().
/// @par Example
/// @code
/// dm::stats::GStudy g;
/// std::println("{}", g.components.size()); // 0
/// @endcode
struct GStudy {
    std::vector<std::size_t> levels;           ///< Levels of each factor (factor 0 is the object of measurement).
    double grand_mean{};                       ///< Mean of all scores.
    std::vector<VarianceComponent> components; ///< All effects, ordered by size (number of factors) then index.
};

/// @brief G-study: estimates variance components of a balanced, fully crossed random-effects design with
/// one observation per cell.
///
/// Mean squares come from the usual ANOVA decomposition; the components solve the expected mean square
/// equations \f$E[MS_S]=\sum_{T\supseteq S}\sigma^2_T\prod_{f\notin T}n_f\f$ from the highest-order interaction
/// down, using the raw (possibly negative) estimates in the substitution and truncating at zero afterwards
/// (Brennan 2001). The highest-order interaction is confounded with residual error. For balanced data with
/// positive estimates these equal REML (`lme4::lmer`) estimates.
/// @param scores Scores in row-major order over the factors (factor 0, e.g. persons, varies slowest).
/// @param levels Number of levels of each factor (at least two factors, each with at least two levels).
/// @return The components.
/// @throws std::invalid_argument For fewer than two factors, a factor with fewer than two levels, a size
/// mismatch or nonfinite scores.
/// @par Example
/// @code
/// const auto g = dm::stats::g_study({1, 2, 3, 3, 5, 6}, {3, 2}); // 3 persons x 2 raters
/// std::println("{:.3f}", g.components[0].variance); // 4.000
/// @endcode
GStudy g_study(const std::vector<double>& scores, const std::vector<std::size_t>& levels);

/// @brief Projected error variances and reliability for a measurement procedure (a D-study).
/// @par Example
/// @code
/// dm::stats::DStudy d{.universe_variance = 1, .relative_error = 0.25};
/// std::println("{}", d.universe_variance / (d.universe_variance + d.relative_error)); // 0.8
/// @endcode
struct DStudy {
    std::vector<double> facet_sizes; ///< Numbers of conditions of facets 1..m-1 averaged over.
    double universe_variance{};      ///< \f$\sigma^2_p\f$, universe-score variance.
    double relative_error{};         ///< \f$\sigma^2_\delta\f$: interactions of persons with the facets.
    double absolute_error{};         ///< \f$\sigma^2_\Delta\f$: also includes facet main effects and their interactions.
    double generalizability{};       ///< \f$E\rho^2=\sigma^2_p/(\sigma^2_p+\sigma^2_\delta)\f$ for relative decisions.
    double dependability{};          ///< \f$\Phi=\sigma^2_p/(\sigma^2_p+\sigma^2_\Delta)\f$ for absolute decisions.
    double relative_sem{};           ///< sqrt(relative_error).
    double absolute_sem{};           ///< sqrt(absolute_error).
};

/// @brief D-study: reliability of the mean over a chosen number of conditions of each facet.
///
/// Each component involving persons and facets T is divided by \f$\prod_{f\in T}n'_f\f$: averaging over more
/// raters or occasions shrinks its contribution to error. With the G-study sizes in a one-facet design,
/// \f$E\rho^2\f$ equals Cronbach's alpha.
/// @param g A G-study.
/// @param facet_sizes n'_f for facets 1..m-1 (positive; non-integers allowed for planning).
/// @return Error variances, coefficients and SEMs.
/// @throws std::invalid_argument If the number of sizes differs from m - 1 or a size is not positive.
/// @par Example
/// @code
/// const auto g = dm::stats::g_study({1, 2, 3, 3, 5, 6}, {3, 2});
/// std::println("{:.3f}", dm::stats::d_study(g, {2}).generalizability); // 0.980
/// @endcode
DStudy d_study(const GStudy& g, const std::vector<double>& facet_sizes);

} // namespace datamunge::stats
