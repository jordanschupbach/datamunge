#pragma once

/// @file gage_rr.hpp
/// @brief Gage repeatability and reproducibility (ANOVA method) for balanced crossed parts-by-operators studies.

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Options for gage_rr().
/// @par Example
/// @code
/// dm::stats::GageOptions o{.tolerance = 2};
/// std::println("{}", o.study_multiplier); // 6
/// @endcode
struct GageOptions {
    double tolerance{std::numeric_limits<double>::quiet_NaN()}; ///< USL - LSL for %tolerance (NaN: not reported).
    double study_multiplier{6};    ///< Study variation = multiplier x SD (6, or 5.15 in older AIAG editions).
    double pool_interaction_above{0.25}; ///< Pool the interaction when its p-value exceeds this (AIAG 0.25; 1 never pools).
};

/// @brief One line of the ANOVA table.
/// @par Example
/// @code
/// dm::stats::GageAnovaRow r{.source = "Part", .df = 9};
/// std::println("{} {}", r.source, r.df); // Part 9
/// @endcode
struct GageAnovaRow {
    std::string source;      ///< "Part", "Operator", "Part x Operator", "Repeatability" (or "Pooled error").
    double df{};             ///< Degrees of freedom.
    double sum_of_squares{}; ///< Sum of squares.
    double mean_square{};    ///< Mean square.
    double f{};              ///< F statistic (NaN for the error line).
    double p_value{};        ///< p-value (NaN for the error line).
};

/// @brief One variance component with its share of the variation.
/// @par Example
/// @code
/// dm::stats::GageComponent c{.source = "Repeatability", .variance = 4};
/// std::println("{}", c.variance); // 4
/// @endcode
struct GageComponent {
    std::string source;              ///< Name of the component.
    double variance{};               ///< Variance component (negative estimates set to 0).
    double percent_contribution{};   ///< 100 x variance / total variance.
    double sd{};                     ///< sqrt(variance).
    double study_variation{};        ///< multiplier x sd.
    double percent_study_variation{}; ///< 100 x sd / total sd.
    double percent_tolerance{};      ///< 100 x study variation / tolerance (NaN without a tolerance).
};

/// @brief Result of gage_rr().
/// @par Example
/// @code
/// dm::stats::GageStudy g;
/// std::println("{}", g.components.size()); // 0
/// @endcode
struct GageStudy {
    std::size_t parts{};          ///< Number of parts.
    std::size_t operators{};      ///< Number of operators (appraisers).
    std::size_t replicates{};     ///< Repeated measurements per part and operator.
    bool interaction_pooled{};    ///< Whether the interaction was pooled into repeatability.
    std::vector<GageAnovaRow> anova;        ///< ANOVA table.
    std::vector<GageComponent> components; ///< Total Gage R&R, Repeatability, Reproducibility, Operator, Operator x Part, Part-to-Part, Total Variation.
    double distinct_categories{};  ///< floor(1.41 x part SD / gage SD), the AIAG number of distinct categories.
};

/// @brief ANOVA gage R&R study of a balanced, fully crossed design.
///
/// The two-way random-effects model \f$y_{ijk}=\mu+P_i+O_j+(PO)_{ij}+e_{ijk}\f$ is fitted by expected mean
/// squares: \f$\sigma^2_e=MS_E\f$, \f$\sigma^2_{PO}=(MS_{PO}-MS_E)/r\f$, \f$\sigma^2_O=(MS_O-MS_{PO})/(pr)\f$ and
/// \f$\sigma^2_P=(MS_P-MS_{PO})/(or)\f$, each truncated at 0. Parts and operators are tested against the
/// interaction. When the interaction's p-value exceeds `pool_interaction_above` it is pooled with error and
/// the additive model is used (AIAG, Minitab). Gage R&R = repeatability + reproducibility (operator +
/// interaction).
/// @param values Measurements ordered by part, then operator, then replicate.
/// @param parts Number of parts (at least 2).
/// @param operators Number of operators (at least 2).
/// @param replicates Replicates per cell (at least 2).
/// @param options Options.
/// @return The study.
/// @throws std::invalid_argument For a size mismatch, too few levels, nonfinite values, a nonpositive
/// tolerance or multiplier, or no variation at all.
/// @par Example
/// @code
/// const auto g = dm::stats::gage_rr({1, 1.2, 1.1, 1.3, 2, 2.1, 2.2, 2.2}, 2, 2, 2);
/// std::println("{}", g.components[0].source); // Total Gage R&R
/// @endcode
GageStudy gage_rr(const std::vector<double>& values, std::size_t parts, std::size_t operators,
                  std::size_t replicates, const GageOptions& options = {});

} // namespace datamunge::stats
