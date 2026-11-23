#pragma once

/// @file repeated_measures_anova.hpp
/// @brief Univariate repeated-measures tests with sphericity corrections.

#include <datamunge/linalg/dense_matrix.hpp>

#include <optional>

namespace datamunge::stats {

/// @brief Reference distribution for a repeated-measures F statistic.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> y(4,2,{2,3,1,4,3,5,2,4});
/// const auto result = dm::stats::repeated_measures_anova(y);
/// std::println("{:.1f}", result.uncorrected.df_denominator); // 3.0
/// @endcode
struct RepeatedMeasuresFTest {
    double epsilon;        ///< Multiplier applied to both uncorrected degrees of freedom.
    double df_numerator;   ///< Numerator degrees of freedom after scaling.
    double df_denominator; ///< Denominator degrees of freedom after scaling.
    double p_value;        ///< Upper tail of the corresponding F distribution.
};

/// @brief Within-subject F statistic and its sphericity corrections.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> y(4,2,{2,3,1,4,3,5,2,4});
/// const auto result = dm::stats::repeated_measures_anova(y);
/// std::println("{:.1f}", result.f_statistic); // 24.0
/// @endcode
struct RepeatedMeasuresANOVAResult {
    double f_statistic;                     ///< Common F statistic before and after correcting degrees of freedom.
    RepeatedMeasuresFTest uncorrected;       ///< Reference F test assuming sphericity.
    RepeatedMeasuresFTest greenhouse_geisser; ///< Greenhouse–Geisser correction.
    std::optional<RepeatedMeasuresFTest> huynh_feldt; ///< Huynh–Feldt correction, absent for a multi-contrast effect with only one residual degree of freedom per contrast.
};

/// @brief Tests equality of the means at all visits in a one-factor repeated-measures design.
/// @param response Complete finite outcomes with one subject per row and one visit per column.
/// @return The time-effect F statistic and the uncorrected and corrected tests.
/// @throws std::invalid_argument Fewer than two subjects or visits, nonfinite data, or zero residual contrast variation.
/// @throws std::length_error A dense workspace exceeds its storage capacity.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> y(4,2,{2,3,1,4,3,5,2,4});
/// const auto result = dm::stats::repeated_measures_anova(y);
/// std::println("{:.1f}", result.greenhouse_geisser.epsilon); // 1.0
/// @endcode
[[nodiscard]] RepeatedMeasuresANOVAResult repeated_measures_anova(const linalg::DenseMatrix<double>& response);

/// @brief Tests a within-subject contrast space using nested between-subject models.
///
/// For each orthonormal within-subject contrast, compares the fitted values of
/// the full and reduced subject-level models. The trace sums of squares give
/// the common F statistic. The full model residual covariance defines the
/// Greenhouse–Geisser and Huynh–Feldt (with the residual-DF correction) epsilons.
/// Results depend on the span of the contrasts. Subjects must be independent
/// and share the same covariance on the tested contrast space.
/// @param response Complete finite outcomes with one subject per row and one condition per column.
/// @param full_subject_design Full-rank subject-level design containing the tested effect.
/// @param reduced_subject_design Full-rank nested design omitting the effect. Supply a matrix with zero columns for the null model.
/// @param within_contrasts Full-row-rank matrix of within-subject contrasts, one row per contrast, with zero row sums.
/// @return The within-subject F statistic and the uncorrected and corrected tests.
/// @throws std::invalid_argument Incompatible, nonfinite or rank-deficient data, non-nested designs, or zero residual contrast variation.
/// @throws std::length_error A dense workspace exceeds its storage capacity.
/// @throws std::runtime_error A required factorization failed to converge.
/// @par Example
/// @code
/// using dm::linalg::DenseMatrix;
/// const DenseMatrix<double> y(4,2,{2,3,1,4,3,5,2,4});
/// const auto result = dm::stats::repeated_measures_anova(y, DenseMatrix<double>(4,1,1), DenseMatrix<double>(4,0),
///     DenseMatrix<double>(1,2,{-1,1}));
/// std::println("{:.1f}", result.huynh_feldt->df_numerator); // 1.0
/// @endcode
/// @par References
/// - Greenhouse and Geisser (1959), https://doi.org/10.1007/BF02289823
/// - Huynh and Feldt (1976), https://doi.org/10.3102/10769986001001069
/// - Lecoutre (1991), https://doi.org/10.3102/10769986016004371
[[nodiscard]] RepeatedMeasuresANOVAResult repeated_measures_anova(const linalg::DenseMatrix<double>& response,
    const linalg::DenseMatrix<double>& full_subject_design, const linalg::DenseMatrix<double>& reduced_subject_design,
    const linalg::DenseMatrix<double>& within_contrasts);

} // namespace datamunge::stats
