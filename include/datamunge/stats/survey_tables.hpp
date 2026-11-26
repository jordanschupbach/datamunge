#pragma once

/// @file survey_tables.hpp
/// @brief Design-based tests of association in two-way tables: Rao–Scott corrections and Wald tests.

#include <datamunge/stats/survey.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief One design-adjusted test of independence.
/// @par Example
/// @code
/// const dm::stats::SurveyTableTest t{.statistic = 3.2, .df1 = 1, .df2 = 30, .p_value = 0.08};
/// std::println("{}", t.df1); // 1
/// @endcode
struct SurveyTableTest {
    double statistic{}; ///< Test statistic (chi-square or F scale, see SurveyTableTests).
    double df1{};       ///< Numerator (or chi-square) degrees of freedom.
    double df2{};       ///< Denominator degrees of freedom; infinity for a chi-square reference.
    double p_value{};   ///< Upper-tail p-value.
};

/// @brief Pearson statistic with its design-based corrections, as `survey::svychisq`.
/// @par Example
/// @code
/// const dm::stats::SurveyTableTests t;
/// std::println("{}", t.design_effects.size()); // 0
/// @endcode
struct SurveyTableTests {
    std::size_t rows{};                     ///< Row categories observed.
    std::size_t cols{};                     ///< Column categories observed.
    std::vector<double> proportions;        ///< Estimated cell proportions, row index fastest (r + rows * c).
    double pearson{};                       ///< Pearson \f$X^2\f$ on the weighted table rescaled to the sample size.
    double naive_p_value{};                 ///< Unadjusted \f$\chi^2_{(R-1)(C-1)}\f$ p-value of the Pearson statistic.
    std::vector<double> design_effects;     ///< Eigenvalues of the generalized design-effect matrix \f$\Delta\f$, decreasing.
    double design_df{};                     ///< Design degrees of freedom \f$\nu\f$ (PSUs minus strata).
    SurveyTableTest first_order;            ///< \f$X^2/\bar\delta\f$ against \f$\chi^2_{(R-1)(C-1)}\f$ (svychisq "Chisq").
    SurveyTableTest second_order;           ///< \f$X^2/\mathrm{tr}\Delta\f$ against \f$F(d,d\nu)\f$, \f$d=(\mathrm{tr}\Delta)^2/\mathrm{tr}\Delta^2\f$ (svychisq "F").
    SurveyTableTest wald;                   ///< Wald statistic divided by its df, against \f$F(\mathrm{df},\nu)\f$ (svychisq "Wald").
    SurveyTableTest adjusted_wald;          ///< Wald scaled by \f$(\nu-\mathrm{df}+1)/(\mathrm{df}\,\nu)\f$, against \f$F(\mathrm{df},\nu-\mathrm{df}+1)\f$ (svychisq "adjWald").
};

/// @brief Tests independence of two categorical variables in a complex survey.
///
/// The Pearson statistic computed on weighted proportions is not \f$\chi^2\f$ under clustering
/// and unequal weights: its null distribution is \f$\sum_k\delta_kW_k\f$ with \f$W_k\sim\chi^2_1\f$
/// and \f$\delta_k\f$ the eigenvalues of the generalized design-effect matrix
/// \f$\Delta=(C^\top D^{-1}C/n)^{-1}(C^\top D^{-1}VD^{-1}C)\f$, where V is the linearized covariance
/// of the cell proportions, D their diagonal and C a basis for the interaction contrasts. The
/// first-order Rao–Scott correction divides by the mean eigenvalue; the second-order correction
/// also matches the variance through an F reference with Satterthwaite degrees of freedom. The Wald
/// test uses the covariance of \f$\hat N_{rc}-\hat N_{r\cdot}\hat N_{\cdot c}/\hat N\f$ directly
/// and is unstable when \f$\nu\f$ is small relative to the table size; the adjusted Wald (Fellegi)
/// form compensates. Categories are the distinct values present; empty rows or columns are dropped.
/// @param design Sampling design.
/// @param rows Row category of each unit.
/// @param cols Column category of each unit.
/// @return Pearson statistic, design effects and the four adjusted tests.
/// @throws std::invalid_argument For an invalid design, mismatched sizes, fewer than two observed
/// categories in either variable, or a design with no degrees of freedom.
/// @throws std::runtime_error If the covariance needed by the Wald tests is singular.
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = std::vector<double>(8, 1.0)};
/// const auto t = dm::stats::survey_table_test(d, {0, 0, 0, 1, 1, 1, 0, 1}, {0, 0, 1, 1, 1, 0, 0, 1});
/// std::println("{:.4f} {:.4f}", t.pearson, t.design_effects[0]); // 2.0000 1.1429
/// @endcode
/// @par References
/// - J. N. K. Rao and A. J. Scott (1984). "On chi-squared tests for multiway contingency tables with
///   cell proportions estimated from survey data." Annals of Statistics 12(1), 46-60.
///   doi:10.1214/aos/1176346391
/// - T. Lumley (2010). Complex Surveys: A Guide to Analysis Using R. Wiley, section 6.3.
/// @see survey_score_covariance()
[[nodiscard]] SurveyTableTests survey_table_test(const SurveyDesign& design, const std::vector<std::size_t>& rows,
                                                 const std::vector<std::size_t>& cols);

} // namespace datamunge::stats
