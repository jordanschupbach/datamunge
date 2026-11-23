#pragma once
/// @file anova_measures.hpp
/// @brief ANOVA effect sizes and intraclass-correlation estimates.

#include <cstddef>
#include <datamunge/linalg/dense_matrix.hpp>

namespace datamunge::stats {

/// @brief Overall and partial ANOVA effect sizes for a term.
/// @par Example
/// @code
/// const auto result = dm::stats::anova_effect_sizes(4, 6, 10, 1, 6);
/// std::print("{:.1f}\n", result.eta_squared); // 0.4
/// @endcode
/// @see anova_effect_sizes()
struct AnovaEffectSizes {
  double eta_squared;          ///< Term sum of squares divided by corrected total sum of squares.
  double partial_eta_squared;  ///< Term sum of squares divided by term plus residual sum of squares.
  double omega_squared;        ///< Bias-corrected term share of total variation.
  double partial_omega_squared;///< Bias-corrected term share relative to term plus residual variation.
};

/// @brief Estimates overall and partial eta and omega squared from ANOVA sums of squares.
///
/// Both omega estimates subtract term degrees of freedom times residual
/// mean square from the numerator. Their denominators add one residual
/// mean square to total or term-plus-residual sum of squares, respectively.
/// Negative omega estimates are retained.
/// @param term_ss Nonnegative finite sum of squares for the term.
/// @param residual_ss Nonnegative finite residual sum of squares.
/// @param total_ss Positive finite corrected total sum of squares.
/// @param term_df Positive finite degrees of freedom for the term.
/// @param residual_df Positive finite residual degrees of freedom.
/// @return All four estimates.
/// @throws std::invalid_argument Invalid sums of squares, degrees of freedom, or a zero partial denominator.
/// @throws std::overflow_error An estimate exceeds finite double range.
/// @par Example
/// @code
/// const auto result = dm::stats::anova_effect_sizes(4, 6, 20, 1, 6);
/// std::print("{:.2f} {:.2f}\n", result.eta_squared, result.partial_eta_squared); // 0.20 0.40
/// @endcode
/// @par References
/// - D. Lakens (2013), https://doi.org/10.3389/fpsyg.2013.00863
/// - D. Lakens, https://lakens.github.io/statistical_inferences/06-effectsize.html
/// @see intraclass_correlation()
AnovaEffectSizes anova_effect_sizes(double term_ss, double residual_ss, double total_ss,
                                   double term_df, double residual_df);

/// @brief ANOVA model and rater population for intraclass correlation.
/// @par Example
/// @code
/// std::print("{}\n", dm::stats::IccModel::OneWayRandom != dm::stats::IccModel::TwoWayRandom); // true
/// @endcode
/// @see intraclass_correlation()
enum class IccModel {
  OneWayRandom, ///< Each target may have a different panel drawn from the rater population.
  TwoWayRandom, ///< The same panel rates every target and represents a rater population.
  TwoWayMixed   ///< The same panel rates every target and the raters are fixed.
};

/// @brief Whether systematic differences across raters count as disagreement.
/// @par Example
/// @code
/// std::print("{}\n", dm::stats::IccDefinition::AbsoluteAgreement != dm::stats::IccDefinition::Consistency); // true
/// @endcode
/// @see intraclass_correlation()
enum class IccDefinition {
  AbsoluteAgreement, ///< Systematic rater differences count as disagreement.
  Consistency         ///< Two-way adjustment removes systematic rater differences.
};

/// @brief Unit whose reliability the intraclass correlation measures.
/// @par Example
/// @code
/// std::print("{}\n", dm::stats::IccUnit::Single != dm::stats::IccUnit::Average); // true
/// @endcode
/// @see intraclass_correlation()
enum class IccUnit {
  Single, ///< Reliability for a single rating.
  Average ///< Reliability for the average across the supplied panel size.
};

/// @brief Intraclass correlation estimate and the choices that define it.
/// @par Example
/// @code
/// const dm::stats::IntraclassCorrelation fit{1, dm::stats::IccModel::OneWayRandom,
///     dm::stats::IccDefinition::AbsoluteAgreement, dm::stats::IccUnit::Single, 3, 2};
/// std::print("{}\n", fit.raters); // 2
/// @endcode
/// @see intraclass_correlation()
struct IntraclassCorrelation {
  double estimate;          ///< Untruncated ANOVA intraclass correlation estimate.
  IccModel model;           ///< Selected ANOVA model.
  IccDefinition definition; ///< Selected agreement definition.
  IccUnit unit;             ///< Selected rating unit.
  std::size_t targets;      ///< Number of targets in the table.
  std::size_t raters;       ///< Number of ratings per target.
};

/// @brief Estimates intraclass correlation from a complete balanced rating matrix.
///
/// One-way random effects support absolute agreement. The two-way models
/// support both definitions and give the same point estimate for a given
/// definition and unit. For the two-way models each column identifies
/// the same rater across targets. Negative estimates are retained.
/// @param ratings At least two target rows and two rating columns, with finite entries.
/// @param model One-way random, two-way random or two-way mixed effects.
/// @param definition Absolute agreement or two-way consistency.
/// @param unit Single or average rating.
/// @return ICC, design choices and dimensions.
/// @throws std::invalid_argument Invalid design or zero denominator.
/// @throws std::overflow_error An estimate exceeds finite double range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> ratings(3, 2, {1, 2, 3, 4, 6, 7});
/// const auto fit = dm::stats::intraclass_correlation(ratings, dm::stats::IccModel::TwoWayMixed,
///     dm::stats::IccDefinition::Consistency, dm::stats::IccUnit::Single);
/// std::print("{}\n", fit.estimate); // 1
/// @endcode
/// @par References
/// - P. E. Shrout and J. L. Fleiss (1979), https://pubmed.ncbi.nlm.nih.gov/18839484/
/// - K. O. McGraw and S. P. Wong (1996), https://doi.org/10.1037/1082-989X.1.1.30
/// @see anova_effect_sizes()
IntraclassCorrelation intraclass_correlation(const linalg::DenseMatrix<double>& ratings, IccModel model,
                                             IccDefinition definition, IccUnit unit);

} // namespace datamunge::stats
