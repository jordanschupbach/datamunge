#pragma once

/// @file sample_moments.hpp
/// @brief Weighted covariance and univariate standardized moments.

#include <datamunge/linalg/dense_matrix.hpp>
#include <optional>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Specifies how the weights enter the sample covariance divisor.
/// @par Example
/// @code
/// const auto type = dm::stats::SampleWeightType::Frequency;
/// std::print("{}\n", type == dm::stats::SampleWeightType::Frequency); // true
/// @endcode
/// @see weighted_sample_moments()
enum class SampleWeightType {
  Analytic,  ///< Relative weights, with divisor W - sum(w^2)/W.
  Frequency, ///< Integer replication counts, with divisor W - 1.
};

/// @brief Weighted mean and sample covariance for each column of a numeric design.
/// @par Example
/// @code
/// const auto result = dm::stats::weighted_sample_moments({2, 1, {1, 3}}, {1, 1});
/// std::print("{:.0f}\n", result.covariance(0, 0)); // 2
/// @endcode
/// @see weighted_sample_moments()
struct WeightedSampleMoments {
  std::vector<double> mean;               ///< Weighted means in column order.
  linalg::DenseMatrix<double> covariance; ///< Symmetric covariance, with variances on the diagonal.
};

/// @brief Computes weighted sample means and covariances, like R's cov.wt for analytic weights.
///
/// With analytic weights the results are invariant to a common weight multiplier.
/// With frequency weights the results equal those obtained by expanding the rows.
/// A zero weight contributes no moments. Work is O(n p^2) for n rows and p columns.
/// @param values Finite numeric matrix with observations in rows and at least one column.
/// @param weights Finite nonnegative row weights. Frequency weights must be integral.
/// @param type Weight interpretation, analytic by default.
/// @return Owned vector of weighted means and symmetric sample covariance.
/// @throws std::invalid_argument Invalid dimensions, weights, data or weight type, nonpositive
/// sample covariance divisor, or an unrepresentable result.
/// @par Example
/// @code
/// const auto result = dm::stats::weighted_sample_moments(
///     {2, 1, {1, 3}}, {2, 1}, dm::stats::SampleWeightType::Frequency);
/// std::print("{:.3f}\n", result.mean.front()); // 1.667
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cov.wt.html
/// @see sample_shape()
WeightedSampleMoments weighted_sample_moments(const linalg::DenseMatrix<double>& values,
                                              const std::vector<double>& weights,
                                              SampleWeightType type = SampleWeightType::Analytic);

/// @brief Standardized third and fourth central moments of a sample.
/// @par Example
/// @code
/// const auto shape = dm::stats::sample_shape({1, 2, 3, 4});
/// std::print("{:.1f}\n", shape.skewness); // 0.0
/// @endcode
/// @see sample_shape()
struct SampleShape {
  double skewness;        ///< Standardized third central moment.
  double excess_kurtosis; ///< Standardized fourth central moment less three.
};

/// @brief Estimates sample skewness and excess kurtosis with optional finite-sample corrections.
///
/// The uncorrected result uses central moments with denominator n.
/// The corrected result uses the adjusted Fisher–Pearson skewness and k-statistic excess.
/// Input is unchanged. Work is linear in its size.
/// @param values Finite nonconstant observations, at least two, or four when corrected.
/// @param bias_correct Apply the finite-sample corrections when true, default false.
/// @return Both standardized moments as dimensionless coefficients.
/// @throws std::invalid_argument Invalid sample size, nonfinite values or zero variance.
/// @par Example
/// @code
/// const auto shape = dm::stats::sample_shape({1, 2, 3, 4}, true);
/// std::print("{:.1f}\n", shape.excess_kurtosis); // -1.2
/// @endcode
/// @par References
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html
/// @see weighted_sample_moments()
SampleShape sample_shape(const std::vector<double>& values, bool bias_correct = false);

/// @brief First four sample L-moments and the standardized third and fourth L-moment ratios.
/// @par Example
/// @code
/// const auto fit = dm::stats::sample_l_moments({0, 1, 2, 3});
/// std::print("{:.1f}\n", fit.location); // 1.5
/// @endcode
/// @see sample_l_moments()
struct SampleLMoments {
  double location;                  ///< First L-moment, the sample mean.
  double scale;                     ///< Second L-moment, half the average pairwise absolute difference.
  double third;                     ///< Third L-moment in observation units.
  double fourth;                    ///< Fourth L-moment in observation units.
  std::optional<double> skewness;    ///< Third L-moment divided by scale, absent for constant data.
  std::optional<double> kurtosis;    ///< Fourth L-moment divided by scale, absent for constant data.
};

/// @brief Estimates L-moments from unbiased probability-weighted sample moments.
///
/// The first four L-moment estimates are unbiased for independent, identically distributed
/// observations with finite population mean. The standardized estimates are ratios of these
/// estimates. Sorts a working copy in O(n log n) time.
/// @param values At least four finite observations, in any order.
/// @return L-moments and dimensionless L-skewness and L-kurtosis.
/// @throws std::invalid_argument Invalid sample size, nonfinite input or unrepresentable result.
/// @par Example
/// @code
/// const auto fit = dm::stats::sample_l_moments({0, 0, 0, 4});
/// std::print("{:.0f} {:.0f} {:.0f}\n", fit.scale, *fit.skewness, *fit.kurtosis); // 1 1 1
/// @endcode
/// @par References
/// - Hosking (1990), L-Moments, doi:10.1111/j.2517-6161.1990.tb01775.x.
/// - https://search.r-project.org/CRAN/refmans/EnvStats/html/lMoment.html
/// @see sample_shape()
SampleLMoments sample_l_moments(const std::vector<double>& values);

/// @brief Weighted empirical Lorenz curve and its relative Gini coefficient.
/// @par Example
/// @code
/// const auto fit = dm::stats::lorenz_curve({0, 1});
/// std::print("{:.1f}\n", fit.gini); // 0.5
/// @endcode
/// @see lorenz_curve()
struct LorenzCurve {
  std::vector<double> population_share; ///< Cumulative share of positive weight.
  std::vector<double> value_share;      ///< Cumulative share of the weighted value.
  double gini;                          ///< Twice the area between equality and the curve, in [0,1].
};

/// @brief Returns cumulative weighted shares in ascending value order and the empirical Gini.
///
/// Omits zero-weight observations. The curve includes (0,0) and (1,1) and is linear
/// between points. The Gini uses the empirical distribution without a sample-size correction.
/// Input is unchanged. Work is O(n log n) with O(n) storage.
/// @param values Nonempty finite nonnegative observations.
/// @param weights Finite nonnegative weights matching values, or empty for unit weights.
/// @return Owned curve and Gini coefficient.
/// @throws std::invalid_argument Invalid inputs or zero total weighted value.
/// @par Example
/// @code
/// const auto fit = dm::stats::lorenz_curve({3, 1}, {1, 3});
/// std::print("{:.2f}\n", fit.value_share[1]); // 0.50
/// @endcode
/// @par References
/// - https://www.oecd.org/en/publications/oecd-handbook-on-the-compilation-of-household-distributional-results-on-income-consumption-and-saving-in-line-with-national-accounts-totals_5a3b9119-en.html
/// @see weighted_sample_moments()
LorenzCurve lorenz_curve(const std::vector<double>& values, const std::vector<double>& weights = {});

/// @brief Inverts Fieller's quadratic inequality for a ratio of correlated estimates.
///
/// Accepts ratios r for which the squared residual (numerator - r * denominator)
/// is at most the critical value squared times its estimated variance. The
/// resulting set can be bounded, unbounded, disconnected, empty or all real.
/// @param numerator Finite numerator estimate.
/// @param denominator Finite denominator estimate.
/// @param numerator_variance Finite nonnegative variance of the numerator estimate.
/// @param denominator_variance Finite nonnegative variance of the denominator estimate.
/// @param covariance Finite covariance consistent with the two variances.
/// @param critical_value Positive finite normal or Student t critical value. The default is
/// the two-sided 95% standard normal critical value.
/// @return Sorted intervals as pairs of inclusive endpoints, possibly signed infinities.
/// @throws std::invalid_argument Invalid estimates, covariance or critical value.
/// @par Example
/// @code
/// const auto set = dm::stats::fieller_confidence_set(1, 2, .01, 0, 0, 1);
/// std::print("{:.2f} {:.2f}\n", set[0].first, set[0].second); // 0.45 0.55
/// @endcode
/// @par References
/// - Fieller (1954), Some Problems in Interval Estimation, doi:10.1111/j.2517-6161.1954.tb00159.x.
/// @see weighted_sample_moments()
std::vector<std::pair<double, double>> fieller_confidence_set(
    double numerator, double denominator, double numerator_variance, double denominator_variance,
    double covariance, double critical_value = 1.959963984540054);

/// @brief Propagates parameter covariance through a vector transformation by the delta method.
///
/// Multiplies J times the parameter covariance times the transpose of J. This is
/// exact for a linear transformation and a first-order approximation otherwise.
/// The caller supplies the Jacobian evaluated at the fitted parameters.
/// @param jacobian Finite q-by-p matrix, one row per derived quantity and one column per parameter.
/// @param covariance Finite symmetric positive-semidefinite p-by-p parameter covariance.
/// @return Owned symmetric q-by-q covariance in the units of the derived quantities.
/// @throws std::invalid_argument Invalid dimensions, nonfinite data, asymmetric covariance,
/// negative input variance or an unrepresentable output.
/// @par Example
/// @code
/// const auto cov = dm::stats::delta_method_covariance({1, 2, {1, 1}}, {2, 2, {1, .5, .5, 2}});
/// std::print("{:.0f}\n", cov(0, 0)); // 4
/// @endcode
/// @par References
/// - https://search.r-project.org/CRAN/refmans/msm/html/deltamethod.html
/// @see fieller_confidence_set()
linalg::DenseMatrix<double> delta_method_covariance(const linalg::DenseMatrix<double>& jacobian,
                                                      const linalg::DenseMatrix<double>& covariance);

} // namespace datamunge::stats
