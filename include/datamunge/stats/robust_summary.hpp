#pragma once

/// @file robust_summary.hpp
/// @brief Robust univariate summaries for finite samples.

#include <cstddef>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Estimates sample quantiles using the nine Hyndman–Fan definitions, like R's quantile().
///
/// Types 1–3 are discontinuous. Types 4–9 interpolate between order statistics.
/// Uses one sorted copy and returns quantiles in the requested probability order.
/// @param values Nonempty finite observations.
/// @param probabilities Finite probabilities in [0, 1]. The endpoints give the sample extremes.
/// @param type Hyndman–Fan type from 1 through 9, default 7 as in R.
/// @return An owned vector of quantiles, in observation units.
/// @throws std::invalid_argument Empty sample, nonfinite input, or invalid probability or type.
/// @par Example
/// @code
/// const auto q = dm::stats::sample_quantiles({0, 10, 20, 40}, {.25, .75});
/// std::print("{:.1f} {:.1f}\n", q[0], q[1]); // 7.5 25.0
/// @endcode
/// @par References
/// - Hyndman and Fan (1996), Sample Quantiles in Statistical Packages, doi:10.1080/00031305.1996.10473566.
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html
/// @see weighted_empirical_quantiles()
std::vector<double> sample_quantiles(const std::vector<double>& values,
                                     const std::vector<double>& probabilities, int type = 7);

/// @brief Inverts the empirical CDF of observations with nonnegative weights.
///
/// Returns the least observed value whose normalized cumulative weight is at least p.
/// Tied observations contribute their combined mass. Observations with weight zero do
/// not contribute to the distribution or either endpoint.
/// @param values Nonempty finite observations.
/// @param weights Corresponding finite nonnegative weights with at least one positive entry.
/// @param probabilities Finite probabilities in [0, 1], in any order.
/// @return An owned vector of quantiles, in observation units and requested order.
/// @throws std::invalid_argument Invalid observations, weights or probabilities.
/// @par Example
/// @code
/// const auto q = dm::stats::weighted_empirical_quantiles({10, 20, 100}, {1, 3, 0}, {.25, .5});
/// std::print("{:.0f} {:.0f}\n", q[0], q[1]); // 10 20
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html (type 1, equal weights).
/// @see sample_quantiles()
std::vector<double> weighted_empirical_quantiles(const std::vector<double>& values,
                                                  const std::vector<double>& weights,
                                                  const std::vector<double>& probabilities);

/// @brief Estimates quantiles by Harrell–Davis beta-weighted order statistics.
///
/// For each interior probability p, the weight of an order statistic is the beta
/// probability across its rank interval, with shapes (n+1)p and (n+1)(1-p).
/// Uses one sorted copy. Probabilities 0 and 1 return the respective sample extreme.
/// @param values Nonempty finite observations.
/// @param probabilities Finite probabilities in [0, 1], in any order.
/// @return An owned vector of quantiles in observation units and requested order.
/// @throws std::invalid_argument Invalid input or a failure to resolve positive beta weights.
/// @par Example
/// @code
/// const auto q = dm::stats::harrell_davis_quantiles({2, 4, 9, 15, 63}, {.5});
/// std::print("{:.4f}\n", q.front()); // 11.9818
/// @endcode
/// @par References
/// - Harrell and Davis (1982), A new distribution-free quantile estimator, doi:10.1093/biomet/69.3.635.
/// @see sample_quantiles()
std::vector<double> harrell_davis_quantiles(const std::vector<double>& values,
                                            const std::vector<double>& probabilities);

/// @brief Distribution-free quantile interval with its achieved binomial coverage.
/// @par Example
/// @code
/// const auto bounds = dm::stats::exact_quantile_interval({0, 1, 2, 3, 4, 5}, .5);
/// std::print("{:.5f}\n", bounds.coverage); // 0.96875
/// @endcode
/// @see exact_quantile_interval()
struct ExactQuantileInterval {
  double lower;           ///< Lower bound in observation units, possibly negative infinity.
  double upper;           ///< Upper bound in observation units, possibly positive infinity.
  std::size_t lower_rank; ///< One-based lower order statistic, or zero for negative infinity.
  std::size_t upper_rank; ///< One-based upper order statistic, or sample size plus one for positive infinity.
  double coverage;        ///< Coverage under independent sampling from a continuous distribution.
};

/// @brief Inverts two binomial tails to give a distribution-free interval for a population quantile.
///
/// For n independent observations from a continuous distribution, the number below
/// the target quantile is binomial(n, probability). Each excluded tail has probability
/// at most (1 - confidence_level)/2. The interval can be conservative due to integer ranks.
/// Sorts a working copy in O(n log n) time.
/// @param values Nonempty finite observations, in any order.
/// @param probability Target population quantile probability in (0,1).
/// @param confidence_level Nominal coverage in (0,1), default 0.95.
/// @return Owned bounds, the selected ranks and their achieved coverage.
/// @throws std::invalid_argument Invalid sample, quantile probability or confidence level.
/// @par Example
/// @code
/// const auto ci = dm::stats::exact_quantile_interval({5, 4, 2, 3, 1, 0}, .5);
/// std::print("{:.0f} {:.0f}\n", ci.lower, ci.upper); // 0 5
/// @endcode
/// @par References
/// - https://support.sas.com/documentation/cdl/en/qcug/68161/HTML/default/qcug_capability_sect012.htm
/// @see sample_quantiles()
ExactQuantileInterval exact_quantile_interval(const std::vector<double>& values,
                                               double probability, double confidence_level = 0.95);

/// @brief One point on an empirical distribution and its simultaneous confidence band.
/// @par Example
/// @code
/// const auto point = dm::stats::EmpiricalCdfBand({1, 2, 3, 4}).evaluate(2);
/// std::print("{:.1f}\n", point.cdf); // 0.5
/// @endcode
/// @see EmpiricalCdfBand
struct EmpiricalCdfPoint {
  double x;     ///< Evaluation point in observation units.
  double cdf;   ///< Fraction of observations less than or equal to x.
  double lower; ///< Simultaneous lower bound, clipped to [0, 1].
  double upper; ///< Simultaneous upper bound, clipped to [0, 1].
};

/// @brief Empirical CDF with a two-sided Dvoretzky–Kiefer–Wolfowitz confidence band.
///
/// The band uses the finite-sample Massart bound for independent, identically
/// distributed observations, including discrete data. The empirical CDF is
/// right-continuous. The fit owns its jump points and uses no shared state.
/// @par Example
/// @code
/// const dm::stats::EmpiricalCdfBand band({1, 1, 3, 4});
/// std::print("{}\n", band.knots().size()); // 3
/// @endcode
/// @par References
/// - Massart (1990), The Tight Constant in the Dvoretzky–Kiefer–Wolfowitz Inequality,
///   doi:10.1214/aop/1176990746.
/// @see empirical_cdf_plot()
class EmpiricalCdfBand {
public:
  /// @brief Builds the CDF and band in O(n log n) time by sorting a copy of the sample.
  /// @param values Nonempty finite observations, in any order.
  /// @param confidence_level Lower bound on simultaneous coverage in (0,1), default 0.95.
  /// @throws std::invalid_argument Invalid observations or confidence level.
  /// @par Example
  /// @code
  /// const dm::stats::EmpiricalCdfBand band({0, 1, 2}, .90);
  /// std::print("{:.0f}\n", band.knots().back().cdf); // 1
  /// @endcode
  /// @see evaluate()
  explicit EmpiricalCdfBand(const std::vector<double>& values, double confidence_level = .95);

  /// @brief Returns one right-hand CDF and band value per distinct sample observation.
  /// @return Borrowed points sorted by increasing x, valid for the lifetime of the object.
  /// @see evaluate()
  [[nodiscard]] const std::vector<EmpiricalCdfPoint>& knots() const noexcept { return knots_; }

  /// @brief Evaluates the step function and clipped band at x in O(log n) time.
  /// @param x Query in observation units, including signed infinities.
  /// @return Owned CDF and band values at x.
  /// @throws std::invalid_argument NaN query.
  /// @par Example
  /// @code
  /// const dm::stats::EmpiricalCdfBand band({1, 1, 3, 4});
  /// std::print("{:.1f}\n", band.evaluate(2).cdf); // 0.5
  /// @endcode
  /// @see knots()
  [[nodiscard]] EmpiricalCdfPoint evaluate(double x) const;

private:
  /// @brief Computes the clipped band around a specified empirical probability.
  /// @param x Point in observation units.
  /// @param cdf Empirical probability in [0, 1].
  /// @return Point with the fitted band half-width applied.
  [[nodiscard]] EmpiricalCdfPoint band_point(double x, double cdf) const noexcept;

  std::vector<EmpiricalCdfPoint> knots_; ///< Owned right-hand values at distinct observations.
  double half_width_;                  ///< Unclipped band half-width.
};

/// @brief Median absolute deviation from a supplied or estimated center, like R's mad().
///
/// The normal-consistency multiplier defaults to R's rounded value, 1.4826.
/// A sample of one observation gives zero when its center is estimated.
/// Uses a working copy and linear expected time.
/// @param values Nonempty finite observations.
/// @param center Center in observation units, defaulting to the sample median.
/// @param scale Finite nonnegative multiplier, default 1.4826. Pass 1 for raw MAD.
/// @return Scaled median absolute deviation in observation units.
/// @throws std::invalid_argument Invalid inputs or a numerically unrepresentable result.
/// @par Example
/// @code
/// std::print("{:.1f}\n", dm::stats::median_absolute_deviation({1, 2, 3, 100})); // 1.5
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mad.html
/// @see trimmed_summary()
double median_absolute_deviation(const std::vector<double>& values,
                                 std::optional<double> center = std::nullopt,
                                 double scale = 1.4826);

/// @brief Location and dispersion from a symmetric trim.
/// @par Example
/// @code
/// const auto fit = dm::stats::trimmed_summary({1, 2, 3, 4, 100}, .2);
/// std::print("{:.1f}\n", fit.winsorized_variance); // 1.0
/// @endcode
/// @see trimmed_summary()
struct TrimmedSummary {
  double trimmed_mean;        ///< Mean of the retained central values, in observation units.
  double winsorized_mean;     ///< Mean after replacing trimmed values by the nearest retained endpoints.
  double winsorized_variance; ///< Sample variance of the replaced data, divided by original size minus one.
};

/// @brief Computes a trimmed mean, Winsorized mean and Winsorized sample variance.
///
/// Removes floor(fraction * size) values from each tail for the trimmed mean.
/// The Winsorized summaries use the full sample after endpoint replacement.
/// Uses a sorted working copy with O(n log n) time and O(n) storage.
/// @param values At least two finite observations.
/// @param fraction Fraction removed from each tail, in [0, 0.5).
/// @return Owned summary. At fraction zero the two means coincide.
/// @throws std::invalid_argument Invalid inputs or a numerically unrepresentable result.
/// @par Example
/// @code
/// const auto fit = dm::stats::trimmed_summary({0, 1, 2, 3, 4, 100}, .2);
/// std::print("{:.1f}\n", fit.trimmed_mean); // 2.5
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/base/html/mean.html
/// @see median_absolute_deviation()
TrimmedSummary trimmed_summary(const std::vector<double>& values, double fraction);

/// @brief Fitted Huber location and scale with an iteration count and convergence flag.
/// @par Example
/// @code
/// const auto fit = dm::stats::huber_location_scale({-2, -1, 0, 1, 2});
/// std::print("{:.0f}\n", fit.location); // 0
/// @endcode
/// @see huber_location_scale()
struct HuberLocationScale {
  double location;         ///< Fitted center in observation units.
  double scale;            ///< Normal-calibrated spread in observation units.
  double tuning_constant;  ///< Cutoff in units of the current fitted scale.
  std::size_t iterations; ///< Number of joint location and scale updates applied.
  bool converged;          ///< Both updates were small relative to the current scale.
};

/// @brief Fits Huber's Proposal 2 for joint estimation of location and scale.
///
/// Starts from the median and normal-scaled MAD. Each iteration recomputes the mean
/// and normal-calibrated sample standard deviation of observations clipped about
/// the previous center. Uses the same degrees-of-freedom convention as MASS hubers().
/// A zero initial MAD returns the median with zero scale and converged=false.
/// If the iteration limit is reached, returns the last iterate with converged=false.
/// @param values At least two finite observations.
/// @param tuning_constant Positive finite clipping threshold, default 1.5.
/// @param tolerance Positive finite tolerance on both changes relative to scale, default 1e-8.
/// @param max_iterations Positive iteration cap, default 200.
/// @return Fitted location, scale, tuning constant and convergence diagnostics.
/// @throws std::invalid_argument Invalid inputs or an unrepresentable result.
/// @par Example
/// @code
/// const auto fit = dm::stats::huber_location_scale({3, 3, 3, 9});
/// std::print("{:.0f} {:.0f} {}\n", fit.location, fit.scale, fit.converged); // 3 0 false
/// @endcode
/// @par References
/// - Huber (1964), Robust Estimation of a Location Parameter, doi:10.1214/aoms/1177703732.
/// - https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/hubers.html
/// @see median_absolute_deviation(), trimmed_summary()
HuberLocationScale huber_location_scale(const std::vector<double>& values,
                                        double tuning_constant = 1.5,
                                        double tolerance = 1e-8,
                                        std::size_t max_iterations = 200);

/// @brief Estimates the Rousseeuw–Croux Qn robust scale from pairwise distances.
///
/// Selects the order statistic with one-based rank choose(floor(n/2)+1, 2)
/// from differences between distinct observations, without storing the pairwise
/// matrix. The normal-consistency constant is 2.21914, as in R's robustbase Qn().
/// @param values Nonempty finite observations. A singleton gives zero.
/// @param finite_correction Apply the robustbase finite-sample correction, default true.
/// @return Qn scale in observation units.
/// @throws std::invalid_argument Invalid sample, unrepresentable rank or scale.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::stats::qn_scale({1, 2, 3, 100}, false)); // 4.438
/// @endcode
/// @par References
/// - https://search.r-project.org/CRAN/refmans/robustbase/html/Qn.html
/// @see sn_scale(), median_absolute_deviation()
double qn_scale(const std::vector<double>& values, bool finite_correction = true);

/// @brief Estimates the Rousseeuw–Croux Sn robust scale from inner and outer medians.
///
/// For each observation, takes the high median of its distances to all
/// observations, including itself. The low median of those values is multiplied
/// by the normal-consistency constant 1.1926. Work is O(n log n).
/// @param values Nonempty finite observations. A singleton gives zero.
/// @param finite_correction Apply the robustbase finite-sample correction, default true.
/// @return Sn scale in observation units.
/// @throws std::invalid_argument Invalid sample or unrepresentable scale.
/// @par Example
/// @code
/// std::print("{:.3f}\n", dm::stats::sn_scale({1, 2, 3, 100}, false)); // 2.385
/// @endcode
/// @par References
/// - https://search.r-project.org/CRAN/refmans/robustbase/html/Sn.html
/// @see qn_scale(), median_absolute_deviation()
double sn_scale(const std::vector<double>& values, bool finite_correction = true);

/// @brief Empirical likelihood ratio test and its fitted observation weights.
/// @par Example
/// @code
/// const auto fit = dm::stats::empirical_likelihood_mean_test({0, 2}, 1);
/// std::print("{:.1f}\n", fit.weights.front()); // 0.5
/// @endcode
/// @see empirical_likelihood_mean_test()
struct EmpiricalLikelihoodMeanTest {
  double statistic;            ///< Twice the negative log likelihood ratio.
  double p_value;              ///< Upper chi-square tail probability with one degree of freedom.
  std::vector<double> weights; ///< Probabilities in observation order, empty at or outside the hull.
};

/// @brief Tests a hypothesized population mean using the empirical likelihood ratio.
///
/// Maximizes the product of positive weights subject to their sum being one and
/// their weighted mean matching the hypothesized mean. Uses the asymptotic
/// chi-square(1) reference for independent sampling with positive finite variance.
/// A hypothesis at or outside the sample convex hull gives infinite statistic and p=0.
/// @param values At least two nonconstant finite observations.
/// @param null_mean Finite mean in observation units.
/// @return Likelihood-ratio statistic, p-value and fitted weights when feasible.
/// @throws std::invalid_argument Invalid observations or hypothesized mean.
/// @par Example
/// @code
/// const auto fit = dm::stats::empirical_likelihood_mean_test({0, 2}, .5);
/// std::print("{:.2f} {:.2f}\n", fit.weights[0], fit.weights[1]); // 0.75 0.25
/// @endcode
/// @par References
/// - Owen (1988), Empirical likelihood ratio confidence intervals for a single functional,
///   doi:10.1093/biomet/75.2.237.
/// @see empirical_likelihood_mean_interval()
EmpiricalLikelihoodMeanTest empirical_likelihood_mean_test(
    const std::vector<double>& values, double null_mean);

/// @brief Empirical likelihood interval for a mean with its likelihood-ratio cutoff.
/// @par Example
/// @code
/// const auto ci = dm::stats::empirical_likelihood_mean_interval({0, 1, 2});
/// std::print("{:.0f}\n", ci.estimate); // 1
/// @endcode
/// @see empirical_likelihood_mean_interval()
struct EmpiricalLikelihoodMeanInterval {
  double estimate; ///< Unconstrained mean.
  double lower;    ///< Lower confidence bound.
  double upper;    ///< Upper confidence bound.
  double cutoff;   ///< One-dimensional chi-square critical value.
};

/// @brief Inverts the empirical likelihood ratio to obtain a confidence interval for the mean.
///
/// Searches both sides of the sample mean for the requested chi-square cutoff.
/// The nominal coverage is asymptotic for independent sampling with finite positive variance.
/// @param values At least two nonconstant finite observations.
/// @param confidence_level Nominal coverage in (0,1), default .95.
/// @return Sample mean, interval bounds and likelihood-ratio cutoff.
/// @throws std::invalid_argument Invalid observations or confidence level.
/// @par Example
/// @code
/// const auto ci = dm::stats::empirical_likelihood_mean_interval({0, 2});
/// std::print("{:.3f}\n", ci.cutoff); // 3.841
/// @endcode
/// @par References
/// - Owen (1988), Empirical likelihood ratio confidence intervals for a single functional,
///   doi:10.1093/biomet/75.2.237.
/// @see empirical_likelihood_mean_test()
EmpiricalLikelihoodMeanInterval empirical_likelihood_mean_interval(
    const std::vector<double>& values, double confidence_level = .95);

} // namespace datamunge::stats
