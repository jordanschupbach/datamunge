#pragma once

/// @file control_chart.hpp
/// @brief Statistical process control charts and native plots.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Limit settings for charts that estimate their process baseline from the input.
///
/// @par Example
/// @code
/// auto chart = dm::stats::c_chart({2, 3, 4, 16}, {.baseline_size = 3});
/// std::print("{}\n", chart.points.back().outside); // true
/// @endcode
/// @see c_chart(), individuals_chart()
struct ControlChartOptions {
    double sigma_multiplier{3.0}; ///< Positive finite limit multiplier, or Normal-equivalent tail setting for g and t.
    std::size_t baseline_size{0}; ///< Number of leading input observations or subgroups used to fit limits. Zero uses all.
};

/// @brief Statistic and control limits at a single plotted position.
///
/// @par Example
/// @code
/// const auto point = dm::stats::c_chart({2, 4}).points.front();
/// std::print("{:.0f}\n", point.center); // 3
/// @endcode
/// @see ControlChart
struct ControlChartPoint {
    std::size_t index; ///< One-based index of the corresponding input observation or subgroup.
    double value;      ///< Plotted statistic in the chart's units.
    double center;     ///< Center line at this position.
    double lower;      ///< Lower control limit at this position.
    double upper;      ///< Upper control limit at this position.
    bool outside;      ///< True when the statistic is strictly below lower or above upper.
};

/// @brief Owned control chart with statistics, per-position limits and signals.
///
/// @par Example
/// @code
/// const auto chart = dm::stats::p_chart({1, 2}, {10, 20});
/// std::print("{}\n", chart.points.size()); // 2
/// @endcode
/// @see ControlChartPoint, ControlChartOptions
struct ControlChart {
    std::string title; ///< Default plot title.
    std::string statistic; ///< Vertical axis label.
    std::vector<ControlChartPoint> points; ///< Ordered records including baseline and monitoring observations.

    /// @brief Builds an editable plot with the statistic, control limits and highlighted signals.
    /// @return An owned native plot, ready to display, compose or save as SVG.
    /// @throws std::bad_alloc If allocation fails.
    /// @par Example
    /// @code
    /// auto plot = dm::stats::c_chart({2, 3, 4}).plot();
    /// std::print("{}\n", !plot.to_svg().empty()); // true
    /// @endcode
    /// @see c_chart(), graphics::RPlot
    [[nodiscard]] graphics::RPlot plot() const;
};

/// @brief Builds a Poisson c-chart for defect counts at constant exposure.
///
/// The baseline arithmetic mean estimates the Poisson count. Limits use its square root.
/// The lower limit is clipped at zero.
/// @param counts Nonempty ordered defect counts.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::c_chart({2, 5, 3});
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/contchar.htm
/// @see u_chart(), ControlChart::plot()
ControlChart c_chart(const std::vector<std::size_t>& counts, ControlChartOptions options = {});

/// @brief Builds a Poisson u-chart for defects per unit of exposure.
///
/// The baseline total count divided by baseline exposure estimates the rate.
/// The lower limit is clipped at zero. Limits vary with exposure.
/// @param counts Nonempty ordered defect counts.
/// @param exposure Positive finite exposures, one per count. Units may be fractional.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::u_chart({2, 5, 3}, {1., 2., 1.});
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/contchar.htm
/// @see c_chart(), laney_u_chart()
ControlChart u_chart(const std::vector<std::size_t>& counts,
                     const std::vector<double>& exposure,
                     ControlChartOptions options = {});

/// @brief Builds a binomial p-chart for the fraction of defective units.
///
/// The baseline defective fraction pools counts and sample sizes. Limits vary with
/// sample size and are clipped to [0, 1].
/// @param counts Ordered counts of defective units, each at most its corresponding sample size.
/// @param sizes Positive inspected sample sizes, one per count.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::p_chart({2, 5, 3}, {20, 30, 20});
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/contchar.htm
/// @see np_chart(), laney_p_chart()
ControlChart p_chart(const std::vector<std::size_t>& counts,
                     const std::vector<std::size_t>& sizes,
                     ControlChartOptions options = {});

/// @brief Builds a binomial np-chart for counts of defective units at fixed sample size.
///
/// The center and standard deviation are expressed in counts. Limits are clipped
/// to the possible count interval.
/// @param counts Ordered counts of defective units, each at most its corresponding sample size.
/// @param size Positive common number of inspected units per subgroup.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::np_chart({2, 5, 3}, 20);
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/contchar.htm
/// @see p_chart()
ControlChart np_chart(const std::vector<std::size_t>& counts,
                      std::size_t size,
                      ControlChartOptions options = {});

/// @brief Builds a Laney p′ chart for binomial proportions.
///
/// Scales the usual standard errors by the mean moving range of baseline standardized
/// residuals divided by d2(2). Requires at least two baseline points and nonzero
/// theoretical variance. Zero estimated dispersion yields coincident limits.
/// @param counts Ordered counts of defective units, each at most its corresponding sample size.
/// @param sizes Positive inspected sample sizes, one per count.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::laney_p_chart({1, 4, 2}, {10, 15, 12});
/// @endcode
/// @par References
/// - https://doi.org/10.1081/QEN-120003555
/// @see p_chart()
ControlChart laney_p_chart(const std::vector<std::size_t>& counts,
                           const std::vector<std::size_t>& sizes,
                           ControlChartOptions options = {});

/// @brief Builds a Laney u′ chart for Poisson rates.
///
/// Scales the usual standard errors by the mean moving range of baseline standardized
/// residuals divided by d2(2). Requires at least two baseline points and nonzero
/// theoretical variance. Zero estimated dispersion yields coincident limits.
/// @param counts Nonempty ordered defect counts.
/// @param exposure Positive finite exposures, one per count. Units may be fractional.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::laney_u_chart({1, 4, 2}, {10, 15, 12});
/// @endcode
/// @par References
/// - https://doi.org/10.1081/QEN-120003555
/// @see u_chart()
ControlChart laney_u_chart(const std::vector<std::size_t>& counts,
                           const std::vector<double>& exposure,
                           ControlChartOptions options = {});

/// @brief Builds an individuals chart with scale estimated from adjacent moving ranges.
///
/// The center is the baseline mean. Scale is baseline mean moving range divided by
/// d2(2). Requires at least two baseline observations.
/// @param values Finite observations in process order.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::individuals_chart({2.1, 2.3, 2.2});
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc322.htm
/// @see moving_range_chart()
ControlChart individuals_chart(const std::vector<double>& values, ControlChartOptions options = {});

/// @brief Builds a moving-range chart for absolute differences of adjacent observations.
///
/// Uses baseline moving ranges and the normal two-observation range constants.
/// Requires at least two baseline observations and clips the lower limit at zero.
/// @param values Finite observations in process order.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per adjacent pair, indexed by its second observation.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::moving_range_chart({2.1, 2.3, 2.2});
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc322.htm
/// @see individuals_chart()
ControlChart moving_range_chart(const std::vector<double>& values, ControlChartOptions options = {});

/// @brief Builds an X-bar chart using subgroup ranges to estimate process scale.
///
/// Each subgroup must have 2 through 10 measurements. Uses the baseline grand mean
/// and total baseline range divided by the sum of the normal range factors d2(n).
/// Limits vary with subgroup size.
/// @param groups Nonempty ordered subgroups of finite observations.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::xbar_chart({{2., 3.}, {3., 4.}});
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm
/// @see r_chart(), xbar_s_chart()
ControlChart xbar_chart(const std::vector<std::vector<double>>& groups, ControlChartOptions options = {});

/// @brief Builds an R-chart for within-subgroup ranges.
///
/// Each subgroup must have 2 through 10 measurements. Uses baseline ranges and
/// normal range moments d2(n), d3(n). Centers and limits vary with subgroup size.
/// The lower limit is clipped at zero.
/// @param groups Nonempty ordered subgroups of finite observations.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::r_chart({{2., 3.}, {3., 4.}});
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm
/// @see xbar_chart(), s_chart()
ControlChart r_chart(const std::vector<std::vector<double>>& groups, ControlChartOptions options = {});

/// @brief Builds an X-bar chart using subgroup sample standard deviations to estimate scale.
///
/// Each subgroup must have at least two measurements. Uses the baseline grand mean
/// and total baseline sample SD divided by the sum of c4(n). Limits vary with subgroup size.
/// @param groups Nonempty ordered subgroups of finite observations.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::xbar_s_chart({{2., 3.}, {3., 4.}});
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm
/// @see s_chart(), xbar_chart()
ControlChart xbar_s_chart(const std::vector<std::vector<double>>& groups, ControlChartOptions options = {});

/// @brief Builds an S-chart for within-subgroup sample standard deviations.
///
/// Each subgroup must have at least two measurements. Uses the sum of baseline sample
/// SDs divided by the sum of c4(n) to estimate scale. Centers and limits use the normal
/// SD moments for each size. The lower limit is clipped at zero.
/// @param groups Nonempty ordered subgroups of finite observations.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::s_chart({{2., 3.}, {3., 4.}});
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm
/// @see xbar_s_chart(), r_chart()
ControlChart s_chart(const std::vector<std::vector<double>>& groups, ControlChartOptions options = {});

/// @brief Builds an exponentially weighted moving average chart with known baseline.
///
/// Starts the EWMA at the target. Limits use the exact startup variance for
/// independent observations with common variance.
/// @param values Finite observations in process order.
/// @param target Finite in-control process mean.
/// @param sigma Positive finite in-control standard deviation of an individual observation.
/// @param lambda Smoothing weight in (0, 1]. Defaults to 0.2.
/// @param k Positive finite limit multiplier. Defaults to three.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::ewma_chart({1., 2., 3.}, 0., 1.);
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm
/// @see moving_average_chart(), cusum_chart()
ControlChart ewma_chart(const std::vector<double>& values,
                        double target,
                        double sigma,
                        double lambda = 0.2,
                        double k = 3.0);

/// @brief Builds a simple moving average chart with known baseline.
///
/// Uses complete trailing windows only. Limits are based on independent
/// observations with common variance. Overlapping plotted windows are dependent.
/// @param values Finite observations in process order.
/// @param window Number of observations per window, from one through input length.
/// @param target Finite in-control process mean.
/// @param sigma Positive finite in-control standard deviation of an individual observation.
/// @param k Positive finite limit multiplier. Defaults to three.
/// @return Owned chart indexed by the final observation of each complete window.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::moving_average_chart({1., 2., 3.}, 2, 0., 1.);
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/contchar.htm
/// @see ewma_chart()
ControlChart moving_average_chart(const std::vector<double>& values,
                                  std::size_t window,
                                  double target,
                                  double sigma,
                                  double k = 3.0);

/// @brief Builds a moving standard deviation chart with known process scale.
///
/// Uses sample SDs from complete trailing windows. Normal calibration uses
/// c4(window) and sqrt(1 - c4(window)^2). Overlapping windows are dependent.
/// Runtime scales with the product of the series length and window size.
/// @param values Finite observations in process order.
/// @param window Window length, from two through the input length.
/// @param sigma Positive finite in-control standard deviation of an observation.
/// @param k Positive finite limit multiplier. Defaults to three.
/// @return An owned chart indexed by the last observation in each window.
/// @throws std::invalid_argument If data, window size, sigma or k are invalid.
/// @throws std::overflow_error If a statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::moving_sd_chart({1., 2., 3.}, 2, 1.);
/// @endcode
/// @par References
/// - https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/movecont.htm
/// @see s_chart(), moving_average_chart()
ControlChart moving_sd_chart(const std::vector<double>& values, std::size_t window, double sigma, double k = 3.0);

/// @brief Builds the two tabular CUSUM charts of standardized upward and downward shifts.
///
/// Starts both one-sided sums at zero and floors them at zero after each update.
/// Sums continue across signals. A sum strictly above the decision interval signals.
/// @param values Finite observations in process order.
/// @param target Finite in-control process mean.
/// @param sigma Positive finite in-control standard deviation of an individual observation.
/// @param reference Nonnegative finite reference allowance in standard deviation units. Defaults to 0.5.
/// @param decision Positive finite decision interval in the same units. Defaults to 5.
/// @return Upward and downward charts, respectively, both plotted as nonnegative magnitudes.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto [up, down] = dm::stats::cusum_chart({1., 2., 3.}, 0., 1.);
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm
/// @see ewma_chart()
std::pair<ControlChart, ControlChart> cusum_chart(const std::vector<double>& values,
                                                  double target,
                                                  double sigma,
                                                  double reference = 0.5,
                                                  double decision = 5.0);

/// @brief Builds a geometric g-chart for the number of non-events between events.
///
/// Fits the geometric mean to the baseline gaps. Uses the fitted median and
/// equal-tail quantile limits. The multiplier k sets each tail to the standard
/// Normal probability below −k, which must be positive at floating-point precision.
/// @param gaps Nonempty ordered counts of non-events between successive events.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::g_chart({5, 8, 4, 11});
/// @endcode
/// @par References
/// - https://doi.org/10.1023/A:1011846412909
/// @see t_chart()
ControlChart g_chart(const std::vector<std::size_t>& gaps, ControlChartOptions options = {});

/// @brief Builds an exponential t-chart for continuous time between events.
///
/// Fits the exponential mean to baseline intervals. Uses the fitted median and
/// equal-tail quantile limits. The multiplier k sets each tail to the standard
/// Normal probability below −k, which must be positive at floating-point precision.
/// @param intervals Nonempty ordered positive finite inter-event durations.
/// @param options Limit multiplier and optional baseline prefix. Defaults to three and the full input.
/// @return Owned chart with one point per input observation or subgroup.
/// @throws std::invalid_argument If an input violates the stated constraints.
/// @throws std::overflow_error If a computed statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto chart = dm::stats::t_chart({0.5, 1., 0.75, 2.});
/// @endcode
/// @par References
/// - https://doi.org/10.1023/A:1011846412909
/// @see g_chart()
ControlChart t_chart(const std::vector<double>& intervals, ControlChartOptions options = {});

/// @brief Builds an individual-observation Hotelling T-squared chart with known baseline parameters.
///
/// The statistic is the squared Mahalanobis distance from the in-control mean.
/// Under independent multivariate Normal observations with known covariance, the
/// upper limit is the chi-squared quantile with one degree of freedom per feature.
/// @param observations Nonempty ordered observations, one finite row per point.
/// @param target Nonempty finite mean vector, with one entry for each feature.
/// @param covariance Finite symmetric positive definite covariance of an observation.
/// @param alpha Upper tail probability, defaults to 0.0027. Must lie in (0, 1) and 1 - alpha must be below 1.
/// @return An owned chart with zero lower limit and a chi-squared upper limit.
/// @throws std::invalid_argument If inputs are invalid, ragged or covariance is not positive definite.
/// @throws std::overflow_error If a statistic or limit cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> covariance(2, 2, {1., 0., 0., 1.});
/// const auto chart = dm::stats::hotelling_t2_chart({{1., 2.}}, {0., 0.}, covariance);
/// std::print("{:.0f}\n", chart.points.front().value); // 5
/// @endcode
/// @par References
/// - https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc341.htm
/// @see individuals_chart()
ControlChart hotelling_t2_chart(const std::vector<std::vector<double>>& observations,
                                const std::vector<double>& target,
                                const linalg::DenseMatrix<double>& covariance, double alpha = 0.0027);

} // namespace datamunge::stats
