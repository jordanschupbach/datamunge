#pragma once

/// @file segmented_regression.hpp
/// @brief Continuous piecewise-linear regression with estimated breakpoints.

#include <datamunge/stats/profile_likelihood.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Controls for fitting ordered breakpoints and profiling their uncertainty.
/// @par Example
/// @code
/// const dm::stats::SegmentedRegressionOptions options{.breakpoints = 2};
/// std::println("{}", options.breakpoints); // 2
/// @endcode
/// @see segmented_regression()
struct SegmentedRegressionOptions {
    std::size_t breakpoints{1};          ///< Number of unknown breakpoints, at least one.
    std::size_t minimum_segment_size{2}; ///< Minimum distinct predictor values in each closed segment.
    std::size_t starts{5};               ///< Number of deterministic starts in a joint search.
    std::size_t max_sweeps{100};         ///< Maximum coordinate sweeps from each start.
    double tolerance{1e-8};             ///< Convergence tolerance relative to the observed predictor range.
    bool profile_intervals{true};       ///< Compute pointwise likelihood-ratio intervals for breakpoint locations.
    double confidence_level{.95};       ///< Confidence level for the breakpoint intervals.
};

/// @brief Fit for a continuous piecewise-linear mean, in the input predictor's units.
/// @par Example
/// @code
/// const auto fit = dm::stats::segmented_regression({0, 1, 2, 3, 4, 5}, {0, 1, 2, 2.5, 2.5, 2.5},
///                                                {.profile_intervals = false});
/// std::println("{:.1f}", fit.breakpoints[0]); // 2.5
/// @endcode
/// @see segmented_regression()
struct SegmentedRegressionResult {
    double origin;                                       ///< Smallest observed predictor value.
    double value_at_origin;                              ///< Fitted mean at origin.
    std::vector<double> breakpoints;                     ///< Knot locations in increasing order.
    std::vector<double> slopes;                          ///< One slope per segment, from left to right.
    std::vector<double> fitted;                           ///< Fitted mean in input row order.
    double residual_sum_squares;                         ///< Ordinary residual sum of squares.
    bool converged;                                      ///< The selected joint search reached tolerance.
    std::vector<ProfileLikelihoodInterval> intervals;   ///< One per breakpoint when profiling is requested and the fit converges.
};

/// @brief Fits a continuous broken line to finite predictor-response pairs.
///
/// For one breakpoint the search checks all admissible gaps and their
/// stationary points. For multiple breakpoints it optimizes one at a time
/// from deterministic starts and returns the best fit found. Repeated
/// predictor values are allowed. The segment-size rule counts distinct
/// predictor values and an observed value at a knot counts in both
/// adjacent segments. The data must contain more observations than the
/// number of coefficients and breakpoints combined.
///
/// Optional intervals refit the slopes and the other breakpoint locations
/// at each candidate. The likelihood-ratio calibration is asymptotic and
/// assumes an identified slope change with independent Gaussian errors.
/// The interval search brackets the contour around the fitted breakpoint.
/// @param predictor Predictor values, in any row order.
/// @param response Observed values in the same row order.
/// @param options Knot count, minimum support and numerical and inference controls.
/// @return Fitted breakpoints, continuous segment slopes, fitted values, and requested intervals.
/// @throws std::invalid_argument Invalid input, options, or insufficient data.
/// @throws std::domain_error A numerically exact fit prevents profiling, or a profile exceeds the main fit likelihood.
/// @throws std::runtime_error Numeric rank loss, or a nuisance search that fails to converge.
/// @throws std::overflow_error A reported quantity exceeds finite double range.
/// @par Example
/// @code
/// const auto fit = dm::stats::segmented_regression({0, 1, 2, 3, 4, 5}, {0, 1, 2, 2.5, 2.5, 2.5},
///                                                {.profile_intervals = false});
/// std::println("{:.1f} {:.1f}", fit.slopes[0], std::abs(fit.slopes[1])); // 1.0 0.0
/// @endcode
/// @par References
/// - Hudson (1966), https://doi.org/10.1080/01621459.1966.10482198
/// - Muggeo (2003), https://doi.org/10.1002/sim.1545
/// @see profile_likelihood_interval()
[[nodiscard]] SegmentedRegressionResult segmented_regression(
    const std::vector<double>& predictor, const std::vector<double>& response,
    const SegmentedRegressionOptions& options = {});

} // namespace datamunge::stats
