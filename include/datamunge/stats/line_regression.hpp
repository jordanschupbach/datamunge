#pragma once

/// @file line_regression.hpp
/// @brief Robust and errors-in-variables estimates for a straight line.

#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Coefficients of the fitted response as a function of the predictor.
/// @par Example
/// @code
/// const auto fit = dm::stats::theil_sen_regression({0, 1, 2}, {1, 3, 5});
/// std::println("{} {}", fit.intercept, fit.slope); // 1 2
/// @endcode
/// @see theil_sen_regression(), repeated_median_regression()
struct LineCoefficients {
    double intercept; ///< Fitted value at predictor zero.
    double slope;     ///< Fitted change in response per unit change in the predictor.
};

/// @brief Fits the median slope of all pairs with distinct predictor values.
///
/// The intercept is the median of the responses after subtracting the fitted
/// slope times each predictor. Even-sized medians average the middle pair.
/// Stores one slope per eligible pair and takes quadratic work in the worst case.
/// @param x Finite predictor values with at least two distinct values.
/// @param y Finite paired responses in predictor order.
/// @return Intercept and slope.
/// @throws std::invalid_argument Mismatched, insufficient, nonfinite or constant predictors, or nonfinite responses.
/// @throws std::overflow_error The fitted coefficients exceed finite double range.
/// @par Example
/// @code
/// const auto fit = dm::stats::theil_sen_regression({0, 1, 2, 3, 4}, {1, 3, 5, 7, 101});
/// std::println("{}", fit.slope); // 2
/// @endcode
/// @par References
/// - Sen (1968), https://doi.org/10.1080/01621459.1968.10480934
/// @see repeated_median_regression(), deming_regression()
LineCoefficients theil_sen_regression(const std::vector<double>& x, const std::vector<double>& y);

/// @brief Fits Siegel's repeated-median slope with a median residual intercept.
///
/// Computes a median slope through each observation using other observations
/// with different predictor values, then takes the median of those medians.
/// The intercept is the median of the responses after removing the slope term.
/// Even-sized medians average the middle pair. Uses linear working storage and
/// quadratic work.
/// @param x Finite predictor values with at least two distinct values.
/// @param y Finite paired responses in predictor order.
/// @return Intercept and slope.
/// @throws std::invalid_argument Mismatched, insufficient, nonfinite or constant predictors, or nonfinite responses.
/// @throws std::overflow_error The fitted coefficients exceed finite double range.
/// @par Example
/// @code
/// const auto fit = dm::stats::repeated_median_regression({0, 0, 1, 2}, {3, 3, 5, 7});
/// std::println("{} {}", fit.intercept, fit.slope); // 3 2
/// @endcode
/// @par References
/// - Siegel (1982), https://doi.org/10.1093/biomet/69.1.242
/// @see theil_sen_regression(), deming_regression()
LineCoefficients repeated_median_regression(const std::vector<double>& x, const std::vector<double>& y);

/// @brief A Deming line in point-direction form with optional slope-intercept coefficients.
/// @par Example
/// @code
/// const auto fit = dm::stats::deming_regression({2, 2, 2}, {1, 2, 3});
/// std::println("{} {}", fit.center_x, fit.direction_x); // 2 0
/// @endcode
/// @see deming_regression()
struct DemingLine {
    double center_x;                             ///< Mean of the predictor measurements.
    double center_y;                             ///< Mean of the response measurements.
    double direction_x;                          ///< Nonnegative predictor component of the unit direction.
    double direction_y;                          ///< Signed response component, positive for exact vertical lines.
    std::optional<LineCoefficients> coefficients; ///< Present for nonvertical lines with finite representable coefficients.
};

/// @brief Fits a straight line with measurement error in both coordinates.
///
/// The estimated direction is the principal axis after dividing the response
/// by the square root of the measurement error variance ratio. The line
/// passes through the measurement means and is returned in original units.
/// A variance ratio of one gives orthogonal least squares.
/// @param x Finite predictor measurements.
/// @param y Finite paired response measurements.
/// @param error_variance_ratio Positive finite response error variance divided by predictor error variance.
/// @return Line center, oriented unit direction and slope-intercept coefficients where representable.
/// @throws std::invalid_argument Invalid pairs or error variance ratio.
/// @throws std::domain_error The scaled scatter has no unique principal direction.
/// @throws std::overflow_error The centered scatter exceeds the working numeric range.
/// @par Example
/// @code
/// const auto fit = dm::stats::deming_regression({0, 1, 2}, {1, 3, 5}, 4);
/// std::println("{:.1f}", fit.coefficients->slope); // 2.0
/// @endcode
/// @par References
/// - Linnet (1998), https://doi.org/10.1093/clinchem/44.5.1024
/// @see DemingLine, LineCoefficients
DemingLine deming_regression(const std::vector<double>& x, const std::vector<double>& y,
                            double error_variance_ratio = 1.0);

} // namespace datamunge::stats
