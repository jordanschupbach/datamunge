#pragma once

/// @file diebold_mariano.hpp
/// @brief Diebold–Mariano test of equal predictive accuracy with the Harvey–Leybourne–Newbold correction.

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Long-run variance estimator for the mean loss differential.
/// @par Example
/// @code
/// const auto v = dm::stats::DmVariance::bartlett;
/// std::println("{}", v == dm::stats::DmVariance::acf); // false
/// @endcode
enum class DmVariance {
    acf,      ///< Truncated sum \f$\gamma_0+2\sum_{j=1}^{h-1}\gamma_j\f$ (the original test; can be negative).
    bartlett, ///< Bartlett weights \f$\gamma_0+2\sum_{j=1}^{h-1}(1-j/h)\gamma_j\f$ (always nonnegative).
};

/// @brief Options for the Diebold–Mariano test.
/// @par Example
/// @code
/// const dm::stats::DieboldMarianoOptions options{.horizon = 4};
/// std::println("{}", options.power); // 2
/// @endcode
struct DieboldMarianoOptions {
    std::size_t horizon{1};      ///< Forecast horizon h; errors of h-step forecasts are MA(h-1) under optimality.
    double power{2};             ///< Loss \f$|e|^{\text{power}}\f$ (ignored for diebold_mariano_test_differentials()).
    Alternative alternative{Alternative::TwoSided}; ///< Less: method 1 is more accurate (mean differential below 0).
    DmVariance variance{DmVariance::acf}; ///< Long-run variance estimator.
};

/// @brief Result of the Diebold–Mariano test.
/// @par Example
/// @code
/// const dm::stats::DieboldMarianoTest t;
/// std::println("{}", t.horizon_used); // 0
/// @endcode
struct DieboldMarianoTest {
    double statistic{};          ///< Corrected statistic \f$k\,\bar d/\sqrt{\hat V}\f$.
    double p_value{};            ///< From the t distribution with n - 1 degrees of freedom.
    double mean_differential{};  ///< \f$\bar d\f$, mean of loss 1 minus loss 2.
    double variance{};           ///< \f$\hat V\f$, estimated variance of \f$\bar d\f$.
    double correction{};         ///< Harvey–Leybourne–Newbold factor \f$k=\sqrt{(n+1-2h+h(h-1)/n)/n}\f$.
    std::size_t horizon_used{};  ///< Horizon after any fallback to 1 (when the acf variance is not positive).
    std::size_t observations{};  ///< n.
};

/// @brief Diebold–Mariano test of equal expected loss from two forecast-error series, as R's `forecast::dm.test`.
///
/// Forms \f$d_t=|e_{1t}|^p-|e_{2t}|^p\f$ and tests \f$E[d_t]=0\f$ with
/// \f$\hat V=\big(\hat\gamma_0+2\sum_{j<h}w_j\hat\gamma_j\big)/n\f$, using autocovariances with divisor n.
/// The statistic is scaled by Harvey, Leybourne and Newbold's small-sample factor and referred
/// to \f$t_{n-1}\f$. If the acf variance is not positive for h > 1, the test is repeated with
/// h = 1, as R does. The test compares forecasts, not models: with nested models estimated on
/// expanding windows its null distribution is not standard.
/// @param e1 Errors of method 1.
/// @param e2 Errors of method 2 (same length).
/// @param options Horizon, loss power, alternative and variance estimator.
/// @return Statistic, p-value and components.
/// @throws std::invalid_argument For mismatched or too short series (fewer than 3), nonfinite values,
/// h = 0 or h > n, a nonpositive power, or a zero variance at h = 1.
/// @par Example
/// @code
/// std::vector<double> e1, e2;
/// for (int t = 0; t < 100; ++t) {
///     const double common = std::sin(1.3 * t);
///     e1.push_back(common + .2 * std::cos(2.1 * t));
///     e2.push_back(1.5 * common);
/// }
/// const auto r = dm::stats::diebold_mariano_test(e1, e2);
/// std::println("{:.2f} {}", r.statistic, r.p_value < .001); // -12.44 true
/// @endcode
/// @par References
/// - F. X. Diebold and R. S. Mariano (1995). "Comparing predictive accuracy." Journal of Business
///   and Economic Statistics 13(3), 253-263. doi:10.1080/07350015.1995.10524599
/// - D. Harvey, S. Leybourne and P. Newbold (1997). "Testing the equality of prediction mean
///   squared errors." International Journal of Forecasting 13(2), 281-291.
///   doi:10.1016/S0169-2070(96)00719-4
/// @see diebold_mariano_test_differentials()
[[nodiscard]] DieboldMarianoTest diebold_mariano_test(const std::vector<double>& e1, const std::vector<double>& e2,
                                                      const DieboldMarianoOptions& options = {});

/// @brief Diebold–Mariano test on precomputed loss differentials \f$d_t=L(e_{1t})-L(e_{2t})\f$ for any loss.
/// @param d Loss differentials.
/// @param options Horizon, alternative and variance estimator (power unused).
/// @return Statistic, p-value and components.
/// @throws std::invalid_argument As diebold_mariano_test().
/// @par Example
/// @code
/// const auto r = dm::stats::diebold_mariano_test_differentials({.5, .2, .9, .4, .1, .7, .3, .6});
/// std::println("{:.3f}", r.mean_differential); // 0.462
/// @endcode
/// @see diebold_mariano_test()
[[nodiscard]] DieboldMarianoTest diebold_mariano_test_differentials(const std::vector<double>& d,
                                                                    const DieboldMarianoOptions& options = {});

} // namespace datamunge::stats
