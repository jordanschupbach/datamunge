#pragma once

/// @file partial_correlation.hpp
/// @brief Partial and semipartial (part) correlations from data or from a covariance matrix, with t tests,
/// Fisher-z intervals and optional bootstrap intervals.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief Correlation type used after adjustment.
/// @par Example
/// @code
/// const auto m = dm::stats::PartialMethod::spearman;
/// std::println("{}", m == dm::stats::PartialMethod::spearman); // true
/// @endcode
enum class PartialMethod {
    pearson,  ///< Linear (product-moment) correlation.
    spearman, ///< Pearson correlation of ranks (average ranks for ties), adjusting ranks for ranked covariates.
};

/// @brief Options for partial_correlation() and semipartial_correlation().
/// @par Example
/// @code
/// dm::stats::PartialOptions o{.bootstrap = 200};
/// std::println("{}", o.bootstrap); // 200
/// @endcode
struct PartialOptions {
    PartialMethod method{PartialMethod::pearson}; ///< Correlation type.
    double confidence{0.95};                     ///< Interval level.
    std::size_t bootstrap{0};                    ///< Percentile bootstrap replicates (0: none).
    std::uint64_t seed{1};                       ///< Bootstrap seed.
};

/// @brief An adjusted correlation with its test and intervals.
/// @par Example
/// @code
/// dm::stats::AdjustedCorrelation r{.estimate = 0.3};
/// std::println("{}", r.estimate); // 0.3
/// @endcode
struct AdjustedCorrelation {
    double estimate{};    ///< Partial or semipartial correlation.
    double statistic{};   ///< \f$t=r_p\sqrt{(n-2-k)/(1-r_p^2)}\f$ with the partial correlation \f$r_p\f$ (the coefficient test).
    double df{};          ///< n - 2 - k.
    double p_value{};     ///< Two-sided t-test p-value.
    double lower{};       ///< Fisher-z interval \f$\tanh(\operatorname{atanh}r\mp z/\sqrt{n-3-k})\f$ (partial only; NaN for semipartial).
    double upper{};       ///< Upper Fisher-z limit.
    double bootstrap_lower{std::numeric_limits<double>::quiet_NaN()}; ///< Percentile bootstrap lower limit.
    double bootstrap_upper{std::numeric_limits<double>::quiet_NaN()}; ///< Percentile bootstrap upper limit.
    std::size_t n{};          ///< Complete observations.
    std::size_t covariates{}; ///< Number k of covariates.
};

/// @brief Partial correlation of x and y given covariates Z: the correlation of the residuals of x and y after
/// least-squares regression of each on Z (with intercept).
///
/// Equivalent to \f$-P_{xy}/\sqrt{P_{xx}P_{yy}}\f$ for the inverse P of the joint covariance matrix
/// (`ppcor::pcor.test`). The t test and Fisher interval are exact / first-order under multivariate normality.
/// @param x First variable.
/// @param y Second variable.
/// @param covariates Covariate columns (each of length n; may be empty for the ordinary correlation).
/// @param options Options.
/// @return The estimate with test and intervals.
/// @throws std::invalid_argument On length mismatch, nonfinite values, n <= k + 3, or a variable with no
/// residual variation.
/// @par Example
/// @code
/// const std::vector<double> x{1, 3, 2, 5, 4, 6}, y{2, 1, 4, 3, 6, 5}, z{1, 2, 3, 4, 5, 6};
/// std::println("{:.3f}", dm::stats::partial_correlation(x, y, {z}).estimate); // -0.955
/// @endcode
AdjustedCorrelation partial_correlation(const std::vector<double>& x, const std::vector<double>& y,
                                        const std::vector<std::vector<double>>& covariates,
                                        const PartialOptions& options = {});

/// @brief Semipartial (part) correlation: x with the residual of y on Z (Z removed from y only).
///
/// Its square is the increase in \f$R^2\f$ when y is added to a regression of x on Z. Estimates equal
/// `ppcor::spcor.test`. The reported t test is the test of y's coefficient in that regression, which is the
/// partial-correlation t test; ppcor instead uses \f$sr\sqrt{(n-2-k)/(1-sr^2)}\f$, which is too conservative
/// whenever Z predicts x. No Fisher-z interval is given (lower/upper are NaN); request bootstrap replicates.
/// @param x Variable left unadjusted.
/// @param y Variable adjusted for Z.
/// @param covariates Covariate columns.
/// @param options Options.
/// @return The estimate with test and intervals.
/// @throws std::invalid_argument As for partial_correlation().
/// @par Example
/// @code
/// const std::vector<double> x{1, 3, 2, 5, 4, 6}, y{2, 1, 4, 3, 6, 5}, z{1, 2, 3, 4, 5, 6};
/// std::println("{:.3f}", dm::stats::semipartial_correlation(x, y, {z}).estimate); // -0.443
/// @endcode
AdjustedCorrelation semipartial_correlation(const std::vector<double>& x, const std::vector<double>& y,
                                            const std::vector<std::vector<double>>& covariates,
                                            const PartialOptions& options = {});

/// @brief All pairwise partial and semipartial correlations given the remaining variables.
/// @par Example
/// @code
/// dm::stats::PartialCorrelationMatrix m;
/// std::println("{}", m.n); // 0
/// @endcode
struct PartialCorrelationMatrix {
    linalg::DenseMatrix<double> partial;          ///< (i, j): partial correlation of i and j given all others.
    linalg::DenseMatrix<double> partial_p_value;  ///< t-test p-values (df n - p).
    linalg::DenseMatrix<double> semipartial;      ///< (i, j): x_i with x_j residualized on the others (ppcor::spcor).
    linalg::DenseMatrix<double> semipartial_p_value; ///< Coefficient t-test p-values (equal to the partial ones).
    std::size_t n{};                              ///< Sample size used for the tests.
};

/// @brief Partial and semipartial correlation matrices from a covariance (or correlation) matrix.
/// @param covariance p x p positive-definite covariance matrix (p >= 2).
/// @param n Sample size behind it (n > p) for the tests.
/// @return The matrices (diagonals 1, p-values 0 on the diagonal).
/// @throws std::invalid_argument If the matrix is not square, p < 2, not positive definite, or n <= p.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> r(3, 3, {1, .5, .5, .5, 1, .5, .5, .5, 1});
/// std::println("{:.3f}", dm::stats::partial_correlation_matrix(r, 50).partial(0, 1)); // 0.333
/// @endcode
PartialCorrelationMatrix partial_correlation_matrix(const linalg::DenseMatrix<double>& covariance, std::size_t n);

} // namespace datamunge::stats
