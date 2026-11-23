#pragma once

/// @file conditional_logistic.hpp
/// @brief Exact logistic likelihood conditional on the number of cases per stratum.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Stopping diagnostic for exact conditional logistic regression.
/// @par Example
/// @code
/// const auto status = dm::stats::ConditionalLogisticStatus::Converged;
/// std::println("{}", status == dm::stats::ConditionalLogisticStatus::Converged); // true
/// @endcode
/// @see conditional_logistic()
enum class ConditionalLogisticStatus {
    Converged,       ///< The Newton step met tolerance.
    IterationLimit,  ///< The fit used all allowed updates without reaching tolerance.
    Separated,       ///< A numerical likelihood recession direction was found.
    NumericalFailure ///< Information lost numerical rank or the line search stalled.
};

/// @brief Iteration controls for maximizing the conditional likelihood.
/// @par Example
/// @code
/// const dm::stats::ConditionalLogisticOptions options{.max_iterations = 150};
/// std::println("{}", options.max_iterations); // 150
/// @endcode
/// @see conditional_logistic()
struct ConditionalLogisticOptions {
    std::size_t max_iterations{100}; ///< Maximum number of accepted Newton updates.
    double tolerance{1e-8};          ///< Absolute Newton step tolerance in within-stratum log odds, in (0, 1).
};

/// @brief Fitted conditional coefficients and matched-set probabilities.
/// @par Example
/// @code
/// const auto fit = dm::stats::conditional_logistic(dm::linalg::DenseMatrix<double>(4, 1, {0, 1, 0, 1}),
///                                                  {0, 1, 1, 0}, {10, 10, 20, 20});
/// std::println("{:.1f}", fit.fitted_probabilities[0]); // 0.5
/// @endcode
/// @see conditional_logistic()
struct ConditionalLogisticResult {
    std::vector<double> coefficients;             ///< Within-stratum log odds coefficients in design-column order.
    linalg::DenseMatrix<double> covariance;       ///< Inverse conditional information when converged, empty otherwise.
    std::vector<double> fitted_probabilities;     ///< Conditional probability of being a case, in input row order.
    double log_likelihood;                        ///< Conditional log likelihood at the returned coefficients.
    double null_log_likelihood;                   ///< Conditional log likelihood with all coefficients zero.
    std::size_t informative_strata;              ///< Number of strata containing both cases and controls.
    std::size_t iterations;                      ///< Accepted Newton updates.
    ConditionalLogisticStatus status;            ///< Numerical stopping diagnostic.
};

/// @brief Fits logistic regression with stratum intercepts eliminated by conditioning.
///
/// Each stratum contribution conditions on its observed case total. A
/// log-space recursion evaluates the exact partition function, its score, and
/// its information for any number of cases per stratum. Strata containing
/// only cases or only controls contribute no information. Supply only
/// predictors that are identifiable from comparisons within mixed strata,
/// with no intercept column. Fitted probabilities condition on the fixed
/// case total in each stratum and therefore sum to that total.
/// A stratum of size n with m members in its smaller outcome class takes
/// O(n * m * p^2) work per likelihood evaluation for p predictors.
/// On an unsuccessful fit, the coefficients and probabilities describe
/// the last finite iterate.
/// @param predictors Finite predictor rows without an intercept.
/// @param cases Zero for a control and one for a case, in input row order.
/// @param strata One stratum label per row. Equal labels may occur in any row order.
/// @param options Positive numerical stopping controls.
/// @return Conditional coefficients, fit summary and probabilities. Check status before inference.
/// @throws std::invalid_argument Invalid inputs, no mixed strata, or predictors aliased within strata.
/// @throws std::length_error A forward table cannot be represented in memory.
/// @throws std::overflow_error A result exceeds finite double range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6, 1, {0, 1, 0, 1, 0, 1});
/// const auto fit = dm::stats::conditional_logistic(x, {0, 1, 0, 1, 1, 0}, {1, 1, 2, 2, 3, 3});
/// std::println("{:.3f}", fit.coefficients[0]); // 0.693
/// @endcode
/// @par References
/// - Gail, Lubin and Rubinstein (1981), https://doi.org/10.1093/biomet/68.3.703
/// - R survival, https://cran.r-project.org/web/packages/survival/refman/survival.html#clogit
[[nodiscard]] ConditionalLogisticResult conditional_logistic(
    const linalg::DenseMatrix<double>& predictors, const std::vector<double>& cases,
    const std::vector<std::int64_t>& strata, const ConditionalLogisticOptions& options = {});

} // namespace datamunge::stats
