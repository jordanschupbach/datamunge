#pragma once

/// @file ordinal_regression.hpp
/// @brief Proportional-odds regression and a parallel-slopes score test.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Numerical outcome of an ordinal logistic fit.
/// @par Example
/// @code
/// const auto status = dm::stats::OrdinalRegressionStatus::Converged;
/// std::println("{}", status == dm::stats::OrdinalRegressionStatus::Converged); // true
/// @endcode
enum class OrdinalRegressionStatus {
    Converged,       ///< The change in cumulative logits met tolerance.
    IterationLimit,  ///< The allowed number of updates was used.
    Separated,       ///< The solver found a numerical likelihood recession direction.
    NumericalFailure ///< The information lost rank or the line search stalled.
};

/// @brief Prior weights and optimizer controls for proportional odds.
/// @par Example
/// @code
/// const dm::stats::OrdinalRegressionOptions options{.max_iterations = 120};
/// std::println("{}", options.max_iterations); // 120
/// @endcode
struct OrdinalRegressionOptions {
    std::vector<double> weights{};    ///< Positive finite case weights. Empty means unit weights.
    std::size_t max_iterations{100}; ///< Maximum accepted Newton updates.
    double tolerance{1e-8};          ///< Absolute tolerance in cumulative log odds, in (0, 1).
    bool test_parallel_slopes{true}; ///< Compute the score test when it is estimable.
};

/// @brief Rao score test against threshold-specific slopes.
/// @par Example
/// @code
/// const dm::stats::ParallelSlopesTest test{4, 2, .135335283};
/// std::println("{}", test.degrees_of_freedom); // 2
/// @endcode
struct ParallelSlopesTest {
    double statistic;                ///< Efficient score quadratic form.
    std::size_t degrees_of_freedom;  ///< Number of added slope parameters.
    double p_value;                  ///< Upper chi-square tail probability.
};

/// @brief Ordinal coefficients, ordered thresholds, and fitted category probabilities.
/// @par Example
/// @code
/// const auto fit = dm::stats::ordinal_regression(dm::linalg::DenseMatrix<double>(4, 0), {0, 1, 1, 2});
/// std::println("{:.2f}", fit.fitted_probabilities(0, 1)); // 0.50
/// @endcode
struct OrdinalRegressionResult {
    std::vector<std::int64_t> levels;           ///< Increasing outcome labels.
    std::vector<double> cutpoints;              ///< Increasing threshold intercepts.
    std::vector<double> coefficients;          ///< Common slopes, in design-column order.
    linalg::DenseMatrix<double> covariance;     ///< Cutpoints first, then slopes. Empty unless converged.
    linalg::DenseMatrix<double> fitted_probabilities; ///< Rows in input order and columns in level order.
    double log_likelihood;                     ///< Weighted log likelihood at the returned iterate.
    std::size_t iterations;                    ///< Accepted Newton updates.
    OrdinalRegressionStatus status;            ///< Numerical stopping reason.
    std::optional<ParallelSlopesTest> parallel_slopes; ///< Absent if disabled, inapplicable, or numerically inestimable.

    /// @brief Predicts category probabilities in outcome-level order.
    /// @param design Finite predictors with the same columns as the fitted design.
    /// @return Probability rows with one column per outcome level.
    /// @throws std::invalid_argument Incompatible design or invalid coefficients or cutpoints.
    /// @throws std::overflow_error A query logit exceeds finite working precision.
    /// @par Example
    /// @code
    /// const auto fit = dm::stats::ordinal_regression(dm::linalg::DenseMatrix<double>(4, 0), {0, 1, 1, 2});
    /// std::println("{:.2f}", fit.predict(dm::linalg::DenseMatrix<double>(1, 0))(0, 0)); // 0.25
    /// @endcode
    [[nodiscard]] linalg::DenseMatrix<double> predict(const linalg::DenseMatrix<double>& design) const;
};

/// @brief Fits cumulative logits of the form cutpoint minus design times slope.
///
/// The model estimates strictly ordered cutpoints and a shared slope vector.
/// An optional Rao score test uses expected information at the proportional-odds
/// fit to test all threshold-specific slope deviations jointly. The test is
/// asymptotic and requires at least three outcome levels and one predictor.
/// @param design Finite predictor matrix without a constant column. A zero-column matrix fits only cutpoints.
/// @param labels Ordered outcomes, with increasing integers denoting higher categories.
/// @param options Case weights and numerical controls.
/// @return Ordered fit and probabilities, with covariance and an optional test for converged fits.
/// @throws std::invalid_argument Invalid input, fewer than two observed levels, or aliased design.
/// @throws std::length_error Information matrix exceeds representable capacity.
/// @throws std::overflow_error A result or threshold spacing cannot be represented in double precision.
/// @par Example
/// @code
/// const auto fit = dm::stats::ordinal_regression(dm::linalg::DenseMatrix<double>(4, 0), {0, 1, 1, 2});
/// std::println("{:.3f}", fit.cutpoints.front()); // -1.099
/// @endcode
/// @par References
/// - McCullagh (1980), https://doi.org/10.1111/j.2517-6161.1980.tb01109.x
/// - Peterson and Harrell (1990), https://doi.org/10.2307/2347760
[[nodiscard]] OrdinalRegressionResult ordinal_regression(
    const linalg::DenseMatrix<double>& design, const std::vector<std::int64_t>& labels,
    const OrdinalRegressionOptions& options = {});

} // namespace datamunge::stats
