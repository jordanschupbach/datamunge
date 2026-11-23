#pragma once

/// @file multinomial_logistic.hpp
/// @brief Baseline-category logistic regression and joint coefficient inference.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Numerical outcome of a multinomial logistic fit.
/// @par Example
/// @code
/// const auto status = dm::stats::MultinomialLogisticStatus::Converged;
/// std::println("{}", status == dm::stats::MultinomialLogisticStatus::Converged); // true
/// @endcode
/// @see multinomial_logistic()
enum class MultinomialLogisticStatus {
    Converged,       ///< Newton step reached the requested tolerance.
    IterationLimit,  ///< The allowed number of updates was used.
    Separated,       ///< A numerical likelihood recession direction was found.
    NumericalFailure ///< Information lost numerical rank or the line search stalled.
};

/// @brief Baseline, case weights, and convergence settings.
/// @par Example
/// @code
/// const dm::stats::MultinomialLogisticOptions options{.baseline = 2};
/// std::println("{}", *options.baseline); // 2
/// @endcode
/// @see multinomial_logistic()
struct MultinomialLogisticOptions {
    std::optional<std::int64_t> baseline{}; ///< Baseline label. Defaults to the smallest class label.
    std::vector<double> weights{};         ///< Positive finite case weights. Empty uses unit weights.
    std::size_t max_iterations{100};       ///< Maximum accepted Newton updates.
    double tolerance{1e-8};                ///< Absolute Newton step tolerance in baseline log odds, in (0, 1).
};

/// @brief A multinomial logistic fit with probabilities and a joint covariance matrix.
/// @par Example
/// @code
/// const auto fit = dm::stats::multinomial_logistic(dm::linalg::DenseMatrix<double>(6, 1, 1),
///     std::vector<std::int64_t>{1, 1, 2, 2, 2, 3});
/// std::println("{:.2f}", fit.fitted_probabilities(0, 1)); // 0.50
/// @endcode
/// @see multinomial_logistic()
struct MultinomialLogisticResult {
    std::vector<std::int64_t> classes;       ///< Class labels in increasing order, including the baseline.
    std::int64_t baseline;                  ///< Label whose entire coefficient row is zero by construction.
    linalg::DenseMatrix<double> coefficients; ///< One row per nonbaseline class in class order, columns in design order.
    linalg::DenseMatrix<double> covariance;   ///< Joint covariance in row-major coefficient order. Empty unless converged.
    linalg::DenseMatrix<double> fitted_probabilities; ///< One row per observation, columns in class order.
    double log_likelihood;                   ///< Weighted log-likelihood kernel at the returned coefficients.
    std::size_t iterations;                   ///< Accepted updates.
    MultinomialLogisticStatus status;         ///< Diagnostic for the returned fit.

    /// @brief Predicts multinomial probabilities for new design rows.
    /// @param design Finite design with the fitted columns, including any supplied intercept.
    /// @return Probability rows with columns in classes order.
    /// @throws std::invalid_argument Incompatible design, class or coefficient shape, or nonfinite input.
    /// @throws std::overflow_error A linear predictor exceeds finite working range.
    /// @par Example
    /// @code
    /// const auto fit = dm::stats::multinomial_logistic(dm::linalg::DenseMatrix<double>(4, 1, 1),
    ///     std::vector<std::int64_t>{0, 1, 1, 1});
    /// std::println("{:.2f}", fit.predict(dm::linalg::DenseMatrix<double>(1, 1, 1))(0, 1)); // 0.75
    /// @endcode
    [[nodiscard]] linalg::DenseMatrix<double> predict(const linalg::DenseMatrix<double>& design) const;
};

/// @brief Fits baseline-category log odds for individual class labels.
///
/// The model uses a softmax with a zero logit for the selected baseline.
/// A line search safeguards Newton steps. The covariance is the inverse
/// joint information, including cross-class blocks. A fit that stops
/// without convergence returns the last finite iterate and a diagnostic.
/// @param design Full-rank finite design. Include any desired intercept column.
/// @param labels One integer class label per row, with at least two different labels.
/// @param options Baseline, case weights, and optimizer controls.
/// @return Class-aligned coefficients and probabilities, with covariance when converged.
/// @throws std::invalid_argument Incompatible dimensions, baseline, weights, controls, or rank.
/// @throws std::length_error Information matrix dimensions exceed capacity.
/// @throws std::overflow_error A result exceeds finite double range.
/// @par Example
/// @code
/// const auto fit = dm::stats::multinomial_logistic(dm::linalg::DenseMatrix<double>(6, 1, 1),
///     std::vector<std::int64_t>{1, 1, 2, 2, 2, 3}, {.baseline = 2});
/// std::println("{}", fit.baseline); // 2
/// @endcode
/// @par References
/// - Venables and Ripley (2002), Modern Applied Statistics with S.
/// - R nnet, https://stat.ethz.ch/R-manual/R-devel/library/nnet/html/multinom.html
[[nodiscard]] MultinomialLogisticResult multinomial_logistic(
    const linalg::DenseMatrix<double>& design, const std::vector<std::int64_t>& labels,
    const MultinomialLogisticOptions& options = {});

/// @brief Fits baseline-category log odds for rows of multinomial counts.
///
/// Count columns are labeled from zero. Probabilities and covariance use
/// the same ordering as the individual-label overload. A zero-total row
/// contributes no likelihood. The reported log-likelihood kernel omits
/// multinomial combinatorial constants.
/// @param design Full-rank finite design among the rows with positive totals.
/// @param counts Nonnegative integer counts, with positive aggregate count in each class.
/// @param options Baseline column label, case weights, and optimizer controls.
/// @return Fit, with one predicted probability per row and class.
/// @throws std::invalid_argument Invalid inputs, counts, missing classes, or rank.
/// @throws std::length_error Information matrix dimensions exceed capacity.
/// @throws std::overflow_error A result exceeds finite double range.
/// @par Example
/// @code
/// const auto fit = dm::stats::multinomial_logistic(dm::linalg::DenseMatrix<double>(1, 1, 1),
///     dm::linalg::DenseMatrix<double>(1, 3, {2, 3, 1}));
/// std::println("{:.2f}", fit.fitted_probabilities(0, 1)); // 0.50
/// @endcode
[[nodiscard]] MultinomialLogisticResult multinomial_logistic(
    const linalg::DenseMatrix<double>& design, const linalg::DenseMatrix<double>& counts,
    const MultinomialLogisticOptions& options = {});

} // namespace datamunge::stats
