#pragma once

/// @file mmrm.hpp
/// @brief Gaussian repeated-measures likelihood with an unstructured covariance.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Optimization outcome for the unstructured covariance fit.
/// @par Example
/// @code
/// const auto status = dm::stats::MMRMStatus::Converged;
/// std::println("{}", status == dm::stats::MMRMStatus::Converged); // true
/// @endcode
enum class MMRMStatus {
    Converged,       ///< Covariance score reached the requested tolerance.
    IterationLimit,  ///< The best candidate reached the allowed update count.
    NumericalFailure ///< The best candidate failed the numerical line search.
};

/// @brief Options for repeated-measures maximum likelihood.
/// @par Example
/// @code
/// const dm::stats::MMRMOptions options{.reml = false};
/// std::println("{}", options.reml); // false
/// @endcode
struct MMRMOptions {
    bool reml{true};                 ///< Use restricted likelihood, or ML when false.
    std::size_t max_iterations{400}; ///< Accepted covariance updates per starting point.
    double tolerance{1e-9};          ///< Absolute covariance-score tolerance per observed row.
};

/// @brief Fixed effects and the unstructured covariance across visits.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_mmrm({0,1, 1,3, 2,2, 3,4}, {0,0,1,1,2,2,3,3}, {0,1,0,1,0,1,0,1});
/// std::println("{:.1f}", fit.coefficients[1]); // 2.5
/// @endcode
struct MMRMFit {
    std::vector<double> coefficients;              ///< Fixed effects in design order, or visit order for the means overload.
    linalg::DenseMatrix<double> covariance;        ///< Fixed-effect covariance conditional on the fitted visit covariance.
    std::vector<std::size_t> visit_levels;          ///< Ascending visit labels.
    linalg::DenseMatrix<double> visit_covariance; ///< Estimated within-subject covariance in visit order.
    std::vector<double> fitted;                    ///< Marginal fitted means at the observed rows.
    double log_likelihood;                        ///< Profiled Gaussian ML or restricted log likelihood.
    MMRMStatus status;                            ///< Numerical optimization outcome.

    /// @brief Predicts marginal means for supplied fixed-effect rows.
    /// @param design Finite rows with the fitted coefficient columns.
    /// @return Mean outcomes in query order.
    /// @throws std::invalid_argument Invalid shape or nonfinite query or coefficient.
    /// @throws std::overflow_error A predicted mean exceeds finite range.
    /// @par Example
    /// @code
    /// const auto fit = dm::stats::fit_mmrm({0,1, 1,3, 2,2, 3,4}, {0,0,1,1,2,2,3,3}, {0,1,0,1,0,1,0,1});
    /// std::println("{:.1f}", fit.predict(dm::linalg::DenseMatrix<double>(1,2,{1,0}))[0]); // 1.5
    /// @endcode
    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& design) const;
};

/// @brief Fits an unstructured Gaussian repeated-measures model with a supplied fixed design.
///
/// Each subject contributes the covariance submatrix for their observed
/// visits. Omit rows for missing outcomes. Visit labels can be nonconsecutive,
/// and rows can be in any order. Each visit pair needs joint observations
/// in at least one subject. A log-Cholesky parameterization keeps the visit
/// covariance positive definite while fixed effects are profiled by GLS.
/// The likelihood assumes independent subjects and ignorable missingness.
/// Check the returned status before using the estimates for inference.
/// @param design Finite full-rank fixed predictors, including any desired intercept or visit effects.
/// @param response Finite observed responses only.
/// @param subject Subject label for each observed row.
/// @param visit Visit label for each row, unique within a subject.
/// @param options ML or REML and the covariance optimizer controls.
/// @return Estimates, fitted means, and convergence status.
/// @throws std::invalid_argument Invalid shapes, nonidentifiable visits or means, or invalid controls.
/// @throws std::length_error A dense working matrix exceeds storage capacity.
/// @throws std::runtime_error No usable positive-definite fit was found.
/// @throws std::overflow_error A fitted result exceeds the finite reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(8,2,{1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1});
/// const auto fit = dm::stats::fit_mmrm(x, {0,1, 1,3, 2,2, 3,4}, {0,0,1,1,2,2,3,3}, {0,1,0,1,0,1,0,1});
/// std::println("{}", fit.visit_covariance.rows()); // 2
/// @endcode
/// @par References
/// - Jennrich and Schluchter (1986), https://doi.org/10.2307/2530695
[[nodiscard]] MMRMFit fit_mmrm(const linalg::DenseMatrix<double>& design, const std::vector<double>& response,
    const std::vector<std::size_t>& subject, const std::vector<std::size_t>& visit,
    const MMRMOptions& options = {});

/// @brief Fits one marginal mean per visit using the same unstructured covariance.
/// @param response Finite observed responses only.
/// @param subject Subject label for each observed row.
/// @param visit Visit label for each row, unique within a subject.
/// @param options ML or REML and the covariance optimizer controls.
/// @return Means in ascending visit order, and other fit fields as in the design overload.
/// @throws std::invalid_argument Invalid shapes, nonidentifiable visits or means, or invalid controls.
/// @throws std::length_error A dense working matrix exceeds storage capacity.
/// @throws std::runtime_error No usable positive-definite fit was found.
/// @throws std::overflow_error A fitted result exceeds the finite reporting range.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_mmrm({0,1, 1,3, 2,2, 3,4}, {0,0,1,1,2,2,3,3}, {0,1,0,1,0,1,0,1});
/// std::println("{:.1f}", fit.coefficients[0]); // 1.5
/// @endcode
[[nodiscard]] MMRMFit fit_mmrm(const std::vector<double>& response, const std::vector<std::size_t>& subject,
    const std::vector<std::size_t>& visit, const MMRMOptions& options = {});

} // namespace datamunge::stats
