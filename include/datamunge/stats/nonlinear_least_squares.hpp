#pragma once

/// @file nonlinear_least_squares.hpp
/// @brief Nonlinear weighted least squares with covariance and likelihood-ratio intervals.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/profile_likelihood.hpp>

#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief Profile a particular parameter using its own search scale and domain.
/// @par Example
/// @code
/// const dm::stats::NonlinearProfileRequest request{1, {.lower_bound = 0, .initial_step = .1}};
/// std::println("{}", request.parameter); // 1
/// @endcode
/// @see nonlinear_least_squares()
struct NonlinearProfileRequest {
    std::size_t parameter;               ///< Zero-based parameter index.
    ProfileLikelihoodOptions options{};  ///< Confidence level, bounds, and search settings for this parameter.
};

/// @brief Weights, derivatives, and controls for a damped least-squares solve.
/// @par Example
/// @code
/// const dm::stats::NonlinearLeastSquaresOptions options{.weights = {1, 2, 1}};
/// std::println("{}", options.weights.size()); // 3
/// @endcode
/// @see nonlinear_least_squares()
struct NonlinearLeastSquaresOptions {
    std::vector<double> weights{}; ///< Finite positive relative observation precisions. Empty uses equal precision.
    std::function<linalg::DenseMatrix<double>(const std::vector<double>&)> jacobian{}; ///< Optional derivative of the mean vector, observations by parameters.
    std::vector<NonlinearProfileRequest> profiles{}; ///< Profile requests in desired output order.
    std::size_t max_iterations{300};    ///< Trial budget for the main fit and each nuisance refit.
    double gradient_tolerance{1e-8};    ///< Target score in normalized residual and Jacobian-column units.
};

/// @brief Nonlinear regression fit and requested pointwise profile intervals.
/// @par Example
/// @code
/// const auto fit = dm::stats::nonlinear_least_squares({1, 2, 3}, {0}, [](const auto& b) {
///     return std::vector<double>(3, b[0]);
/// });
/// std::println("{:.0f}", fit.parameters[0]); // 2
/// @endcode
/// @see nonlinear_least_squares()
struct NonlinearLeastSquaresResult {
    std::vector<double> parameters;       ///< Fitted parameters in starting-vector order.
    linalg::DenseMatrix<double> covariance; ///< Linearized parameter covariance at the fit.
    std::vector<double> fitted;           ///< Mean vector at the fit.
    std::vector<double> residuals;        ///< Unweighted observed minus fitted values.
    double residual_sum_squares;          ///< Weighted residual sum of squares.
    double residual_variance;             ///< Weighted residual sum of squares divided by n minus p.
    std::vector<ProfileLikelihoodInterval> intervals; ///< In profile-request order.
};

/// @brief Fits a differentiable user-supplied mean function to a finite response.
///
/// The mean callback takes the parameter vector and returns one prediction
/// per observation. Use a start near the desired minimum. Damped Gauss–Newton
/// uses either the supplied Jacobian or finite differences. Inference
/// assumes independent Gaussian errors with common dispersion and the
/// specified relative precisions. Each requested interval fixes one
/// parameter, starts a new nuisance fit at the unrestricted estimates,
/// profiles out the dispersion, and inverts a chi-square likelihood ratio.
/// Bounds in a profile request constrain endpoint search for that parameter.
/// The optimizer itself is unconstrained. Return nonfinite means outside the
/// model domain to reject trial points, or reparameterize as needed.
/// On step stagnation the solver accepts a score within the floating-point
/// resolution floor of 8e-8 or the requested tolerance, whichever is larger.
/// @param response Observed responses, with more observations than parameters.
/// @param initial Starting parameters in model order.
/// @param mean Function mapping a parameter vector to mean predictions.
/// @param options Observation weights, optional Jacobian, profile requests, and optimizer controls.
/// @return Fit, covariance, and requested profile intervals.
/// @throws std::invalid_argument Invalid input, callback output shape, or numerical option.
/// @throws std::domain_error Invalid predictions or derivatives at the current iterate, or numerically zero SSE when profiling.
/// @throws std::runtime_error Iteration or factorization failure, or an unidentified parameter at the fit.
/// @throws std::overflow_error A reported quantity exceeds finite double range.
/// @par Example
/// @code
/// const auto fit = dm::stats::nonlinear_least_squares({2, 3, 4}, {1}, [](const auto& b) {
///     return std::vector<double>(3, b[0]);
/// }, {.profiles = {{0}}});
/// std::println("{:.1f}", fit.parameters[0]); // 3.0
/// @endcode
/// @par References
/// - Bates and Watts (1988), Nonlinear Regression Analysis and Its Applications.
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/nls.html
/// @see profile_likelihood_interval()
[[nodiscard]] NonlinearLeastSquaresResult nonlinear_least_squares(
    const std::vector<double>& response, const std::vector<double>& initial,
    const std::function<std::vector<double>(const std::vector<double>&)>& mean,
    const NonlinearLeastSquaresOptions& options = {});

} // namespace datamunge::stats
