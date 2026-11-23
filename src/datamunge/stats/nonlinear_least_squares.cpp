/// @file nonlinear_least_squares.cpp
/// @brief Damped least squares, linearized covariance, and nuisance refits.

#include <datamunge/stats/nonlinear_least_squares.hpp>

#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Working-precision optimization and inference for one nonlinear fit.
namespace {

/// @brief A dense matrix in working precision.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief A user-supplied mean function evaluated on the observed predictors.
using MeanFunction = std::function<std::vector<double>(const std::vector<double>&)>;

/// @brief Values and scaled weighted residuals at one parameter vector.
struct Evaluation {
    std::vector<double> mean;            ///< Fitted mean at each observation.
    std::vector<long double> residuals;  ///< Residuals in normalized weighted coordinates.
    long double ss;                      ///< Sum of squares, infinite for an invalid trial.
};

/// @brief Weighted response and reusable callback state.
struct Problem {
    const std::vector<double>& response;          ///< Observations to fit.
    const MeanFunction& mean;                     ///< Mean callback, borrowed for the fit.
    const NonlinearLeastSquaresOptions& options;  ///< Derivatives and stopping controls.
    std::vector<long double> weights;             ///< Square roots of relative weights divided by maximum weight.
    long double response_scale;                   ///< Common residual scale.

    /// @brief Evaluates the callback, allowing nonfinite trial predictions.
    std::vector<double> predictions(const std::vector<double>& parameters) const {
        auto result = mean(parameters);
        if (result.size() != response.size())
            throw std::invalid_argument("nonlinear_least_squares: mean size differs from response size");
        return result;
    }

    /// @brief Evaluates a trial objective and retains the predictions and working residuals.
    Evaluation evaluate(const std::vector<double>& parameters) const {
        Evaluation result{predictions(parameters), std::vector<long double>(response.size()), 0};
        for (std::size_t i = 0; i < response.size(); ++i) {
            if (!std::isfinite(result.mean[i])) {
                result.ss = std::numeric_limits<long double>::infinity();
                break;
            }
            const auto residual = static_cast<long double>(response[i]) - result.mean[i];
            result.residuals[i] = (residual / response_scale) * weights[i];
            result.ss += result.residuals[i] * result.residuals[i];
        }
        return result;
    }

    /// @brief Builds only the derivative columns for free parameters.
    Matrix jacobian(const std::vector<double>& parameters, const std::vector<double>& center,
                    const std::vector<std::size_t>& free) const {
        Matrix result(response.size(), free.size());
        if (options.jacobian) {
            const auto supplied = options.jacobian(parameters);
            if (supplied.rows() != response.size() || supplied.cols() != parameters.size())
                throw std::invalid_argument("nonlinear_least_squares: Jacobian shape differs from model shape");
            for (std::size_t i = 0; i < response.size(); ++i)
                for (std::size_t j = 0; j < free.size(); ++j)
                    result(i, j) = (static_cast<long double>(supplied(i, free[j])) / response_scale) * weights[i];
        } else {
            auto perturbed = parameters;
            const double relative_step = std::cbrt(std::numeric_limits<double>::epsilon());
            for (std::size_t j = 0; j < free.size(); ++j) {
                const auto column = free[j];
                const double base = parameters[column];
                const auto step = relative_step * std::max(1.0, std::abs(base));
                const auto high = base + step;
                const auto low = base - step;
                perturbed[column] = std::isfinite(high) ? high : base;
                const auto plus = std::isfinite(high) ? predictions(perturbed) : center;
                perturbed[column] = std::isfinite(low) ? low : base;
                const auto minus = std::isfinite(low) ? predictions(perturbed) : center;
                perturbed[column] = base;
                for (std::size_t i = 0; i < response.size(); ++i) {
                    const bool plus_ok = std::isfinite(high) && std::isfinite(plus[i]);
                    const bool minus_ok = std::isfinite(low) && std::isfinite(minus[i]);
                    if (!plus_ok && !minus_ok)
                        throw std::domain_error("nonlinear_least_squares: cannot evaluate a derivative at the iterate");
                    const auto delta_y = static_cast<long double>(plus_ok ? plus[i] : center[i]) -
                        (minus_ok ? minus[i] : center[i]);
                    const auto delta_x = static_cast<long double>(plus_ok ? high : base) - (minus_ok ? low : base);
                    result(i, j) = (delta_y / response_scale) * weights[i] / delta_x;
                }
            }
        }
        if (!std::ranges::all_of(result.data(), [](long double x) { return std::isfinite(x); }))
            throw std::domain_error("nonlinear_least_squares: nonfinite derivative at the iterate");
        return result;
    }
};

/// @brief Normalizes derivative columns and returns their undo scales.
std::vector<long double> scale_jacobian(Matrix& jacobian) {
    std::vector<long double> scales(jacobian.cols());
    for (std::size_t j = 0; j < jacobian.cols(); ++j) {
        for (std::size_t i = 0; i < jacobian.rows(); ++i)
            scales[j] = std::hypot(scales[j], jacobian(i, j));
        if (scales[j] == 0)
            scales[j] = 1;
        for (std::size_t i = 0; i < jacobian.rows(); ++i)
            jacobian(i, j) /= scales[j];
    }
    return scales;
}

/// @brief Largest residual projection onto any unit-norm Jacobian column.
long double normalized_score(const Matrix& jacobian, const Evaluation& current) {
    long double score = 0;
    for (std::size_t j = 0; j < jacobian.cols(); ++j) {
        long double dot = 0;
        for (std::size_t i = 0; i < jacobian.rows(); ++i)
            dot += jacobian(i, j) * current.residuals[i];
        score = std::max(score, std::abs(dot));
    }
    const auto norm = std::sqrt(current.ss);
    const auto floor = 16 * std::numeric_limits<double>::epsilon() * std::sqrt(static_cast<long double>(jacobian.rows()));
    return norm <= floor ? 0 : score / norm;
}

/// @brief A local least-squares solution, possibly with one fixed parameter.
struct Solution {
    std::vector<double> parameters; ///< Final full parameter vector.
    Evaluation evaluation;         ///< Objective and predictions at the final vector.
};

/// @brief Solves in scaled SVD coordinates and increases damping on unsuccessful trials.
Solution fit(const Problem& problem, std::vector<double> parameters, const std::vector<std::size_t>& free) {
    auto current = problem.evaluate(parameters);
    if (!std::isfinite(current.ss))
        throw std::domain_error("nonlinear_least_squares: invalid mean at the current parameters");
    if (free.empty())
        return {std::move(parameters), std::move(current)};
    long double damping = 1e-3L;
    std::size_t trials = 0;
    while (trials < problem.options.max_iterations) {
        auto jacobian = problem.jacobian(parameters, current.mean, free);
        const auto scales = scale_jacobian(jacobian);
        const auto score = normalized_score(jacobian, current);
        if (score <= problem.options.gradient_tolerance)
            return {std::move(parameters), std::move(current)};
        const auto factor = linalg::svd(jacobian);
        if (!factor.converged)
            throw std::runtime_error("nonlinear_least_squares: Jacobian SVD reached its sweep limit");
        std::vector<long double> components(free.size());
        for (std::size_t j = 0; j < free.size(); ++j)
            for (std::size_t i = 0; i < current.residuals.size(); ++i)
                components[j] += factor.U(i, j) * current.residuals[i];
        for (; trials < problem.options.max_iterations;) {
            ++trials;
            auto candidate = parameters;
            std::vector<long double> damped(free.size());
            for (std::size_t j = 0; j < free.size(); ++j)
                damped[j] = factor.s[j] * components[j] / (factor.s[j] * factor.s[j] + damping);
            bool valid = true;
            for (std::size_t j = 0; j < free.size(); ++j) {
                long double shift = 0;
                for (std::size_t k = 0; k < free.size(); ++k)
                    shift += factor.V(j, k) * damped[k];
                const auto point = parameters[free[j]] + shift / scales[j];
                if (std::isfinite(point) && std::abs(point) <= std::numeric_limits<double>::max())
                    candidate[free[j]] = static_cast<double>(point);
                else
                    valid = false;
            }
            if (valid && candidate != parameters) {
                auto next = problem.evaluate(candidate);
                if (next.ss < current.ss) {
                    parameters = std::move(candidate);
                    current = std::move(next);
                    damping = std::max(damping * .3L, 1e-18L);
                    break;
                }
            }
            damping *= 10;
            if (!std::isfinite(damping) || (valid && candidate == parameters)) {
                if (score <= std::max<long double>(problem.options.gradient_tolerance, 8e-8L))
                    return {std::move(parameters), std::move(current)};
                throw std::runtime_error("nonlinear_least_squares: the solver stalled before reaching stationarity");
            }
        }
    }
    auto jacobian = problem.jacobian(parameters, current.mean, free);
    scale_jacobian(jacobian);
    if (normalized_score(jacobian, current) <= problem.options.gradient_tolerance)
        return {std::move(parameters), std::move(current)};
    throw std::runtime_error("nonlinear_least_squares: iteration budget exhausted");
}

/// @brief Narrows one finite fit or inference quantity.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("nonlinear_least_squares: result exceeds finite double range");
    return static_cast<double>(value);
}

} // namespace

NonlinearLeastSquaresResult nonlinear_least_squares(const std::vector<double>& response,
    const std::vector<double>& initial,
    const std::function<std::vector<double>(const std::vector<double>&)>& mean,
    const NonlinearLeastSquaresOptions& options) {
    const auto n = response.size();
    const auto p = initial.size();
    const auto finite = [](double x) { return std::isfinite(x); };
    const auto positive = [&](double x) { return x > 0 && finite(x); };
    if (p == 0 || n <= p || !mean || !std::ranges::all_of(response, finite) ||
        !std::ranges::all_of(initial, finite) || (!options.weights.empty() && options.weights.size() != n) ||
        !std::ranges::all_of(options.weights, positive) || options.max_iterations == 0 ||
        !positive(options.gradient_tolerance) || options.gradient_tolerance >= 1 ||
        std::ranges::any_of(options.profiles, [p](const auto& request) { return request.parameter >= p; }))
        throw std::invalid_argument("nonlinear_least_squares: invalid response, parameters, weights, or controls");
    const double max_weight = options.weights.empty() ? 1 : std::ranges::max(options.weights);
    Problem problem{response, mean, options, std::vector<long double>(n, 1), 0};
    for (std::size_t i = 0; i < n; ++i) {
        problem.response_scale = std::max(problem.response_scale, std::abs(static_cast<long double>(response[i])));
        if (!options.weights.empty())
            problem.weights[i] = std::sqrt(static_cast<long double>(options.weights[i]) / max_weight);
    }
    if (problem.response_scale == 0)
        problem.response_scale = 1;
    std::vector<std::size_t> columns(p);
    std::iota(columns.begin(), columns.end(), 0u);
    const auto solution = fit(problem, initial, columns);
    auto jacobian = problem.jacobian(solution.parameters, solution.evaluation.mean, columns);
    const auto scales = scale_jacobian(jacobian);
    const auto factor = linalg::svd(jacobian);
    const auto cutoff = static_cast<long double>(n) * std::numeric_limits<double>::epsilon();
    if (factor.rank(cutoff) != p || !factor.converged)
        throw std::runtime_error("nonlinear_least_squares: Jacobian does not identify all parameters at the fit");
    const auto scaled_variance = solution.evaluation.ss / (n - p);
    const auto sum_sq = solution.evaluation.ss * problem.response_scale * problem.response_scale * max_weight;
    NonlinearLeastSquaresResult result{solution.parameters, linalg::DenseMatrix<double>(p, p),
        solution.evaluation.mean, {}, output(sum_sq), output(sum_sq / (n - p)), {}};
    for (std::size_t i = 0; i < n; ++i)
        result.residuals.push_back(output(static_cast<long double>(response[i]) - result.fitted[i]));
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            long double entry = 0;
            for (std::size_t k = 0; k < p; ++k)
                entry += (factor.V(i, k) / factor.s[k]) * (factor.V(j, k) / factor.s[k]);
            result.covariance(i, j) = result.covariance(j, i) = output(entry * scaled_variance / scales[i] / scales[j]);
        }
    for (const auto& request : options.profiles) {
        const auto baseline = solution.evaluation.ss;
        const auto floor = 16 * std::numeric_limits<double>::epsilon();
        if (baseline <= n * floor * floor)
            throw std::domain_error("nonlinear_least_squares: positive residual dispersion is needed for profiling");
        const auto parameter = request.parameter;
        auto nuisance = columns;
        nuisance.erase(nuisance.begin() + parameter);
        const auto estimate = result.parameters[parameter];
        const auto profile = [&](double fixed) {
            if (fixed == estimate)
                return 0.0;
            auto start = result.parameters;
            start[parameter] = fixed;
            const auto restricted_ss = nuisance.empty() ? problem.evaluate(start).ss
                : fit(problem, std::move(start), nuisance).evaluation.ss;
            const auto relative_change = (restricted_ss - baseline) / baseline;
            if (relative_change < 0 && relative_change >= -64 * std::numeric_limits<double>::epsilon())
                return 0.0;
            return static_cast<double>(-.5L * n * std::log1p(relative_change));
        };
        result.intervals.push_back(profile_likelihood_interval(estimate, profile, request.options));
    }
    return result;
}

} // namespace datamunge::stats
