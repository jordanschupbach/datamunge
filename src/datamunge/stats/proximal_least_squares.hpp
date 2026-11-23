#pragma once

/// @file proximal_least_squares.hpp
/// @brief Shared preparation and accelerated proximal steps for penalized least squares.

#include <datamunge/linalg/dense_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats::detail {

/// @brief Extended-precision vector for penalized least squares.
using ProximalVector = std::vector<long double>;

/// @brief Weighted least-squares data and the transform back to original units.
struct PenalizedLeastSquares {
    linalg::DenseMatrix<long double> x; ///< Centered, optionally scaled predictors times square-root weights.
    ProximalVector y;                   ///< Centered responses times square-root weights.
    ProximalVector centers;             ///< Weighted column means before preprocessing.
    ProximalVector scales;              ///< Weighted RMS scales, or one for unscaled or constant columns.
    long double intercept;              ///< Weighted mean of the original response.
    long double curvature;              ///< Largest squared working column norm, or zero for constant design.
};

/// @brief Smooth quadratic loss and gradient in working units.
struct QuadraticEvaluation {
    long double loss;           ///< Half the weighted mean squared error.
    ProximalVector gradient;    ///< Derivative with respect to working slopes.
};

/// @brief Final evaluation and stopping information from a proximal fit.
struct ProximalSolution {
    QuadraticEvaluation state; ///< Loss and gradient at the returned slopes.
    std::size_t iterations;    ///< Number of accepted updates.
    bool converged;            ///< Whether the caller's stopping criterion was met.
};

/// @brief Converts a finite working scalar to public precision.
inline double proximal_output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("penalized least squares: value exceeds the finite reporting range");
    return static_cast<double>(value);
}

/// @brief Applies the scalar proximal operation for an absolute-value penalty.
inline long double soft_threshold(long double value, long double threshold) {
    return std::copysign(std::max(std::abs(value) - threshold, 0.0L), value);
}

/// @brief Centers a validated design and response and applies normalized case weights.
inline PenalizedLeastSquares prepare_penalized_least_squares(const linalg::DenseMatrix<double>& design,
    const std::vector<double>& response, const std::vector<double>& case_weights, bool standardize) {
    const auto n = design.rows(), d = design.cols();
    PenalizedLeastSquares result{linalg::DenseMatrix<long double>(n, d), ProximalVector(n), ProximalVector(d),
                                  ProximalVector(d, 1), 0, 0};
    ProximalVector weights(n, 1);
    if (!case_weights.empty()) {
        const long double largest = std::ranges::max(case_weights);
        for (std::size_t i = 0; i < n; ++i)
            weights[i] = case_weights[i] / largest;
    }
    const auto total = std::accumulate(weights.begin(), weights.end(), 0.0L);
    const long double response_origin = response.front();
    for (std::size_t i = 0; i < n; ++i) {
        weights[i] /= total;
        result.intercept += weights[i] * (response[i] - response_origin);
        for (std::size_t j = 0; j < d; ++j)
            result.centers[j] += weights[i] * (static_cast<long double>(design(i, j)) - design(0, j));
    }
    for (std::size_t i = 0; i < n; ++i) {
        weights[i] = std::sqrt(weights[i]);
        result.y[i] = weights[i] * ((response[i] - response_origin) - result.intercept);
    }
    result.intercept += response_origin;
    for (std::size_t j = 0; j < d; ++j) {
        const long double origin = design(0, j);
        long double norm = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto value = weights[i] * ((design(i, j) - origin) - result.centers[j]);
            result.x(i, j) = value;
            norm = std::hypot(norm, value);
        }
        result.centers[j] += origin;
        if (standardize && norm > 0) {
            result.scales[j] = norm;
            for (std::size_t i = 0; i < n; ++i)
                result.x(i, j) /= norm;
            result.curvature = std::max(result.curvature, 1.0L);
        } else {
            result.curvature = std::max(result.curvature, norm * norm);
        }
    }
    if (!std::isfinite(result.curvature))
        throw std::overflow_error("penalized least squares: predictor norm exceeds the working range");
    return result;
}

/// @brief Evaluates the smooth objective without explicitly forming a feature Gram matrix.
inline QuadraticEvaluation evaluate_quadratic(const PenalizedLeastSquares& problem, const ProximalVector& beta) {
    const auto d = beta.size();
    QuadraticEvaluation result{0, ProximalVector(d)};
    for (std::size_t i = 0; i < problem.x.rows(); ++i) {
        long double residual = -problem.y[i];
        for (std::size_t j = 0; j < d; ++j)
            residual += problem.x(i, j) * beta[j];
        result.loss += residual * residual / 2;
        for (std::size_t j = 0; j < d; ++j)
            result.gradient[j] += problem.x(i, j) * residual;
    }
    if (!std::isfinite(result.loss) ||
        !std::ranges::all_of(result.gradient, [](long double value) { return std::isfinite(value); }))
        throw std::overflow_error("penalized least squares: loss or gradient exceeds the working range");
    return result;
}

/// @brief Returns one valid proximal step by curvature backtracking for the quadratic loss.
/// @tparam Prox Callback that applies the proximal operator in place at the supplied curvature.
template <typename Prox>
ProximalVector proximal_step(const PenalizedLeastSquares& problem, const ProximalVector& origin,
    const ProximalVector& gradient, long double& curvature, const Prox& prox) {
    for (int trial = 0; trial < 128 && std::isfinite(curvature); ++trial, curvature *= 2) {
        ProximalVector candidate(origin.size()), delta(origin.size());
        for (std::size_t j = 0; j < origin.size(); ++j)
            candidate[j] = origin[j] - gradient[j] / curvature;
        prox(candidate, curvature);
        long double distance = 0;
        for (std::size_t j = 0; j < origin.size(); ++j) {
            delta[j] = candidate[j] - origin[j];
            distance += delta[j] * delta[j];
        }
        const auto change = problem.x * delta;
        const auto actual = std::inner_product(change.begin(), change.end(), change.begin(), 0.0L);
        if (std::isfinite(actual) && actual <= curvature * distance *
            (1 + 64 * std::numeric_limits<long double>::epsilon()))
            return candidate;
    }
    throw std::runtime_error("penalized least squares: could not bound quadratic curvature");
}

/// @brief Accelerates proximal descent with gradient restart and a caller-supplied stopping check.
/// @tparam Prox Callback that applies the proximal operator in place at the supplied curvature.
/// @tparam Stationary Callback that tests stationarity given current coefficients and smooth gradient.
/// @param problem Weighted quadratic objective.
/// @param beta Starting slopes, replaced with the final slopes.
/// @param curvature Current curvature estimate, updated by backtracking.
/// @param limit Cap on accepted updates.
/// @param prox Proximal operator.
/// @param stationary Numerical stopping condition.
/// @return Final loss and gradient, accepted step count and convergence flag.
template <typename Prox, typename Stationary>
ProximalSolution solve_proximal(const PenalizedLeastSquares& problem, ProximalVector& beta, long double& curvature,
    std::size_t limit, const Prox& prox, const Stationary& stationary) {
    auto origin = beta;
    auto state = evaluate_quadratic(problem, beta);
    long double acceleration = 1;
    auto done = stationary(beta, state.gradient);
    std::size_t iterations = 0;
    while (!done && iterations < limit) {
        const auto trial = evaluate_quadratic(problem, origin);
        auto candidate = proximal_step(problem, origin, trial.gradient, curvature, prox);
        long double restart = 0;
        for (std::size_t j = 0; j < beta.size(); ++j)
            restart += (origin[j] - candidate[j]) * (candidate[j] - beta[j]);
        const auto next = restart > 0 ? 1 : (1 + std::sqrt(1 + 4 * acceleration * acceleration)) / 2;
        const auto momentum = restart > 0 ? 0 : (acceleration - 1) / next;
        for (std::size_t j = 0; j < beta.size(); ++j)
            origin[j] = candidate[j] + momentum * (candidate[j] - beta[j]);
        acceleration = next;
        beta = std::move(candidate);
        state = evaluate_quadratic(problem, beta);
        done = stationary(beta, state.gradient);
        ++iterations;
    }
    return {std::move(state), iterations, done};
}

/// @brief Predicts using restored coefficients on finite raw predictor rows.
inline std::vector<double> predict_penalized_least_squares(const linalg::DenseMatrix<double>& design,
    const std::vector<double>& coefficients, double intercept) {
    const auto finite = [](double value) { return std::isfinite(value); };
    if (design.cols() != coefficients.size() || !std::isfinite(intercept) ||
        !std::ranges::all_of(design.data(), finite) || !std::ranges::all_of(coefficients, finite))
        throw std::invalid_argument("penalized least squares: invalid prediction dimensions or data");
    std::vector<double> result(design.rows());
    for (std::size_t i = 0; i < design.rows(); ++i) {
        long double mean = intercept;
        for (std::size_t j = 0; j < design.cols(); ++j)
            mean = std::fma(static_cast<long double>(design(i, j)), coefficients[j], mean);
        result[i] = proximal_output(mean);
    }
    return result;
}

} // namespace datamunge::stats::detail
