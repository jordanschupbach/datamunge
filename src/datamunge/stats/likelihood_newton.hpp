#pragma once

/// @file likelihood_newton.hpp
/// @brief Shared safeguarded Newton and Fisher-scoring maximization.

#include <datamunge/linalg/cholesky.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace datamunge::stats::detail {

/// @brief Change in model predictors and evidence of a likelihood recession direction.
struct LikelihoodStep {
    long double maximum; ///< Largest absolute identifiable predictor change.
    bool separates;      ///< The step is a numerical likelihood recession direction.
};

/// @brief Final finite iterate and its information factor.
/// @tparam Status Model-specific stopping enumeration.
template <typename Status>
struct LikelihoodFit {
    std::vector<long double> beta;                      ///< Working-scale coefficient vector.
    long double value;                                 ///< Log likelihood at beta.
    long double null_value;                            ///< Log likelihood at the starting coefficients.
    linalg::CholeskyDecomposition<long double> factor; ///< Factor of information at beta.
    std::size_t iterations;                            ///< Number of accepted updates.
    Status status;                                    ///< Numerical stopping reason.
};

/// @brief Safeguards Newton or Fisher-scoring steps in identifiable predictor units.
/// @tparam Status Model stopping enumeration with convergence, limit, and numerical-failure values. Separation is optional.
/// @tparam Evaluate Callback returning the log likelihood, score and lower triangle of observed or expected information.
/// @tparam Effect Callback returning the predictor change and a separation diagnostic.
/// @param beta Finite starting coefficients inside the model domain.
/// @param max_iterations Maximum accepted Newton updates.
/// @param tolerance Absolute predictor-change tolerance.
/// @param evaluate Likelihood callback in working coordinates.
/// @param step_effect Model-specific check of a Newton step in data coordinates.
/// @return Fit, or no value when the starting information is numerically singular.
template <typename Status, typename Evaluate, typename Effect>
std::optional<LikelihoodFit<Status>> maximize_likelihood(std::vector<long double> beta, std::size_t max_iterations,
    double tolerance, const Evaluate& evaluate, const Effect& step_effect) {
    const auto finite = [](long double x) { return std::isfinite(x); };
    const auto dimension = beta.size();
    auto state = evaluate(beta);
    if (!std::isfinite(state.value) || state.info.rows() != dimension || state.info.cols() != dimension ||
        state.score.size() != dimension)
        return std::nullopt;
    const auto null_value = state.value;
    auto factor = linalg::cholesky(state.info);
    long double largest = 0, smallest = std::numeric_limits<long double>::infinity();
    for (std::size_t j = 0; j < dimension; ++j) {
        largest = std::max(largest, factor.L(j, j));
        smallest = std::min(smallest, factor.L(j, j));
    }
    if (!factor.ok || smallest <= largest * std::sqrt(64 * std::numeric_limits<long double>::epsilon()))
        return std::nullopt;

    auto status = Status::IterationLimit;
    std::size_t iterations = 0;
    for (;;) {
        if (!factor.ok) {
            status = Status::NumericalFailure;
            break;
        }
        const auto step = factor.solve(state.score);
        if (!std::ranges::all_of(step, finite)) {
            status = Status::NumericalFailure;
            break;
        }
        const auto effect = step_effect(step);
        if (!std::isfinite(effect.maximum)) {
            status = Status::NumericalFailure;
            break;
        }
        if (effect.maximum <= tolerance) {
            status = Status::Converged;
            break;
        }
        if constexpr (requires { Status::Separated; }) {
            if (effect.separates) {
                status = Status::Separated;
                break;
            }
        }
        if (iterations == max_iterations)
            break;
        const auto predicted = std::inner_product(step.begin(), step.end(), state.score.begin(), 0.0L);
        auto scale = effect.maximum > 5 ? 5 / effect.maximum : 1.0L;
        const auto rounding = 64 * std::numeric_limits<long double>::epsilon() * std::max(1.0L, std::abs(state.value));
        bool accepted = false;
        for (int trial = 0; trial < 64 && predicted > 0 && std::isfinite(predicted); ++trial, scale /= 2) {
            auto candidate = beta;
            for (std::size_t j = 0; j < dimension; ++j)
                candidate[j] += scale * step[j];
            if (candidate == beta)
                break;
            if (!std::ranges::all_of(candidate, finite))
                continue;
            auto next = evaluate(candidate);
            const auto required_gain = 1e-4L * scale * predicted;
            if (!(next.value - state.value >= required_gain - rounding))
                continue;
            auto next_factor = linalg::cholesky(next.info);
            if (scale * predicted <= rounding && next_factor.ok) {
                // Near roundoff, compare the local information-weighted
                // score norms to avoid accepting a cycle of trial steps.
                const auto next_step = next_factor.solve(next.score);
                const auto next_norm = std::inner_product(next_step.begin(), next_step.end(), next.score.begin(), 0.0L);
                if (!(next_norm < predicted))
                    continue;
            }
            beta = std::move(candidate);
            state = std::move(next);
            factor = std::move(next_factor);
            ++iterations;
            accepted = true;
            break;
        }
        if (!accepted) {
            status = Status::NumericalFailure;
            break;
        }
    }
    return LikelihoodFit<Status>{std::move(beta), state.value, null_value, std::move(factor), iterations, status};
}

/// @brief Uses a zero start for a likelihood defined at zero coefficients.
template <typename Status, typename Evaluate, typename Effect>
std::optional<LikelihoodFit<Status>> maximize_likelihood(std::size_t dimension, std::size_t max_iterations,
    double tolerance, const Evaluate& evaluate, const Effect& step_effect) {
    return maximize_likelihood<Status>(std::vector<long double>(dimension), max_iterations, tolerance,
                                       evaluate, step_effect);
}

} // namespace datamunge::stats::detail
