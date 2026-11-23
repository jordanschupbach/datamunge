#pragma once

/// @file likelihood_bfgs.hpp
/// @brief Safeguarded BFGS maximization and shared inverse information updates.

#include <datamunge/linalg/dense_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace datamunge::stats::detail {

/// @brief Updates the inverse information estimate after a step with positive curvature.
inline void update_likelihood_metric(linalg::DenseMatrix<long double>& metric,
    const std::vector<long double>& old_beta, const std::vector<long double>& beta,
    const std::vector<long double>& old_score, const std::vector<long double>& score) {
    const auto d = beta.size();
    std::vector<long double> change(d), step(d);
    for (std::size_t i = 0; i < d; ++i) {
        change[i] = old_score[i] - score[i];
        step[i] = beta[i] - old_beta[i];
    }
    const auto dot = [](const std::vector<long double>& a, const std::vector<long double>& b) {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0L);
    };
    const auto curvature = dot(change, step);
    if (!(curvature > 64 * std::numeric_limits<long double>::epsilon() *
                      std::sqrt(dot(change, change)) * std::sqrt(dot(step, step))))
        return;
    const auto projected = metric * change;
    const auto scale = 1 + dot(change, projected) / curvature;
    for (std::size_t j = 0; j < d; ++j)
        for (std::size_t k = 0; k <= j; ++k) {
            const auto increment = (step[j] / curvature) * (scale * step[k] - projected[k])
                                   - projected[j] * (step[k] / curvature);
            metric(k, j) = metric(j, k) = metric(j, k) + increment;
        }
}

/// @brief Result of unconstrained likelihood maximization.
template<class Evaluation, class Status>
struct LikelihoodOptimum {
    std::vector<long double> parameters; ///< Working parameters at the best iterate.
    Evaluation state;                    ///< Likelihood and score at the fitted parameters.
    Status status;                       ///< Numerical stopping reason.
};

/// @brief Maximum absolute value in a working vector.
inline long double likelihood_score_norm(const std::vector<long double>& x) {
    long double result = 0;
    for (const auto value : x)
        result = std::max(result, std::abs(value));
    return result;
}

/// @brief Safeguarded quasi-Newton maximization for one starting point.
template<class Status, class Evaluate>
auto maximize_likelihood(std::vector<long double> parameters, std::size_t max_iterations, long double tolerance,
    const Evaluate& evaluate) {
    auto state = evaluate(parameters);
    const auto dot = [](const std::vector<long double>& a, const std::vector<long double>& b) {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0L);
    };
    auto status = Status::NumericalFailure;
    auto metric = linalg::DenseMatrix<long double>::identity(parameters.size());
    for (std::size_t iter = 0; !state.score.empty(); ++iter) {
        const auto score_size = likelihood_score_norm(state.score);
        if (score_size <= tolerance) {
            status = Status::Converged;
            break;
        }
        if (iter == max_iterations) {
            status = Status::IterationLimit;
            break;
        }
        const auto rounding = 64 * std::numeric_limits<long double>::epsilon() * std::max(1.0L, std::abs(state.value));
        bool accepted = false;
        for (int restart = 0; restart < 2 && !accepted; ++restart) {
            if (restart)
                metric = linalg::DenseMatrix<long double>::identity(parameters.size());
            const auto direction = metric * state.score;
            const auto projected = dot(direction, state.score);
            if (!(projected > 0) || !std::isfinite(projected))
                continue;
            auto step = std::min(1.0L, 3 / likelihood_score_norm(direction));
            for (int attempt = 0; attempt < 72; ++attempt, step *= .5L) {
                auto trial = parameters;
                for (std::size_t j = 0; j < trial.size(); ++j)
                    trial[j] += step * direction[j];
                if (trial == parameters)
                    break;
                auto next = evaluate(trial);
                if (next.score.empty() || next.value - state.value < 1e-4L * step * projected - rounding ||
                    (step * projected <= rounding && likelihood_score_norm(next.score) >= score_size))
                    continue;
                update_likelihood_metric(metric, parameters, trial, state.score, next.score);
                parameters = std::move(trial);
                state = std::move(next);
                accepted = true;
                break;
            }
        }
        if (!accepted)
            break;
    }
    return LikelihoodOptimum<decltype(state), Status>{std::move(parameters), std::move(state), status};
}

} // namespace datamunge::stats::detail
