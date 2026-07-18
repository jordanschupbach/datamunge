#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::bayes::detail {

struct LeapfrogState {
    std::vector<double> q;
    std::vector<double> p;
};

/// @brief One full leapfrog step (half-step momentum, full-step position, half-step
/// momentum) against an identity mass matrix. Self-contained -- recomputes the gradient at
/// both half-steps rather than caching it across calls, trading a redundant gradient
/// evaluation per step for a simpler (and, for NUTS's recursive tree building, much easier
/// to get right) implementation. Shared by HMC and NUTS.
inline LeapfrogState leapfrog_step(optim::DifferentiableFunction& log_posterior, const LeapfrogState& state,
                                    const double step_size) {
    const std::size_t d = state.q.size();
    std::vector<double> p = state.p;
    std::vector<double> grad = log_posterior.gradient(state.q);
    for (std::size_t j = 0; j < d; ++j) p[j] += 0.5 * step_size * grad[j];

    std::vector<double> q = state.q;
    for (std::size_t j = 0; j < d; ++j) q[j] += step_size * p[j];

    grad = log_posterior.gradient(q);
    for (std::size_t j = 0; j < d; ++j) p[j] += 0.5 * step_size * grad[j];

    return LeapfrogState{std::move(q), std::move(p)};
}

/// @brief H(q, p) = -log_posterior(q) + 0.5*|p|^2 (identity mass matrix).
inline double hamiltonian(optim::DifferentiableFunction& log_posterior, const std::vector<double>& q,
                           const std::vector<double>& p) {
    double kinetic = 0.0;
    for (const double v : p) kinetic += v * v;
    return -log_posterior.evaluate(q) + 0.5 * kinetic;
}

} // namespace datamunge::bayes::detail
