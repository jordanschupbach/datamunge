#pragma once

/// \file grover.hpp
/// \brief Grover's algorithm: quadratic-speedup unstructured search.
///
/// To find a marked item among \f$N=2^n\f$ unstructured possibilities, a classical search
/// needs \f$O(N)\f$ queries; Grover's algorithm needs only \f$O(\sqrt N)\f$. Starting from a
/// uniform superposition, each iteration applies the *oracle* (a phase flip on the marked
/// state) followed by the *diffusion* operator (inversion about the mean amplitude). Together
/// they rotate the state toward the marked basis state by a fixed angle, so after about
/// \f$\frac{\pi}{4}\sqrt N\f$ iterations the marked state's probability is near 1. This module
/// runs the exact state-vector iteration and can trace the success probability per step.

#include <datamunge/algorithms/quantum_sim.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// Outcome of a Grover search.
struct GroverResult {
    std::size_t measured;       ///< Most probable basis state after the iterations.
    double      success_prob;   ///< Probability of the marked state.
    int         iterations;     ///< Number of Grover iterations applied.
};

namespace detail {
/// One Grover iteration: oracle phase flip on `target`, then inversion about the mean.
inline void grover_step(QuantumState& s, std::size_t target) {
    s[target] = -s[target];                              // oracle
    Complex mean{0, 0};
    for (const auto& z : s) mean += z;
    mean /= static_cast<double>(s.size());
    for (auto& z : s) z = 2.0 * mean - z;                // diffusion (inversion about mean)
}
}  // namespace detail

/// \brief Search for `target` among 2^n items with the optimal number of Grover iterations.
inline GroverResult grover(int n, std::size_t target) {
    const std::size_t N = std::size_t{1} << n;
    QuantumState      s = qstate_zero(n);
    for (int q = 0; q < n; ++q) h_gate(s, q);

    int iters = static_cast<int>(std::round(M_PI / 4.0 * std::sqrt(static_cast<double>(N))));
    for (int t = 0; t < iters; ++t) detail::grover_step(s, target);
    return {most_probable(s), std::norm(s[target]), iters};
}

/// \brief Success probability of the marked state after 0, 1, ..., `max_iters` iterations.
///
/// Useful for showing the sinusoidal amplitude and why over-iterating *lowers* the success
/// probability (the state rotates past the target).
inline std::vector<double> grover_success_trace(int n, std::size_t target, int max_iters) {
    const std::size_t N = std::size_t{1} << n;
    QuantumState      s = qstate_zero(n);
    for (int q = 0; q < n; ++q) h_gate(s, q);

    std::vector<double> trace;
    trace.push_back(std::norm(s[target]));
    for (int t = 0; t < max_iters; ++t) {
        detail::grover_step(s, target);
        trace.push_back(std::norm(s[target]));
    }
    return trace;
}

}  // namespace datamunge::algorithms
