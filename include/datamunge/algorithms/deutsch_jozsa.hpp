#pragma once

/// \file deutsch_jozsa.hpp
/// \brief The Deutsch-Jozsa algorithm: constant vs balanced in a single query.
///
/// Given a black-box Boolean function \f$f:\{0,1\}^n\to\{0,1\}\f$ *promised* to be either
/// *constant* (same output for all inputs) or *balanced* (0 on exactly half the inputs),
/// Deutsch-Jozsa decides which with a *single* evaluation -- versus up to \f$2^{n-1}+1\f$
/// classically. It puts all inputs in superposition, applies \f$f\f$ as a phase
/// \f$(-1)^{f(x)}\f$ (phase kickback), and interferes the results with Hadamards: constant
/// \f$f\f$ refocuses all amplitude onto \f$|0\cdots0\rangle\f$, balanced \f$f\f$ cancels it
/// completely. This module runs the circuit exactly on the state-vector simulator.

#include <datamunge/algorithms/quantum_sim.hpp>

#include <cstddef>
#include <functional>

namespace datamunge::algorithms {

/// Outcome of the Deutsch-Jozsa circuit.
struct DeutschJozsaResult {
    bool   constant;        ///< True if f was determined constant, false if balanced.
    double prob_all_zero;   ///< Probability of measuring |0..0> (1 constant, 0 balanced).
};

/// \brief Decide whether \c f (on \c n bits) is constant or balanced in one quantum query.
///
/// \param f  the promised oracle: returns 0/1 for each input in [0, 2^n).
inline DeutschJozsaResult deutsch_jozsa(int n, const std::function<int(std::size_t)>& f) {
    QuantumState s = qstate_zero(n);
    for (int q = 0; q < n; ++q) h_gate(s, q);          // uniform superposition of all inputs
    for (std::size_t x = 0; x < s.size(); ++x)
        if (f(x) & 1) s[x] = -s[x];                    // phase oracle (-1)^{f(x)}
    for (int q = 0; q < n; ++q) h_gate(s, q);          // interfere
    double p0 = std::norm(s[0]);
    return {p0 > 0.5, p0};
}

}  // namespace datamunge::algorithms
