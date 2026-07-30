#pragma once

/// \file bernstein_vazirani.hpp
/// \brief The Bernstein-Vazirani algorithm: recover a hidden bit string in one query.
///
/// Given a black box computing \f$f(x)=s\cdot x \bmod 2\f$ (the parity of \f$x\f$ masked by a
/// hidden string \f$s\in\{0,1\}^n\f$), Bernstein-Vazirani recovers all \f$n\f$ bits of \f$s\f$
/// with a *single* quantum query -- versus \f$n\f$ classically (one per bit). The circuit is
/// the Deutsch-Jozsa circuit with this specific oracle: after the phase kickback and the
/// closing Hadamards, the register lands *exactly* on the basis state \f$|s\rangle\f$, so one
/// measurement reads out the secret. This module runs it exactly on the state-vector simulator.

#include <datamunge/algorithms/quantum_sim.hpp>

#include <bitset>
#include <cstddef>

namespace datamunge::algorithms {

/// \brief Recover the hidden string \c secret (on \c n bits) with one Bernstein-Vazirani query.
///
/// The oracle used is \f$f(x)=\operatorname{parity}(x \wedge s)\f$. Returns the measured basis
/// state, which equals \c secret with certainty.
inline std::size_t bernstein_vazirani(int n, std::size_t secret) {
    QuantumState s = qstate_zero(n);
    for (int q = 0; q < n; ++q) h_gate(s, q);
    for (std::size_t x = 0; x < s.size(); ++x) {
        // f(x) = parity of (x & secret)
        std::size_t masked = x & secret;
        int         parity = 0;
        while (masked) { parity ^= 1; masked &= masked - 1; }
        if (parity) s[x] = -s[x];                       // phase oracle
    }
    for (int q = 0; q < n; ++q) h_gate(s, q);
    return most_probable(s);                            // == secret with probability 1
}

}  // namespace datamunge::algorithms
