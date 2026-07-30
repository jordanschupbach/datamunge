#pragma once

/// \file phase_estimation.hpp
/// \brief Quantum phase estimation: read out the eigenphase of a unitary.
///
/// Given a unitary \f$U\f$ and one of its eigenstates \f$|\psi\rangle\f$ with
/// \f$U|\psi\rangle=e^{2\pi i\varphi}|\psi\rangle\f$, *quantum phase estimation* recovers the
/// phase \f$\varphi\in[0,1)\f$ to \f$t\f$ bits of precision. A register of \f$t\f$ counting
/// qubits is put in superposition; controlled-\f$U^{2^j}\f$ operations kick the phase
/// \f$e^{2\pi i\varphi 2^j}\f$ back onto counting qubit \f$j\f$; an inverse QFT then
/// concentrates the amplitude on the integer nearest \f$2^t\varphi\f$. QPE is the engine
/// behind Shor's factoring and quantum chemistry energy estimates. This module simulates it
/// on the counting register (the eigenstate contributes only through the kicked-back phase).

#include <datamunge/algorithms/qft.hpp>
#include <datamunge/algorithms/quantum_sim.hpp>

#include <cmath>
#include <cstddef>

namespace datamunge::algorithms {

/// Result of phase estimation.
struct PhaseEstimationResult {
    double      estimate;     ///< Best t-bit estimate of the phase in [0, 1).
    std::size_t measured;     ///< The measured integer (estimate = measured / 2^t).
    double      probability;  ///< Probability of that outcome.
};

/// \brief Estimate the eigenphase \c phi to \c t bits via quantum phase estimation.
///
/// Simulates the counting register: Hadamards, phase kickback of \f$2\pi\varphi 2^j\f$ onto
/// each counting qubit \f$j\f$, then an inverse QFT. Exact when \f$\varphi\f$ is a multiple of
/// \f$2^{-t}\f$; otherwise it returns the nearest representable value with high probability.
inline PhaseEstimationResult phase_estimation(int t, double phi) {
    QuantumState s = qstate_zero(t);
    for (int q = 0; q < t; ++q) h_gate(s, q);

    // Controlled-U^{2^j} on the eigenstate kicks back phase 2*pi*phi*2^j onto counting qubit j.
    for (int q = 0; q < t; ++q)
        phase_gate(s, q, 2.0 * M_PI * phi * static_cast<double>(std::size_t{1} << q));

    inverse_qft(s, t);

    std::size_t m = most_probable(s);
    return {static_cast<double>(m) / static_cast<double>(std::size_t{1} << t), m, std::norm(s[m])};
}

}  // namespace datamunge::algorithms
