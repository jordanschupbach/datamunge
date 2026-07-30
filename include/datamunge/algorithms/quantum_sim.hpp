#pragma once

/// \file quantum_sim.hpp
/// \brief A minimal state-vector quantum simulator: qubits, gates, and measurement.
///
/// This is the shared substrate for the quantum-algorithm reports (Deutsch-Jozsa,
/// Bernstein-Vazirani, Grover, Simon, QFT, phase estimation). An \f$n\f$-qubit register is a
/// unit vector of \f$2^n\f$ complex amplitudes; a basis state is indexed by the integer whose
/// bit \f$q\f$ is the value of qubit \f$q\f$ (qubit 0 = least significant bit). Gates are
/// unitary maps applied in place: single-qubit gates (Hadamard, Pauli-X, phase), the
/// two-qubit CNOT and controlled-phase, and helpers to read out measurement probabilities.
/// Everything is exact (no sampling) so the algorithm reports can print deterministic
/// amplitudes and probabilities.

#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

using Complex      = std::complex<double>;
using QuantumState = std::vector<Complex>;

/// \brief The \f$n\f$-qubit ground state \f$|0\cdots0\rangle\f$.
inline QuantumState qstate_zero(int n) {
    QuantumState s(std::size_t{1} << n, Complex{0.0, 0.0});
    s[0] = Complex{1.0, 0.0};
    return s;
}

/// \brief Apply an arbitrary single-qubit gate [[m00,m01],[m10,m11]] to qubit \c q.
inline void apply_single(QuantumState& s, int q, Complex m00, Complex m01, Complex m10, Complex m11) {
    const std::size_t step = std::size_t{1} << q;
    for (std::size_t i = 0; i < s.size(); ++i)
        if ((i & step) == 0) {
            Complex a = s[i], b = s[i | step];
            s[i]        = m00 * a + m01 * b;
            s[i | step] = m10 * a + m11 * b;
        }
}

/// \brief Hadamard gate on qubit \c q.
inline void h_gate(QuantumState& s, int q) {
    const double r = 1.0 / std::sqrt(2.0);
    apply_single(s, q, r, r, r, -r);
}

/// \brief Pauli-X (bit flip) on qubit \c q.
inline void x_gate(QuantumState& s, int q) {
    apply_single(s, q, Complex{0, 0}, Complex{1, 0}, Complex{1, 0}, Complex{0, 0});
}

/// \brief Phase gate: multiply the \f$|1\rangle\f$ component of qubit \c q by \f$e^{i\theta}\f$.
inline void phase_gate(QuantumState& s, int q, double theta) {
    apply_single(s, q, Complex{1, 0}, Complex{0, 0}, Complex{0, 0}, std::polar(1.0, theta));
}

/// \brief Controlled-NOT: flip \c target when \c control is \f$|1\rangle\f$.
inline void cnot(QuantumState& s, int control, int target) {
    const std::size_t cs = std::size_t{1} << control, ts = std::size_t{1} << target;
    for (std::size_t i = 0; i < s.size(); ++i)
        if ((i & cs) && !(i & ts)) std::swap(s[i], s[i | ts]);
}

/// \brief Controlled-phase: multiply by \f$e^{i\theta}\f$ when both qubits are \f$|1\rangle\f$.
inline void controlled_phase(QuantumState& s, int control, int target, double theta) {
    const std::size_t cs = std::size_t{1} << control, ts = std::size_t{1} << target;
    const Complex     p  = std::polar(1.0, theta);
    for (std::size_t i = 0; i < s.size(); ++i)
        if ((i & cs) && (i & ts)) s[i] *= p;
}

/// \brief Swap two qubits (used e.g. for the QFT's bit reversal).
inline void swap_qubits(QuantumState& s, int a, int b) {
    if (a == b) return;
    cnot(s, a, b);
    cnot(s, b, a);
    cnot(s, a, b);
}

/// \brief Probability of measuring qubit \c q as \f$|1\rangle\f$.
inline double prob_one(const QuantumState& s, int q) {
    const std::size_t step = std::size_t{1} << q;
    double            p    = 0.0;
    for (std::size_t i = 0; i < s.size(); ++i)
        if (i & step) p += std::norm(s[i]);
    return p;
}

/// \brief Full measurement distribution: \f$|amp_i|^2\f$ over all basis states.
inline std::vector<double> probabilities(const QuantumState& s) {
    std::vector<double> p(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) p[i] = std::norm(s[i]);
    return p;
}

/// \brief Index of the most probable basis state (argmax of \f$|amp_i|^2\f$).
inline std::size_t most_probable(const QuantumState& s) {
    std::size_t best = 0;
    double      bp   = -1.0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        double p = std::norm(s[i]);
        if (p > bp) { bp = p; best = i; }
    }
    return best;
}

}  // namespace datamunge::algorithms
