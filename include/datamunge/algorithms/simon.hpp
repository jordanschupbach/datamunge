#pragma once

/// \file simon.hpp
/// \brief Simon's algorithm: find a hidden XOR period with exponential speedup.
///
/// Simon's problem gives a 2-to-1 black box \f$f\f$ with a hidden period \f$s\neq 0\f$ such
/// that \f$f(x)=f(y)\iff y=x\oplus s\f$, and asks for \f$s\f$. Classically this needs
/// \f$\Theta(2^{n/2})\f$ queries; Simon's quantum algorithm needs only \f$O(n)\f$. Each run
/// entangles an input register with \f$f\f$, and after Hadamards the input register is
/// measured to yield a random string \f$y\f$ *orthogonal* to \f$s\f$ (\f$y\cdot s=0\f$).
/// Collecting \f$n-1\f$ independent such \f$y\f$ and solving the linear system over
/// \f$\mathrm{GF}(2)\f$ recovers \f$s\f$. This module simulates the circuit exactly and does
/// the GF(2) solve.

#include <datamunge/algorithms/quantum_sim.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// Result of Simon's algorithm.
struct SimonResult {
    std::size_t              recovered_period;  ///< The hidden string s, recovered.
    std::vector<std::size_t> observed_y;        ///< The measured y (all satisfy y . s = 0).
};

namespace detail {
/// Find a nonzero \c s over GF(2) orthogonal to every row (i.e. solving `rows * s = 0`).
inline std::size_t gf2_orthogonal_vector(std::vector<std::size_t> mat, int n) {
    std::vector<int> where(n, -1);  // pivot row for each column, or -1 if free
    int r = 0;
    for (int col = 0; col < n && r < static_cast<int>(mat.size()); ++col) {
        int sel = -1;
        for (int i = r; i < static_cast<int>(mat.size()); ++i)
            if ((mat[i] >> col) & 1) { sel = i; break; }
        if (sel < 0) continue;
        std::swap(mat[r], mat[sel]);
        for (int i = 0; i < static_cast<int>(mat.size()); ++i)
            if (i != r && ((mat[i] >> col) & 1)) mat[i] ^= mat[r];
        where[col] = r;
        ++r;
    }
    // A free column gives the nullspace direction (period).
    for (int f = 0; f < n; ++f)
        if (where[f] == -1) {
            std::size_t s = std::size_t{1} << f;             // free variable = 1
            for (int c = 0; c < n; ++c)
                if (where[c] != -1 && ((mat[where[c]] >> f) & 1)) s |= std::size_t{1} << c;
            return s;
        }
    return 0;
}
}  // namespace detail

/// \brief Simulate Simon's algorithm for a function with the given hidden \c period on \c n bits.
///
/// Uses the canonical 2-to-1 function \f$f(x)=\min(x, x\oplus s)\f$. Returns the recovered
/// period and the set of orthogonal strings \f$y\f$ the measurement can yield.
inline SimonResult simon(int n, std::size_t period) {
    const int         total = 2 * n;                 // n input + n output qubits
    const std::size_t Nin   = std::size_t{1} << n;
    QuantumState      s     = qstate_zero(total);

    for (int q = 0; q < n; ++q) h_gate(s, q);         // superpose all inputs, output |0>

    // Oracle |x>|0> -> |x>|f(x)>, with f(x) = min(x, x^period) in the output register (high bits).
    QuantumState out(s.size(), Complex{0, 0});
    for (std::size_t x = 0; x < Nin; ++x) {
        std::size_t fx  = std::min(x, x ^ period);
        out[x | (fx << n)] = s[x];
    }
    s = out;

    for (int q = 0; q < n; ++q) h_gate(s, q);         // Hadamard the input register

    // Input-register marginal: the y with nonzero probability are exactly those with y . s = 0.
    std::vector<double> py(Nin, 0.0);
    for (std::size_t i = 0; i < s.size(); ++i) py[i & (Nin - 1)] += std::norm(s[i]);

    std::vector<std::size_t> ys;
    for (std::size_t y = 1; y < Nin; ++y)             // skip the trivial y = 0
        if (py[y] > 1e-9) ys.push_back(y);

    return {detail::gf2_orthogonal_vector(ys, n), ys};
}

}  // namespace datamunge::algorithms
