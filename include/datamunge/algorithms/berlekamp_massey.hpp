#pragma once

/// \file berlekamp_massey.hpp
/// \brief The Berlekamp-Massey algorithm: shortest LFSR for a binary sequence.
///
/// Given a binary sequence, the Berlekamp-Massey algorithm (1968) finds the *shortest* linear
/// feedback shift register (LFSR) that generates it -- equivalently, the minimal-degree
/// *connection polynomial* of the linear recurrence the sequence satisfies. It processes the
/// sequence left to right, maintaining a current LFSR and, whenever the next bit disagrees with
/// the LFSR's prediction (a nonzero *discrepancy*), correcting the polynomial using a saved
/// earlier version. The register's final length is the sequence's *linear complexity*. The
/// algorithm is the decoding core of BCH and Reed-Solomon codes (finding the error-locator
/// polynomial) and a classic measure of pseudorandomness. This module implements it over
/// \f$\mathrm{GF}(2)\f$.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// Result of Berlekamp-Massey over GF(2).
struct BerlekampMasseyResult {
    int              length;      ///< LFSR length L (the sequence's linear complexity).
    std::vector<int> polynomial;  ///< Connection polynomial C(x) coefficients, C[0]=1.
};

/// \brief Shortest LFSR generating the binary sequence \c s (values 0/1), over GF(2).
inline BerlekampMasseyResult berlekamp_massey(const std::vector<int>& s) {
    const int        n = static_cast<int>(s.size());
    std::vector<int> c(n + 1, 0), b(n + 1, 0);
    c[0] = b[0] = 1;
    int L = 0, m = 1;   // L = current LFSR length, m = steps since last length change

    for (int i = 0; i < n; ++i) {
        int d = s[i] & 1;                               // discrepancy
        for (int j = 1; j <= L; ++j) d ^= (c[j] & s[i - j]);
        if (d == 0) {
            ++m;                                        // prediction correct: nothing to do
        } else if (2 * L <= i) {
            std::vector<int> t = c;                     // save copy before update
            for (int j = 0; j + m <= n; ++j) c[j + m] ^= b[j];
            L = i + 1 - L;
            b = t;
            m = 1;
        } else {
            for (int j = 0; j + m <= n; ++j) c[j + m] ^= b[j];
            ++m;
        }
    }

    c.resize(L + 1);
    return {L, c};
}

}  // namespace datamunge::algorithms
