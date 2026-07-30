#pragma once

/// \file hamming_code.hpp
/// \brief The Hamming(7,4) code: encode 4 data bits into 7 with 3 parity bits so any
///        single-bit error can be detected *and corrected* (Hamming 1950).
///
/// Hamming codes are the first family of *error-correcting* codes. The (7,4) code adds
/// three parity bits to four data bits; the parity bits are placed at the power-of-two
/// positions 1, 2, 4 and each checks a specific subset of positions so that, on
/// decoding, the three recomputed parities form a 3-bit *syndrome* whose value is
/// *exactly the position of the flipped bit* (0 meaning "no error"). Flipping that bit
/// back corrects the error. The code has minimum distance 3, so it corrects any single
/// error (and detects any double error). It is the textbook introduction to coding
/// theory and the ancestor of the SECDED codes protecting computer memory.

#include <array>
#include <cstddef>

namespace datamunge::algorithms {

/// Result of decoding a 7-bit Hamming codeword.
struct Hamming74Decoded {
    std::array<int, 4> data;            ///< Recovered 4 data bits.
    int                error_position;  ///< 1-based position of the corrected bit, or 0 if none.
    bool               corrected;       ///< Whether a single-bit error was corrected.
};

/// \brief Encode 4 data bits \f$(d_1,d_2,d_3,d_4)\f$ into a 7-bit codeword.
///
/// Layout (1-based): position 1 = p1, 2 = p2, 3 = d1, 4 = p4, 5 = d2, 6 = d3, 7 = d4.
/// Parity p1 covers {1,3,5,7}, p2 covers {2,3,6,7}, p4 covers {4,5,6,7} (even parity).
inline std::array<int, 7> hamming74_encode(const std::array<int, 4>& d) {
    std::array<int, 7> c{};
    c[2] = d[0];  // position 3
    c[4] = d[1];  // position 5
    c[5] = d[2];  // position 6
    c[6] = d[3];  // position 7
    c[0] = c[2] ^ c[4] ^ c[6];  // p1 = d1 ^ d2 ^ d4
    c[1] = c[2] ^ c[5] ^ c[6];  // p2 = d1 ^ d3 ^ d4
    c[3] = c[4] ^ c[5] ^ c[6];  // p4 = d2 ^ d3 ^ d4
    return c;
}

/// \brief Decode a 7-bit codeword, correcting any single-bit error via the syndrome.
inline Hamming74Decoded hamming74_decode(std::array<int, 7> c) {
    // Recompute parities; the syndrome is the 1-based position of the error (0 = none).
    const int s1 = c[0] ^ c[2] ^ c[4] ^ c[6];
    const int s2 = c[1] ^ c[2] ^ c[5] ^ c[6];
    const int s4 = c[3] ^ c[4] ^ c[5] ^ c[6];
    const int syndrome = s1 * 1 + s2 * 2 + s4 * 4;

    Hamming74Decoded out;
    out.error_position = syndrome;
    out.corrected      = false;
    if (syndrome != 0) {
        c[syndrome - 1] ^= 1;  // flip the erroneous bit back
        out.corrected = true;
    }
    out.data = {c[2], c[4], c[5], c[6]};
    return out;
}

/// Compute just the 3-bit syndrome of a codeword (0 means no detected single-bit error).
inline int hamming74_syndrome(const std::array<int, 7>& c) {
    const int s1 = c[0] ^ c[2] ^ c[4] ^ c[6];
    const int s2 = c[1] ^ c[2] ^ c[5] ^ c[6];
    const int s4 = c[3] ^ c[4] ^ c[5] ^ c[6];
    return s1 + 2 * s2 + 4 * s4;
}

}  // namespace datamunge::algorithms
