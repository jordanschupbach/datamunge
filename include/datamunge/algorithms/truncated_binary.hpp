#pragma once

/// \file truncated_binary.hpp
/// \brief Truncated binary encoding: near-optimal fixed codes for a finite alphabet.
///
/// To encode the integers \f$0..n-1\f$ when \f$n\f$ is not a power of two, plain binary would
/// waste the unused codewords. *Truncated binary* encoding fixes this: with
/// \f$k=\lfloor\log_2 n\rfloor\f$, the first \f$u=2^{k+1}-n\f$ values are written in \f$k\f$
/// bits and the remaining \f$n-u\f$ in \f$k+1\f$ bits (offset by \f$u\f$). Every codeword is
/// within one bit of \f$\log_2 n\f$, making it the optimal fixed code for a uniform alphabet
/// and the remainder part of Golomb coding. This module encodes and decodes truncated-binary
/// symbols for a known alphabet size.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// \brief Append the truncated-binary codeword for value \c v in [0, n) to a bit vector.
inline void truncated_binary_encode_value(std::uint64_t v, std::uint64_t n, std::vector<bool>& out) {
    int k = 0;
    while ((std::uint64_t{1} << (k + 1)) <= n) ++k;      // k = floor(log2 n)
    std::uint64_t u = (std::uint64_t{1} << (k + 1)) - n;  // number of short (k-bit) codewords
    if (v < u) {
        for (int i = k - 1; i >= 0; --i) out.push_back((v >> i) & 1);       // k bits
    } else {
        std::uint64_t x = v + u;
        for (int i = k; i >= 0; --i) out.push_back((x >> i) & 1);           // k+1 bits
    }
}

/// \brief Encode a sequence of symbols in [0, n) as one truncated-binary bit stream.
inline std::vector<bool> truncated_binary_encode(const std::vector<std::uint64_t>& values, std::uint64_t n) {
    std::vector<bool> out;
    for (std::uint64_t v : values) truncated_binary_encode_value(v, n, out);
    return out;
}

/// \brief Decode \c count truncated-binary symbols (alphabet size \c n) from a bit stream.
inline std::vector<std::uint64_t> truncated_binary_decode(const std::vector<bool>& bits, std::uint64_t n, int count) {
    int k = 0;
    while ((std::uint64_t{1} << (k + 1)) <= n) ++k;
    std::uint64_t u = (std::uint64_t{1} << (k + 1)) - n;

    std::vector<std::uint64_t> out;
    std::size_t                pos = 0;
    for (int c = 0; c < count && pos < bits.size(); ++c) {
        std::uint64_t x = 0;
        for (int i = 0; i < k; ++i) x = (x << 1) | bits[pos++];   // read k bits
        if (x < u) {
            out.push_back(x);
        } else {
            x = (x << 1) | bits[pos++];                           // one more bit
            out.push_back(x - u);
        }
    }
    return out;
}

}  // namespace datamunge::algorithms
