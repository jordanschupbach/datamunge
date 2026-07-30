#pragma once

/// \file levenshtein_coding.hpp
/// \brief Levenshtein coding: a recursive universal code for non-negative integers.
///
/// Levenshtein coding (Vladimir Levenshtein, 1968) is a *universal* code whose length grows
/// like the *iterated* logarithm of the value -- asymptotically shorter than Elias gamma. It
/// encodes a number by repeatedly stripping the leading 1 of its binary representation and
/// recording, in a unary count prefix, how many times it did so. Decoding replays the recursion:
/// read the count, then rebuild the value one binary block at a time. This module encodes and
/// decodes Levenshtein bit streams.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// \brief Append the Levenshtein codeword for \c n to a bit vector.
inline void levenshtein_encode_value(std::uint64_t n, std::vector<bool>& out) {
    if (n == 0) { out.push_back(false); return; }   // 0 -> "0"

    std::vector<std::vector<bool>> chunks;          // binary blocks (leading 1 stripped)
    int           C   = 1;
    std::uint64_t num = n;
    while (true) {
        int m = 0;                                  // bits after the leading 1
        for (std::uint64_t t = num; t > 1; t >>= 1) ++m;
        std::vector<bool> chunk;
        for (int i = m - 1; i >= 0; --i) chunk.push_back((num >> i) & 1);   // num without leading 1
        chunks.push_back(chunk);
        if (m == 0) break;
        ++C;
        num = m;
    }
    for (int i = 0; i < C; ++i) out.push_back(true);   // C ones ...
    out.push_back(false);                              // ... then a zero
    for (int i = (int)chunks.size() - 1; i >= 0; --i)  // value blocks, outermost first
        for (bool b : chunks[i]) out.push_back(b);
}

/// \brief Encode a sequence of non-negative integers as one Levenshtein bit stream.
inline std::vector<bool> levenshtein_encode(const std::vector<std::uint64_t>& values) {
    std::vector<bool> out;
    for (std::uint64_t n : values) levenshtein_encode_value(n, out);
    return out;
}

/// \brief Decode \c count Levenshtein codewords from a bit stream.
inline std::vector<std::uint64_t> levenshtein_decode(const std::vector<bool>& bits, int count) {
    std::vector<std::uint64_t> out;
    std::size_t                pos = 0;
    for (int c = 0; c < count && pos < bits.size(); ++c) {
        int C = 0;
        while (pos < bits.size() && bits[pos]) { ++C; ++pos; }
        ++pos;                                    // skip the zero
        if (C == 0) { out.push_back(0); continue; }
        std::uint64_t N = 1;
        for (int i = 0; i < C - 1; ++i) {
            std::uint64_t v = 0;
            for (std::uint64_t j = 0; j < N; ++j) v = (v << 1) | bits[pos++];
            N = v + (std::uint64_t{1} << N);       // rebuild: prepend the implicit leading 1
        }
        out.push_back(N);
    }
    return out;
}

}  // namespace datamunge::algorithms
