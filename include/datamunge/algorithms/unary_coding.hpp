#pragma once

/// \file unary_coding.hpp
/// \brief Unary coding: the simplest variable-length integer code.
///
/// Unary coding represents a non-negative integer \f$n\f$ as \f$n\f$ ones followed by a
/// terminating zero (so \f$0\to\texttt{0}\f$, \f$3\to\texttt{1110}\f$). It is trivially
/// self-delimiting and needs \f$n+1\f$ bits, which is wildly inefficient for large values but
/// *optimal* for a source where \f$P(n)=2^{-(n+1)}\f$ -- and it is the quotient part of Golomb,
/// Rice, and Elias codes. This module encodes and decodes unary bit streams.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// \brief Append the unary codeword for \c n (n ones then a zero) to a bit vector.
inline void unary_encode_value(std::uint64_t n, std::vector<bool>& out) {
    for (std::uint64_t i = 0; i < n; ++i) out.push_back(true);
    out.push_back(false);
}

/// \brief Encode a sequence of non-negative integers as a unary bit stream.
inline std::vector<bool> unary_encode(const std::vector<std::uint64_t>& values) {
    std::vector<bool> out;
    for (std::uint64_t n : values) unary_encode_value(n, out);
    return out;
}

/// \brief Decode \c count unary codewords from a bit stream.
inline std::vector<std::uint64_t> unary_decode(const std::vector<bool>& bits, int count) {
    std::vector<std::uint64_t> out;
    std::size_t                pos = 0;
    for (int c = 0; c < count && pos < bits.size(); ++c) {
        std::uint64_t n = 0;
        while (pos < bits.size() && bits[pos]) { ++n; ++pos; }
        ++pos;   // skip the terminating zero
        out.push_back(n);
    }
    return out;
}

}  // namespace datamunge::algorithms
