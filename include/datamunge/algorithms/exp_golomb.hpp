#pragma once

/// \file exp_golomb.hpp
/// \brief Exponential-Golomb coding: universal codes for non-negative integers.
///
/// Exponential-Golomb (Exp-Golomb) codes are self-delimiting universal codes for non-negative
/// integers, used pervasively in video compression (H.264/H.265 syntax elements). The order-0
/// code writes \f$n\f$ as \f$m=\lfloor\log_2(n+1)\rfloor\f$ leading zeros, followed by the
/// \f$(m+1)\f$-bit binary of \f$n+1\f$ -- so the codeword length grows only logarithmically with
/// \f$n\f$. An order-\f$k\f$ variant appends \f$k\f$ raw low bits, biasing the code toward
/// slightly larger typical values. This module encodes and decodes Exp-Golomb bit streams.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// \brief Append the order-0 Exp-Golomb codeword for \c n to a bit vector.
inline void exp_golomb0_encode_value(std::uint64_t n, std::vector<bool>& out) {
    std::uint64_t v = n + 1;                 // map 0->1 so it always has a leading 1
    int           m = 0;
    while ((v >> (m + 1)) != 0) ++m;         // m = floor(log2(v))
    for (int i = 0; i < m; ++i) out.push_back(false);   // m leading zeros
    for (int i = m; i >= 0; --i) out.push_back((v >> i) & 1);  // (m+1)-bit value
}

/// \brief Encode a sequence with order-\c k Exp-Golomb (k raw low bits + order-0 of the high part).
inline std::vector<bool> exp_golomb_encode(const std::vector<std::uint64_t>& values, int k = 0) {
    std::vector<bool> out;
    for (std::uint64_t n : values) {
        exp_golomb0_encode_value(n >> k, out);
        for (int i = k - 1; i >= 0; --i) out.push_back((n >> i) & 1);   // k low bits verbatim
    }
    return out;
}

/// \brief Decode \c count order-\c k Exp-Golomb codewords from a bit stream.
inline std::vector<std::uint64_t> exp_golomb_decode(const std::vector<bool>& bits, int count, int k = 0) {
    std::vector<std::uint64_t> out;
    std::size_t                pos = 0;
    for (int c = 0; c < count && pos < bits.size(); ++c) {
        int m = 0;
        while (pos < bits.size() && !bits[pos]) { ++m; ++pos; }   // count leading zeros
        std::uint64_t v = 1;
        ++pos;                                                     // consume the leading 1
        for (int i = 0; i < m; ++i) v = (v << 1) | bits[pos++];    // read m more bits
        std::uint64_t hi = v - 1;
        std::uint64_t lo = 0;
        for (int i = 0; i < k; ++i) lo = (lo << 1) | bits[pos++];
        out.push_back((hi << k) | lo);
    }
    return out;
}

}  // namespace datamunge::algorithms
