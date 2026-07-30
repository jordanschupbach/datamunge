#pragma once

/// \file elias_gamma.hpp
/// \brief Elias gamma coding: a universal, self-delimiting code for positive integers
///        (Elias 1975).
///
/// When encoding a stream of positive integers whose magnitudes are unknown in advance,
/// a fixed field width is wasteful. *Universal codes* spend bits in proportion to a
/// number's size while remaining *self-delimiting* (no separators needed). *Elias gamma*
/// codes \f$n\ge 1\f$ as: write \f$\lfloor\log_2 n\rfloor\f$ zeros, then the
/// \f$(\lfloor\log_2 n\rfloor+1)\f$-bit binary of \f$n\f$ (which begins with a 1). The
/// leading run of zeros tells the decoder how many more bits to read, so the code is
/// prefix-free and needs \f$2\lfloor\log_2 n\rfloor+1\f$ bits. It is optimal (to within a
/// constant) when the integers follow a roughly \f$1/n^2\f$ distribution and is a building
/// block of many compressors (e.g. inverted-index posting lists).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// \brief Elias-gamma-encode a single positive integer to a bit string.
inline std::string elias_gamma_encode_one(std::uint64_t n) {
    if (n == 0) return "";  // gamma is defined for n >= 1
    int nbits = 0;
    while ((static_cast<std::uint64_t>(1) << (nbits + 1)) <= n) ++nbits;  // floor(log2 n)
    std::string out(nbits, '0');  // the unary length prefix
    for (int b = nbits; b >= 0; --b) out += ((n >> b) & 1u) ? '1' : '0';   // the binary of n
    return out;
}

/// \brief Elias-gamma-encode a sequence of positive integers to one concatenated bit string.
inline std::string elias_gamma_encode(const std::vector<std::uint64_t>& values) {
    std::string out;
    for (std::uint64_t v : values) out += elias_gamma_encode_one(v);
    return out;
}

/// \brief Decode an Elias gamma bit string back to the sequence of integers.
inline std::vector<std::uint64_t> elias_gamma_decode(const std::string& bits) {
    std::vector<std::uint64_t> out;
    std::size_t                i = 0;
    while (i < bits.size()) {
        int zeros = 0;
        while (i < bits.size() && bits[i] == '0') { ++zeros; ++i; }
        if (i >= bits.size()) break;  // trailing padding zeros, no value follows
        std::uint64_t value = 1;      // the leading 1 we are now at
        ++i;
        for (int k = 0; k < zeros && i < bits.size(); ++k, ++i) value = (value << 1) | (bits[i] == '1' ? 1u : 0u);
        out.push_back(value);
    }
    return out;
}

}  // namespace datamunge::algorithms
