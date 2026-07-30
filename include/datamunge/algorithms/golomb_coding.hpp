#pragma once

/// \file golomb_coding.hpp
/// \brief Golomb coding: optimal entropy coding for geometrically distributed integers.
///
/// Golomb coding (Solomon Golomb, 1966) encodes non-negative integers with a tunable parameter
/// \f$M\f$. A value \f$n\f$ is split into a quotient \f$q=\lfloor n/M\rfloor\f$ and remainder
/// \f$r=n\bmod M\f$: the quotient is written in *unary* (\f$q\f$ ones then a zero) and the
/// remainder in *truncated binary*. When the source is geometrically distributed -- small values
/// far more common than large -- and \f$M\f$ is chosen to match the distribution, Golomb coding
/// is *optimal*, which is why it appears in run-length and residual coding (FLAC, JPEG-LS). This
/// module encodes and decodes bit streams of Golomb codewords.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace detail {
inline int ilog2_ceil(int x) {   // smallest b with 2^b >= x
    int b = 0;
    while ((1 << b) < x) ++b;
    return b;
}
}  // namespace detail

/// \brief Append the Golomb codeword for \c n (with parameter \c M) to a bit vector.
inline void golomb_encode_value(std::uint64_t n, int M, std::vector<bool>& out) {
    std::uint64_t q = n / M;
    std::uint64_t r = n % M;
    for (std::uint64_t i = 0; i < q; ++i) out.push_back(true);   // unary quotient
    out.push_back(false);                                        // terminator

    int b = detail::ilog2_ceil(M);
    int cutoff = (1 << b) - M;                                   // truncated-binary split point
    if (M == (1 << b)) {                                        // M is a power of two (Rice case)
        for (int i = b - 1; i >= 0; --i) out.push_back((r >> i) & 1);
    } else if ((int)r < cutoff) {
        for (int i = b - 2; i >= 0; --i) out.push_back((r >> i) & 1);   // short code, b-1 bits
    } else {
        std::uint64_t v = r + cutoff;                            // long code, b bits
        for (int i = b - 1; i >= 0; --i) out.push_back((v >> i) & 1);
    }
}

/// \brief Encode a sequence of non-negative integers as one Golomb bit stream.
inline std::vector<bool> golomb_encode(const std::vector<std::uint64_t>& values, int M) {
    std::vector<bool> out;
    for (std::uint64_t n : values) golomb_encode_value(n, M, out);
    return out;
}

/// \brief Decode \c count Golomb codewords (parameter \c M) from a bit stream.
inline std::vector<std::uint64_t> golomb_decode(const std::vector<bool>& bits, int M, int count) {
    std::vector<std::uint64_t> out;
    std::size_t                pos = 0;
    int                        b = detail::ilog2_ceil(M), cutoff = (1 << b) - M;

    for (int c = 0; c < count && pos < bits.size(); ++c) {
        std::uint64_t q = 0;
        while (pos < bits.size() && bits[pos]) { ++q; ++pos; }   // count unary ones
        ++pos;                                                    // skip the terminating zero

        std::uint64_t r = 0;
        if (M == (1 << b)) {
            for (int i = 0; i < b; ++i) r = (r << 1) | bits[pos++];
        } else {
            for (int i = 0; i < b - 1; ++i) r = (r << 1) | bits[pos++];   // read b-1 bits
            if ((int)r >= cutoff) r = (r << 1) | bits[pos++];             // one more if needed
            if ((int)r >= cutoff) r -= cutoff;                           // undo the offset
        }
        out.push_back(q * M + r);
    }
    return out;
}

/// \brief Rice coding: Golomb with \f$M=2^k\f$ (remainder is a plain k-bit field).
inline std::vector<bool> rice_encode(const std::vector<std::uint64_t>& values, int k) {
    return golomb_encode(values, 1 << k);
}
/// \brief Decode a Rice bit stream with parameter k.
inline std::vector<std::uint64_t> rice_decode(const std::vector<bool>& bits, int k, int count) {
    return golomb_decode(bits, 1 << k, count);
}

}  // namespace datamunge::algorithms
