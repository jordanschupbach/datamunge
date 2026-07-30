#pragma once

/// \file fibonacci_coding.hpp
/// \brief Fibonacci coding: a self-delimiting universal code via Zeckendorf's theorem.
///
/// Fibonacci coding represents a positive integer using *Zeckendorf's theorem*: every positive
/// integer is a unique sum of non-consecutive Fibonacci numbers. The code writes a bit per
/// Fibonacci number (from smallest up) marking which are used, then appends a terminating
/// \f$1\f$. Because no two consecutive Fibonacci numbers are used, the only place two
/// consecutive \f$1\f$s appear is at the very end -- the appended \f$1\f$ next to the highest set
/// bit -- so \f$11\f$ is an unambiguous end-of-codeword marker. The result is *self-delimiting*
/// and robust to bit errors (an error cannot propagate past the next \f$11\f$). This module
/// encodes and decodes Fibonacci bit streams.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace detail {
/// Fibonacci numbers for coding (1, 2, 3, 5, 8, ...) up to at least \c limit.
inline std::vector<std::uint64_t> fib_table(std::uint64_t limit) {
    std::vector<std::uint64_t> f = {1, 2};
    while (f.back() <= limit) f.push_back(f[f.size() - 1] + f[f.size() - 2]);
    return f;
}
}  // namespace detail

/// \brief Append the Fibonacci codeword for \c n (>= 1) to a bit vector.
inline void fibonacci_encode_value(std::uint64_t n, std::vector<bool>& out) {
    auto f = detail::fib_table(n);
    // Largest index with f[idx] <= n.
    int idx = static_cast<int>(f.size()) - 1;
    while (idx >= 0 && f[idx] > n) --idx;

    std::vector<bool> code(idx + 1, false);   // bit i marks Fibonacci f[i]
    for (int i = idx; i >= 0; --i)
        if (f[i] <= n) { code[i] = true; n -= f[i]; }   // greedy Zeckendorf

    for (bool b : code) out.push_back(b);     // least-significant Fibonacci first
    out.push_back(true);                      // terminating 1 -> creates the "11" delimiter
}

/// \brief Encode a sequence of positive integers as one Fibonacci bit stream.
inline std::vector<bool> fibonacci_encode(const std::vector<std::uint64_t>& values) {
    std::vector<bool> out;
    for (std::uint64_t n : values) fibonacci_encode_value(n, out);
    return out;
}

/// \brief Decode all Fibonacci codewords from a bit stream (delimited by "11").
inline std::vector<std::uint64_t> fibonacci_decode(const std::vector<bool>& bits) {
    std::vector<std::uint64_t>       out;
    static const auto                f = detail::fib_table(~0ull >> 2);
    std::uint64_t                    value = 0;
    int                              i     = 0;    // Fibonacci index within the current codeword
    bool                             prev  = false;

    for (std::size_t p = 0; p < bits.size(); ++p) {
        if (bits[p] && prev) {                     // "11" -> end of codeword
            out.push_back(value);
            value = 0; i = 0; prev = false;
        } else {
            if (bits[p]) value += f[i];
            ++i;
            prev = bits[p];
        }
    }
    return out;
}

}  // namespace datamunge::algorithms
