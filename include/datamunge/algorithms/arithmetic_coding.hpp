#pragma once

/// \file arithmetic_coding.hpp
/// \brief Arithmetic coding: near-optimal entropy coding to a fractional bit budget.
///
/// Arithmetic coding represents an entire message as a single number in \f$[0,1)\f$. Starting
/// from the full interval, each symbol *narrows* it to the sub-interval whose width is the
/// symbol's probability; the final interval identifies the message, and any number inside it
/// (written in as few bits as possible) is the code. Unlike Huffman coding -- which must spend a
/// whole bit per symbol -- arithmetic coding approaches the Shannon entropy arbitrarily closely,
/// spending *fractional* bits per symbol, which matters most for skewed alphabets. This module
/// implements the classic 32-bit integer coder with carry-free renormalization (E1/E2/E3
/// scaling and the bit-plus-follow rule).

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace detail {
constexpr std::uint64_t AC_MASK     = 0xFFFFFFFFull;
constexpr std::uint64_t AC_HALF     = 0x80000000ull;
constexpr std::uint64_t AC_QUARTER  = 0x40000000ull;
constexpr std::uint64_t AC_THREE_Q  = 0xC0000000ull;

inline std::vector<std::uint64_t> cumulative(const std::vector<std::uint32_t>& freq) {
    std::vector<std::uint64_t> cum(freq.size() + 1, 0);
    for (std::size_t i = 0; i < freq.size(); ++i) cum[i + 1] = cum[i] + freq[i];
    return cum;
}
}  // namespace detail

/// \brief Encode a symbol sequence under a static frequency model to a bit stream.
///
/// \param symbols  symbol indices (0..freq.size()-1).
/// \param freq     frequency (weight) of each symbol; the total should be well under 2^30.
inline std::vector<bool> arithmetic_encode(const std::vector<int>&           symbols,
                                           const std::vector<std::uint32_t>& freq) {
    using namespace detail;
    auto          cum   = cumulative(freq);
    std::uint64_t total = cum.back();
    std::uint64_t low = 0, high = AC_MASK;
    int           pending = 0;
    std::vector<bool> out;

    auto emit = [&](int bit) {
        out.push_back(bit);
        while (pending > 0) { out.push_back(bit ^ 1); --pending; }   // bit-plus-follow
    };

    for (int s : symbols) {
        std::uint64_t range = high - low + 1;
        high = low + range * cum[s + 1] / total - 1;
        low  = low + range * cum[s] / total;
        for (;;) {
            if (high < AC_HALF)            { emit(0); }
            else if (low >= AC_HALF)       { emit(1); low -= AC_HALF; high -= AC_HALF; }
            else if (low >= AC_QUARTER && high < AC_THREE_Q) { ++pending; low -= AC_QUARTER; high -= AC_QUARTER; }
            else break;
            low  = (low << 1) & AC_MASK;
            high = ((high << 1) | 1) & AC_MASK;
        }
    }
    ++pending;                                   // flush: one more bit disambiguates the interval
    emit(low < AC_QUARTER ? 0 : 1);
    return out;
}

/// \brief Decode \c count symbols from a bit stream under the same frequency model.
inline std::vector<int> arithmetic_decode(const std::vector<bool>&          bits,
                                          const std::vector<std::uint32_t>& freq, int count) {
    using namespace detail;
    auto          cum   = cumulative(freq);
    std::uint64_t total = cum.back();
    std::uint64_t low = 0, high = AC_MASK, value = 0;
    std::size_t   pos = 0;

    auto readbit = [&]() -> std::uint64_t { return pos < bits.size() ? (bits[pos++] ? 1 : 0) : 0; };
    for (int i = 0; i < 32; ++i) value = (value << 1) | readbit();

    std::vector<int> out;
    for (int c = 0; c < count; ++c) {
        std::uint64_t range  = high - low + 1;
        std::uint64_t scaled = ((value - low + 1) * total - 1) / range;   // where value sits
        int           s      = 0;
        while (s + 1 < (int)cum.size() && cum[s + 1] <= scaled) ++s;
        out.push_back(s);

        high = low + range * cum[s + 1] / total - 1;
        low  = low + range * cum[s] / total;
        for (;;) {
            if (high < AC_HALF)            { /* shift in a 0 region */ }
            else if (low >= AC_HALF)       { low -= AC_HALF; high -= AC_HALF; value -= AC_HALF; }
            else if (low >= AC_QUARTER && high < AC_THREE_Q) { low -= AC_QUARTER; high -= AC_QUARTER; value -= AC_QUARTER; }
            else break;
            low   = (low << 1) & AC_MASK;
            high  = ((high << 1) | 1) & AC_MASK;
            value = ((value << 1) | readbit()) & AC_MASK;
        }
    }
    return out;
}

}  // namespace datamunge::algorithms
