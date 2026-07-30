#pragma once

/// \file range_coding.hpp
/// \brief Range coding: byte-oriented arithmetic coding.
///
/// Range coding is arithmetic coding viewed as narrowing an integer *range* rather than a
/// \f$[0,1)\f$ interval, and renormalized a *byte* at a time instead of a bit at a time. Each
/// symbol scales the current range by its probability and shifts the low end; when the range
/// gets small enough, the top byte of the low end is settled and shifted out. Byte-wise
/// renormalization makes it fast, and the carry-free variant (Subbotin's) avoids the carry
/// propagation that complicates a naive arithmetic coder. It achieves the same near-entropy
/// compression as arithmetic coding and was historically used to sidestep arithmetic-coding
/// patents (it is the coder in LZMA, among others). This module implements the carry-free range
/// coder.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace detail {
constexpr std::uint32_t RC_TOP = 1u << 24;
constexpr std::uint32_t RC_BOT = 1u << 16;

inline std::vector<std::uint32_t> rc_cumulative(const std::vector<std::uint32_t>& freq) {
    std::vector<std::uint32_t> cum(freq.size() + 1, 0);
    for (std::size_t i = 0; i < freq.size(); ++i) cum[i + 1] = cum[i] + freq[i];
    return cum;
}
}  // namespace detail

/// \brief Encode a symbol sequence under a static frequency model to a byte stream.
inline std::vector<std::uint8_t> range_encode(const std::vector<int>&           symbols,
                                              const std::vector<std::uint32_t>& freq) {
    using namespace detail;
    auto          cum   = rc_cumulative(freq);
    std::uint32_t total = cum.back();
    std::uint32_t low = 0, range = 0xFFFFFFFFu;
    std::vector<std::uint8_t> out;

    for (int s : symbols) {
        range /= total;
        low += cum[s] * range;
        range *= freq[s];
        while ((low ^ (low + range)) < RC_TOP ||
               (range < RC_BOT && ((range = (0u - low) & (RC_BOT - 1)), true))) {
            out.push_back(static_cast<std::uint8_t>(low >> 24));
            low <<= 8;
            range <<= 8;
        }
    }
    for (int i = 0; i < 4; ++i) { out.push_back(static_cast<std::uint8_t>(low >> 24)); low <<= 8; }
    return out;
}

/// \brief Decode \c count symbols from a range-coded byte stream under the same model.
inline std::vector<int> range_decode(const std::vector<std::uint8_t>&  in,
                                     const std::vector<std::uint32_t>& freq, int count) {
    using namespace detail;
    auto          cum   = rc_cumulative(freq);
    std::uint32_t total = cum.back();
    std::uint32_t low = 0, range = 0xFFFFFFFFu, code = 0;
    std::size_t   pos = 0;
    auto          nextbyte = [&]() -> std::uint32_t { return pos < in.size() ? in[pos++] : 0; };
    for (int i = 0; i < 4; ++i) code = (code << 8) | nextbyte();

    std::vector<int> out;
    for (int c = 0; c < count; ++c) {
        range /= total;
        std::uint32_t val = (code - low) / range;
        if (val >= total) val = total - 1;
        int s = 0;
        while (s + 1 < (int)cum.size() && cum[s + 1] <= val) ++s;
        low += cum[s] * range;
        range *= freq[s];
        while ((low ^ (low + range)) < RC_TOP ||
               (range < RC_BOT && ((range = (0u - low) & (RC_BOT - 1)), true))) {
            code = (code << 8) | nextbyte();
            low <<= 8;
            range <<= 8;
        }
        out.push_back(s);
    }
    return out;
}

}  // namespace datamunge::algorithms
