#pragma once

/// \file salsa20.hpp
/// \brief The Salsa20 stream cipher (Bernstein, eSTREAM portfolio).
///
/// Salsa20 is the predecessor of ChaCha20 and shares its shape: a 512-bit state of sixteen
/// 32-bit words -- four "expand 32-byte k" constants, eight key words, two nonce words, two
/// counter words -- stirred by 20 ARX rounds and fed forward. Its quarter-round differs
/// from ChaCha's in the rotation amounts and update order:
/// \f[
///   b\oplus\!=\text{ROTL}_{7}(a+d),\ \ c\oplus\!=\text{ROTL}_{9}(b+a),\ \
///   d\oplus\!=\text{ROTL}_{13}(c+b),\ \ a\oplus\!=\text{ROTL}_{18}(d+c),
/// \f]
/// and it alternates *column* rounds with *row* rounds rather than columns and diagonals.
/// ChaCha20 later reorganised these operations for better diffusion per round, but Salsa20
/// remains a clean, fast, well-analysed design. This implementation verifies the round-trip
/// property and Bernstein's documented Salsa20 core example.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace datamunge::algorithms {

namespace detail {

inline std::uint32_t salsa_rotl(std::uint32_t x, int c) {
    return (x << c) | (x >> (32 - c));
}

inline void salsa_quarter(std::uint32_t& a, std::uint32_t& b, std::uint32_t& c,
                          std::uint32_t& d) {
    b ^= salsa_rotl(a + d, 7);
    c ^= salsa_rotl(b + a, 9);
    d ^= salsa_rotl(c + b, 13);
    a ^= salsa_rotl(d + c, 18);
}

inline std::uint32_t salsa_load32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

}  // namespace detail

/// \brief The Salsa20 core: 20 rounds plus feed-forward on a 64-byte state block.
/// Exposed because it is the unit Bernstein's specification gives a test vector for.
inline std::array<std::uint8_t, 64> salsa20_core(const std::array<std::uint8_t, 64>& in) {
    std::uint32_t s[16], x[16];
    for (int i = 0; i < 16; ++i) {
        s[i] = detail::salsa_load32(in.data() + 4 * i);
        x[i] = s[i];
    }
    for (int r = 0; r < 10; ++r) {
        // column round
        detail::salsa_quarter(x[0], x[4], x[8], x[12]);
        detail::salsa_quarter(x[5], x[9], x[13], x[1]);
        detail::salsa_quarter(x[10], x[14], x[2], x[6]);
        detail::salsa_quarter(x[15], x[3], x[7], x[11]);
        // row round
        detail::salsa_quarter(x[0], x[1], x[2], x[3]);
        detail::salsa_quarter(x[5], x[6], x[7], x[4]);
        detail::salsa_quarter(x[10], x[11], x[8], x[9]);
        detail::salsa_quarter(x[15], x[12], x[13], x[14]);
    }
    std::array<std::uint8_t, 64> out{};
    for (int i = 0; i < 16; ++i) {
        std::uint32_t v = x[i] + s[i];
        out[4 * i + 0] = static_cast<std::uint8_t>(v & 0xFF);
        out[4 * i + 1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
        out[4 * i + 2] = static_cast<std::uint8_t>((v >> 16) & 0xFF);
        out[4 * i + 3] = static_cast<std::uint8_t>((v >> 24) & 0xFF);
    }
    return out;
}

/// \brief Produce one 64-byte Salsa20 keystream block for a 256-bit key.
/// \param key   32-byte key.
/// \param counter 64-bit block counter.
/// \param nonce 8-byte nonce.
inline std::array<std::uint8_t, 64> salsa20_block(const std::array<std::uint8_t, 32>& key,
                                                  std::uint64_t counter,
                                                  const std::array<std::uint8_t, 8>& nonce) {
    // Assemble the 64-byte input block in Salsa20 word layout, then run the core.
    std::uint32_t s[16];
    s[0] = 0x61707865u;  // "expa"
    s[5] = 0x3320646eu;  // "nd 3"
    s[10] = 0x79622d32u; // "2-by"
    s[15] = 0x6b206574u; // "te k"
    for (int i = 0; i < 4; ++i) s[1 + i] = detail::salsa_load32(key.data() + 4 * i);
    for (int i = 0; i < 4; ++i) s[11 + i] = detail::salsa_load32(key.data() + 16 + 4 * i);
    s[6] = detail::salsa_load32(nonce.data());
    s[7] = detail::salsa_load32(nonce.data() + 4);
    s[8] = static_cast<std::uint32_t>(counter & 0xFFFFFFFFu);
    s[9] = static_cast<std::uint32_t>(counter >> 32);

    std::array<std::uint8_t, 64> in{};
    for (int i = 0; i < 16; ++i) {
        in[4 * i + 0] = static_cast<std::uint8_t>(s[i] & 0xFF);
        in[4 * i + 1] = static_cast<std::uint8_t>((s[i] >> 8) & 0xFF);
        in[4 * i + 2] = static_cast<std::uint8_t>((s[i] >> 16) & 0xFF);
        in[4 * i + 3] = static_cast<std::uint8_t>((s[i] >> 24) & 0xFF);
    }
    return salsa20_core(in);
}

/// \brief Encrypt (or decrypt) \p data with Salsa20; decryption is the same operation.
inline std::string salsa20_encrypt(const std::array<std::uint8_t, 32>& key, std::uint64_t counter,
                                   const std::array<std::uint8_t, 8>& nonce,
                                   const std::string& data) {
    std::string out = data;
    for (std::size_t i = 0; i < out.size(); i += 64) {
        auto ks = salsa20_block(key, counter + i / 64, nonce);
        std::size_t n = std::min<std::size_t>(64, out.size() - i);
        for (std::size_t j = 0; j < n; ++j)
            out[i + j] = static_cast<char>(static_cast<std::uint8_t>(out[i + j]) ^ ks[j]);
    }
    return out;
}

}  // namespace datamunge::algorithms
