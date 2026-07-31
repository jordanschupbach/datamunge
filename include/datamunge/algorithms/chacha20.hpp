#pragma once

/// \file chacha20.hpp
/// \brief The ChaCha20 stream cipher (RFC 8439, Bernstein).
///
/// ChaCha20 turns a 256-bit key, a 96-bit nonce, and a 32-bit block counter into a
/// pseudorandom *keystream*; encryption is XOR of the plaintext with that stream. Its core
/// is a 512-bit state of sixteen 32-bit words -- four constants, eight key words, one
/// counter, three nonce words -- stirred by 20 rounds of the ARX *quarter-round*
/// (add, rotate, XOR):
/// \f[
///   a\mathrel{+}=b;\ d\oplus\!=a;\ d=\text{ROTL}_{16}d;\quad
///   c\mathrel{+}=d;\ b\oplus\!=c;\ b=\text{ROTL}_{12}b;\ \dots
/// \f]
/// applied down the four columns and then the four diagonals, ten times. The stirred state
/// is added back to the original (feed-forward) and serialised little-endian as a 64-byte
/// keystream block. ChaCha20 has no known practical attack and is the stream cipher in
/// TLS 1.3, WireGuard, and SSH. This implementation reproduces the RFC 8439 test vectors.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

inline std::uint32_t chacha_rotl(std::uint32_t x, int c) {
    return (x << c) | (x >> (32 - c));
}

inline void chacha_quarter(std::uint32_t& a, std::uint32_t& b, std::uint32_t& c,
                           std::uint32_t& d) {
    a += b; d ^= a; d = chacha_rotl(d, 16);
    c += d; b ^= c; b = chacha_rotl(b, 12);
    a += b; d ^= a; d = chacha_rotl(d, 8);
    c += d; b ^= c; b = chacha_rotl(b, 7);
}

inline std::uint32_t chacha_load32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

}  // namespace detail

/// \brief Produce one 64-byte ChaCha20 keystream block.
/// \param key   32-byte key.
/// \param counter 32-bit block counter.
/// \param nonce 12-byte nonce.
inline std::array<std::uint8_t, 64> chacha20_block(const std::array<std::uint8_t, 32>& key,
                                                   std::uint32_t counter,
                                                   const std::array<std::uint8_t, 12>& nonce) {
    std::uint32_t s[16];
    s[0] = 0x61707865u;  // "expa"
    s[1] = 0x3320646eu;  // "nd 3"
    s[2] = 0x79622d32u;  // "2-by"
    s[3] = 0x6b206574u;  // "te k"
    for (int i = 0; i < 8; ++i) s[4 + i] = detail::chacha_load32(key.data() + 4 * i);
    s[12] = counter;
    for (int i = 0; i < 3; ++i) s[13 + i] = detail::chacha_load32(nonce.data() + 4 * i);

    std::uint32_t x[16];
    for (int i = 0; i < 16; ++i) x[i] = s[i];
    for (int r = 0; r < 10; ++r) {
        detail::chacha_quarter(x[0], x[4], x[8], x[12]);
        detail::chacha_quarter(x[1], x[5], x[9], x[13]);
        detail::chacha_quarter(x[2], x[6], x[10], x[14]);
        detail::chacha_quarter(x[3], x[7], x[11], x[15]);
        detail::chacha_quarter(x[0], x[5], x[10], x[15]);
        detail::chacha_quarter(x[1], x[6], x[11], x[12]);
        detail::chacha_quarter(x[2], x[7], x[8], x[13]);
        detail::chacha_quarter(x[3], x[4], x[9], x[14]);
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

/// \brief Encrypt (or decrypt) \p data with ChaCha20 starting at block \p counter.
/// Because it is a stream cipher, decryption is the identical operation.
inline std::string chacha20_encrypt(const std::array<std::uint8_t, 32>& key,
                                    std::uint32_t counter,
                                    const std::array<std::uint8_t, 12>& nonce,
                                    const std::string& data) {
    std::string out = data;
    for (std::size_t i = 0; i < out.size(); i += 64) {
        auto ks = chacha20_block(key, counter + static_cast<std::uint32_t>(i / 64), nonce);
        std::size_t n = std::min<std::size_t>(64, out.size() - i);
        for (std::size_t j = 0; j < n; ++j)
            out[i + j] = static_cast<char>(static_cast<std::uint8_t>(out[i + j]) ^ ks[j]);
    }
    return out;
}

}  // namespace datamunge::algorithms
