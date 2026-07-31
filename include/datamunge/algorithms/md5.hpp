#pragma once

/// \file md5.hpp
/// \brief The MD5 message-digest algorithm (RFC 1321).
///
/// MD5 (Rivest, 1992) compresses an arbitrary-length message into a fixed 128-bit digest.
/// It processes the padded message in 512-bit blocks, running a 64-step
/// Merkle-Damgard compression over four 32-bit state words \f$(A,B,C,D)\f$. Each step
/// applies one of four nonlinear round functions, adds a message word and a
/// per-step sine-derived constant \f$K_i = \lfloor 2^{32}\,|\sin(i+1)|\rfloor\f$, rotates
/// left, and accumulates.
///
/// MD5 is now *cryptographically broken*: practical collisions are known (Wang 2004),
/// so it must never be used for security (signatures, integrity against an adversary).
/// It survives as a fast non-adversarial checksum and as a clean teaching example of the
/// Merkle-Damgard construction. This implementation reproduces the RFC 1321 test suite.

#include <array>
#include <cstdint>
#include <cstddef>
#include <string>

#include <datamunge/algorithms/hex_encoding.hpp>

namespace datamunge::algorithms {

namespace detail {

/// Per-step left-rotation amounts s[0..63].
inline constexpr int md5_s[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

/// Per-step additive constants K[i] = floor(2^32 * |sin(i+1)|).
inline constexpr std::uint32_t md5_k[64] = {
    0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu, 0x4787c62au,
    0xa8304613u, 0xfd469501u, 0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu,
    0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u, 0xf61e2562u, 0xc040b340u,
    0x265e5a51u, 0xe9b6c7aau, 0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
    0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu, 0xa9e3e905u, 0xfcefa3f8u,
    0x676f02d9u, 0x8d2a4c8au, 0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu,
    0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u, 0x289b7ec6u, 0xeaa127fau,
    0xd4ef3085u, 0x04881d05u, 0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
    0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u,
    0xffeff47du, 0x85845dd1u, 0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
    0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u};

inline std::uint32_t md5_rotl(std::uint32_t x, int c) {
    return (x << c) | (x >> (32 - c));
}

}  // namespace detail

/// \brief Compute the 128-bit MD5 digest of a byte string.
/// \returns the 16 digest bytes in RFC 1321 output order.
inline std::array<std::uint8_t, 16> md5(const std::string& message) {
    std::uint32_t a0 = 0x67452301u, b0 = 0xefcdab89u, c0 = 0x98badcfeu, d0 = 0x10325476u;

    // Padding: append 0x80, then zeros, until length r 56 (mod 64), then 64-bit length.
    std::string msg = message;
    const std::uint64_t bit_len = static_cast<std::uint64_t>(message.size()) * 8u;
    msg.push_back(static_cast<char>(0x80));
    while (msg.size() % 64 != 56) msg.push_back(0);
    for (int i = 0; i < 8; ++i) msg.push_back(static_cast<char>((bit_len >> (8 * i)) & 0xFF));

    for (std::size_t off = 0; off < msg.size(); off += 64) {
        std::uint32_t m[16];
        for (int i = 0; i < 16; ++i) {
            m[i] = static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i])) |
                   (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 1])) << 8) |
                   (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 2])) << 16) |
                   (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 3])) << 24);
        }

        std::uint32_t a = a0, b = b0, c = c0, d = d0;
        for (int i = 0; i < 64; ++i) {
            std::uint32_t f;
            int g;
            if (i < 16) {
                f = (b & c) | (~b & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | (~d & c);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3 * i + 5) % 16;
            } else {
                f = c ^ (b | ~d);
                g = (7 * i) % 16;
            }
            f = f + a + detail::md5_k[i] + m[g];
            a = d;
            d = c;
            c = b;
            b = b + detail::md5_rotl(f, detail::md5_s[i]);
        }
        a0 += a;
        b0 += b;
        c0 += c;
        d0 += d;
    }

    std::array<std::uint8_t, 16> digest{};
    const std::uint32_t words[4] = {a0, b0, c0, d0};
    for (int w = 0; w < 4; ++w)
        for (int i = 0; i < 4; ++i)
            digest[4 * w + i] = static_cast<std::uint8_t>((words[w] >> (8 * i)) & 0xFF);
    return digest;
}

/// \brief MD5 digest of a string as a 32-character lowercase hex string.
inline std::string md5_hex(const std::string& message) { return to_hex(md5(message)); }

}  // namespace datamunge::algorithms
