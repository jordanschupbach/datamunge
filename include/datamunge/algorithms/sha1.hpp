#pragma once

/// \file sha1.hpp
/// \brief The SHA-1 secure hash algorithm (FIPS 180, US NIST).
///
/// SHA-1 produces a 160-bit digest from an arbitrary-length message. Like MD5 it is a
/// Merkle-Damgard construction over 512-bit blocks, but with a 160-bit state
/// \f$(h_0,\dots,h_4)\f$ and 80 rounds. Each block first expands its sixteen 32-bit words
/// into eighty via the recurrence
/// \f[
///   W_t = \text{ROTL}_1\!\left(W_{t-3}\oplus W_{t-8}\oplus W_{t-14}\oplus W_{t-16}\right),
/// \f]
/// then mixes them through four 20-round phases with distinct round functions and
/// constants.
///
/// SHA-1 is *deprecated for security*: a practical collision (SHAttered, 2017) makes it
/// unsafe for signatures or integrity against an adversary. It remains useful as a
/// content fingerprint (e.g. git object ids) and as a clear illustration of the
/// message-schedule expansion. This implementation reproduces the FIPS 180 test vectors.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <datamunge/algorithms/hex_encoding.hpp>

namespace datamunge::algorithms {

namespace detail {

inline std::uint32_t sha1_rotl(std::uint32_t x, int c) {
    return (x << c) | (x >> (32 - c));
}

}  // namespace detail

/// \brief Compute the 160-bit SHA-1 digest of a byte string.
/// \returns the 20 digest bytes in big-endian FIPS 180 order.
inline std::array<std::uint8_t, 20> sha1(const std::string& message) {
    std::uint32_t h0 = 0x67452301u, h1 = 0xEFCDAB89u, h2 = 0x98BADCFEu, h3 = 0x10325476u,
                  h4 = 0xC3D2E1F0u;

    // Padding: append 0x80, zeros to length r 56 (mod 64), then 64-bit big-endian length.
    std::string msg = message;
    const std::uint64_t bit_len = static_cast<std::uint64_t>(message.size()) * 8u;
    msg.push_back(static_cast<char>(0x80));
    while (msg.size() % 64 != 56) msg.push_back(0);
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<char>((bit_len >> (8 * i)) & 0xFF));

    for (std::size_t off = 0; off < msg.size(); off += 64) {
        std::uint32_t w[80];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i])) << 24) |
                   (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 1])) << 16) |
                   (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 2])) << 8) |
                   (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 3])));
        }
        for (int i = 16; i < 80; ++i)
            w[i] = detail::sha1_rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

        std::uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; ++i) {
            std::uint32_t f, k;
            if (i < 20) {
                f = (b & c) | (~b & d);
                k = 0x5A827999u;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1u;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDCu;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6u;
            }
            std::uint32_t temp = detail::sha1_rotl(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = detail::sha1_rotl(b, 30);
            b = a;
            a = temp;
        }
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }

    std::array<std::uint8_t, 20> digest{};
    const std::uint32_t hs[5] = {h0, h1, h2, h3, h4};
    for (int w = 0; w < 5; ++w)
        for (int i = 0; i < 4; ++i)
            digest[4 * w + i] = static_cast<std::uint8_t>((hs[w] >> (24 - 8 * i)) & 0xFF);
    return digest;
}

/// \brief SHA-1 digest of a string as a 40-character lowercase hex string.
inline std::string sha1_hex(const std::string& message) { return to_hex(sha1(message)); }

}  // namespace datamunge::algorithms
