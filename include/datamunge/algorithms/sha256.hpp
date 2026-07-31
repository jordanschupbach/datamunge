#pragma once

/// \file sha256.hpp
/// \brief The SHA-256 secure hash algorithm (FIPS 180-4, SHA-2 family).
///
/// SHA-256 produces a 256-bit digest. It is a Merkle-Damgard hash over 512-bit
/// blocks with eight 32-bit state words \f$(a,\dots,h)\f$ and 64 rounds. Each block
/// expands its sixteen message words to sixty-four with
/// \f[
///   W_t = \sigma_1(W_{t-2}) + W_{t-7} + \sigma_0(W_{t-15}) + W_{t-16},
/// \f]
/// then applies the round function using the majority/choice mixers and the big-sigma
/// rotations, adding per-round constants \f$K_t\f$ (fractional parts of the cube roots of
/// the first 64 primes). The initial state words are the fractional parts of the square
/// roots of the first 8 primes.
///
/// Unlike MD5 and SHA-1, SHA-256 has *no known practical collision or preimage attack*
/// and is a current standard for digital signatures, HMAC, TLS, and blockchains. This
/// implementation reproduces the FIPS 180-4 test vectors.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <datamunge/algorithms/hex_encoding.hpp>

namespace datamunge::algorithms {

namespace detail {

inline constexpr std::uint32_t sha256_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

inline std::uint32_t sha256_rotr(std::uint32_t x, int c) {
    return (x >> c) | (x << (32 - c));
}

}  // namespace detail

/// \brief Compute the 256-bit SHA-256 digest of a byte string.
/// \returns the 32 digest bytes in big-endian FIPS 180-4 order.
inline std::array<std::uint8_t, 32> sha256(const std::string& message) {
    std::uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                          0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

    std::string msg = message;
    const std::uint64_t bit_len = static_cast<std::uint64_t>(message.size()) * 8u;
    msg.push_back(static_cast<char>(0x80));
    while (msg.size() % 64 != 56) msg.push_back(0);
    for (int i = 7; i >= 0; --i) msg.push_back(static_cast<char>((bit_len >> (8 * i)) & 0xFF));

    for (std::size_t off = 0; off < msg.size(); off += 64) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i])) << 24) |
                   (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 1])) << 16) |
                   (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 2])) << 8) |
                   (static_cast<std::uint32_t>(static_cast<std::uint8_t>(msg[off + 4 * i + 3])));
        }
        for (int i = 16; i < 64; ++i) {
            std::uint32_t s0 = detail::sha256_rotr(w[i - 15], 7) ^
                               detail::sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            std::uint32_t s1 = detail::sha256_rotr(w[i - 2], 17) ^
                               detail::sha256_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6],
                      hh = h[7];
        for (int i = 0; i < 64; ++i) {
            std::uint32_t S1 = detail::sha256_rotr(e, 6) ^ detail::sha256_rotr(e, 11) ^
                               detail::sha256_rotr(e, 25);
            std::uint32_t ch = (e & f) ^ (~e & g);
            std::uint32_t t1 = hh + S1 + ch + detail::sha256_k[i] + w[i];
            std::uint32_t S0 = detail::sha256_rotr(a, 2) ^ detail::sha256_rotr(a, 13) ^
                               detail::sha256_rotr(a, 22);
            std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            std::uint32_t t2 = S0 + maj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    std::array<std::uint8_t, 32> digest{};
    for (int w = 0; w < 8; ++w)
        for (int i = 0; i < 4; ++i)
            digest[4 * w + i] = static_cast<std::uint8_t>((h[w] >> (24 - 8 * i)) & 0xFF);
    return digest;
}

/// \brief SHA-256 digest of a string as a 64-character lowercase hex string.
inline std::string sha256_hex(const std::string& message) { return to_hex(sha256(message)); }

}  // namespace datamunge::algorithms
