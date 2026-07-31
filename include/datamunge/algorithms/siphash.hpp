#pragma once

/// \file siphash.hpp
/// \brief SipHash-2-4, a fast keyed pseudorandom function (Aumasson & Bernstein, 2012).
///
/// SipHash is a *keyed* hash designed for short inputs: given a 128-bit secret key it maps a
/// message to a 64-bit value that an attacker cannot predict without the key. Its purpose is
/// to stop *hash-flooding* denial-of-service -- an adversary deliberately colliding a hash
/// table's buckets -- so it is the default hash for the tables in Python, Rust, and many
/// language runtimes. Internally it keeps four 64-bit words \f$(v_0,v_1,v_2,v_3)\f$ seeded
/// from the key, and stirs them with an ARX *SipRound*
/// \f[
///   v_0\mathrel{+}=v_1,\ v_1=\text{ROTL}_{13}v_1,\ v_1\oplus\!=v_0,\ v_0=\text{ROTL}_{32}v_0,\ \dots
/// \f]
/// It absorbs the message eight bytes at a time with two SipRounds each (the "2"),
/// appends the length, and finalises with four more rounds (the "4"), hence *SipHash-2-4*.
/// This implementation reproduces the reference test vector from the SipHash paper.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace datamunge::algorithms {

namespace detail {

inline std::uint64_t sip_rotl(std::uint64_t x, int c) {
    return (x << c) | (x >> (64 - c));
}

inline void sip_round(std::uint64_t& v0, std::uint64_t& v1, std::uint64_t& v2,
                      std::uint64_t& v3) {
    v0 += v1; v1 = sip_rotl(v1, 13); v1 ^= v0; v0 = sip_rotl(v0, 32);
    v2 += v3; v3 = sip_rotl(v3, 16); v3 ^= v2;
    v0 += v3; v3 = sip_rotl(v3, 21); v3 ^= v0;
    v2 += v1; v1 = sip_rotl(v1, 17); v1 ^= v2; v2 = sip_rotl(v2, 32);
}

}  // namespace detail

/// \brief SipHash-2-4 of \p message under the 16-byte key \p key.
/// \returns the 64-bit keyed hash.
inline std::uint64_t siphash24(const std::array<std::uint8_t, 16>& key,
                               const std::string& message) {
    auto load64 = [](const std::uint8_t* p) {
        std::uint64_t r = 0;
        for (int i = 0; i < 8; ++i) r |= static_cast<std::uint64_t>(p[i]) << (8 * i);
        return r;
    };
    const std::uint64_t k0 = load64(key.data());
    const std::uint64_t k1 = load64(key.data() + 8);

    std::uint64_t v0 = 0x736f6d6570736575ULL ^ k0;
    std::uint64_t v1 = 0x646f72616e646f6dULL ^ k1;
    std::uint64_t v2 = 0x6c7967656e657261ULL ^ k0;
    std::uint64_t v3 = 0x7465646279746573ULL ^ k1;

    const std::size_t len = message.size();
    const auto* data = reinterpret_cast<const std::uint8_t*>(message.data());
    const std::size_t end = len - (len % 8);

    for (std::size_t off = 0; off < end; off += 8) {
        std::uint64_t m = load64(data + off);
        v3 ^= m;
        detail::sip_round(v0, v1, v2, v3);
        detail::sip_round(v0, v1, v2, v3);
        v0 ^= m;
    }

    // Last block: remaining bytes plus the message length in the top byte.
    std::uint64_t b = static_cast<std::uint64_t>(len) << 56;
    for (std::size_t i = 0; i < (len % 8); ++i)
        b |= static_cast<std::uint64_t>(data[end + i]) << (8 * i);
    v3 ^= b;
    detail::sip_round(v0, v1, v2, v3);
    detail::sip_round(v0, v1, v2, v3);
    v0 ^= b;

    v2 ^= 0xff;
    detail::sip_round(v0, v1, v2, v3);
    detail::sip_round(v0, v1, v2, v3);
    detail::sip_round(v0, v1, v2, v3);
    detail::sip_round(v0, v1, v2, v3);
    return v0 ^ v1 ^ v2 ^ v3;
}

}  // namespace datamunge::algorithms
