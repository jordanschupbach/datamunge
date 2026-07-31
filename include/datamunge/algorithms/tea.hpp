#pragma once

/// \file tea.hpp
/// \brief The Tiny Encryption Algorithm (TEA; Wheeler & Needham, 1994).
///
/// TEA is a 64-bit block cipher with a 128-bit key, famous for how little code it takes: a
/// balanced *Feistel network* of 64 rounds (applied as 32 double-rounds) using only
/// addition, XOR, and shifts -- no S-boxes, no tables. A "magic" constant
/// \f$\delta = \lfloor 2^{32}/\phi\rfloor = \texttt{0x9E3779B9}\f$ (derived from the golden
/// ratio) is accumulated into a running \c sum each double-round to break symmetry between
/// rounds. Each half is updated from the other via
/// \f[
///   v_0 \mathrel{+}= ((v_1\!\ll\!4)+k_0)\oplus(v_1+\text{sum})\oplus((v_1\!\gg\!5)+k_1),
/// \f]
/// and symmetrically for \f$v_1\f$. Decryption runs the same structure with \c sum counting
/// down. TEA is easy to implement but has known weaknesses (equivalent keys, related-key
/// attacks) fixed in XTEA/XXTEA; it is a teaching-grade cipher, not for production. This
/// implementation verifies exact round-trip and the avalanche effect.

#include <array>
#include <cstdint>

namespace datamunge::algorithms {

/// \brief Encrypt a 64-bit block \p v (two 32-bit words) under the 128-bit key \p k.
inline std::array<std::uint32_t, 2> tea_encrypt(std::array<std::uint32_t, 2> v,
                                                const std::array<std::uint32_t, 4>& k) {
    std::uint32_t v0 = v[0], v1 = v[1], sum = 0;
    const std::uint32_t delta = 0x9E3779B9u;
    for (int i = 0; i < 32; ++i) {
        sum += delta;
        v0 += ((v1 << 4) + k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + k[1]);
        v1 += ((v0 << 4) + k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + k[3]);
    }
    return {v0, v1};
}

/// \brief Decrypt a 64-bit block \p v under the 128-bit key \p k (inverse of tea_encrypt).
inline std::array<std::uint32_t, 2> tea_decrypt(std::array<std::uint32_t, 2> v,
                                                const std::array<std::uint32_t, 4>& k) {
    std::uint32_t v0 = v[0], v1 = v[1];
    const std::uint32_t delta = 0x9E3779B9u;
    std::uint32_t sum = 0xC6EF3720u;  // delta * 32
    for (int i = 0; i < 32; ++i) {
        v1 -= ((v0 << 4) + k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + k[3]);
        v0 -= ((v1 << 4) + k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + k[1]);
        sum -= delta;
    }
    return {v0, v1};
}

}  // namespace datamunge::algorithms
