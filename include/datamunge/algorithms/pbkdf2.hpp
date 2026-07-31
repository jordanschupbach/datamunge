#pragma once

/// \file pbkdf2.hpp
/// \brief PBKDF2 password-based key derivation (RFC 2898 / PKCS #5), over HMAC-SHA-256.
///
/// A password is a poor key: short, low-entropy, and guessable. PBKDF2 *stretches* it into
/// a strong key by iterating a pseudorandom function (here HMAC-SHA-256) thousands of times,
/// so that each guess an attacker tries costs the same thousands of hashes. For output block
/// \f$i\f$ it computes
/// \f[
///   T_i = U_1 \oplus U_2 \oplus \dots \oplus U_c,\qquad
///   U_1 = \mathrm{PRF}(P,\ S\,\|\,\mathrm{INT32BE}(i)),\quad U_j = \mathrm{PRF}(P, U_{j-1}),
/// \f]
/// concatenating the \f$T_i\f$ and truncating to the desired length. The *salt* \f$S\f$ makes
/// every derivation unique, defeating rainbow tables; the *iteration count* \f$c\f$ sets the
/// cost. This implementation reproduces the standard PBKDF2-HMAC-SHA256 test vectors.
///
/// PBKDF2 is a sound, widely-deployed KDF (WPA2, LUKS, 1Password), but it is only
/// CPU-hard -- memory-hard functions (scrypt, Argon2) resist GPU/ASIC cracking better and
/// are preferred for new password-hashing designs.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <datamunge/algorithms/hex_encoding.hpp>
#include <datamunge/algorithms/hmac.hpp>

namespace datamunge::algorithms {

/// \brief Derive a \p dk_len-byte key from \p password and \p salt with \p iterations rounds.
/// Uses HMAC-SHA-256 as the underlying pseudorandom function (32-byte blocks).
inline std::vector<std::uint8_t> pbkdf2_hmac_sha256(const std::string& password,
                                                    const std::string& salt,
                                                    std::uint32_t iterations,
                                                    std::size_t dk_len) {
    constexpr std::size_t hlen = 32;  // HMAC-SHA-256 output size
    std::vector<std::uint8_t> dk;
    dk.reserve(dk_len);

    const std::size_t blocks = (dk_len + hlen - 1) / hlen;
    for (std::uint32_t i = 1; i <= blocks; ++i) {
        // U1 = PRF(P, S || INT32BE(i))
        std::string msg = salt;
        msg.push_back(static_cast<char>((i >> 24) & 0xFF));
        msg.push_back(static_cast<char>((i >> 16) & 0xFF));
        msg.push_back(static_cast<char>((i >> 8) & 0xFF));
        msg.push_back(static_cast<char>(i & 0xFF));

        auto u = hmac_sha256(password, msg);
        std::array<std::uint8_t, hlen> t = u;  // running XOR accumulator T_i
        for (std::uint32_t j = 1; j < iterations; ++j) {
            std::string prev(reinterpret_cast<const char*>(u.data()), u.size());
            u = hmac_sha256(password, prev);
            for (std::size_t b = 0; b < hlen; ++b) t[b] ^= u[b];
        }
        for (std::size_t b = 0; b < hlen && dk.size() < dk_len; ++b) dk.push_back(t[b]);
    }
    return dk;
}

/// \brief PBKDF2-HMAC-SHA256 derived key as a lowercase hex string.
inline std::string pbkdf2_hmac_sha256_hex(const std::string& password, const std::string& salt,
                                          std::uint32_t iterations, std::size_t dk_len) {
    auto dk = pbkdf2_hmac_sha256(password, salt, iterations, dk_len);
    static const char* hexd = "0123456789abcdef";
    std::string out;
    out.reserve(2 * dk.size());
    for (std::uint8_t byte : dk) {
        out.push_back(hexd[byte >> 4]);
        out.push_back(hexd[byte & 0xF]);
    }
    return out;
}

}  // namespace datamunge::algorithms
