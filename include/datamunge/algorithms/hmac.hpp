#pragma once

/// \file hmac.hpp
/// \brief HMAC keyed-hash message authentication code (RFC 2104), instantiated with SHA-256.
///
/// A message authentication code lets two parties sharing a secret key \f$K\f$ verify that
/// a message was neither forged nor altered. HMAC builds a MAC from any Merkle-Damgard
/// hash \f$H\f$ by nesting two keyed hashes:
/// \f[
///   \mathrm{HMAC}(K,m) = H\big((K'\oplus \text{opad}) \,\|\, H((K'\oplus \text{ipad}) \,\|\, m)\big),
/// \f]
/// where \f$K'\f$ is the key padded (or hashed then padded) to the hash's block size \f$B\f$,
/// \f$\text{ipad}=0x36^{B}\f$ and \f$\text{opad}=0x5c^{B}\f$. The two-layer construction
/// makes HMAC provably secure against length-extension forgeries that plague a naive
/// \f$H(K\,\|\,m)\f$, given only that the underlying compression function is a PRF.
///
/// This implementation uses SHA-256 (block size 64 bytes, digest 32 bytes) and reproduces
/// the RFC 4231 HMAC-SHA-256 test vectors.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <datamunge/algorithms/hex_encoding.hpp>
#include <datamunge/algorithms/sha256.hpp>

namespace datamunge::algorithms {

/// \brief Compute HMAC-SHA-256 of \p message under secret \p key.
/// \returns the 32-byte authentication tag.
inline std::array<std::uint8_t, 32> hmac_sha256(const std::string& key,
                                                const std::string& message) {
    constexpr std::size_t block_size = 64;  // SHA-256 block size in bytes

    // Keys longer than the block are hashed down first; then zero-pad to the block size.
    std::string k_prime;
    if (key.size() > block_size) {
        auto kh = sha256(key);
        k_prime.assign(reinterpret_cast<const char*>(kh.data()), kh.size());
    } else {
        k_prime = key;
    }
    k_prime.resize(block_size, '\0');

    std::string inner_pad(block_size, '\0'), outer_pad(block_size, '\0');
    for (std::size_t i = 0; i < block_size; ++i) {
        auto kb = static_cast<std::uint8_t>(k_prime[i]);
        inner_pad[i] = static_cast<char>(kb ^ 0x36);
        outer_pad[i] = static_cast<char>(kb ^ 0x5c);
    }

    auto inner = sha256(inner_pad + message);
    std::string inner_bytes(reinterpret_cast<const char*>(inner.data()), inner.size());
    return sha256(outer_pad + inner_bytes);
}

/// \brief HMAC-SHA-256 tag as a 64-character lowercase hex string.
inline std::string hmac_sha256_hex(const std::string& key, const std::string& message) {
    return to_hex(hmac_sha256(key, message));
}

}  // namespace datamunge::algorithms
