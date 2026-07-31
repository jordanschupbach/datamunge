#pragma once

/// \file hex_encoding.hpp
/// \brief Shared helper: render a fixed-size byte digest as a lowercase hex string.
///
/// Used by the cryptographic hash headers (MD5, SHA-1, SHA-256, HMAC) so they present
/// digests in the conventional lowercase hexadecimal form found in test vectors.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace datamunge::algorithms {

/// \brief Lowercase hexadecimal string of an N-byte digest.
template <std::size_t N>
inline std::string to_hex(const std::array<std::uint8_t, N>& digest) {
    static const char* hexd = "0123456789abcdef";
    std::string out;
    out.reserve(2 * N);
    for (std::uint8_t byte : digest) {
        out.push_back(hexd[byte >> 4]);
        out.push_back(hexd[byte & 0xF]);
    }
    return out;
}

}  // namespace datamunge::algorithms
