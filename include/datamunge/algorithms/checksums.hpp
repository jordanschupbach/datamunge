#pragma once

/// \file checksums.hpp
/// \brief Three classic error-detecting checksums: CRC-32, Adler-32, and Fletcher-16.
///
/// A *checksum* condenses a message into a short tag appended to it; the receiver
/// recomputes the tag and, if it differs, knows the message was corrupted. The three
/// here trade off strength for speed:
///   - *CRC-32* (cyclic redundancy check) treats the message as a polynomial over
///     GF(2) and returns the remainder modulo a fixed generator polynomial. CRCs
///     catch *all* burst errors up to the check width and are the standard for
///     Ethernet, ZIP, PNG, and disk sectors. This is the reflected IEEE 802.3 variant.
///   - *Adler-32* (Adler 1995, used in zlib) sums the bytes and the running sums modulo
///     the prime 65521 -- much faster than CRC but weaker on short messages.
///   - *Fletcher-16* (Fletcher 1982) is a position-sensitive two-byte running sum,
///     stronger than a plain sum and cheaper than a CRC.
/// Each is a deterministic function of the byte stream and is verified below against
/// its published reference value.

#include <cstddef>
#include <cstdint>
#include <string>

namespace datamunge::algorithms {

/// \brief CRC-32 (reflected IEEE 802.3 / zlib polynomial 0xEDB88320).
///
/// crc32("123456789") == 0xCBF43926.
inline std::uint32_t crc32(const std::string& data) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (unsigned char byte : data) {
        crc ^= byte;
        for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (~(crc & 1u) + 1u));
    }
    return crc ^ 0xFFFFFFFFu;
}

/// \brief Adler-32 (zlib), the checksum of RFC 1950.
///
/// adler32("Wikipedia") == 0x11E60398.
inline std::uint32_t adler32(const std::string& data) {
    constexpr std::uint32_t kMod = 65521u;
    std::uint32_t           a = 1, b = 0;
    for (unsigned char byte : data) {
        a = (a + byte) % kMod;
        b = (b + a) % kMod;
    }
    return (b << 16) | a;
}

/// \brief Fletcher-16 checksum.
///
/// fletcher16("abcde") == 0xC8F0.
inline std::uint16_t fletcher16(const std::string& data) {
    std::uint16_t sum1 = 0, sum2 = 0;
    for (unsigned char byte : data) {
        sum1 = static_cast<std::uint16_t>((sum1 + byte) % 255);
        sum2 = static_cast<std::uint16_t>((sum2 + sum1) % 255);
    }
    return static_cast<std::uint16_t>((sum2 << 8) | sum1);
}

}  // namespace datamunge::algorithms
