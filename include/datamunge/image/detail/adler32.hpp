#pragma once

#include <cstddef>
#include <cstdint>

namespace datamunge::image::detail {

/// @brief The Adler-32 checksum of @p data (RFC 1950's zlib stream trailer algorithm): two
///        16-bit running sums mod 65521 (the largest prime below 2^16), packed as (B << 16) |
///        A. Deliberately not incremental (unlike crc32_update()) -- the zlib wrapper only
///        ever needs it over one complete buffer at a time, so there's no call site that
///        benefits from a resumable form.
[[nodiscard]] inline std::uint32_t adler32(const unsigned char* data, std::size_t length) {
    constexpr std::uint32_t kModAdler = 65521;
    std::uint32_t a = 1, b = 0;
    for (std::size_t i = 0; i < length; ++i) {
        a = (a + data[i]) % kModAdler;
        b = (b + a) % kModAdler;
    }
    return (b << 16) | a;
}

} // namespace datamunge::image::detail
