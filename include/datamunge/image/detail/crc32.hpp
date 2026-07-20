#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace datamunge::image::detail {

/// @brief The standard reflected CRC-32 (IEEE 802.3) lookup table -- polynomial 0xEDB88320,
///        the same algorithm PNG's own spec (Annex D) and zlib's crc32() use. Built once, at
///        program startup, from the bit-reversed polynomial.
[[nodiscard]] inline const std::array<std::uint32_t, 256>& crc32_table() {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[n] = c;
        }
        return t;
    }();
    return table;
}

/// @brief The CRC-32 of @p data, continuing from a previous partial checksum @p crc (pass 0 to
///        start a new checksum) -- the incremental-update form needed to checksum a PNG
///        chunk's type and data as two separate calls without concatenating them first.
[[nodiscard]] inline std::uint32_t crc32_update(std::uint32_t crc, const unsigned char* data, std::size_t length) {
    const auto& table = crc32_table();
    std::uint32_t c = crc ^ 0xFFFFFFFFu;
    for (std::size_t i = 0; i < length; ++i) {
        c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

} // namespace datamunge::image::detail
