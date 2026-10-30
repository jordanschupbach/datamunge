#pragma once

/// @file adler32.hpp
/// @brief Raw-buffer Adler-32 for the zlib codec, forwarding to algorithms::adler32().

#include <datamunge/algorithms/checksums.hpp>

#include <cstddef>
#include <cstdint>

/// @brief Internal helpers of the image codecs.
namespace datamunge::image::detail {

/// @brief The Adler-32 checksum of @p data (RFC 1950's zlib stream trailer): forwards to
///        algorithms::adler32() for a raw buffer.
[[nodiscard]] inline std::uint32_t adler32(const unsigned char* data, std::size_t length) {
    return algorithms::adler32({reinterpret_cast<const char*>(data), length});
}

} // namespace datamunge::image::detail
