#pragma once

#include <cstddef>
#include <vector>

namespace datamunge::image::detail {

/// @brief Default cap on the output of the unsized decompressors (1 GiB) -- a guard against
///        decompression bombs.
inline constexpr std::size_t kDefaultMaxInflateOutput = std::size_t{1} << 30;

/// @brief Output limit for the unsized decompressors. A distinct type (rather than a bare
///        size_t) so it can never be confused with the sized overloads' expected_size.
struct InflateLimit {
    std::size_t max_output = kDefaultMaxInflateOutput;
};

/// @brief Decompresses a zlib-wrapped (RFC 1950: 2-byte header, raw DEFLATE/RFC 1951 stream,
///        4-byte big-endian Adler-32 trailer) buffer into exactly @p expected_size bytes.
///        Handles stored, fixed-Huffman and dynamic-Huffman blocks (any standard encoder).
///        Throws std::runtime_error on any format violation (bad header, invalid Huffman code,
///        corrupt stored length, back-reference before the output start, Adler-32 mismatch) or
///        when the decompressed size differs from @p expected_size.
[[nodiscard]] std::vector<unsigned char> zlib_decompress(const std::vector<unsigned char>& compressed, std::size_t expected_size);

/// @brief Decompresses a zlib stream of unknown decompressed size; the output grows as needed.
///        Throws std::runtime_error on any format violation, or if the output would exceed
///        @p limit.max_output bytes.
[[nodiscard]] std::vector<unsigned char> zlib_decompress(const std::vector<unsigned char>& compressed, InflateLimit limit = {});

/// @brief Decompresses a raw DEFLATE (RFC 1951) stream of @p size bytes -- no zlib header or
///        trailer -- into exactly @p expected_size bytes (the payload format of gzip members and
///        ZIP method-8 entries). Throws std::runtime_error on any format violation or a size
///        mismatch. Bytes after the final block are ignored.
[[nodiscard]] std::vector<unsigned char> deflate_decompress(const unsigned char* data, std::size_t size,
                                                            std::size_t expected_size);

/// @brief Decompresses a raw DEFLATE stream of unknown decompressed size. Throws
///        std::runtime_error on any format violation, or if the output would exceed
///        @p limit.max_output bytes. Bytes after the final block are ignored.
[[nodiscard]] std::vector<unsigned char> deflate_decompress(const unsigned char* data, std::size_t size, InflateLimit limit = {});

/// @brief Compresses @p data into a raw DEFLATE (RFC 1951) stream: LZ77 (hash chains, 32 KiB
///        window, lazy matching) plus Huffman coding, choosing per block the smallest of
///        stored / fixed / dynamic. @p level follows zlib: 0 = stored blocks only, 1..9 trade
///        speed for ratio (-1 = default 6). Output is deterministic. Throws
///        std::invalid_argument for a level outside -1..9.
[[nodiscard]] std::vector<unsigned char> deflate_compress(const std::vector<unsigned char>& data, int level = 6);

/// @brief Compresses @p data into a zlib (RFC 1950) stream: header, deflate_compress() payload,
///        Adler-32 trailer. Level semantics as for deflate_compress(); level 0 reproduces the
///        old stored-only output byte for byte.
[[nodiscard]] std::vector<unsigned char> zlib_compress(const std::vector<unsigned char>& data, int level = 6);

} // namespace datamunge::image::detail
