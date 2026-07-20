#pragma once

#include <cstddef>
#include <vector>

namespace datamunge::image::detail {

/// @brief Decompresses a zlib-wrapped (RFC 1950: 2-byte header, then a raw DEFLATE/RFC 1951
///        stream, then a 4-byte big-endian Adler-32 trailer) buffer into exactly @p
///        expected_size bytes. A full DEFLATE decoder -- handles stored, fixed-Huffman, and
///        dynamic-Huffman blocks, so it correctly reads zlib streams produced by ANY standard
///        encoder (this codebase's own zlib_compress() only ever emits stored blocks, but
///        PNGs found in the wild will use real Huffman+LZ77 compression). Throws
///        std::runtime_error on any format violation: a bad header, an invalid Huffman code,
///        a corrupt stored-block length, a back-reference pointing before the start of the
///        output, a Adler-32 mismatch, or a final size that doesn't match @p expected_size.
[[nodiscard]] std::vector<unsigned char> zlib_decompress(const std::vector<unsigned char>& compressed, std::size_t expected_size);

/// @brief Compresses @p data into a valid zlib stream using ONLY DEFLATE "stored"
///        (uncompressed) blocks -- no Huffman coding or LZ77 matching, so this produces no
///        actual compression (roughly a 5-byte-per-64KB-block overhead over the input size).
///        Still fully RFC 1951/1950-compliant output, readable by any standard zlib/DEFLATE
///        consumer (this codebase's own zlib_decompress() included). The deliberate tradeoff
///        for keeping the ENCODER simple: this module's read path needs the general decoder
///        regardless (to read arbitrary real-world PNGs), so the encoder gets to take the
///        cheapest correct route rather than re-deriving a compressor to match it.
[[nodiscard]] std::vector<unsigned char> zlib_compress(const std::vector<unsigned char>& data);

} // namespace datamunge::image::detail
