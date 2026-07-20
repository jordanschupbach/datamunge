#pragma once

#include <datamunge/image/image.hpp>

#include <string>

namespace datamunge::image {

/// @brief Reads a PNG file as an Image (mode matches the file's color type: Grayscale,
///        GrayscaleAlpha, RGB, or RGBA). Only 8-bit-depth, non-interlaced, non-palette PNGs
///        are supported (the overwhelming majority of real-world PNGs); indexed-color
///        (palette), 16-bit-depth, and Adam7-interlaced files throw std::runtime_error, as
///        does a corrupt file (bad signature, chunk CRC mismatch, or truncated/invalid zlib
///        stream).
[[nodiscard]] Image read_png(const std::string& path);

/// @brief Writes @p img as an 8-bit-depth, non-interlaced PNG, using its ImageMode directly as
///        the PNG color type (Grayscale/GrayscaleAlpha/RGB/RGBA all have a direct PNG
///        equivalent, so no conversion is ever needed). Every scanline is written with filter
///        type 0 (None) -- simpler and always correct, at the cost of a somewhat larger file
///        than a filter-optimizing encoder would produce; any standard PNG decoder still reads
///        the result correctly.
void write_png(const Image& img, const std::string& path);

} // namespace datamunge::image
