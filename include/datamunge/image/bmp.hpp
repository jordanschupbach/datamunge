#pragma once

#include <datamunge/image/image.hpp>

#include <string>

namespace datamunge::image {

/// @brief Reads an uncompressed (BI_RGB) Windows BMP file as an RGB image. Accepts 24-bit
///        (BGR), 32-bit (BGRx -- the 4th byte is ignored, never treated as alpha, since plain
///        BITMAPINFOHEADER 32-bit files don't reliably agree on what that byte means), and
///        8-bit palette (indexed-color) files; any other bit depth or a compressed
///        (non-BI_RGB) file throws std::runtime_error.
[[nodiscard]] Image read_bmp(const std::string& path);

/// @brief Writes @p img as an uncompressed 24-bit BGR Windows BMP (BITMAPINFOHEADER, BI_RGB),
///        converting to RGB first if it isn't already (see to_rgb()) -- any alpha channel is
///        dropped, matching write_ppm()'s same RGB-only-on-disk tradeoff.
void write_bmp(const Image& img, const std::string& path);

} // namespace datamunge::image
