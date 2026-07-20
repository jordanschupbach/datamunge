#pragma once

#include <datamunge/image/image.hpp>

#include <cstdint>
#include <string>

namespace datamunge::image {

/// @brief Reads a binary (P6) Netpbm color image. Only 8-bit-per-channel (maxval <= 255)
///        files are supported. Throws std::runtime_error on a malformed file or I/O failure.
[[nodiscard]] Image read_ppm(const std::string& path);

/// @brief Writes @p img as a binary (P6) Netpbm color image, converting to RGB first if it
///        isn't already (see to_rgb()).
void write_ppm(const Image& img, const std::string& path);

/// @brief Reads a binary (P5) Netpbm grayscale image. Only 8-bit (maxval <= 255) files are
///        supported. Throws std::runtime_error on a malformed file or I/O failure.
[[nodiscard]] Image read_pgm(const std::string& path);

/// @brief Writes @p img as a binary (P5) Netpbm grayscale image, converting to Grayscale first
///        if it isn't already (see to_grayscale()).
void write_pgm(const Image& img, const std::string& path);

/// @brief Reads a binary (P4) Netpbm bitmap image (1 bit per pixel: set bits are black) as a
///        Grayscale image (0 = black, 255 = white). Throws std::runtime_error on a malformed
///        file or I/O failure.
[[nodiscard]] Image read_pbm(const std::string& path);

/// @brief Writes @p img as a binary (P4) Netpbm bitmap, converting to grayscale first and then
///        thresholding: pixels with luma strictly below @p threshold become set (black) bits.
void write_pbm(const Image& img, const std::string& path, std::uint8_t threshold = 128);

} // namespace datamunge::image
