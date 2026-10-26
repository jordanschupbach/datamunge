#pragma once

#include <datamunge/image/image.hpp>

#include <string>

namespace datamunge::image {

/// @brief Reads an image file of any supported format, chosen by its leading bytes rather than
///        its extension: PNG (read_png), JPEG (read_jpeg with default options: Grayscale or RGB),
///        BMP (read_bmp), and binary Netpbm P6/P5/P4 (read_ppm/read_pgm/read_pbm). Throws
///        std::runtime_error for an unreadable file or an unrecognised signature.
[[nodiscard]] Image read_image(const std::string& path);

}  // namespace datamunge::image
