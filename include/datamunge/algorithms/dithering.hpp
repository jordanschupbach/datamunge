#pragma once

/// @file dithering.hpp
/// @brief Binary-valued halftoning: native raster error diffusion and ordered Bayer thresholds.
/// Floyd-Steinberg uses an owned working copy and discards outside error shares;
/// arithmetic tone equality and a universal perceptual/speed ranking are not implied.

#include <datamunge/algorithms/floyd_steinberg.hpp>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// \brief Ordered (Bayer) dithering of a grayscale image to 1-bit using a 4x4 Bayer matrix.
inline std::vector<std::vector<int>> ordered_dither(const std::vector<std::vector<double>>& image) {
    static const int bayer4[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};
    const std::size_t rows = image.size();
    const std::size_t cols = rows ? image.front().size() : 0;
    std::vector<std::vector<int>> out(rows, std::vector<int>(cols, 0));
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) {
            // Threshold from the tiled Bayer matrix, scaled to 0..255.
            const double threshold = (bayer4[i % 4][j % 4] + 0.5) / 16.0 * 255.0;
            out[i][j] = image[i][j] > threshold ? 255 : 0;
        }
    return out;
}

}  // namespace datamunge::algorithms
