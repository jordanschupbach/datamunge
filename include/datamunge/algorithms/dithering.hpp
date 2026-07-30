#pragma once

/// \file dithering.hpp
/// \brief Two halftoning algorithms that render a grayscale image in fewer levels
///        (here 1-bit black/white) while preserving apparent tone: Floyd-Steinberg
///        error diffusion and ordered (Bayer) dithering.
///
/// Reducing an image to few levels naively (thresholding) destroys detail; *dithering*
/// instead arranges the coarse levels so that, at a distance, the average matches the
/// original. Two classic approaches:
///   - *Floyd-Steinberg* (1976) *error diffusion*: quantize a pixel, then push the
///     quantization error to not-yet-visited neighbors with weights 7/16, 3/16, 5/16,
///     1/16, so errors cancel over a region. It gives high-quality, organic-looking
///     halftones and is the default "dither" in image tools.
///   - *Ordered (Bayer) dithering*: compare each pixel to a fixed threshold from a small
///     tiled *Bayer matrix*. It is stateless, parallel, and fast, with a characteristic
///     cross-hatch texture, used in real-time graphics and printing.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// \brief Floyd-Steinberg error-diffusion dithering of a grayscale image to 1-bit.
///
/// \param image  Grayscale intensities (0-255).
/// \return a binary image (0 or 255) whose local average approximates the input.
inline std::vector<std::vector<int>> floyd_steinberg_dither(std::vector<std::vector<double>> image) {
    const std::size_t rows = image.size();
    const std::size_t cols = rows ? image.front().size() : 0;
    std::vector<std::vector<int>> out(rows, std::vector<int>(cols, 0));
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) {
            const double old_val = image[i][j];
            const int    newv    = old_val < 128.0 ? 0 : 255;  // nearest of the two levels
            out[i][j]            = newv;
            const double err = old_val - newv;
            // Diffuse the error to future neighbors (Floyd-Steinberg weights).
            if (j + 1 < cols) image[i][j + 1] += err * 7.0 / 16.0;
            if (i + 1 < rows) {
                if (j > 0) image[i + 1][j - 1] += err * 3.0 / 16.0;
                image[i + 1][j] += err * 5.0 / 16.0;
                if (j + 1 < cols) image[i + 1][j + 1] += err * 1.0 / 16.0;
            }
        }
    return out;
}

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
