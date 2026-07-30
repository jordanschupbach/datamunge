#pragma once

/// \file histogram_equalization.hpp
/// \brief Histogram equalization: enhance image contrast by remapping intensities so
///        their histogram is (approximately) uniform.
///
/// A low-contrast image has its intensities bunched into a narrow band. *Histogram
/// equalization* spreads them across the full range by remapping each level through the
/// *cumulative distribution function* (CDF) of the intensity histogram. Because the CDF
/// is steep where pixels are common, equalization stretches the busy tonal ranges and
/// compresses the empty ones, maximizing global contrast and often revealing hidden
/// detail. The transform is
/// \f[
///   T(v) = \operatorname{round}\!\Big(\frac{\mathrm{cdf}(v) - \mathrm{cdf}_{\min}}
///        {N - \mathrm{cdf}_{\min}}\,(L-1)\Big),
/// \f]
/// with \f$L=256\f$ levels. It is a standard contrast-enhancement step in photography,
/// medical imaging, and computer vision (where adaptive/local variants like CLAHE refine
/// it).

#include <array>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// \brief Equalize the histogram of an 8-bit grayscale image (levels 0-255).
///
/// \param image  Grayscale intensities (values clamped to 0-255).
/// \return the contrast-enhanced image and, optionally usable, the 256-entry mapping.
inline std::vector<std::vector<int>> histogram_equalize(const std::vector<std::vector<int>>& image) {
    const std::size_t rows = image.size();
    const std::size_t cols = rows ? image.front().size() : 0;
    std::vector<std::vector<int>> out(rows, std::vector<int>(cols, 0));
    if (rows == 0 || cols == 0) return out;

    std::array<std::size_t, 256> hist{};
    for (const auto& row : image)
        for (int v : row) {
            int c = v < 0 ? 0 : (v > 255 ? 255 : v);
            ++hist[static_cast<std::size_t>(c)];
        }

    // Cumulative distribution and its first nonzero value.
    std::array<std::size_t, 256> cdf{};
    std::size_t                  running = 0, cdf_min = 0;
    bool                         found_min = false;
    for (int v = 0; v < 256; ++v) {
        running += hist[static_cast<std::size_t>(v)];
        cdf[static_cast<std::size_t>(v)] = running;
        if (!found_min && running > 0) { cdf_min = running; found_min = true; }
    }
    const std::size_t n = rows * cols;

    // Build the 0..255 remapping through the normalized CDF.
    std::array<int, 256> map{};
    const double         denom = static_cast<double>(n - cdf_min);
    for (int v = 0; v < 256; ++v) {
        if (denom <= 0.0) { map[static_cast<std::size_t>(v)] = v; continue; }
        double t = (static_cast<double>(cdf[static_cast<std::size_t>(v)]) - static_cast<double>(cdf_min)) / denom * 255.0;
        if (t < 0) t = 0;
        if (t > 255) t = 255;
        map[static_cast<std::size_t>(v)] = static_cast<int>(t + 0.5);
    }
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) {
            int c = image[i][j] < 0 ? 0 : (image[i][j] > 255 ? 255 : image[i][j]);
            out[i][j] = map[static_cast<std::size_t>(c)];
        }
    return out;
}

}  // namespace datamunge::algorithms
