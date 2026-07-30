#pragma once

/// \file median_filter.hpp
/// \brief The median filter: remove impulse (salt-and-pepper) noise by replacing each
///        pixel with the median of its neighborhood.
///
/// A mean/blur filter smears sharp outliers across their neighbors; the *median filter*
/// instead replaces each pixel by the *median* of the values in a window around it.
/// Because the median is a rank statistic, a few extreme values (salt = 255,
/// pepper = 0) fall to the ends of the sorted window and are discarded, so impulse noise
/// is removed *while edges stay sharp* -- the median of a step is still the step. It is
/// the standard nonlinear denoiser for salt-and-pepper corruption and a building block
/// of many image pipelines.

#include <algorithm>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// \brief Median-filter a grayscale image with a square window of the given \p radius.
///
/// Each output pixel is the median of the input pixels in the \f$(2r{+}1)\times(2r{+}1)\f$
/// window centered on it (clamped at the borders).
inline std::vector<std::vector<int>> median_filter(const std::vector<std::vector<int>>& image, int radius = 1) {
    const int rows = static_cast<int>(image.size());
    const int cols = rows ? static_cast<int>(image.front().size()) : 0;
    std::vector<std::vector<int>> out(static_cast<std::size_t>(rows), std::vector<int>(static_cast<std::size_t>(cols), 0));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            std::vector<int> window;
            for (int di = -radius; di <= radius; ++di)
                for (int dj = -radius; dj <= radius; ++dj) {
                    const int r = std::min(std::max(i + di, 0), rows - 1);  // clamp to border
                    const int c = std::min(std::max(j + dj, 0), cols - 1);
                    window.push_back(image[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)]);
                }
            std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
            out[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = window[window.size() / 2];
        }
    return out;
}

}  // namespace datamunge::algorithms
