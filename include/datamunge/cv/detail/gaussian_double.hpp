#pragma once

#include <datamunge/image/color.hpp>
#include <datamunge/image/image.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::cv::detail {

/// @brief Grayscale luma of every pixel of @p img as plain doubles (row-major), for algorithms
///        that need to accumulate/difference intensities without image::Image's uint8
///        [0,255] clamping (squared gradients, scale-space differences, etc. routinely exceed
///        that range or go negative).
[[nodiscard]] inline std::vector<double> to_double_luma(const image::Image& img) {
    std::vector<double> out(static_cast<std::size_t>(img.width()) * static_cast<std::size_t>(img.height()));
    std::size_t i = 0;
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) {
            const image::Pixel p = img.get_pixel(x, y);
            out[i++] = image::rgb_to_gray(p.r, p.g, p.b);
        }
    }
    return out;
}

/// @brief Separable Gaussian blur of a row-major @p width x @p height double buffer, with
///        clamp-to-edge borders (matches image::convolve()'s border handling). Kept as a
///        distinct double-precision implementation (rather than reusing image::gaussian_blur)
///        specifically because callers here (structure-tensor smoothing, DoG scale space) need
///        values that routinely fall outside [0, 255] or go negative.
[[nodiscard]] inline std::vector<double> smooth_doubles(const std::vector<double>& data, int width, int height, double sigma) {
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    const int size = 2 * radius + 1;
    std::vector<double> kernel(static_cast<std::size_t>(size));
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double v = std::exp(-(i * i) / (2.0 * sigma * sigma));
        kernel[static_cast<std::size_t>(i + radius)] = v;
        sum += v;
    }
    for (double& v : kernel) v /= sum;

    std::vector<double> tmp(data.size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                const int sx = std::clamp(x + k, 0, width - 1);
                acc += kernel[static_cast<std::size_t>(k + radius)] * data[static_cast<std::size_t>(y) * width + static_cast<std::size_t>(sx)];
            }
            tmp[static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)] = acc;
        }
    }

    std::vector<double> out(data.size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; ++k) {
                const int sy = std::clamp(y + k, 0, height - 1);
                acc += kernel[static_cast<std::size_t>(k + radius)] * tmp[static_cast<std::size_t>(sy) * width + static_cast<std::size_t>(x)];
            }
            out[static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)] = acc;
        }
    }
    return out;
}

} // namespace datamunge::cv::detail
