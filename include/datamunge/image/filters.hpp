#pragma once

#include <datamunge/image/color.hpp>
#include <datamunge/image/image.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::image {

/// @brief A square @p size x @p size convolution kernel, row-major (kernel[ky * size + kx]).
using Kernel = std::vector<double>;

/// @brief A normalized (sums to 1) box blur kernel of the given odd size.
[[nodiscard]] inline Kernel box_kernel(int size) {
    if (size <= 0 || size % 2 == 0) {
        throw std::invalid_argument("box_kernel: size must be a positive odd number");
    }
    return Kernel(static_cast<std::size_t>(size) * static_cast<std::size_t>(size), 1.0 / (size * size));
}

/// @brief A normalized 2D Gaussian kernel with standard deviation @p sigma; the kernel radius
///        is chosen as ceil(3*sigma) (covers >99.7% of the distribution's mass), size = 2*radius+1.
[[nodiscard]] inline Kernel gaussian_kernel(double sigma) {
    if (sigma <= 0.0) {
        throw std::invalid_argument("gaussian_kernel: sigma must be positive");
    }
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    const int size = 2 * radius + 1;
    Kernel kernel(static_cast<std::size_t>(size) * static_cast<std::size_t>(size));

    double sum = 0.0;
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            const double value = std::exp(-(kx * kx + ky * ky) / (2.0 * sigma * sigma));
            kernel[static_cast<std::size_t>((ky + radius) * size + (kx + radius))] = value;
            sum += value;
        }
    }
    for (double& value : kernel) value /= sum;
    return kernel;
}

/// @brief Applies a square convolution @p kernel to every channel of @p src (alpha included),
///        clamping accesses at the border (edge-replicate padding, avoiding any darkening
///        artifact a zero-padded border would introduce). @p kernel's side length must be odd.
[[nodiscard]] inline Image convolve(const Image& src, const Kernel& kernel) {
    const int size = static_cast<int>(std::lround(std::sqrt(static_cast<double>(kernel.size()))));
    if (size % 2 == 0 || static_cast<std::size_t>(size) * static_cast<std::size_t>(size) != kernel.size()) {
        throw std::invalid_argument("convolve: kernel must be a square odd-sized matrix");
    }
    const int radius = size / 2;
    const int channels = src.channels();

    Image out(src.width(), src.height(), src.mode());
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            double accum[4] = {0.0, 0.0, 0.0, 0.0};
            for (int ky = -radius; ky <= radius; ++ky) {
                const int sy = std::clamp(y + ky, 0, src.height() - 1);
                for (int kx = -radius; kx <= radius; ++kx) {
                    const int sx = std::clamp(x + kx, 0, src.width() - 1);
                    const double weight = kernel[static_cast<std::size_t>((ky + radius) * size + (kx + radius))];
                    const Pixel p = src.get_pixel(sx, sy);
                    const std::uint8_t fields[4] = {p.r, p.g, p.b, p.a};
                    for (int c = 0; c < channels; ++c) accum[c] += weight * fields[c];
                }
            }
            const auto clamp_byte = [](double v) { return static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L)); };
            Pixel result = src.get_pixel(x, y);
            std::uint8_t* fields[4] = {&result.r, &result.g, &result.b, &result.a};
            for (int c = 0; c < channels; ++c) *fields[c] = clamp_byte(accum[c]);
            out.set_pixel(x, y, result);
        }
    }
    return out;
}

[[nodiscard]] inline Image box_blur(const Image& src, int radius) { return convolve(src, box_kernel(2 * radius + 1)); }

[[nodiscard]] inline Image gaussian_blur(const Image& src, double sigma) { return convolve(src, gaussian_kernel(sigma)); }

/// @brief Sharpens @p src via an unsharp-mask kernel: the center weight grows (and the
///        neighbor weights, already negative, grow more negative) as @p amount increases past
///        its neutral value of 0 (0 = no change).
[[nodiscard]] inline Image sharpen(const Image& src, double amount = 1.0) {
    const double center = 1.0 + 4.0 * amount;
    const double edge = -amount;
    const Kernel kernel = {0.0, edge, 0.0, edge, center, edge, 0.0, edge, 0.0};
    return convolve(src, kernel);
}

/// @brief Sobel gradient-magnitude edge detection: converts to grayscale first, then computes
///        the horizontal and vertical Sobel gradients at every pixel and returns their
///        Euclidean magnitude (clamped to [0, 255]) as a new Grayscale image.
[[nodiscard]] inline Image sobel_edges(const Image& src) {
    const Image gray = to_grayscale(src);
    static constexpr int kGx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static constexpr int kGy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    Image out(gray.width(), gray.height(), ImageMode::Grayscale);
    for (int y = 0; y < gray.height(); ++y) {
        for (int x = 0; x < gray.width(); ++x) {
            int gx = 0, gy = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                const int sy = std::clamp(y + ky, 0, gray.height() - 1);
                for (int kx = -1; kx <= 1; ++kx) {
                    const int sx = std::clamp(x + kx, 0, gray.width() - 1);
                    const int value = gray.get_pixel(sx, sy).r;
                    gx += kGx[ky + 1][kx + 1] * value;
                    gy += kGy[ky + 1][kx + 1] * value;
                }
            }
            const double magnitude = std::sqrt(static_cast<double>(gx) * gx + static_cast<double>(gy) * gy);
            const std::uint8_t v = static_cast<std::uint8_t>(std::clamp(std::lround(magnitude), 0L, 255L));
            out.set_pixel(x, y, Pixel{v, v, v, 255});
        }
    }
    return out;
}

/// @brief Adds @p delta to every color channel (alpha untouched), clamped to [0, 255].
///        Negative @p delta darkens, positive brightens.
[[nodiscard]] inline Image adjust_brightness(const Image& src, int delta) {
    Image out(src.width(), src.height(), src.mode());
    const auto clamp_byte = [&](int v) { return static_cast<std::uint8_t>(std::clamp(v + delta, 0, 255)); };
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            const Pixel p = src.get_pixel(x, y);
            out.set_pixel(x, y, Pixel{clamp_byte(p.r), clamp_byte(p.g), clamp_byte(p.b), p.a});
        }
    }
    return out;
}

/// @brief Scales every color channel's distance from mid-gray (128) by @p factor (alpha
///        untouched), clamped to [0, 255]. factor=1 is a no-op; factor=0 collapses to solid
///        gray; factor>1 increases contrast.
[[nodiscard]] inline Image adjust_contrast(const Image& src, double factor) {
    if (factor < 0.0) {
        throw std::invalid_argument("adjust_contrast: factor must be non-negative");
    }
    Image out(src.width(), src.height(), src.mode());
    const auto scale_byte = [&](std::uint8_t v) {
        return static_cast<std::uint8_t>(std::clamp(std::lround((v - 128.0) * factor + 128.0), 0L, 255L));
    };
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            const Pixel p = src.get_pixel(x, y);
            out.set_pixel(x, y, Pixel{scale_byte(p.r), scale_byte(p.g), scale_byte(p.b), p.a});
        }
    }
    return out;
}

/// @brief A binary (black/white) Grayscale image: pixels whose luma (rgb_to_gray) is >=
///        @p level become 255 (white), everything else becomes 0 (black).
[[nodiscard]] inline Image threshold(const Image& src, std::uint8_t level) {
    const Image gray = to_grayscale(src);
    Image out(gray.width(), gray.height(), ImageMode::Grayscale);
    for (int y = 0; y < gray.height(); ++y) {
        for (int x = 0; x < gray.width(); ++x) {
            const std::uint8_t v = (gray.get_pixel(x, y).r >= level) ? 255 : 0;
            out.set_pixel(x, y, Pixel{v, v, v, 255});
        }
    }
    return out;
}

} // namespace datamunge::image
