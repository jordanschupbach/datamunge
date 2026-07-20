#pragma once

#include <datamunge/image/image.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace datamunge::image {

/// @brief A pixel color in HSV space: h in [0, 360), s and v in [0, 1].
struct HSV {
    double h{0.0};
    double s{0.0};
    double v{0.0};
};

/// @brief The standard luma-weighted RGB -> grayscale conversion (ITU-R BT.601 coefficients,
///        the same weights Pillow's "L" conversion and most other libraries use).
[[nodiscard]] inline std::uint8_t rgb_to_gray(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    const double luma = 0.299 * r + 0.587 * g + 0.114 * b;
    return static_cast<std::uint8_t>(std::clamp(std::lround(luma), 0L, 255L));
}

[[nodiscard]] inline HSV rgb_to_hsv(std::uint8_t r8, std::uint8_t g8, std::uint8_t b8) {
    const double r = r8 / 255.0, g = g8 / 255.0, b = b8 / 255.0;
    const double max_c = std::max({r, g, b});
    const double min_c = std::min({r, g, b});
    const double delta = max_c - min_c;

    HSV hsv;
    hsv.v = max_c;
    hsv.s = (max_c == 0.0) ? 0.0 : delta / max_c;

    if (delta == 0.0) {
        hsv.h = 0.0;
    } else if (max_c == r) {
        hsv.h = 60.0 * std::fmod((g - b) / delta, 6.0);
    } else if (max_c == g) {
        hsv.h = 60.0 * ((b - r) / delta + 2.0);
    } else {
        hsv.h = 60.0 * ((r - g) / delta + 4.0);
    }
    if (hsv.h < 0.0) hsv.h += 360.0;
    return hsv;
}

[[nodiscard]] inline Pixel hsv_to_rgb(const HSV& hsv) {
    const double c = hsv.v * hsv.s;
    const double h_prime = hsv.h / 60.0;
    const double x = c * (1.0 - std::abs(std::fmod(h_prime, 2.0) - 1.0));
    const double m = hsv.v - c;

    double r1 = 0.0, g1 = 0.0, b1 = 0.0;
    if (h_prime < 1.0) {
        r1 = c;
        g1 = x;
    } else if (h_prime < 2.0) {
        r1 = x;
        g1 = c;
    } else if (h_prime < 3.0) {
        g1 = c;
        b1 = x;
    } else if (h_prime < 4.0) {
        g1 = x;
        b1 = c;
    } else if (h_prime < 5.0) {
        r1 = x;
        b1 = c;
    } else {
        r1 = c;
        b1 = x;
    }

    const auto to_byte = [](double v) { return static_cast<std::uint8_t>(std::clamp(std::lround((v) * 255.0), 0L, 255L)); };
    return Pixel{to_byte(r1 + m), to_byte(g1 + m), to_byte(b1 + m), 255};
}

/// @brief A new image converted to Grayscale mode. Any existing alpha channel is dropped.
[[nodiscard]] inline Image to_grayscale(const Image& src) {
    Image out(src.width(), src.height(), ImageMode::Grayscale);
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            const Pixel p = src.get_pixel(x, y);
            out.set_pixel(x, y, Pixel{rgb_to_gray(p.r, p.g, p.b), 0, 0, 255});
        }
    }
    return out;
}

/// @brief A new image converted to RGB mode. Grayscale sources are expanded (r=g=b); any
///        existing alpha channel is dropped.
[[nodiscard]] inline Image to_rgb(const Image& src) {
    Image out(src.width(), src.height(), ImageMode::RGB);
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            const Pixel p = src.get_pixel(x, y);
            out.set_pixel(x, y, Pixel{p.r, p.g, p.b, 255});
        }
    }
    return out;
}

/// @brief A new RGBA image with the same colors as @p src and every pixel's alpha set to
///        @p src's existing alpha (255 for modes with no alpha channel).
[[nodiscard]] inline Image to_rgba(const Image& src) {
    Image out(src.width(), src.height(), ImageMode::RGBA);
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            out.set_pixel(x, y, src.get_pixel(x, y));
        }
    }
    return out;
}

/// @brief One 8-bit single-channel plane, e.g. as produced by split_channels().
using Channel = std::vector<std::uint8_t>;

/// @brief Splits @p src into one Grayscale-mode Image per channel, in the source's own
///        channel order (so RGB -> {R, G, B}, RGBA -> {R, G, B, A}, etc.).
[[nodiscard]] inline std::vector<Image> split_channels(const Image& src) {
    std::vector<Image> planes(static_cast<std::size_t>(src.channels()), Image(src.width(), src.height(), ImageMode::Grayscale));
    const auto& data = src.data();
    const int n = src.channels();
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            const std::size_t base = (static_cast<std::size_t>(y) * static_cast<std::size_t>(src.width()) + static_cast<std::size_t>(x)) *
                                      static_cast<std::size_t>(n);
            for (int c = 0; c < n; ++c) {
                planes[static_cast<std::size_t>(c)].set_pixel(x, y, Pixel{data[base + static_cast<std::size_t>(c)], 0, 0, 255});
            }
        }
    }
    return planes;
}

/// @brief The inverse of split_channels(): combines Grayscale-mode @p planes (same width and
///        height, one plane per output channel, in channel order) into a single image of the
///        matching mode. Throws std::invalid_argument if the plane count isn't 1-4 or the
///        planes' dimensions disagree.
[[nodiscard]] inline Image merge_channels(const std::vector<Image>& planes) {
    if (planes.empty() || planes.size() > 4) {
        throw std::invalid_argument("merge_channels: expected 1 to 4 planes");
    }
    const int width = planes.front().width();
    const int height = planes.front().height();
    for (const auto& plane : planes) {
        if (plane.width() != width || plane.height() != height) {
            throw std::invalid_argument("merge_channels: all planes must share the same dimensions");
        }
    }

    Image out(width, height, static_cast<ImageMode>(planes.size()));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Pixel p;
            std::uint8_t* fields[4] = {&p.r, &p.g, &p.b, &p.a};
            p.a = 255;
            for (std::size_t c = 0; c < planes.size(); ++c) {
                *fields[c] = planes[c].get_pixel(x, y).r;
            }
            out.set_pixel(x, y, p);
        }
    }
    return out;
}

} // namespace datamunge::image
