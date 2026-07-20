#pragma once

#include <datamunge/image/image.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::image {

enum class ResampleFilter { Nearest, Bilinear };

namespace detail {
[[nodiscard]] inline std::uint8_t lerp_byte(std::uint8_t a, std::uint8_t b, double t) {
    return static_cast<std::uint8_t>(std::clamp(std::lround(a + t * (b - a)), 0L, 255L));
}

[[nodiscard]] inline Pixel bilinear_sample(const Image& src, double x, double y) {
    const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0, src.width() - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, src.height() - 1);
    const int x1 = std::min(x0 + 1, src.width() - 1);
    const int y1 = std::min(y0 + 1, src.height() - 1);
    const double tx = x - x0;
    const double ty = y - y0;

    const Pixel p00 = src.get_pixel(x0, y0);
    const Pixel p10 = src.get_pixel(x1, y0);
    const Pixel p01 = src.get_pixel(x0, y1);
    const Pixel p11 = src.get_pixel(x1, y1);

    const auto mix = [&](std::uint8_t Pixel::*field) {
        const std::uint8_t top = lerp_byte(p00.*field, p10.*field, tx);
        const std::uint8_t bottom = lerp_byte(p01.*field, p11.*field, tx);
        return lerp_byte(top, bottom, ty);
    };

    return Pixel{mix(&Pixel::r), mix(&Pixel::g), mix(&Pixel::b), mix(&Pixel::a)};
}
} // namespace detail

/// @brief A new image of size @p new_width x @p new_height, resampled from @p src. Nearest
///        maps each destination pixel to its closest source pixel (blocky, but exact for
///        integer scale factors); Bilinear interpolates the four surrounding source pixels
///        (smoother, the usual default for photographic content).
[[nodiscard]] inline Image resize(const Image& src, int new_width, int new_height, ResampleFilter filter = ResampleFilter::Bilinear) {
    if (new_width <= 0 || new_height <= 0) {
        throw std::invalid_argument("resize: new dimensions must be positive");
    }
    if (src.width() == 0 || src.height() == 0) {
        throw std::invalid_argument("resize: source image must be non-empty");
    }

    Image out(new_width, new_height, src.mode());
    const double scale_x = static_cast<double>(src.width()) / new_width;
    const double scale_y = static_cast<double>(src.height()) / new_height;

    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            if (filter == ResampleFilter::Nearest) {
                const int sx = std::min(static_cast<int>((x + 0.5) * scale_x), src.width() - 1);
                const int sy = std::min(static_cast<int>((y + 0.5) * scale_y), src.height() - 1);
                out.set_pixel(x, y, src.get_pixel(sx, sy));
            } else {
                const double sx = (x + 0.5) * scale_x - 0.5;
                const double sy = (y + 0.5) * scale_y - 0.5;
                out.set_pixel(x, y, detail::bilinear_sample(src, sx, sy));
            }
        }
    }
    return out;
}

/// @brief The sub-image [x, x+width) x [y, y+height) of @p src. Throws std::out_of_range if
///        the requested rectangle isn't fully contained in @p src.
[[nodiscard]] inline Image crop(const Image& src, int x, int y, int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("crop: width and height must be positive");
    }
    if (x < 0 || y < 0 || x + width > src.width() || y + height > src.height()) {
        throw std::out_of_range("crop: rectangle extends outside the source image");
    }

    Image out(width, height, src.mode());
    for (int dy = 0; dy < height; ++dy) {
        for (int dx = 0; dx < width; ++dx) {
            out.set_pixel(dx, dy, src.get_pixel(x + dx, y + dy));
        }
    }
    return out;
}

/// @brief A new image with every row reversed left-to-right (mirror image).
[[nodiscard]] inline Image flip_horizontal(const Image& src) {
    Image out(src.width(), src.height(), src.mode());
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            out.set_pixel(src.width() - 1 - x, y, src.get_pixel(x, y));
        }
    }
    return out;
}

/// @brief A new image with every column reversed top-to-bottom (upside down).
[[nodiscard]] inline Image flip_vertical(const Image& src) {
    Image out(src.width(), src.height(), src.mode());
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            out.set_pixel(x, src.height() - 1 - y, src.get_pixel(x, y));
        }
    }
    return out;
}

/// @brief A new image rotated clockwise by exactly 90, 180, or 270 degrees (any other value
///        throws std::invalid_argument -- use rotate() for arbitrary angles). 90/270 swap
///        width and height; 180 keeps them.
[[nodiscard]] inline Image rotate90(const Image& src, int degrees) {
    const int normalized = ((degrees % 360) + 360) % 360;
    if (normalized == 0) {
        return src;
    }
    if (normalized == 180) {
        return flip_vertical(flip_horizontal(src));
    }
    if (normalized != 90 && normalized != 270) {
        throw std::invalid_argument("rotate90: degrees must be a multiple of 90");
    }

    Image out(src.height(), src.width(), src.mode());
    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            const Pixel p = src.get_pixel(x, y);
            if (normalized == 90) {
                out.set_pixel(src.height() - 1 - y, x, p);
            } else { // 270
                out.set_pixel(y, src.width() - 1 - x, p);
            }
        }
    }
    return out;
}

/// @brief A new image rotated clockwise by an arbitrary angle (in degrees) about its center,
///        via inverse-mapping + bilinear resampling. The output canvas is exactly large enough
///        to contain the fully-rotated source (like Pillow's expand=True), and pixels sourced
///        from outside the original image are filled with @p fill_color.
[[nodiscard]] inline Image rotate(const Image& src, double degrees, const Pixel& fill_color = Pixel()) {
    if (src.width() == 0 || src.height() == 0) {
        throw std::invalid_argument("rotate: source image must be non-empty");
    }
    constexpr double kPi = 3.14159265358979323846;
    const double radians = degrees * kPi / 180.0;
    const double cos_a = std::cos(radians);
    const double sin_a = std::sin(radians);

    const double w = src.width(), h = src.height();
    const double new_w = std::abs(w * cos_a) + std::abs(h * sin_a);
    const double new_h = std::abs(w * sin_a) + std::abs(h * cos_a);
    const int out_w = std::max(1, static_cast<int>(std::round(new_w)));
    const int out_h = std::max(1, static_cast<int>(std::round(new_h)));

    Image out(out_w, out_h, src.mode(), fill_color);
    const double src_cx = w / 2.0, src_cy = h / 2.0;
    const double out_cx = out_w / 2.0, out_cy = out_h / 2.0;

    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            // Inverse-map the destination pixel back into source space (rotate by -degrees).
            const double dx = x + 0.5 - out_cx;
            const double dy = y + 0.5 - out_cy;
            const double sx = dx * cos_a + dy * sin_a + src_cx - 0.5;
            const double sy = -dx * sin_a + dy * cos_a + src_cy - 0.5;

            if (sx >= 0.0 && sx <= w - 1.0 && sy >= 0.0 && sy <= h - 1.0) {
                out.set_pixel(x, y, detail::bilinear_sample(src, sx, sy));
            }
        }
    }
    return out;
}

} // namespace datamunge::image
