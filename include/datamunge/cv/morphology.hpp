#pragma once

#include <datamunge/image/color.hpp>
#include <datamunge/image/image.hpp>

#include <cstdlib>

namespace datamunge::cv {

/// @brief The neighborhood shape morphological operations probe around each pixel.
enum class StructuringElement { Square, Cross };

namespace detail {
[[nodiscard]] inline bool in_structuring_element(StructuringElement se, int dx, int dy, int radius) {
    if (se == StructuringElement::Square) {
        return std::abs(dx) <= radius && std::abs(dy) <= radius;
    }
    return dx == 0 || dy == 0; // Cross (plus-shape, 4-connectivity), still bounded to [-radius, radius]
}

[[nodiscard]] inline bool is_foreground(const image::Image& img, int x, int y) { return img.get_pixel(x, y).r >= 128; }
} // namespace detail

/// @brief Binary erosion: a pixel stays foreground (luma >= 128, matching image::threshold()'s
///        convention) only if EVERY pixel within the structuring element centered on it is
///        also foreground -- shrinks foreground regions and removes thin protrusions. Returns
///        a Grayscale image with 0/255 values. Pixels outside the image count as background.
[[nodiscard]] inline image::Image erode(const image::Image& img, StructuringElement se = StructuringElement::Square, int radius = 1) {
    const image::Image gray = image::to_grayscale(img);
    image::Image out(gray.width(), gray.height(), image::ImageMode::Grayscale);
    for (int y = 0; y < gray.height(); ++y) {
        for (int x = 0; x < gray.width(); ++x) {
            bool all_fg = true;
            for (int dy = -radius; dy <= radius && all_fg; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (!detail::in_structuring_element(se, dx, dy, radius)) continue;
                    const int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= gray.width() || ny < 0 || ny >= gray.height() || !detail::is_foreground(gray, nx, ny)) {
                        all_fg = false;
                        break;
                    }
                }
            }
            const std::uint8_t v = all_fg ? 255 : 0;
            out.set_pixel(x, y, image::Pixel{v, v, v, 255});
        }
    }
    return out;
}

/// @brief Binary dilation: a pixel becomes foreground if ANY pixel within the structuring
///        element centered on it is foreground -- grows foreground regions and fills small
///        gaps. Returns a Grayscale image with 0/255 values.
[[nodiscard]] inline image::Image dilate(const image::Image& img, StructuringElement se = StructuringElement::Square, int radius = 1) {
    const image::Image gray = image::to_grayscale(img);
    image::Image out(gray.width(), gray.height(), image::ImageMode::Grayscale);
    for (int y = 0; y < gray.height(); ++y) {
        for (int x = 0; x < gray.width(); ++x) {
            bool any_fg = false;
            for (int dy = -radius; dy <= radius && !any_fg; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (!detail::in_structuring_element(se, dx, dy, radius)) continue;
                    const int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < gray.width() && ny >= 0 && ny < gray.height() && detail::is_foreground(gray, nx, ny)) {
                        any_fg = true;
                        break;
                    }
                }
            }
            const std::uint8_t v = any_fg ? 255 : 0;
            out.set_pixel(x, y, image::Pixel{v, v, v, 255});
        }
    }
    return out;
}

/// @brief Morphological opening (erode then dilate): removes small foreground specks and thin
///        protrusions while leaving the size of larger regions roughly unchanged.
[[nodiscard]] inline image::Image morphological_open(const image::Image& img, StructuringElement se = StructuringElement::Square,
                                                       int radius = 1) {
    return dilate(erode(img, se, radius), se, radius);
}

/// @brief Morphological closing (dilate then erode): fills small holes and gaps in foreground
///        regions while leaving their overall size roughly unchanged.
[[nodiscard]] inline image::Image morphological_close(const image::Image& img, StructuringElement se = StructuringElement::Square,
                                                        int radius = 1) {
    return erode(dilate(img, se, radius), se, radius);
}

} // namespace datamunge::cv
