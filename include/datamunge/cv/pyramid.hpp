#pragma once

#include <datamunge/image/filters.hpp>
#include <datamunge/image/image.hpp>
#include <datamunge/image/transform.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace datamunge::cv {

/// @brief The classical Gaussian pyramid: level 0 is @p img itself; each subsequent level is
///        the previous level Gaussian-blurred (sigma = @p sigma) then downsampled by 2x
///        (bilinear resize) -- a coarse-to-fine multi-resolution representation used
///        throughout this module (e.g. as the natural foundation for a pyramidal optical-flow
///        extension, though lucas_kanade_optical_flow() itself stays single-level). Stops
///        early (producing fewer than @p levels images) if a level's dimensions would shrink
///        below 1x1. Throws std::invalid_argument for levels < 1.
[[nodiscard]] inline std::vector<image::Image> gaussian_pyramid(const image::Image& img, int levels, double sigma = 1.0) {
    if (levels < 1) {
        throw std::invalid_argument("gaussian_pyramid: levels must be at least 1");
    }
    std::vector<image::Image> pyramid;
    pyramid.push_back(img);
    for (int i = 1; i < levels; ++i) {
        const image::Image& prev = pyramid.back();
        if (prev.width() <= 1 || prev.height() <= 1) break;
        const image::Image blurred = image::gaussian_blur(prev, sigma);
        const int new_w = std::max(1, prev.width() / 2);
        const int new_h = std::max(1, prev.height() / 2);
        pyramid.push_back(image::resize(blurred, new_w, new_h, image::ResampleFilter::Bilinear));
    }
    return pyramid;
}

namespace detail {
/// @brief Per-channel signed difference (cur - upsampled), re-centered at 128 and clamped to
///        [0, 255] so it stores in an ordinary Image -- the standard way to make a Laplacian
///        pyramid level (which is naturally signed) visualizable/storable without introducing
///        a separate signed-pixel image type just for this.
inline image::Image signed_difference_image(const image::Image& cur, const image::Image& upsampled) {
    image::Image out(cur.width(), cur.height(), cur.mode());
    const int channels = cur.channels();
    for (int y = 0; y < cur.height(); ++y) {
        for (int x = 0; x < cur.width(); ++x) {
            const image::Pixel a = cur.get_pixel(x, y);
            const image::Pixel b = upsampled.get_pixel(x, y);
            const int af[4] = {a.r, a.g, a.b, a.a};
            const int bf[4] = {b.r, b.g, b.b, b.a};
            image::Pixel result;
            std::uint8_t* rf[4] = {&result.r, &result.g, &result.b, &result.a};
            result.a = 255;
            for (int c = 0; c < channels; ++c) {
                *rf[c] = static_cast<std::uint8_t>(std::clamp(af[c] - bf[c] + 128, 0, 255));
            }
            out.set_pixel(x, y, result);
        }
    }
    return out;
}
} // namespace detail

/// @brief The Laplacian pyramid: for every level except the last, level_i = (Gaussian level i)
///        minus (Gaussian level i+1 upsampled back to level i's size) -- the detail lost by
///        blurring+downsampling at that scale, re-centered at 128 (see
///        detail::signed_difference_image()) so it stores as an ordinary image; the final
///        level is the smallest Gaussian level kept as-is (the residual low-frequency image
///        the whole pyramid was built from). Reconstructing the original from a Laplacian
///        pyramid is the classical use (progressively add each un-recentered level back onto
///        the upsampled reconstruction-so-far) but isn't implemented here -- this module only
///        needed the decomposition direction.
[[nodiscard]] inline std::vector<image::Image> laplacian_pyramid(const image::Image& img, int levels, double sigma = 1.0) {
    const auto gaussians = gaussian_pyramid(img, levels, sigma);
    std::vector<image::Image> laplacians;
    for (std::size_t i = 0; i + 1 < gaussians.size(); ++i) {
        const image::Image upsampled =
            image::resize(gaussians[i + 1], gaussians[i].width(), gaussians[i].height(), image::ResampleFilter::Bilinear);
        laplacians.push_back(detail::signed_difference_image(gaussians[i], upsampled));
    }
    laplacians.push_back(gaussians.back());
    return laplacians;
}

} // namespace datamunge::cv
