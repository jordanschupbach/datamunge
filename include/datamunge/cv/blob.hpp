#pragma once

#include <datamunge/cv/detail/gaussian_double.hpp>
#include <datamunge/image/image.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::cv {

/// @brief A detected blob keypoint: its pixel location, the Gaussian scale (sigma) at which it
///        was found, and the Difference-of-Gaussians response at that scale (higher magnitude
///        is a stronger blob).
struct KeyPoint {
    int x{0};
    int y{0};
    double scale{0.0};
    double response{0.0};
};

/// @brief Difference-of-Gaussians blob detection (the detector SIFT keypoints are built on,
///        without the orientation/descriptor stages): builds @p num_scales + 1 Gaussian-
///        blurred copies of @p img at geometrically-spaced sigma_i = @p sigma0 * 2^(i /
///        num_scales), subtracts consecutive pairs to get @p num_scales DoG images, then keeps
///        every point that is a strict extremum (min or max) among its 26 neighbors in the
///        resulting (x, y, scale) volume -- 8 in its own DoG layer plus 9 in each adjacent
///        layer -- and whose |response| exceeds @p threshold. A single octave only (no
///        image-halving between octaves like full SIFT) -- simpler, and sufficient for
///        detecting blobs across a single order-of-magnitude size range.
[[nodiscard]] inline std::vector<KeyPoint> dog_blobs(const image::Image& img, int num_scales = 4, double sigma0 = 1.6,
                                                       double threshold = 3.0) {
    if (num_scales < 2) {
        throw std::invalid_argument("dog_blobs: num_scales must be at least 2");
    }
    const int w = img.width();
    const int h = img.height();
    const std::vector<double> base = detail::to_double_luma(img);

    std::vector<std::vector<double>> gaussians;
    std::vector<double> sigmas;
    for (int i = 0; i <= num_scales; ++i) {
        const double sigma = sigma0 * std::pow(2.0, static_cast<double>(i) / num_scales);
        gaussians.push_back(detail::smooth_doubles(base, w, h, sigma));
        sigmas.push_back(sigma);
    }

    std::vector<std::vector<double>> dogs;
    for (int i = 0; i < num_scales; ++i) {
        std::vector<double> d(gaussians[static_cast<std::size_t>(i)].size());
        for (std::size_t j = 0; j < d.size(); ++j) {
            d[j] = gaussians[static_cast<std::size_t>(i + 1)][j] - gaussians[static_cast<std::size_t>(i)][j];
        }
        dogs.push_back(std::move(d));
    }

    const auto at = [&](int layer, int x, int y) { return dogs[static_cast<std::size_t>(layer)][static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)]; };

    std::vector<KeyPoint> keypoints;
    for (int layer = 1; layer < num_scales - 1; ++layer) {
        for (int y = 1; y < h - 1; ++y) {
            for (int x = 1; x < w - 1; ++x) {
                const double value = at(layer, x, y);
                if (std::abs(value) < threshold) continue;

                bool is_max = true, is_min = true;
                for (int dl = -1; dl <= 1 && (is_max || is_min); ++dl) {
                    for (int dy = -1; dy <= 1 && (is_max || is_min); ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dl == 0 && dy == 0 && dx == 0) continue;
                            const double neighbor = at(layer + dl, x + dx, y + dy);
                            if (neighbor >= value) is_max = false;
                            if (neighbor <= value) is_min = false;
                        }
                    }
                }

                if (is_max || is_min) {
                    keypoints.push_back(KeyPoint{x, y, sigmas[static_cast<std::size_t>(layer)], value});
                }
            }
        }
    }
    return keypoints;
}

} // namespace datamunge::cv
