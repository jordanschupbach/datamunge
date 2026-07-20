#pragma once

#include <datamunge/cv/detail/gaussian_double.hpp>
#include <datamunge/image/image.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace datamunge::cv {

/// @brief A detected corner: its pixel location and a detector-specific response score (higher
///        is a stronger corner). Meant to be independently sortable/filterable by the caller.
struct Corner {
    int x{0};
    int y{0};
    double response{0.0};
};

namespace detail {

struct StructureTensorField {
    std::vector<double> sxx;
    std::vector<double> syy;
    std::vector<double> sxy;
};

/// @brief The Gaussian-smoothed second-moment (structure) tensor at every pixel: Sobel
///        gradients Ix/Iy, their per-pixel outer product Ix^2/Iy^2/IxIy, then smoothed with a
///        Gaussian window of the given @p sigma -- the shared computation behind both Harris
///        and Shi-Tomasi corner response, which differ only in how they combine these three
///        fields into a single scalar score.
inline StructureTensorField structure_tensor(const image::Image& img, double sigma) {
    const int w = img.width();
    const int h = img.height();
    const std::vector<double> gray = to_double_luma(img);

    static constexpr int kGx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static constexpr int kGy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    std::vector<double> ixx(gray.size()), iyy(gray.size()), ixy(gray.size());
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double gx = 0.0, gy = 0.0;
            for (int ky = -1; ky <= 1; ++ky) {
                const int sy = std::clamp(y + ky, 0, h - 1);
                for (int kx = -1; kx <= 1; ++kx) {
                    const int sx = std::clamp(x + kx, 0, w - 1);
                    const double v = gray[static_cast<std::size_t>(sy) * w + static_cast<std::size_t>(sx)];
                    gx += kGx[ky + 1][kx + 1] * v;
                    gy += kGy[ky + 1][kx + 1] * v;
                }
            }
            const std::size_t idx = static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x);
            ixx[idx] = gx * gx;
            iyy[idx] = gy * gy;
            ixy[idx] = gx * gy;
        }
    }

    return StructureTensorField{smooth_doubles(ixx, w, h, sigma), smooth_doubles(iyy, w, h, sigma), smooth_doubles(ixy, w, h, sigma)};
}

/// @brief Keeps only the response-field entries that are a strict local maximum within a
///        (2*radius+1)^2 window AND exceed @p threshold, as a sorted-by-descending-response
///        Corner list -- the standard non-max-suppression + thresholding pass shared by every
///        response-map-based corner detector.
inline std::vector<Corner> extract_local_maxima(const std::vector<double>& response, int width, int height, double threshold, int nms_radius) {
    std::vector<Corner> corners;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double value = response[static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)];
            if (value <= threshold) continue;

            bool is_max = true;
            for (int dy = -nms_radius; dy <= nms_radius && is_max; ++dy) {
                const int ny = y + dy;
                if (ny < 0 || ny >= height) continue;
                for (int dx = -nms_radius; dx <= nms_radius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int nx = x + dx;
                    if (nx < 0 || nx >= width) continue;
                    if (response[static_cast<std::size_t>(ny) * width + static_cast<std::size_t>(nx)] > value) {
                        is_max = false;
                        break;
                    }
                }
            }
            if (is_max) corners.push_back(Corner{x, y, value});
        }
    }
    std::sort(corners.begin(), corners.end(), [](const Corner& a, const Corner& b) { return a.response > b.response; });
    return corners;
}

} // namespace detail

/// @brief Harris corner detection: response = det(M) - k*trace(M)^2 of the structure tensor M
///        at every pixel, thresholded at @p threshold_ratio * (the response map's own max, so
///        callers don't have to guess an absolute scale) and non-max-suppressed within a
///        (2*nms_radius+1)^2 window. @p k is the classical Harris free parameter (0.04-0.06).
[[nodiscard]] inline std::vector<Corner> harris_corners(const image::Image& img, double k = 0.04, double sigma = 1.0,
                                                         double threshold_ratio = 0.01, int nms_radius = 3) {
    const auto field = detail::structure_tensor(img, sigma);
    std::vector<double> response(field.sxx.size());
    for (std::size_t i = 0; i < response.size(); ++i) {
        const double det = field.sxx[i] * field.syy[i] - field.sxy[i] * field.sxy[i];
        const double trace = field.sxx[i] + field.syy[i];
        response[i] = det - k * trace * trace;
    }
    const double max_response = *std::max_element(response.begin(), response.end());
    return detail::extract_local_maxima(response, img.width(), img.height(), threshold_ratio * max_response, nms_radius);
}

/// @brief Shi-Tomasi ("good features to track") corner detection: response = the structure
///        tensor's SMALLER eigenvalue at every pixel (a corner needs strong gradient variation
///        in BOTH directions, so the weaker direction is the limiting factor) -- otherwise
///        identical thresholding/NMS pipeline to harris_corners().
[[nodiscard]] inline std::vector<Corner> shi_tomasi_corners(const image::Image& img, double sigma = 1.0, double threshold_ratio = 0.01,
                                                             int nms_radius = 3) {
    const auto field = detail::structure_tensor(img, sigma);
    std::vector<double> response(field.sxx.size());
    for (std::size_t i = 0; i < response.size(); ++i) {
        const double trace = field.sxx[i] + field.syy[i];
        const double diff = field.sxx[i] - field.syy[i];
        const double discriminant = std::sqrt(diff * diff + 4.0 * field.sxy[i] * field.sxy[i]);
        response[i] = (trace - discriminant) / 2.0; // the smaller of the two eigenvalues
    }
    const double max_response = *std::max_element(response.begin(), response.end());
    return detail::extract_local_maxima(response, img.width(), img.height(), threshold_ratio * max_response, nms_radius);
}

/// @brief FAST corner detection: a pixel is a corner if @p min_contiguous (out of 16) pixels on
///        the Bresenham circle of radius 3 around it are ALL brighter than center+@p threshold
///        or ALL darker than center-@p threshold. This is the direct O(16) full-circle test
///        (no accelerated high-speed pre-rejection on pixels 1/5/9/13) -- simpler to verify
///        correct, at the cost of the constant-factor speed FAST is normally chosen for.
///        Response is the found arc's mean absolute deviation from the center intensity.
[[nodiscard]] inline std::vector<Corner> fast_corners(const image::Image& img, int threshold = 20, int min_contiguous = 9) {
    static constexpr int kCircleX[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
    static constexpr int kCircleY[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};

    const int w = img.width();
    const int h = img.height();
    const std::vector<double> gray = detail::to_double_luma(img);

    std::vector<Corner> corners;
    for (int y = 3; y < h - 3; ++y) {
        for (int x = 3; x < w - 3; ++x) {
            const double center = gray[static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)];
            std::array<int, 16> sign{}; // +1 brighter, -1 darker, 0 neither
            std::array<double, 16> circle_value{};
            for (int i = 0; i < 16; ++i) {
                const double v = gray[static_cast<std::size_t>(y + kCircleY[i]) * w + static_cast<std::size_t>(x + kCircleX[i])];
                circle_value[static_cast<std::size_t>(i)] = v;
                if (v > center + threshold)
                    sign[static_cast<std::size_t>(i)] = 1;
                else if (v < center - threshold)
                    sign[static_cast<std::size_t>(i)] = -1;
            }

            int best_run = 0, best_start = -1, best_dir = 0;
            for (int start = 0; start < 16; ++start) {
                if (sign[static_cast<std::size_t>(start)] == 0) continue;
                const int dir = sign[static_cast<std::size_t>(start)];
                int run = 0;
                for (int i = 0; i < 16; ++i) {
                    if (sign[static_cast<std::size_t>((start + i) % 16)] == dir)
                        ++run;
                    else
                        break;
                }
                if (run > best_run) {
                    best_run = run;
                    best_start = start;
                    best_dir = dir;
                }
            }

            if (best_run >= min_contiguous) {
                double deviation_sum = 0.0;
                for (int i = 0; i < best_run; ++i) {
                    deviation_sum += std::abs(circle_value[static_cast<std::size_t>((best_start + i) % 16)] - center);
                }
                (void)best_dir;
                corners.push_back(Corner{x, y, deviation_sum / best_run});
            }
        }
    }
    std::sort(corners.begin(), corners.end(), [](const Corner& a, const Corner& b) { return a.response > b.response; });
    return corners;
}

} // namespace datamunge::cv
