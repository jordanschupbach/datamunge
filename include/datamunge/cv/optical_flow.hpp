#pragma once

#include <datamunge/cv/detail/gaussian_double.hpp>
#include <datamunge/image/image.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::cv {

/// @brief The estimated motion of one tracked point between two frames. @p valid is false when
///        the point's window was too close to the image border, or too textureless/aperture-
///        ambiguous (the same "does the structure tensor have two strong eigenvalues"
///        criterion shi_tomasi_corners() uses to pick good points to track in the first place
///        -- Lucas-Kanade windows should generally be centered on such points).
struct FlowVector {
    double dx{0.0};
    double dy{0.0};
    bool valid{false};
};

/// @brief Sparse Lucas-Kanade optical flow: for each of @p points, solves the classical 2x2
///        windowed least-squares system (built from Sobel spatial gradients Ix/Iy on @p prev
///        and the temporal gradient It = next - prev, both accumulated over a
///        (2*window_radius+1)^2 window) for the local translation (dx, dy) that best explains
///        the window's brightness change under the standard optical-flow brightness-constancy
///        assumption. Single-level (not the pyramidal coarse-to-fine extension real
///        implementations use for large motions) -- correctly handles motions up to roughly
///        @p window_radius pixels, which is this module's documented scope tradeoff; use
///        gaussian_pyramid() externally and call this once per level if larger motions matter.
[[nodiscard]] inline std::vector<FlowVector> lucas_kanade_optical_flow(const image::Image& prev, const image::Image& next,
                                                                         const std::vector<std::pair<int, int>>& points,
                                                                         int window_radius = 7) {
    if (prev.width() != next.width() || prev.height() != next.height()) {
        throw std::invalid_argument("lucas_kanade_optical_flow: prev and next must have the same dimensions");
    }
    const int w = prev.width();
    const int h = prev.height();
    const std::vector<double> gray_prev = detail::to_double_luma(prev);
    const std::vector<double> gray_next = detail::to_double_luma(next);

    static constexpr int kGx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static constexpr int kGy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    // The raw Sobel kernel's output is 8x the true partial derivative dI/dx (e.g. applied to a
    // pure linear ramp of slope m, it returns 8m, not m) -- harmless for algorithms that only
    // need gradient DIRECTION or relative magnitude (sobel_edges(), the corner detectors), but
    // Lucas-Kanade combines Ix directly with It (an unscaled temporal difference), so an
    // unnormalized Ix silently biases every recovered flow vector by a factor of 8. Divide by
    // the kernel's weight sum (8) here so Ix/Iy are genuine derivative estimates.
    constexpr double kSobelNormalization = 8.0;
    const auto sobel_x = [&](int x, int y) {
        double g = 0.0;
        for (int ky = -1; ky <= 1; ++ky)
            for (int kx = -1; kx <= 1; ++kx)
                g += kGx[ky + 1][kx + 1] * gray_prev[static_cast<std::size_t>(std::clamp(y + ky, 0, h - 1)) * w +
                                                       static_cast<std::size_t>(std::clamp(x + kx, 0, w - 1))];
        return g / kSobelNormalization;
    };
    const auto sobel_y = [&](int x, int y) {
        double g = 0.0;
        for (int ky = -1; ky <= 1; ++ky)
            for (int kx = -1; kx <= 1; ++kx)
                g += kGy[ky + 1][kx + 1] * gray_prev[static_cast<std::size_t>(std::clamp(y + ky, 0, h - 1)) * w +
                                                       static_cast<std::size_t>(std::clamp(x + kx, 0, w - 1))];
        return g / kSobelNormalization;
    };

    std::vector<FlowVector> results;
    results.reserve(points.size());
    for (const auto& [px, py] : points) {
        if (px - window_radius < 0 || px + window_radius >= w || py - window_radius < 0 || py + window_radius >= h) {
            results.push_back(FlowVector{0.0, 0.0, false});
            continue;
        }

        double sxx = 0.0, syy = 0.0, sxy = 0.0, sxt = 0.0, syt = 0.0;
        for (int dy = -window_radius; dy <= window_radius; ++dy) {
            for (int dx = -window_radius; dx <= window_radius; ++dx) {
                const int x = px + dx, y = py + dy;
                const double ix = sobel_x(x, y);
                const double iy = sobel_y(x, y);
                const double it = gray_next[static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)] -
                                   gray_prev[static_cast<std::size_t>(y) * w + static_cast<std::size_t>(x)];
                sxx += ix * ix;
                syy += iy * iy;
                sxy += ix * iy;
                sxt += ix * it;
                syt += iy * it;
            }
        }

        const double det = sxx * syy - sxy * sxy;
        if (std::abs(det) < 1e-6) {
            results.push_back(FlowVector{0.0, 0.0, false}); // aperture problem: not enough texture to solve uniquely
            continue;
        }
        // Solve [[sxx,sxy],[sxy,syy]] * [dx,dy]^T = -[sxt,syt]^T via Cramer's rule.
        const double dx = (-sxt * syy + syt * sxy) / det;
        const double dy = (-syt * sxx + sxt * sxy) / det;
        results.push_back(FlowVector{dx, dy, true});
    }
    return results;
}

} // namespace datamunge::cv
