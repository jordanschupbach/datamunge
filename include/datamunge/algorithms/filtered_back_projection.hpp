#pragma once

// Filtered back-projection (FBP): the classical reconstruction of an image from
// its projections -- the algorithm behind CT scanning. A projection at angle
// theta is the set of line integrals (a "view") of the image; stacked over all
// angles they form the sinogram (the discrete Radon transform). Naive back-
// projection -- smearing each view back across the image -- reconstructs a
// blurred version, because it overweights low frequencies. FBP fixes this by
// convolving each view with a *ramp filter* (Ram-Lak) before back-projecting,
// which exactly cancels the 1/|omega| blur of the back-projection operator. With
// enough angles it recovers the image up to discretization error.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

// Forward Radon transform of an N x N image (row-major; pixel (r,c) has value
// img[r*N + c]). Returns a sinogram sino[angle][detector]; detectors are unit-
// spaced and centered, using linear splatting.
inline std::vector<std::vector<double>>
radon_transform(const std::vector<double>& img, int N, const std::vector<double>& angles,
                int num_detectors) {
    std::vector<std::vector<double>> sino(angles.size(), std::vector<double>(num_detectors, 0.0));
    const double center = (N - 1) / 2.0;
    const double dcen   = (num_detectors - 1) / 2.0;
    for (std::size_t ai = 0; ai < angles.size(); ++ai) {
        const double ct = std::cos(angles[ai]), st = std::sin(angles[ai]);
        auto&        row = sino[ai];
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                const double v = img[r * N + c];
                if (v == 0.0) continue;
                const double x = c - center, y = r - center;
                const double s = x * ct + y * st + dcen; // detector coordinate
                const int    d0 = static_cast<int>(std::floor(s));
                const double f = s - d0;
                if (d0 >= 0 && d0 < num_detectors) row[d0] += v * (1 - f);
                if (d0 + 1 >= 0 && d0 + 1 < num_detectors) row[d0 + 1] += v * f;
            }
    }
    return sino;
}

namespace detail {

// Discrete Ram-Lak ramp filter kernel (unit detector spacing).
inline std::vector<double> ram_lak_kernel(int half) {
    std::vector<double> h(2 * half + 1, 0.0);
    for (int n = -half; n <= half; ++n) {
        double val = 0.0;
        if (n == 0) val = 0.25;
        else if (n % 2 != 0) val = -1.0 / (M_PI * M_PI * static_cast<double>(n) * n);
        h[n + half] = val;
    }
    return h;
}

} // namespace detail

// Reconstruct an N x N image from a sinogram by ramp-filtered back-projection.
inline std::vector<double>
filtered_back_projection(const std::vector<std::vector<double>>& sinogram, int N,
                         const std::vector<double>& angles, int num_detectors) {
    const int          half = num_detectors - 1;
    const auto         h    = detail::ram_lak_kernel(half);

    // 1. Filter each view with the ramp kernel.
    std::vector<std::vector<double>> filtered(angles.size(), std::vector<double>(num_detectors, 0.0));
    for (std::size_t ai = 0; ai < angles.size(); ++ai)
        for (int d = 0; d < num_detectors; ++d) {
            double acc = 0.0;
            for (int dp = 0; dp < num_detectors; ++dp) acc += sinogram[ai][dp] * h[(d - dp) + half];
            filtered[ai][d] = acc;
        }

    // 2. Back-project the filtered views.
    std::vector<double> img(static_cast<std::size_t>(N) * N, 0.0);
    const double        center = (N - 1) / 2.0;
    const double        dcen   = (num_detectors - 1) / 2.0;
    const double        scale  = M_PI / static_cast<double>(angles.size());
    for (std::size_t ai = 0; ai < angles.size(); ++ai) {
        const double ct = std::cos(angles[ai]), st = std::sin(angles[ai]);
        const auto&  fv = filtered[ai];
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                const double x = c - center, y = r - center;
                const double s = x * ct + y * st + dcen;
                const int    d0 = static_cast<int>(std::floor(s));
                const double f = s - d0;
                double       val = 0.0;
                if (d0 >= 0 && d0 < num_detectors) val += fv[d0] * (1 - f);
                if (d0 + 1 >= 0 && d0 + 1 < num_detectors) val += fv[d0 + 1] * f;
                img[r * N + c] += scale * val;
            }
    }
    return img;
}

} // namespace datamunge::algorithms
