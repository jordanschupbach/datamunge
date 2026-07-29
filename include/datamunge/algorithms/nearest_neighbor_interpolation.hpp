#pragma once

// Nearest-neighbor interpolation: the simplest interpolation of all -- return the
// value of the sample whose abscissa is closest to the query. The result is a
// piecewise-constant function that jumps midway between consecutive samples, so it
// exactly reproduces the samples at the nodes and never overshoots or invents
// intermediate values. It is the fastest resampler and the right choice for
// categorical or label data (where averaging is meaningless), and the crude but
// artifact-free option for image and signal magnification.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

// Value at `x` of the piecewise-constant nearest-neighbor interpolant of the
// samples (xs, ys). `xs` must be strictly increasing; ties go to the lower index.
inline double nearest_neighbor_interpolate(const std::vector<double>& xs,
                                           const std::vector<double>& ys, double x) {
    const std::size_t n = xs.size();
    if (n == 0) return 0.0;
    if (x <= xs.front()) return ys.front();
    if (x >= xs.back()) return ys.back();
    // Binary search for the first sample >= x.
    std::size_t lo = 0, hi = n - 1;
    while (lo < hi) {
        const std::size_t mid = (lo + hi) / 2;
        if (xs[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    // Neighbors are xs[lo-1] and xs[lo]; pick the closer (tie -> lower index).
    const double dlo = x - xs[lo - 1], dhi = xs[lo] - x;
    return dhi < dlo ? ys[lo] : ys[lo - 1];
}

} // namespace datamunge::algorithms
