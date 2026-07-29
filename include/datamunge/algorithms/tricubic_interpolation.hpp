#pragma once

// Tricubic interpolation on a regular 3-D grid: the three-dimensional
// generalization of bicubic interpolation. It blends the 4x4x4 neighborhood
// around a query with the separable product of Keys' cubic-convolution kernel
// (a = -1/2) along each axis,
//
//   f(x,y,z) ~ sum_{l,m,n in -1..2} w(tx-l) w(ty-m) w(tz-n) V[i+l, j+m, k+n],
//
// where (tx,ty,tz) are the fractional offsets inside the grid cell. Like its 2-D
// cousin it passes through the grid samples exactly, reproduces trilinear
// functions in the interior, and is C^1 -- the quality choice for resampling
// volumetric data (medical scans, simulation fields, 3-D textures). The grid must
// be uniformly spaced along each axis.

#include <datamunge/algorithms/bicubic_interpolation.hpp> // detail::bicubic_kernel, bicubic_clamp

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

// Interpolate at (x,y,z). `v[i*ny*nz + j*nz + k]` is the sample at
// (xs[i], ys[j], zs[k]); axes are uniformly spaced and strictly increasing.
inline double tricubic_interpolate(const std::vector<double>& xs, const std::vector<double>& ys,
                                   const std::vector<double>& zs, const std::vector<double>& v,
                                   double x, double y, double z) {
    const std::size_t nx = xs.size(), ny = ys.size(), nz = zs.size();
    const double      hx = xs[1] - xs[0], hy = ys[1] - ys[0], hz = zs[1] - zs[0];

    const double fx = (x - xs[0]) / hx, fy = (y - ys[0]) / hy, fz = (z - zs[0]) / hz;
    const long   ix = static_cast<long>(std::floor(fx)), iy = static_cast<long>(std::floor(fy)),
               iz = static_cast<long>(std::floor(fz));
    const double tx = fx - ix, ty = fy - iy, tz = fz - iz;

    double result = 0.0;
    for (int l = -1; l <= 2; ++l) {
        const double      wx = detail::bicubic_kernel(tx - l);
        const std::size_t gi = detail::bicubic_clamp(ix + l, nx);
        double            sy = 0.0;
        for (int m = -1; m <= 2; ++m) {
            const double      wy = detail::bicubic_kernel(ty - m);
            const std::size_t gj = detail::bicubic_clamp(iy + m, ny);
            double            sz = 0.0;
            for (int n = -1; n <= 2; ++n) {
                const double      wz = detail::bicubic_kernel(tz - n);
                const std::size_t gk = detail::bicubic_clamp(iz + n, nz);
                sz += wz * v[(gi * ny + gj) * nz + gk];
            }
            sy += wy * sz;
        }
        result += wx * sy;
    }
    return result;
}

} // namespace datamunge::algorithms
