#pragma once

// Exact Euclidean distance transform (Felzenszwalb & Huttenlocher, 2004): given a
// grid marking some cells as "sites", compute for every cell the Euclidean
// distance to the nearest site. It works by composing one-dimensional squared
// distance transforms -- a lower envelope of parabolas -- along rows then
// columns, in O(rows * cols) total.

#include <cmath>
#include <limits>
#include <vector>

namespace datamunge::geometry {

namespace detail {

// 1-D squared distance transform of a sampled function f: returns d with
// d[q] = min_p ( f[p] + (q-p)^2 ).
inline std::vector<double> dt_1d(const std::vector<double>& f) {
    const int          n = static_cast<int>(f.size());
    std::vector<double> d(n);
    std::vector<int>    v(n, 0);
    std::vector<double> z(n + 1);
    const double        inf = std::numeric_limits<double>::infinity();
    int                 k   = 0;
    v[0] = 0;
    z[0] = -inf;
    z[1] = inf;
    for (int q = 1; q < n; ++q) {
        double s;
        while (true) {
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k]);
            if (s <= z[k]) --k;
            else break;
        }
        ++k;
        v[k]     = q;
        z[k]     = s;
        z[k + 1] = inf;
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < q) ++k;
        const double dq = q - v[k];
        d[q]            = dq * dq + f[v[k]];
    }
    return d;
}

} // namespace detail

// `grid[r][c] != 0` marks a site. Returns the Euclidean distance from each cell to
// the nearest site (infinity if there are no sites).
inline std::vector<std::vector<double>> euclidean_distance_transform(const std::vector<std::vector<char>>& grid) {
    const int    rows = static_cast<int>(grid.size());
    const int    cols = rows ? static_cast<int>(grid[0].size()) : 0;
    // A large FINITE sentinel for "no site" (a true infinity would make the
    // parabola-intersection arithmetic produce NaN); mapped back to infinity at
    // the end for cells no site can reach.
    const double big = 1e20;

    std::vector<std::vector<double>> f(rows, std::vector<double>(cols));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) f[r][c] = grid[r][c] ? 0.0 : big;

    // Transform along columns (each column is a 1-D signal), then along rows.
    for (int c = 0; c < cols; ++c) {
        std::vector<double> col(rows);
        for (int r = 0; r < rows; ++r) col[r] = f[r][c];
        col = detail::dt_1d(col);
        for (int r = 0; r < rows; ++r) f[r][c] = col[r];
    }
    for (int r = 0; r < rows; ++r) {
        f[r] = detail::dt_1d(f[r]);
        for (int c = 0; c < cols; ++c)
            f[r][c] = f[r][c] >= big ? std::numeric_limits<double>::infinity() : std::sqrt(f[r][c]);
    }
    return f;
}

} // namespace datamunge::geometry
