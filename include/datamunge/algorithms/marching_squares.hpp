#pragma once

/// \file marching_squares.hpp
/// \brief Marching squares: extract iso-contours from a 2D scalar field.
///
/// Given a scalar field sampled on a grid and a threshold (iso-value) \f$\tau\f$, marching
/// squares traces the level set \f$f(x,y)=\tau\f$ as a set of line segments. It visits each
/// grid *cell* (a square of four samples), classifies the four corners as above or below
/// \f$\tau\f$ into a 4-bit *case index* (16 cases), and for each edge joining an above corner
/// to a below one places a crossing point by *linear interpolation*:
/// \f[
///   x_{\text{cross}} = x_a + \frac{\tau - f_a}{f_b - f_a}\,(x_b - x_a).
/// \f]
/// The case index selects which crossing points to connect, producing 0, 1, or (for the two
/// *saddle* cases) 2 segments per cell. The union of all segments is the contour. Marching
/// squares is the 2D sibling of marching cubes and underlies contour plots, coastline
/// extraction from elevation grids, and region-boundary tracing in image analysis.

#include <array>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// \brief A contour line segment as {x0, y0, x1, y1}.
using ContourSegment = std::array<double, 4>;

/// \brief Extract iso-contour segments at level \p iso from a row-major scalar \p grid.
/// Grid coordinates: column = x, row = y. Cell (r,c) spans x in [c,c+1], y in [r,r+1].
inline std::vector<ContourSegment> marching_squares(
    const std::vector<std::vector<double>>& grid, double iso) {
    std::vector<ContourSegment> segments;
    const std::size_t rows = grid.size();
    if (rows < 2) return segments;
    const std::size_t cols = grid[0].size();

    auto interp = [iso](double xa, double ya, double fa, double xb, double yb, double fb) {
        double t = (iso - fa) / (fb - fa);
        return std::array<double, 2>{xa + t * (xb - xa), ya + t * (yb - ya)};
    };

    for (std::size_t r = 0; r + 1 < rows; ++r) {
        for (std::size_t c = 0; c + 1 < cols; ++c) {
            double tl = grid[r][c], tr = grid[r][c + 1];
            double br = grid[r + 1][c + 1], bl = grid[r + 1][c];
            double x = static_cast<double>(c), y = static_cast<double>(r);

            int idx = (tl >= iso ? 1 : 0) | (tr >= iso ? 2 : 0) | (br >= iso ? 4 : 0) |
                      (bl >= iso ? 8 : 0);
            if (idx == 0 || idx == 15) continue;

            // Crossing points on the four edges (only valid ones are used).
            auto T = interp(x, y, tl, x + 1, y, tr);          // top: TL-TR
            auto R = interp(x + 1, y, tr, x + 1, y + 1, br);  // right: TR-BR
            auto B = interp(x, y + 1, bl, x + 1, y + 1, br);  // bottom: BL-BR
            auto L = interp(x, y, tl, x, y + 1, bl);          // left: TL-BL

            auto seg = [&](const std::array<double, 2>& p, const std::array<double, 2>& q) {
                segments.push_back({p[0], p[1], q[0], q[1]});
            };

            switch (idx) {
                case 1: case 14: seg(L, T); break;
                case 2: case 13: seg(T, R); break;
                case 3: case 12: seg(L, R); break;
                case 4: case 11: seg(R, B); break;
                case 6: case 9:  seg(T, B); break;
                case 7: case 8:  seg(L, B); break;
                case 5:  seg(L, T); seg(R, B); break;  // saddle: TL,BR above
                case 10: seg(T, R); seg(L, B); break;  // saddle: TR,BL above
                default: break;
            }
        }
    }
    return segments;
}

}  // namespace datamunge::algorithms
