#pragma once

/// \file painters_algorithm.hpp
/// \brief The painter's algorithm: hidden-surface removal by depth-sorted overdraw.
///
/// The *painter's algorithm* renders a 3D scene the way an oil painter works: draw the
/// farthest objects first, then paint nearer ones on top, letting later strokes cover earlier
/// ones. Objects are sorted by depth (distance from the camera) and drawn *back to front*, so
/// whatever ends up visible at a pixel is automatically the nearest surface there -- no
/// per-pixel depth comparison needed. It is the classic alternative to the z-buffer: cheaper
/// per pixel, but it must sort primitives, and it fails on *interpenetrating* or cyclically
/// overlapping polygons (A in front of B in front of C in front of A), which need splitting or
/// Newell's cycle-resolution method (companion report).
///
/// This implementation sorts objects by depth and renders axis-aligned colored rectangles
/// into a small raster to make the overdraw visible.

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace datamunge::algorithms {

/// \brief A colored axis-aligned rectangle at a given depth. Larger \c depth is farther away.
struct DepthRect {
    int x0, y0, x1, y1;  // inclusive pixel bounds
    double depth;        // camera distance; larger = farther
    int color;           // nonzero color id painted into the raster
};

/// \brief Return object indices sorted *back to front* (farthest depth first).
/// Stable so equal-depth objects keep their input order.
inline std::vector<int> painter_order(const std::vector<double>& depths) {
    std::vector<int> order(depths.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return depths[a] > depths[b]; });
    return order;
}

/// \brief Render \p rects into a \p height x \p width raster of color ids (0 = background),
/// painting back to front so nearer rectangles overwrite farther ones.
inline std::vector<std::vector<int>> painter_render(const std::vector<DepthRect>& rects,
                                                    int width, int height) {
    std::vector<std::vector<int>> raster(height, std::vector<int>(width, 0));
    std::vector<double> depths;
    depths.reserve(rects.size());
    for (const auto& r : rects) depths.push_back(r.depth);

    for (int idx : painter_order(depths)) {
        const DepthRect& r = rects[idx];
        for (int y = std::max(0, r.y0); y <= std::min(height - 1, r.y1); ++y)
            for (int x = std::max(0, r.x0); x <= std::min(width - 1, r.x1); ++x)
                raster[y][x] = r.color;  // later (nearer) paint overwrites
    }
    return raster;
}

}  // namespace datamunge::algorithms
