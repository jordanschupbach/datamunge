#pragma once

/// @file class_regions.hpp
/// @brief Shading a plane by the class a two-predictor classifier predicts, as merged
///        rectangles, for the classifiers' plot_decision_regions() methods.

#include <datamunge/plot/plot.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace datamunge::graphics {

/// @brief Mixes @p color with white.
/// @param color     Base colour.
/// @param fraction  Share of white, in \f$[0,1]\f$ (0.8 gives a pale tint).
/// @return The tint.
inline RGB tint(RGB color, double fraction) {
    auto mix = [&](std::uint8_t c) { return static_cast<std::uint8_t>(c + (255 - c) * fraction + 0.5); };
    return RGB{mix(color.r), mix(color.g), mix(color.b)};
}

/// @brief Shades the box @p box = {x_min, x_max, y_min, y_max} by the class @p classify returns
///        at each cell centre of a @p resolution × @p resolution grid, then sets the plot's
///        axis limits to the box.
///
/// Runs of equal cells along a row are merged into one rectangle, so a fine grid stays small
/// (a few hundred polygons for smooth boundaries). Each rectangle overlaps the next row and
/// column by a quarter cell so renderers leave no seams. Draw points after calling this.
/// @param plot        Plot to draw on.
/// @param classify    Class index (into @p fills) at a point; a value outside @p fills leaves the cell empty.
/// @param box         {x_min, x_max, y_min, y_max}.
/// @param resolution  Cells per axis (at least 2).
/// @param fills       Fill colour per class.
/// @throws std::invalid_argument  If the box is not increasing.
///
/// @par Example
/// @code
/// auto plt = dm::graphics::RPlot::create();
/// dm::graphics::add_class_regions(plt, [](double x, double y) { return x + y > 1.0 ? 1u : 0u; },
///                                 {0.0, 1.0, 0.0, 1.0}, 100, {dm::RGB{219, 234, 254}, dm::RGB{254, 226, 226}});
/// plt.save_svg("halves.svg");
/// @endcode
inline void add_class_regions(RPlot& plot, const std::function<std::size_t(double, double)>& classify,
                              const std::array<double, 4>& box, std::size_t resolution, const std::vector<RGB>& fills) {
    const double x_lo = box[0], x_hi = box[1], y_lo = box[2], y_hi = box[3];
    if (!(x_hi > x_lo && y_hi > y_lo)) throw std::invalid_argument("add_class_regions: box must be increasing");
    resolution      = std::max<std::size_t>(resolution, 2);
    const double dx = (x_hi - x_lo) / static_cast<double>(resolution);
    const double dy = (y_hi - y_lo) / static_cast<double>(resolution);
    for (std::size_t gy = 0; gy < resolution; ++gy) {
        const double y         = y_lo + (static_cast<double>(gy) + 0.5) * dy;
        std::size_t  run_start = 0, run_class = SIZE_MAX;
        for (std::size_t gx = 0; gx <= resolution; ++gx) {
            const std::size_t cls = gx < resolution ? classify(x_lo + (static_cast<double>(gx) + 0.5) * dx, y) : SIZE_MAX;
            if (cls == run_class) continue;
            if (run_class < fills.size()) {
                const double left   = x_lo + static_cast<double>(run_start) * dx;
                const double right  = std::min(x_hi, x_lo + (static_cast<double>(gx) + 0.25) * dx);
                const double bottom = y_lo + static_cast<double>(gy) * dy;
                const double top    = std::min(y_hi, bottom + 1.25 * dy);
                plot.polygon({left, right, right, left}, {bottom, bottom, top, top}, fills[run_class], true);
            }
            run_start = gx;
            run_class = cls;
        }
    }
    plot.x_limits(x_lo, x_hi).y_limits(y_lo, y_hi);
}

} // namespace datamunge::graphics
