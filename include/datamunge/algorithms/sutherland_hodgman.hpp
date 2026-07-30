#pragma once

/// \file sutherland_hodgman.hpp
/// \brief Sutherland-Hodgman polygon clipping: clip a polygon against a convex clip
///        region, one edge at a time (Sutherland & Hodgman 1974).
///
/// To clip a filled polygon to a viewport we cannot just clip its edges independently --
/// the result must stay a closed polygon. The *Sutherland-Hodgman* algorithm clips the
/// whole *subject polygon* against each edge of a *convex clip polygon* in turn: for a
/// given clip edge it walks the subject's vertices and, for each edge (prev -> cur),
/// emits output vertices by four cases -- both inside (emit cur), leaving (emit the
/// intersection), entering (emit intersection then cur), both outside (emit nothing).
/// Feeding each edge's output as the next edge's input clips against the full region.
/// It is the standard polygon-to-viewport clipper (here specialized to an axis-aligned
/// rectangle, whose four edges are applied in sequence).

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

using Point2 = std::pair<double, double>;

namespace detail {
// For a rectangle edge given by a keep-predicate and an intersection helper.
enum class Edge { Left, Right, Bottom, Top };

inline bool inside(const Point2& p, Edge e, double xmin, double ymin, double xmax, double ymax) {
    switch (e) {
        case Edge::Left: return p.first >= xmin;
        case Edge::Right: return p.first <= xmax;
        case Edge::Bottom: return p.second >= ymin;
        case Edge::Top: return p.second <= ymax;
    }
    return true;
}

inline Point2 intersect(const Point2& a, const Point2& b, Edge e, double xmin, double ymin, double xmax,
                        double ymax) {
    const double dx = b.first - a.first, dy = b.second - a.second;
    switch (e) {
        case Edge::Left: { double t = (xmin - a.first) / dx; return {xmin, a.second + t * dy}; }
        case Edge::Right: { double t = (xmax - a.first) / dx; return {xmax, a.second + t * dy}; }
        case Edge::Bottom: { double t = (ymin - a.second) / dy; return {a.first + t * dx, ymin}; }
        case Edge::Top: { double t = (ymax - a.second) / dy; return {a.first + t * dx, ymax}; }
    }
    return b;
}
}  // namespace detail

/// \brief Clip \p polygon against the rectangle [xmin,xmax]x[ymin,ymax].
///
/// \return the clipped polygon's vertices (empty if fully outside).
inline std::vector<Point2> sutherland_hodgman_clip(const std::vector<Point2>& polygon, double xmin,
                                                   double ymin, double xmax, double ymax) {
    std::vector<Point2> output = polygon;
    for (detail::Edge e : {detail::Edge::Left, detail::Edge::Right, detail::Edge::Bottom, detail::Edge::Top}) {
        if (output.empty()) break;
        std::vector<Point2> input;
        input.swap(output);
        for (std::size_t i = 0; i < input.size(); ++i) {
            const Point2& cur  = input[i];
            const Point2& prev = input[(i + input.size() - 1) % input.size()];
            const bool    cin  = detail::inside(cur, e, xmin, ymin, xmax, ymax);
            const bool    pin  = detail::inside(prev, e, xmin, ymin, xmax, ymax);
            if (cin) {
                if (!pin) output.push_back(detail::intersect(prev, cur, e, xmin, ymin, xmax, ymax));
                output.push_back(cur);
            } else if (pin) {
                output.push_back(detail::intersect(prev, cur, e, xmin, ymin, xmax, ymax));
            }
        }
    }
    return output;
}

}  // namespace datamunge::algorithms
