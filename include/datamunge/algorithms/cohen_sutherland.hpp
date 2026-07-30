#pragma once

/// \file cohen_sutherland.hpp
/// \brief Cohen-Sutherland line clipping: clip a segment to an axis-aligned rectangle
///        using region outcodes.
///
/// Rendering clips primitives to the viewport. For a line segment, the *Cohen-Sutherland*
/// algorithm assigns each endpoint a 4-bit *outcode* recording which of the four
/// half-planes (left/right/below/above the clip rectangle) it lies outside. Two quick
/// bitwise tests then handle the common cases without any arithmetic: if the OR of the
/// codes is 0 both endpoints are inside (*trivially accept*); if the AND is nonzero both
/// lie outside the same edge so the whole segment is out (*trivially reject*). Only the
/// remaining cases compute an intersection with one clip edge, replace the outside
/// endpoint, and repeat. The outcodes make clipping mostly branch-and-test, which is why
/// it was the classic viewport clipper.

#include <cstdint>

namespace datamunge::algorithms {

/// Result of clipping a segment to a rectangle.
struct ClippedSegment {
    bool   visible = false;  ///< False if the segment is entirely outside.
    double x0 = 0, y0 = 0, x1 = 0, y1 = 0;  ///< The (possibly shortened) visible segment.
};

namespace detail {
constexpr int kInside = 0, kLeft = 1, kRight = 2, kBottom = 4, kTop = 8;
inline int    outcode(double x, double y, double xmin, double ymin, double xmax, double ymax) {
    int code = kInside;
    if (x < xmin) code |= kLeft;
    else if (x > xmax) code |= kRight;
    if (y < ymin) code |= kBottom;
    else if (y > ymax) code |= kTop;
    return code;
}
}  // namespace detail

/// \brief Clip segment (x0,y0)-(x1,y1) to the rectangle [xmin,xmax]x[ymin,ymax].
inline ClippedSegment cohen_sutherland_clip(double x0, double y0, double x1, double y1, double xmin,
                                            double ymin, double xmax, double ymax) {
    int c0 = detail::outcode(x0, y0, xmin, ymin, xmax, ymax);
    int c1 = detail::outcode(x1, y1, xmin, ymin, xmax, ymax);

    for (;;) {
        if ((c0 | c1) == 0) return {true, x0, y0, x1, y1};   // trivially accept
        if ((c0 & c1) != 0) return {false, 0, 0, 0, 0};       // trivially reject
        // Pick an endpoint that is outside and clip it to one edge.
        const int  out = c0 ? c0 : c1;
        double     x = 0, y = 0;
        if (out & detail::kTop) {
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0);
            y = ymax;
        } else if (out & detail::kBottom) {
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0);
            y = ymin;
        } else if (out & detail::kRight) {
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0);
            x = xmax;
        } else {  // kLeft
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0);
            x = xmin;
        }
        if (out == c0) {
            x0 = x; y0 = y;
            c0 = detail::outcode(x0, y0, xmin, ymin, xmax, ymax);
        } else {
            x1 = x; y1 = y;
            c1 = detail::outcode(x1, y1, xmin, ymin, xmax, ymax);
        }
    }
}

}  // namespace datamunge::algorithms
