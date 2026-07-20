#pragma once

#include <datamunge/image/image.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

namespace datamunge::image {

namespace detail {
inline void set_pixel_clipped(Image& img, int x, int y, const Pixel& color) {
    if (x >= 0 && x < img.width() && y >= 0 && y < img.height()) {
        img.set_pixel(x, y, color);
    }
}

inline void stamp(Image& img, int cx, int cy, const Pixel& color, int thickness) {
    const int half = thickness / 2;
    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            set_pixel_clipped(img, cx + dx, cy + dy, color);
        }
    }
}
} // namespace detail

/// @brief Draws a straight line from (x0,y0) to (x1,y1) directly onto @p img (Bresenham's
///        integer algorithm). @p thickness > 1 stamps a (thickness x thickness) square at
///        every line pixel -- simple, not a true round/mitered stroke, but sufficient for this
///        module's scope. Points (and any square stamps) outside @p img's bounds are silently
///        clipped rather than throwing, matching typical drawing-library behavior for
///        partially-offscreen shapes.
inline void draw_line(Image& img, int x0, int y0, int x1, int y1, const Pixel& color, int thickness = 1) {
    const int dx = std::abs(x1 - x0);
    const int dy = -std::abs(y1 - y0);
    const int sx = (x0 < x1) ? 1 : -1;
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    int x = x0, y = y0;
    while (true) {
        if (thickness <= 1) {
            detail::set_pixel_clipped(img, x, y, color);
        } else {
            detail::stamp(img, x, y, color, thickness);
        }
        if (x == x1 && y == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y += sy;
        }
    }
}

/// @brief Draws the rectangle with corners (x0,y0) and (x1,y1) (either diagonal order works)
///        onto @p img, either as a one-pixel-wide outline or filled solid.
inline void draw_rectangle(Image& img, int x0, int y0, int x1, int y1, const Pixel& color, bool filled = false) {
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);

    if (filled) {
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                detail::set_pixel_clipped(img, x, y, color);
            }
        }
    } else {
        for (int x = x0; x <= x1; ++x) {
            detail::set_pixel_clipped(img, x, y0, color);
            detail::set_pixel_clipped(img, x, y1, color);
        }
        for (int y = y0; y <= y1; ++y) {
            detail::set_pixel_clipped(img, x0, y, color);
            detail::set_pixel_clipped(img, x1, y, color);
        }
    }
}

/// @brief Draws a circle of the given @p radius centered at (cx,cy) onto @p img, either as a
///        one-pixel-wide outline (the midpoint circle algorithm) or filled solid (bounding-box
///        scan with a squared-distance test).
inline void draw_circle(Image& img, int cx, int cy, int radius, const Pixel& color, bool filled = false) {
    if (radius < 0) {
        return;
    }

    if (filled) {
        const long r_sq = static_cast<long>(radius) * radius;
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                if (static_cast<long>(x) * x + static_cast<long>(y) * y <= r_sq) {
                    detail::set_pixel_clipped(img, cx + x, cy + y, color);
                }
            }
        }
        return;
    }

    int x = radius, y = 0, err = 0;
    while (x >= y) {
        detail::set_pixel_clipped(img, cx + x, cy + y, color);
        detail::set_pixel_clipped(img, cx + y, cy + x, color);
        detail::set_pixel_clipped(img, cx - y, cy + x, color);
        detail::set_pixel_clipped(img, cx - x, cy + y, color);
        detail::set_pixel_clipped(img, cx - x, cy - y, color);
        detail::set_pixel_clipped(img, cx - y, cy - x, color);
        detail::set_pixel_clipped(img, cx + y, cy - x, color);
        detail::set_pixel_clipped(img, cx + x, cy - y, color);

        if (err <= 0) {
            y += 1;
            err += 2 * y + 1;
        }
        if (err > 0) {
            x -= 1;
            err -= 2 * x + 1;
        }
    }
}

/// @brief Draws the polygon defined by @p points (in order, implicitly closed back to the
///        first point) onto @p img. Outline mode connects consecutive vertices with
///        draw_line(); filled mode uses a standard even-odd scanline fill. Fewer than 3 points
///        draws nothing.
inline void draw_polygon(Image& img, const std::vector<std::pair<int, int>>& points, const Pixel& color, bool filled = false) {
    const std::size_t n = points.size();
    if (n < 3) {
        return;
    }

    if (!filled) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto& [x0, y0] = points[i];
            const auto& [x1, y1] = points[(i + 1) % n];
            draw_line(img, x0, y0, x1, y1, color);
        }
        return;
    }

    int min_y = points[0].second, max_y = points[0].second;
    for (const auto& [px, py] : points) {
        min_y = std::min(min_y, py);
        max_y = std::max(max_y, py);
    }

    for (int y = min_y; y <= max_y; ++y) {
        std::vector<double> intersections;
        for (std::size_t i = 0; i < n; ++i) {
            const auto& [x0, y0] = points[i];
            const auto& [x1, y1] = points[(i + 1) % n];
            if ((y0 <= y && y1 > y) || (y1 <= y && y0 > y)) {
                const double t = static_cast<double>(y - y0) / (y1 - y0);
                intersections.push_back(x0 + t * (x1 - x0));
            }
        }
        std::sort(intersections.begin(), intersections.end());
        for (std::size_t i = 0; i + 1 < intersections.size(); i += 2) {
            const int x_start = static_cast<int>(std::ceil(intersections[i] - 0.5));
            const int x_end = static_cast<int>(std::floor(intersections[i + 1] - 0.5));
            for (int x = x_start; x <= x_end; ++x) {
                detail::set_pixel_clipped(img, x, y, color);
            }
        }
    }
}

} // namespace datamunge::image
