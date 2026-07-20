#pragma once

#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace datamunge::geometry {

struct Circle {
    Point2D center;
    double radius{0.0};
};

namespace detail {

inline Circle circle_from_one_point(const Point2D& p) { return Circle{p, 0.0}; }

inline Circle circle_from_two_points(const Point2D& a, const Point2D& b) {
    const Point2D center{(a.x + b.x) / 2.0, (a.y + b.y) / 2.0};
    return Circle{center, distance(center, a)};
}

inline Circle circle_from_three_points(const Point2D& a, const Point2D& b, const Point2D& c) {
    const double d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if (std::abs(d) < 1e-12) {
        // Collinear -- no unique circumcircle; the minimum circle through all three is the
        // one with the farthest-apart pair as its diameter (the middle point necessarily
        // falls between them on the line).
        const double dab = squared_distance(a, b), dbc = squared_distance(b, c), dac = squared_distance(a, c);
        if (dab >= dbc && dab >= dac) return circle_from_two_points(a, b);
        if (dbc >= dab && dbc >= dac) return circle_from_two_points(b, c);
        return circle_from_two_points(a, c);
    }
    const double a2 = a.x * a.x + a.y * a.y, b2 = b.x * b.x + b.y * b.y, c2 = c.x * c.x + c.y * c.y;
    const Point2D center{(a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d,
                          (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d};
    return Circle{center, distance(center, a)};
}

inline bool circle_contains(const Circle& circle, const Point2D& p) {
    const double eps = 1e-9 * std::max(1.0, circle.radius);
    return distance(circle.center, p) <= circle.radius + eps;
}

} // namespace detail

/// @brief The smallest circle enclosing every point in @p points (Welzl's algorithm, the
///        classical "incremental with boundary point sets" formulation: a point violating the
///        current circle must lie on the boundary of the true minimal circle, so it's fixed
///        in place and the problem re-solved for the remaining points). Points are shuffled
///        first (fixed seed, for determinism) since the algorithm's expected running time
///        depends on processing order; this implementation skips the further "move-to-front"
///        optimization real Welzl uses for guaranteed expected-linear time; O(n^3) worst case
///        here is still polynomial and far simpler to get right, matching this module's usual
///        complexity/simplicity tradeoff. Throws std::invalid_argument for an empty input.
[[nodiscard]] inline Circle min_enclosing_circle(std::vector<Point2D> points) {
    if (points.empty()) {
        throw std::invalid_argument("min_enclosing_circle: points must be non-empty");
    }

    std::mt19937_64 rng(42);
    std::shuffle(points.begin(), points.end(), rng);

    const std::size_t n = points.size();
    Circle circle = detail::circle_from_one_point(points[0]);

    for (std::size_t i = 1; i < n; ++i) {
        if (detail::circle_contains(circle, points[i])) continue;
        circle = detail::circle_from_one_point(points[i]);
        for (std::size_t j = 0; j < i; ++j) {
            if (detail::circle_contains(circle, points[j])) continue;
            circle = detail::circle_from_two_points(points[i], points[j]);
            for (std::size_t k = 0; k < j; ++k) {
                if (detail::circle_contains(circle, points[k])) continue;
                circle = detail::circle_from_three_points(points[i], points[j], points[k]);
            }
        }
    }

    return circle;
}

} // namespace datamunge::geometry
