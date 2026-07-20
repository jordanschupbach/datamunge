#pragma once

#include <datamunge/geometry/point2d.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::geometry {

struct ClosestPairResult {
    Point2D a;
    Point2D b;
    double distance{0.0};
};

/// @brief The closest pair of points by brute-force all-pairs comparison, O(n^2). The classic
///        divide-and-conquer algorithm for this problem is O(n log n); this simpler version
///        is the deliberate choice for a "basic" library component (easy to verify correct,
///        no merge-step edge cases to get subtly wrong) -- fine for the modest point counts
///        this module targets, not a competitive replacement for CGAL at scale.
[[nodiscard]] inline ClosestPairResult closest_pair(const std::vector<Point2D>& points) {
    if (points.size() < 2) {
        throw std::invalid_argument("closest_pair: need at least 2 points");
    }

    ClosestPairResult best;
    best.distance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = i + 1; j < points.size(); ++j) {
            const double d = distance(points[i], points[j]);
            if (d < best.distance) {
                best = ClosestPairResult{points[i], points[j], d};
            }
        }
    }
    return best;
}

} // namespace datamunge::geometry
