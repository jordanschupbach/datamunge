#pragma once

/// @file bentley_ottmann.hpp
/// @brief Bentley-Ottmann sweep-line algorithm for all pairwise crossings of line segments, a
///        brute-force reference, and parallel-coordinate segments for rank-correlation data.
///
/// A vertical sweep line moves left to right. Only segments adjacent in the sweep-line order
/// can cross next, so each crossing is found by a check between neighbours. The status is a
/// balanced search tree (`std::set`), giving \f$O((n+k)\log n)\f$ time for \f$n\f$ segments and
/// \f$k\f$ crossings, against \f$O(n^2)\f$ for testing every pair.
///
/// Segments are assumed to have no vertical members and no shared end points. Three or more
/// segments through one point are handled: the sweep reports every pair among them and reverses
/// their order in one event. Only proper crossings (interiors meeting at a single point) are
/// reported.

#include <datamunge/geometry/point2d.hpp>

#include <cstddef>
#include <vector>

/// @brief Computational geometry: points, segments, hulls, triangulations and sweeps.
namespace datamunge::geometry {

/// A line segment between two points.
struct Segment {
    Point2D a;  ///< One end point.
    Point2D b;  ///< The other end point.
};

/// One crossing found by bentley_ottmann_sweep() or segment_intersections_brute().
struct SegmentIntersection {
    Point2D point;       ///< Where the two segments cross.
    int     first  = 0;  ///< Index of one segment (the smaller index).
    int     second = 0;  ///< Index of the other segment.
};

/// Result of bentley_ottmann_sweep().
struct SweepResult {
    std::vector<SegmentIntersection> intersections;           ///< Crossings in sweep order (increasing x).
    std::size_t                      events             = 0;  ///< Events processed: 2n end points plus one per crossing point.
    std::size_t                      intersection_tests = 0;  ///< Pairs of segments tested for a crossing.
    std::size_t                      max_status         = 0;  ///< Most segments crossing the sweep line at once.
};

/// @brief Implementation helpers.
namespace detail {

/// @brief Proper intersection (strict crossing) of segments AB and CD.
/// @param A first end of the first segment.
/// @param B second end of the first segment.
/// @param C first end of the second segment.
/// @param D second end of the second segment.
/// @param out set to the crossing point when there is one.
/// @return whether the interiors cross at a single point.
bool proper_intersection(const Point2D& A, const Point2D& B, const Point2D& C, const Point2D& D, Point2D& out);

}  // namespace detail

/// @brief All proper crossings of @p segments by the Bentley-Ottmann sweep, with work counters.
/// @param segments segments, none vertical and no two sharing an end point.
/// @return the crossings in sweep order, each with the indices of its two segments.
SweepResult bentley_ottmann_sweep(const std::vector<Segment>& segments);

/// @brief Crossing points of @p segments by the Bentley-Ottmann sweep (see bentley_ottmann_sweep()).
/// @param segs segments, none vertical and no two sharing an end point.
/// @return the crossing points in sweep order.
std::vector<Point2D> bentley_ottmann(std::vector<Segment> segs);

/// @brief All proper crossings by testing every pair, \f$O(n^2)\f$; a reference for the sweep.
/// @param segments the segments.
/// @return the crossings, ordered by (first, second).
std::vector<SegmentIntersection> segment_intersections_brute(const std::vector<Segment>& segments);

/// @brief Parallel-coordinate segments for two variables: observation \f$i\f$ becomes the segment
///        from \f$(0,u_i)\f$ to \f$(1,v_i)\f$, where \f$u\f$ and \f$v\f$ are @p left and @p right
///        rescaled to [0, 1].
///
/// Two segments cross exactly when the pair is discordant (\f$(u_i-u_j)(v_i-v_j)<0\f$). Tied values
/// would share an end point, so @p tie_break times the row index is added after rescaling,
/// which orders tied observations by row.
/// @param left values on the left axis.
/// @param right values on the right axis, the same length.
/// @param tie_break offset per row index added to separate ties (0 keeps the data exact).
/// @return one segment per observation.
/// @throws std::invalid_argument if the lengths differ.
std::vector<Segment> parallel_coordinate_segments(const std::vector<double>& left, const std::vector<double>& right,
                                                  double tie_break = 1e-9);

}  // namespace datamunge::geometry
