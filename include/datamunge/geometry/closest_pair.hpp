#pragma once
/// @file closest_pair.hpp
/// @brief Legacy closest-point result/call forwarding to the checked indexed all-pairs kernel.
#include <datamunge/geometry/closest_pair_search.hpp>
#include <limits>
#include <stdexcept>
#include <vector>

namespace datamunge::geometry {

/// @brief Owned closest-point coordinate copies and legacy finite-double distance.
/// Original field order/types are retained. Equal measured distances keep the
/// earliest original index pair in the scan. Default storage alone is not a
/// valid search result. No input references or global mutable state are retained.
/// @par Example
/// @code
/// const auto r=dm::geometry::closest_pair({{0,0},{3,4}});
/// std::print("{}\n",r.distance); // 5
/// @endcode
/// @see closest_pair(), ClosestPairSelection
struct ClosestPairResult {
    Point2D a; ///< Owned coordinate copy of the lower-index selected input record.
    Point2D b; ///< Owned coordinate copy of the higher-index selected input record.
    double distance{0.0}; ///< Checked finite-double conversion of the selected widened distance.
};

/// @brief Return the closest point copies and finite-double distance using the checked all-pairs kernel.
/// Validates at least two finite points, visits every original i<j pair and compares
/// widened hypot distances. Equal rounded distances keep the lexicographically
/// first index pair. Duplicates are valid distinct records and can yield zero.
/// No early stop, sorting or spatial pruning; O(n^2) work and O(1) auxiliary storage.
/// No input mutation/shared state. Widened comparison can change former false
/// zero/infinite results and last-bit ties; it is not an exact-real predicate.
/// The old call/result layout remains; indexed/traced variants retain input ids.
/// @param points Borrowed original finite planar coordinates, length at least two.
/// @return Owned selected point copies and checked finite-double measured distance.
/// @throws std::invalid_argument If point count is too small or any coordinate is nonfinite.
/// @throws std::length_error If complete pair count cannot fit size_t.
/// @throws std::overflow_error If arithmetic/counters or the selected legacy distance cannot fit.
/// @par Example
/// @code
/// const auto r=dm::geometry::closest_pair({{0,0},{3,4},{0,1}});
/// std::print("{}\n",r.distance); // 1
/// @endcode
/// @see closest_pair_indexed(), closest_pair_trace()
[[nodiscard]] inline ClosestPairResult closest_pair(const std::vector<Point2D>& points) {
    const auto selected=closest_pair_indexed(points);
    if(selected.distance>std::numeric_limits<double>::max())throw std::overflow_error("closest_pair: selected distance exceeds double range");
    return {selected.a,selected.b,static_cast<double>(selected.distance)};
}

} // namespace datamunge::geometry
