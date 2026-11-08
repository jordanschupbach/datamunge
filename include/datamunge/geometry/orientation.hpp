#pragma once
/// @file orientation.hpp
/// @brief Exact planar orientation signs for finite IEEE binary64 coordinates.
#include <datamunge/geometry/point2d.hpp>
namespace datamunge::geometry {
/// @brief Exact sign of (b-a) cross (c-a) for the supplied finite binary64 values.
/// Returns +1 for a left turn, -1 for a right turn and zero for exact collinearity
/// of the stored coordinates. Uses fixed integer arithmetic on decoded dyadics,
/// with no tolerance, floating products or dependence on long-double width.
/// Handles signed zero, subnormals and the entire finite double range. This is
/// exact for represented inputs, not for unknown unrounded measurements. Requires
/// IEEE 754 binary64 doubles, checked when building the implementation. Constant
/// bounded work/storage (132 32-bit words per signed-sum side); no global state.
/// @param a First finite point.
/// @param b Second finite point.
/// @param c Third finite point.
/// @return Exact orientation sign in {-1,0,1}.
/// @throws std::invalid_argument Any nonfinite coordinate.
/// @par Example
/// @code
/// std::cout << dm::geometry::orientation_sign({0,0},{1,0},{0,1}); // 1
/// @endcode
/// @par References
/// - Determinant-sign robustness motivation: Shewchuk (1997), Adaptive Precision
///   Floating-Point Arithmetic and Fast Robust Geometric Predicates,
///   https://www.cs.cmu.edu/~quake/robust.html . This implementation uses fixed
///   dyadic integer sums rather than that paper's adaptive expansion algorithm.
/// @see cross(), convex_hull()
int orientation_sign(const Point2D& a, const Point2D& b, const Point2D& c);
} // namespace datamunge::geometry
