#pragma once
/// @file orientation.hpp
/// @brief Exact planar orientation signs for finite IEEE binary64 coordinates.
#include <datamunge/geometry/point2d.hpp>
/// @brief Canonical checked native producers and owned algorithm values.
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
/// @brief Return the exact sign of (b-a) cross (p-q) for finite represented binary64 points.
/// Compares signed areas of p and q relative to the same directed baseline without subtracting rounded coordinates or determinants. Uses eight exact dyadic products and bounded 132-word integer accumulators; O(1) bounded work/storage. Suitable for farthest-point decisions with fixed baseline; exact for supplied stored values, not unknown measurements.
/// @param a First baseline endpoint.
/// @param b Second baseline endpoint.
/// @param p First candidate.
/// @param q Second candidate.
/// @return Sign in {-1,0,1}; positive means p has larger signed baseline area.
/// @throws std::invalid_argument Any nonfinite coordinate.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::cross_difference_sign({0,0},{1,0},{0,2},{0,1})); // 1
/// @endcode
/// @par References
/// - Exact determinant motivation: Shewchuk (1997), https://www.cs.cmu.edu/~quake/robust.html; this implementation uses fixed dyadic integer sums.
/// @see orientation_sign()
int cross_difference_sign(const Point2D& a, const Point2D& b, const Point2D& p, const Point2D& q);
/// @brief Approximate a quotient of two exact represented-coordinate cross determinants.
/// The numerator is (b-a) cross (p-q), denominator (d-c) cross (r-s). Accumulates both determinants exactly in the same bounded dyadic-integer kernel used by cross_difference_sign, then forms a widened floating quotient from normalized leading magnitude words. No correctly rounded rational-coordinate promise. Exact zero denominator rejects; nonzero quotient overflow/underflow rejects rather than changing a geometric predicate. O(1) bounded storage/work, borrowed inputs unchanged, no shared state.
/// @param a First numerator baseline endpoint.
/// @param b Second numerator baseline endpoint.
/// @param p First numerator difference point.
/// @param q Second numerator difference point.
/// @param c First denominator baseline endpoint.
/// @param d Second denominator baseline endpoint.
/// @param r First denominator difference point.
/// @param s Second denominator difference point.
/// @return Finite widened approximate determinant quotient; exact zero numerator gives zero.
/// @throws std::invalid_argument Any nonfinite coordinate.
/// @throws std::domain_error Exact zero denominator.
/// @throws std::overflow_error Nonzero quotient overflows or underflows widened finite range.
/// @par Example
/// @code
/// auto t=dm::geometry::cross_difference_ratio({0,0},{1,0},{0,1},{0,0},{0,0},{2,0},{0,1},{0,0});
/// std::print("{}\n",t); // 0.5
/// @endcode
/// @par References
/// - Robust determinant motivation: Shewchuk (1997), https://www.cs.cmu.edu/~quake/robust.html ; bounded exact dyadic sums and approximate quotient here.
/// @see cross_difference_sign()
long double cross_difference_ratio(
    const Point2D& a,
    const Point2D& b,
    const Point2D& p,
    const Point2D& q,
    const Point2D& c,
    const Point2D& d,
    const Point2D& r,
    const Point2D& s);
} // namespace datamunge::geometry
