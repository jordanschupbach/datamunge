#pragma once
/// @file
/// @brief Exact circumcircle determinant sign for finite stored binary64 coordinates.
#include <datamunge/geometry/orientation.hpp>
namespace datamunge::geometry {
/// @brief Exact sign of the four-row determinant [x,y,x*x+y*y,1].
/// For counterclockwise noncollinear a,b,c, positive means d strictly inside
/// their circumcircle, negative outside and zero exactly on it. Clockwise a,b,c
/// reverses the meaning. For collinear a,b,c the determinant still has its exact
/// algebraic sign, but no circle interpretation. Decoded dyadic products and
/// fixed integer sums handle signed zero, subnormals and the entire finite
/// binary64 range, without floating products, tolerance or long-double assumptions.
/// Exactness concerns represented inputs, not unknown pre-rounding measurements.
/// Constant bounded work/storage: two 264-word uint32 accumulators plus short
/// mantissa products, no allocation or shared mutable state. Requires IEEE binary64.
/// @param a First finite planar point.
/// @param b Second finite planar point.
/// @param c Third finite planar point.
/// @param d Finite query point.
/// @return Exact determinant sign in {-1,0,1}, with orientation-dependent circle meaning.
/// @throws std::invalid_argument Any nonfinite coordinate.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::geometry::incircle_sign({0,0},{4,0},{0,4},{1,1}),dm::geometry::incircle_sign({0,0},{4,0},{0,4},{4,4})); // 1 0
/// @endcode
/// @par References
/// - J. R. Shewchuk (1997), Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates, DCG 18, 305–363, https://www.cs.cmu.edu/~quake/robust.html . This implementation uses bounded integer sums of decoded dyadics, rather than adaptive floating expansions.
/// @see orientation_sign(), delaunay_triangulation()
int incircle_sign(const Point2D& a,const Point2D& b,const Point2D& c,const Point2D& d);
} // namespace datamunge::geometry
