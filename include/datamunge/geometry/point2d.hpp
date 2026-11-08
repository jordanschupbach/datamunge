#pragma once
/// @file point2d.hpp
/// @brief Planar point storage and the original double-valued vector primitives.
#include <cmath>
namespace datamunge::geometry {
/// @brief A copied point or free vector with ordinary double coordinates.
/// Construction does not validate finiteness; checked algorithms document their
/// own input contracts. Default coordinates are zero. Equality is numeric, so
/// signed zeros compare equal and NaN coordinates do not compare equal.
/// @par Example
/// @code
/// const dm::geometry::Point2D p{3,4};
/// std::cout << p.x << ' ' << p.y; // 3 4
/// @endcode
/// @see squared_distance(), distance(), cross(), orientation_sign()
struct Point2D {
    double x{0.0}; ///< Horizontal coordinate, default zero.
    double y{0.0}; ///< Vertical coordinate, default zero.

    /// @brief Compare both coordinates with ordinary double equality.
    /// @param other Point to compare without mutation.
    /// @return Both coordinate comparisons are true; signed zeros compare equal.
    /// @par Example
    /// @code
    /// std::cout << (dm::geometry::Point2D{0,1}==dm::geometry::Point2D{-0.,1}); // 1
    /// @endcode
    /// @see operator!=()
    bool operator==(const Point2D& other) const { return x==other.x && y==other.y; }
    /// @brief Negate coordinate equality, including unequal NaN coordinates.
    /// @param other Point to compare without mutation.
    /// @return Logical negation of operator==().
    /// @par Example
    /// @code
    /// std::cout << (dm::geometry::Point2D{0,1}!=dm::geometry::Point2D{0,2}); // 1
    /// @endcode
    /// @see operator==()
    bool operator!=(const Point2D& other) const { return !(*this==other); }
};
/// @brief Evaluate squared Euclidean distance using the original double arithmetic.
/// No finite/range validation. Differences, products and sum can overflow,
/// underflow or lose low bits; NaNs propagate according to floating arithmetic.
/// @param a First point.
/// @param b Second point.
/// @return Rounded squared distance, possibly infinity/NaN or underflowed zero.
/// @par Example
/// @code
/// std::cout << dm::geometry::squared_distance({0,0},{3,4}); // 25
/// @endcode
/// @see distance()
inline double squared_distance(const Point2D& a,const Point2D& b) {
    const double dx=a.x-b.x,dy=a.y-b.y;return dx*dx+dy*dy;
}
/// @brief Return sqrt(squared_distance(a,b)) with its original range limitations.
/// A finite mathematical distance may overflow through the squared intermediate;
/// this convenience primitive performs no checked hypot or input validation.
/// @param a First point.
/// @param b Second point.
/// @return Rounded square root of the double-valued squared distance.
/// @par Example
/// @code
/// std::cout << dm::geometry::distance({0,0},{3,4}); // 5
/// @endcode
/// @see squared_distance()
inline double distance(const Point2D& a,const Point2D& b) { return std::sqrt(squared_distance(a,b)); }
/// @brief Evaluate the original double determinant (b-a) cross (c-a).
/// A positive exact determinant means counterclockwise, but this approximation
/// can cancel, overflow or underflow and its returned sign is not certified.
/// No tolerance or finite validation is applied. orientation_sign() supplies an
/// exact sign for finite stored binary64 points; this primitive remains unchanged.
/// @param a First point.
/// @param b Second point.
/// @param c Third point.
/// @return Rounded double determinant, possibly infinity/NaN or a misleading zero.
/// @par Example
/// @code
/// std::cout << dm::geometry::cross({0,0},{1,0},{0,1}); // 1
/// @endcode
/// @see orientation_sign()
inline double cross(const Point2D& a,const Point2D& b,const Point2D& c) {
    return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
}
} // namespace datamunge::geometry
