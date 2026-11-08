#pragma once
/// @file layout_geometry.hpp
/// @brief Shared planar point and disk records, retaining graph-layout field layouts.
#include <cstddef>
namespace datamunge::algorithms {
/// @brief Owned planar coordinates shared by graph layouts and polygon rasterization.
/// Units follow the producer. This plain record does not validate manually assigned
/// nonfinite coordinates. Indexing keeps the legacy rule: zero selects x and every
/// other index selects y. No pointer ownership or shared mutable state is involved.
/// @par Example
/// @code
/// dm::algorithms::Point2D p{1,2};
/// std::print("{} {}\n",p[0],p[1]); // 1 2
/// @endcode
/// @see Circle2D, force_directed_layout()
struct Point2D {
  double x{0.0}; ///< Horizontal coordinate in the producer's units.
  double y{0.0}; ///< Vertical coordinate in the producer's units.

  /// @brief Borrow a mutable coordinate, preserving legacy nonzero-index behavior.
  /// @param index Zero selects x; every other value selects y.
  /// @return Reference owned by this point, invalidated when the point is destroyed/moved.
  /// @par Example
  /// @code
  /// dm::algorithms::Point2D p; p[0]=3;
  /// std::print("{}\n",p.x); // 3
  /// @endcode
  double& operator[](std::size_t index) noexcept {return index==0?x:y;}
  /// @brief Borrow a read-only coordinate, preserving legacy nonzero-index behavior.
  /// @param index Zero selects x; every other value selects y.
  /// @return Const reference owned by this point, invalidated when the point is destroyed/moved.
  /// @par Example
  /// @code
  /// const dm::algorithms::Point2D p{1,2};
  /// std::print("{}\n",p[9]); // 2
  /// @endcode
  const double& operator[](std::size_t index) const noexcept {return index==0?x:y;}
};
/// @brief Owned planar disk with an independently stored centre and radius.
/// Defaults retain centre (0,0), radius one and the original field order. This
/// plain record does not enforce positivity, finite coordinates or non-overlap;
/// circle_packing_metrics() validates data before measuring it. Returned layout
/// radii are expressed in the returned coordinate units.
/// @par Example
/// @code
/// const dm::algorithms::Circle2D c{{1,2},0.5};
/// std::print("{} {}\n",c.center.x,c.radius); // 1 0.5
/// @endcode
/// @see circle_packing_metrics(), circle_packing_layout()
struct Circle2D {
  Point2D center; ///< Owned planar centre.
  double radius{1.0}; ///< Radius in the same units; plain storage is not validated.
};
}
