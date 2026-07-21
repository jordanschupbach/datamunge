#pragma once

#include <datamunge/geometry/bounding_box.hpp>
#include <datamunge/geometry/point2d.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::gis {

using geometry::BoundingBox;
using geometry::Point2D;

/// @brief The subset of ESRI shapefile shape types this module supports. Z/M-valued variants
///        (PointZ, PolygonM, MultiPatch, ...) and Null-shape records are not represented here --
///        see shapefile.hpp's read_shp() for how they're handled while reading.
enum class ShapeType { Null, Point, PolyLine, Polygon, MultiPoint };

[[nodiscard]] inline const char* to_string(const ShapeType type) {
  switch (type) {
    case ShapeType::Point:
      return "point";
    case ShapeType::PolyLine:
      return "polyline";
    case ShapeType::Polygon:
      return "polygon";
    case ShapeType::MultiPoint:
      return "multipoint";
    default:
      return "null";
  }
}

/// @brief One shapefile record's geometry.
///
/// - Point: `points` holds exactly one element; `parts` is unused.
/// - MultiPoint: `points` holds every point in the record; `parts` is unused.
/// - PolyLine: `parts` holds one entry per line (a "polyline" can have several disconnected
///   pieces sharing one record); `points` is unused.
/// - Polygon: `parts` holds one entry per ring. Per the ESRI shapefile spec (the *opposite*
///   winding convention from GeoJSON's right-hand rule): outer/boundary rings are clockwise
///   (negative signed area under datamunge::geometry::signed_polygon_area's CCW-positive
///   convention), holes are counterclockwise (positive signed area). A single Polygon record
///   may contain more than one outer ring (a multi-polygon), each with its own holes.
struct Shape {
  ShapeType type{ShapeType::Null};
  std::vector<Point2D> points;
  std::vector<std::vector<Point2D>> parts;
  BoundingBox bounds{};

  [[nodiscard]] std::size_t num_parts() const { return parts.size(); }
};

} // namespace datamunge::gis
