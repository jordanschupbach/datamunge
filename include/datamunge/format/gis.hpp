#pragma once

/// @file gis.hpp
/// @brief std::format / std::print support for datamunge::gis.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Shapes and layers print a one-line summary (type, counts, bounds); geometry is never dumped.
///
/// @par Example
/// @code
/// dm::gis::Shape s;
/// s.type = dm::gis::ShapeType::Point;
/// s.points = {{1, 2}};
/// std::print("{}\n", s);   // Shape(type=point, points=1, parts=0, bounds=BoundingBox(min=Point2D(0, 0), max=Point2D(0, 0)))
/// @endcode

#include <datamunge/format/geometry.hpp>
#include <datamunge/format/text_formatter.hpp>
#include <datamunge/gis/gis.hpp>

#include <format>
#include <string>

/// @brief Prints a ShapeType via the library's gis::to_string() ("point", "polyline", "polygon", ...).
template <>
struct std::formatter<datamunge::gis::ShapeType> : datamunge::format_detail::text_formatter<datamunge::gis::ShapeType> {
  /// @brief The library's lowercase shape-type name.
  /// @param type The shape type to print.
  /// @return gis::to_string(type).
  static std::string text(datamunge::gis::ShapeType type) { return datamunge::gis::to_string(type); }
};

/// @brief Prints a Shape as "Shape(type=.., points=.., parts=.., bounds=..)".
template <>
struct std::formatter<datamunge::gis::Shape> : datamunge::format_detail::text_formatter<datamunge::gis::Shape> {
  /// @brief The record's type, point/part counts, and bounding box.
  /// @param s The shape to print.
  /// @return The one-line description.
  static std::string text(const datamunge::gis::Shape& s) {
    return std::format("Shape(type={}, points={}, parts={}, bounds={})", s.type, s.points.size(), s.parts.size(),
                       s.bounds);
  }
};

/// @brief Prints ShapefileData as "ShapefileData(type=.., shapes=.., bounds=..)".
template <>
struct std::formatter<datamunge::gis::ShapefileData>
    : datamunge::format_detail::text_formatter<datamunge::gis::ShapefileData> {
  /// @brief The file's shape type, record count, and bounding box.
  /// @param d The parsed data to print.
  /// @return The one-line description.
  static std::string text(const datamunge::gis::ShapefileData& d) {
    return std::format("ShapefileData(type={}, shapes={}, bounds={})", d.shape_type, d.shapes.size(), d.bounds);
  }
};

/// @brief Prints a ShapeLayer as "ShapeLayer(type=.., shapes=.., attributes=rows x cols, bounds=..)".
template <>
struct std::formatter<datamunge::gis::ShapeLayer>
    : datamunge::format_detail::text_formatter<datamunge::gis::ShapeLayer> {
  /// @brief The layer's shape type, record count, attribute-table shape, and bounding box.
  /// @param layer The layer to print.
  /// @return The one-line description.
  static std::string text(const datamunge::gis::ShapeLayer& layer) {
    const auto& attrs = layer.attributes();
    return std::format("ShapeLayer(type={}, shapes={}, attributes={} x {}, bounds={})", layer.shape_type(),
                       layer.size(), attrs.nrows(), attrs.ncols(), layer.bounds());
  }
};
