#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/gis/shape.hpp>

#include <string>
#include <vector>

namespace datamunge::gis {

/// @brief The parsed contents of a ".shp" file.
struct ShapefileData {
  ShapeType shape_type{ShapeType::Null};
  std::vector<Shape> shapes;
  BoundingBox bounds{};
};

/// @brief Reads a ".shp" file's geometry. `path` may be given with or without the ".shp"
///        extension. Only Null/Point/PolyLine/Polygon/MultiPoint records are supported --
///        Z/M-valued variants and MultiPatch throw std::runtime_error naming the unsupported
///        shape type. Null-shape records are skipped (not included in the returned `shapes`),
///        matching most GIS tooling's treatment of them as "no geometry for this row" rather
///        than an error.
[[nodiscard]] ShapefileData read_shp(const std::string& path);

/// @brief Reads a ".dbf" attribute table into a DataFrame, one row per record (including
///        records marked deleted, so row order always matches read_shp()'s shape order 1:1),
///        column order/names taken from the field descriptors. Numeric ('N'/'F') fields become
///        numeric columns (blank/whitespace-only values are null); every other field type
///        ('C' character, 'D' date as raw "YYYYMMDD" text, 'L' logical as a raw "T"/"F"/"?"
///        character, ...) becomes a string column, right-trimmed, with an empty result treated
///        as null. `path` may be given with or without the ".dbf" extension.
[[nodiscard]] dstruct::DataFrame read_dbf(const std::string& path);

} // namespace datamunge::gis
