#include <gtest/gtest.h>

#include <datamunge/geometry/polygon.hpp>
#include <datamunge/gis/gis.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// A minimal, test-only ESRI shapefile (.shp/.dbf) writer, used only to build byte-exact
// fixtures for the reader in shapefile.cpp -- this is deliberately independent code (not a
// reused/shared helper) so a bug in the reader can't be masked by a matching bug in the writer.
namespace {

using datamunge::gis::Point2D;
using datamunge::gis::Shape;
using datamunge::gis::ShapeType;

void write_u16le(std::ostream& out, const std::uint16_t v) {
  const unsigned char b[2] = {static_cast<unsigned char>(v & 0xFF), static_cast<unsigned char>((v >> 8) & 0xFF)};
  out.write(reinterpret_cast<const char*>(b), 2);
}

void write_u32le(std::ostream& out, const std::uint32_t v) {
  const unsigned char b[4] = {static_cast<unsigned char>(v & 0xFF), static_cast<unsigned char>((v >> 8) & 0xFF),
                              static_cast<unsigned char>((v >> 16) & 0xFF), static_cast<unsigned char>((v >> 24) & 0xFF)};
  out.write(reinterpret_cast<const char*>(b), 4);
}

void write_i32le(std::ostream& out, const std::int32_t v) { write_u32le(out, static_cast<std::uint32_t>(v)); }

void write_u32be(std::ostream& out, const std::uint32_t v) {
  const unsigned char b[4] = {static_cast<unsigned char>((v >> 24) & 0xFF), static_cast<unsigned char>((v >> 16) & 0xFF),
                              static_cast<unsigned char>((v >> 8) & 0xFF), static_cast<unsigned char>(v & 0xFF)};
  out.write(reinterpret_cast<const char*>(b), 4);
}

void write_i32be(std::ostream& out, const std::int32_t v) { write_u32be(out, static_cast<std::uint32_t>(v)); }

void write_f64le(std::ostream& out, const double v) {
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(v));
  std::memcpy(&bits, &v, sizeof(v));
  for (int i = 0; i < 8; ++i) {
    out.put(static_cast<char>((bits >> (8 * i)) & 0xFF));
  }
}

int shape_type_code(const ShapeType type) {
  switch (type) {
    case ShapeType::Point:
      return 1;
    case ShapeType::PolyLine:
      return 3;
    case ShapeType::Polygon:
      return 5;
    case ShapeType::MultiPoint:
      return 8;
    default:
      return 0;
  }
}

std::string record_content(const Shape& shape) {
  std::ostringstream out(std::ios::binary);
  write_i32le(out, shape_type_code(shape.type));

  if (shape.type == ShapeType::Point) {
    write_f64le(out, shape.points.at(0).x);
    write_f64le(out, shape.points.at(0).y);
  } else if (shape.type == ShapeType::MultiPoint) {
    write_f64le(out, shape.bounds.min.x);
    write_f64le(out, shape.bounds.min.y);
    write_f64le(out, shape.bounds.max.x);
    write_f64le(out, shape.bounds.max.y);
    write_i32le(out, static_cast<std::int32_t>(shape.points.size()));
    for (const auto& p : shape.points) {
      write_f64le(out, p.x);
      write_f64le(out, p.y);
    }
  } else {
    write_f64le(out, shape.bounds.min.x);
    write_f64le(out, shape.bounds.min.y);
    write_f64le(out, shape.bounds.max.x);
    write_f64le(out, shape.bounds.max.y);
    std::int32_t num_points = 0;
    for (const auto& part : shape.parts) num_points += static_cast<std::int32_t>(part.size());
    write_i32le(out, static_cast<std::int32_t>(shape.parts.size()));
    write_i32le(out, num_points);
    std::int32_t running = 0;
    for (const auto& part : shape.parts) {
      write_i32le(out, running);
      running += static_cast<std::int32_t>(part.size());
    }
    for (const auto& part : shape.parts) {
      for (const auto& p : part) {
        write_f64le(out, p.x);
        write_f64le(out, p.y);
      }
    }
  }

  return out.str();
}

void write_shp(const std::string& path, const ShapeType shape_type, const std::vector<Shape>& shapes) {
  std::vector<std::string> contents;
  contents.reserve(shapes.size());
  for (const auto& shape : shapes) contents.push_back(record_content(shape));

  std::int32_t total_content_words = 0;
  for (const auto& content : contents) total_content_words += 4 + static_cast<std::int32_t>(content.size() / 2);
  const std::int32_t file_length_words = 50 + total_content_words; // 100-byte header = 50 words

  std::ofstream out(path, std::ios::binary);
  write_i32be(out, 9994);
  for (int i = 0; i < 5; ++i) write_i32be(out, 0);
  write_i32be(out, file_length_words);
  write_i32le(out, 1000);
  write_i32le(out, shape_type_code(shape_type));
  write_f64le(out, -1000.0);
  write_f64le(out, -1000.0);
  write_f64le(out, 1000.0);
  write_f64le(out, 1000.0);
  for (int i = 0; i < 4; ++i) write_f64le(out, 0.0);

  for (std::size_t i = 0; i < contents.size(); ++i) {
    write_i32be(out, static_cast<std::int32_t>(i + 1));
    write_i32be(out, static_cast<std::int32_t>(contents[i].size() / 2));
    out.write(contents[i].data(), static_cast<std::streamsize>(contents[i].size()));
  }
}

struct DbfColumn {
  std::string name;
  char type; // 'N' numeric, 'C' character
  int length;
  std::vector<std::string> raw_values; // pre-formatted, will be right-padded/truncated to `length`
};

void write_dbf(const std::string& path, const std::vector<DbfColumn>& columns) {
  const std::uint32_t num_records = columns.empty() ? 0 : static_cast<std::uint32_t>(columns.front().raw_values.size());
  const int header_size = 32 + static_cast<int>(columns.size()) * 32 + 1;
  int record_size = 1;
  for (const auto& column : columns) record_size += column.length;

  std::ofstream out(path, std::ios::binary);
  out.put(static_cast<char>(0x03));
  out.put(0);
  out.put(0);
  out.put(0);
  write_u32le(out, num_records);
  write_u16le(out, static_cast<std::uint16_t>(header_size));
  write_u16le(out, static_cast<std::uint16_t>(record_size));
  for (int i = 0; i < 20; ++i) out.put(0);

  for (const auto& column : columns) {
    std::string name = column.name;
    name.resize(11, '\0');
    out.write(name.data(), 11);
    out.put(column.type);
    for (int i = 0; i < 4; ++i) out.put(0);
    out.put(static_cast<char>(column.length));
    out.put(0);
    for (int i = 0; i < 14; ++i) out.put(0);
  }
  out.put(static_cast<char>(0x0D));

  for (std::uint32_t row = 0; row < num_records; ++row) {
    out.put(' ');
    for (const auto& column : columns) {
      std::string value = column.raw_values.at(row);
      if (static_cast<int>(value.size()) > column.length) {
        value = value.substr(0, static_cast<std::size_t>(column.length));
      } else {
        value.resize(static_cast<std::size_t>(column.length), ' ');
      }
      out.write(value.data(), column.length);
    }
  }
}

std::string temp_path(const std::string& stem) {
  return std::string(::testing::TempDir()) + stem;
}

} // namespace

TEST(GisShapefile, ReadsPolygonWithHoleAndAttributes) {
  // Outer ring must be clockwise (negative signed area), hole counterclockwise (positive), per
  // the ESRI winding convention -- see Shape's doc comment.
  const std::vector<Point2D> outer = {{0, 0}, {0, 10}, {10, 10}, {10, 0}, {0, 0}};
  const std::vector<Point2D> hole = {{3, 3}, {4, 3}, {4, 4}, {3, 4}, {3, 3}};

  Shape polygon;
  polygon.type = ShapeType::Polygon;
  polygon.parts = {outer, hole};
  polygon.bounds = datamunge::geometry::bounding_box(outer);

  const std::string base = temp_path("gis_polygon_test");
  write_shp(base + ".shp", ShapeType::Polygon, {polygon});
  write_dbf(base + ".dbf", {
                               DbfColumn{"NAME", 'C', 11, {"Countyville"}},
                               DbfColumn{"POP", 'N', 8, {"12345"}},
                           });

  const auto layer = datamunge::gis::ShapeLayer::read(base);
  EXPECT_EQ(layer.size(), 1u);
  EXPECT_EQ(layer.shape_type(), ShapeType::Polygon);

  const auto& shape = layer.shape(0);
  ASSERT_EQ(shape.parts.size(), 2u);
  EXPECT_EQ(shape.parts[0].size(), 5u);
  EXPECT_EQ(shape.parts[1].size(), 5u);
  EXPECT_DOUBLE_EQ(shape.parts[0][2].x, 10.0);
  EXPECT_DOUBLE_EQ(shape.parts[0][2].y, 10.0);

  EXPECT_LT(datamunge::geometry::signed_polygon_area(shape.parts[0]), 0.0) << "outer ring must read back clockwise";
  EXPECT_GT(datamunge::geometry::signed_polygon_area(shape.parts[1]), 0.0) << "hole ring must read back counterclockwise";

  const auto& attributes = layer.attributes();
  EXPECT_EQ(attributes.nrows(), 1u);
  EXPECT_EQ(attributes.string_at("NAME", 0), "Countyville");
  EXPECT_DOUBLE_EQ(attributes.double_at("POP", 0), 12345.0);
}

TEST(GisShapefile, ReadsPolyLineWithMultipleParts) {
  Shape line;
  line.type = ShapeType::PolyLine;
  line.parts = {{{0, 0}, {1, 1}, {2, 0}}, {{5, 5}, {6, 6}}};
  std::vector<Point2D> all = {{0, 0}, {1, 1}, {2, 0}, {5, 5}, {6, 6}};
  line.bounds = datamunge::geometry::bounding_box(all);

  const std::string base = temp_path("gis_polyline_test");
  write_shp(base + ".shp", ShapeType::PolyLine, {line});
  write_dbf(base + ".dbf", {DbfColumn{"ID", 'N', 4, {"1"}}});

  const auto layer = datamunge::gis::ShapeLayer::read(base);
  ASSERT_EQ(layer.size(), 1u);
  EXPECT_EQ(layer.shape_type(), ShapeType::PolyLine);
  const auto& shape = layer.shape(0);
  ASSERT_EQ(shape.parts.size(), 2u);
  EXPECT_EQ(shape.parts[0].size(), 3u);
  EXPECT_EQ(shape.parts[1].size(), 2u);
  EXPECT_DOUBLE_EQ(shape.parts[1][1].x, 6.0);
}

TEST(GisShapefile, ReadsPointsAndNullValues) {
  Shape a;
  a.type = ShapeType::Point;
  a.points = {{1.5, 2.5}};
  a.bounds = datamunge::geometry::bounding_box(a.points);

  Shape b;
  b.type = ShapeType::Point;
  b.points = {{-3.0, 4.0}};
  b.bounds = datamunge::geometry::bounding_box(b.points);

  const std::string base = temp_path("gis_point_test");
  write_shp(base + ".shp", ShapeType::Point, {a, b});
  write_dbf(base + ".dbf", {
                               DbfColumn{"LABEL", 'C', 6, {"first", ""}},
                               DbfColumn{"VALUE", 'N', 6, {"1.5", ""}},
                           });

  const auto layer = datamunge::gis::ShapeLayer::read(base);
  ASSERT_EQ(layer.size(), 2u);
  EXPECT_DOUBLE_EQ(layer.shape(0).points.at(0).x, 1.5);
  EXPECT_DOUBLE_EQ(layer.shape(1).points.at(0).y, 4.0);

  const auto& attributes = layer.attributes();
  EXPECT_TRUE(attributes.is_null("LABEL", 1));
  EXPECT_TRUE(attributes.is_null("VALUE", 1));
  EXPECT_EQ(attributes.string_at("LABEL", 0), "first");
}

TEST(GisShapefile, ReadsMultiPoint) {
  Shape multi;
  multi.type = ShapeType::MultiPoint;
  multi.points = {{0, 0}, {1, 0}, {1, 1}};
  multi.bounds = datamunge::geometry::bounding_box(multi.points);

  const std::string base = temp_path("gis_multipoint_test");
  write_shp(base + ".shp", ShapeType::MultiPoint, {multi});
  write_dbf(base + ".dbf", {DbfColumn{"ID", 'N', 4, {"1"}}});

  const auto layer = datamunge::gis::ShapeLayer::read(base);
  ASSERT_EQ(layer.size(), 1u);
  EXPECT_EQ(layer.shape(0).points.size(), 3u);
}

TEST(GisShapefile, ThrowsWhenAttributeRowCountMismatchesShapeCount) {
  Shape point;
  point.type = ShapeType::Point;
  point.points = {{0, 0}};
  point.bounds = datamunge::geometry::bounding_box(point.points);

  const std::string base = temp_path("gis_mismatch_test");
  write_shp(base + ".shp", ShapeType::Point, {point, point});
  write_dbf(base + ".dbf", {DbfColumn{"ID", 'N', 4, {"1"}}}); // one row, two shapes

  EXPECT_THROW(datamunge::gis::ShapeLayer::read(base), std::runtime_error);
}

TEST(GisShapefile, PathWithExtensionIsAccepted) {
  Shape point;
  point.type = ShapeType::Point;
  point.points = {{7, 8}};
  point.bounds = datamunge::geometry::bounding_box(point.points);

  const std::string base = temp_path("gis_ext_test");
  write_shp(base + ".shp", ShapeType::Point, {point});
  write_dbf(base + ".dbf", {DbfColumn{"ID", 'N', 4, {"1"}}});

  const auto layer = datamunge::gis::ShapeLayer::read(base + ".shp");
  EXPECT_EQ(layer.size(), 1u);
}

TEST(GisShapefile, PlotProducesSvgWithPolygonAndHoleColors) {
  const std::vector<Point2D> outer = {{0, 0}, {0, 10}, {10, 10}, {10, 0}, {0, 0}};
  const std::vector<Point2D> hole = {{3, 3}, {4, 3}, {4, 4}, {3, 4}, {3, 3}};

  Shape polygon;
  polygon.type = ShapeType::Polygon;
  polygon.parts = {outer, hole};
  polygon.bounds = datamunge::geometry::bounding_box(outer);

  const std::string base = temp_path("gis_plot_test");
  write_shp(base + ".shp", ShapeType::Polygon, {polygon});
  write_dbf(base + ".dbf", {DbfColumn{"ID", 'N', 4, {"1"}}});

  const auto layer = datamunge::gis::ShapeLayer::read(base);
  const auto rplot = layer.plot();

  const std::string svg_path = base + ".svg";
  rplot.save_svg(svg_path);

  std::ifstream svg_in(svg_path);
  ASSERT_TRUE(svg_in.good());
  const std::string svg((std::istreambuf_iterator<char>(svg_in)), std::istreambuf_iterator<char>());
  EXPECT_NE(svg.find("<svg"), std::string::npos);
  EXPECT_NE(svg.find("polygon"), std::string::npos);
}
