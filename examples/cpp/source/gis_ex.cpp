// Demonstrates datamunge::gis: reading a shapefile (.shp geometry + .dbf attributes) into a
// ShapeLayer and drawing it as a map. Real .shp/.dbf files are large binary bundles that don't
// belong in this repo, so this example first writes a tiny synthetic shapefile by hand (two
// "counties": one plain square, one square with a lake-shaped hole) using the same ESRI
// shapefile byte layout ShapeLayer::read() expects, then reads it back through the public API.

#include <datamunge/gis/gis.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using datamunge::gis::Point2D;
using datamunge::gis::Shape;
using datamunge::gis::ShapeType;

namespace {

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
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>((bits >> (8 * i)) & 0xFF));
}

std::string polygon_record(const Shape& shape) {
  std::ostringstream out(std::ios::binary);
  write_i32le(out, 5); // shape type: Polygon
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
  return out.str();
}

void write_counties_shp(const std::string& path, const std::vector<Shape>& shapes) {
  std::vector<std::string> contents;
  for (const auto& shape : shapes) contents.push_back(polygon_record(shape));

  std::int32_t total_words = 0;
  for (const auto& content : contents) total_words += 4 + static_cast<std::int32_t>(content.size() / 2);

  double xmin = shapes.front().bounds.min.x, ymin = shapes.front().bounds.min.y;
  double xmax = shapes.front().bounds.max.x, ymax = shapes.front().bounds.max.y;
  for (const auto& shape : shapes) {
    xmin = std::min(xmin, shape.bounds.min.x);
    ymin = std::min(ymin, shape.bounds.min.y);
    xmax = std::max(xmax, shape.bounds.max.x);
    ymax = std::max(ymax, shape.bounds.max.y);
  }

  std::ofstream out(path, std::ios::binary);
  write_i32be(out, 9994);
  for (int i = 0; i < 5; ++i) write_i32be(out, 0);
  write_i32be(out, 50 + total_words);
  write_i32le(out, 1000);
  write_i32le(out, 5); // Polygon
  write_f64le(out, xmin);
  write_f64le(out, ymin);
  write_f64le(out, xmax);
  write_f64le(out, ymax);
  for (int i = 0; i < 4; ++i) write_f64le(out, 0.0);

  for (std::size_t i = 0; i < contents.size(); ++i) {
    write_i32be(out, static_cast<std::int32_t>(i + 1));
    write_i32be(out, static_cast<std::int32_t>(contents[i].size() / 2));
    out.write(contents[i].data(), static_cast<std::streamsize>(contents[i].size()));
  }
}

void write_counties_dbf(const std::string& path, const std::vector<std::string>& names,
                        const std::vector<std::string>& populations) {
  struct Field {
    std::string name;
    char type;
    int length;
  };
  const std::vector<Field> fields = {{"NAME", 'C', 12}, {"POP", 'N', 8}};

  const auto num_records = static_cast<std::uint32_t>(names.size());
  const int header_size = 32 + static_cast<int>(fields.size()) * 32 + 1;
  int record_size = 1;
  for (const auto& field : fields) record_size += field.length;

  std::ofstream out(path, std::ios::binary);
  out.put(static_cast<char>(0x03));
  out.put(0);
  out.put(0);
  out.put(0);
  write_u32le(out, num_records);
  write_u16le(out, static_cast<std::uint16_t>(header_size));
  write_u16le(out, static_cast<std::uint16_t>(record_size));
  for (int i = 0; i < 20; ++i) out.put(0);
  for (const auto& field : fields) {
    std::string name = field.name;
    name.resize(11, '\0');
    out.write(name.data(), 11);
    out.put(field.type);
    for (int i = 0; i < 4; ++i) out.put(0);
    out.put(static_cast<char>(field.length));
    out.put(0);
    for (int i = 0; i < 14; ++i) out.put(0);
  }
  out.put(static_cast<char>(0x0D));

  for (std::uint32_t row = 0; row < num_records; ++row) {
    out.put(' ');
    auto write_field = [&](std::string value, const int length) {
      value.resize(static_cast<std::size_t>(length), ' ');
      out.write(value.data(), length);
    };
    write_field(names[row], fields[0].length);
    write_field(populations[row], fields[1].length);
  }
}

} // namespace

int main() {
  const auto temp_dir = std::filesystem::temp_directory_path();
  const std::string base = (temp_dir / "datamunge_gis_ex_counties").string();

  // Shape 1: a plain square county. Shape 2: a county with a lake-shaped hole -- per the ESRI
  // winding convention, outer rings are clockwise, holes counterclockwise (see Shape's doc
  // comment in gis/shape.hpp).
  Shape plain;
  plain.type = ShapeType::Polygon;
  plain.parts = {{{10, 0}, {10, 10}, {20, 10}, {20, 0}, {10, 0}}};
  plain.bounds = datamunge::geometry::bounding_box(plain.parts[0]);

  Shape with_lake;
  with_lake.type = ShapeType::Polygon;
  const std::vector<Point2D> outer = {{0, 0}, {0, 10}, {10, 10}, {10, 0}, {0, 0}};
  const std::vector<Point2D> lake = {{3, 3}, {4, 3}, {4, 4}, {3, 4}, {3, 3}};
  with_lake.parts = {outer, lake};
  with_lake.bounds = datamunge::geometry::bounding_box(outer);

  write_counties_shp(base + ".shp", {with_lake, plain});
  write_counties_dbf(base + ".dbf", {"Lakeside", "Plainview"}, {"48231", "19876"});

  const auto counties = datamunge::gis::ShapeLayer::read(base);

  std::cout << "shapes = " << counties.size() << ", shape_type = " << datamunge::gis::to_string(counties.shape_type())
            << "\n";
  std::cout << "bounds = (" << counties.bounds().min.x << ", " << counties.bounds().min.y << ") - (" << counties.bounds().max.x
            << ", " << counties.bounds().max.y << ")\n\n";

  std::cout << "attributes\n" << counties.attributes().to_string() << "\n\n";

  for (std::size_t i = 0; i < counties.size(); ++i) {
    const auto& shape = counties.shape(i);
    std::cout << counties.attributes().string_at("NAME", i) << ": " << shape.num_parts() << " ring(s)\n";
  }

  const auto map = counties.plot();
  const std::string svg_path = (temp_dir / "datamunge_gis_ex_map.svg").string();
  map.save_svg(svg_path);
  std::cout << "\nmap saved to " << svg_path << "\n";

  return 0;
}
