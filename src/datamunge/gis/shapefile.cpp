#include <datamunge/gis/shapefile.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>

namespace datamunge::gis {

namespace {

std::uint32_t read_u32le(std::istream& in) {
  unsigned char b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

std::int32_t read_i32le(std::istream& in) { return static_cast<std::int32_t>(read_u32le(in)); }

std::uint32_t read_u32be(std::istream& in) {
  unsigned char b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  return (static_cast<std::uint32_t>(b[0]) << 24) | (static_cast<std::uint32_t>(b[1]) << 16) |
         (static_cast<std::uint32_t>(b[2]) << 8) | static_cast<std::uint32_t>(b[3]);
}

std::int32_t read_i32be(std::istream& in) { return static_cast<std::int32_t>(read_u32be(in)); }

double read_f64le(std::istream& in) {
  unsigned char b[8];
  in.read(reinterpret_cast<char*>(b), 8);
  std::uint64_t bits = 0;
  for (int i = 7; i >= 0; --i) {
    bits = (bits << 8) | b[i];
  }
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void require_read(const std::istream& in, const std::string& context, const std::string& path) {
  if (!in) {
    throw std::runtime_error(context + ": unexpected end of file in '" + path + "'");
  }
}

std::string trim(const std::string& text) {
  const auto begin = text.find_first_not_of(" \t\r\n\0", 0, 5);
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = text.find_last_not_of(" \t\r\n\0", std::string::npos, 5);
  return text.substr(begin, end - begin + 1);
}

std::string strip_extension(const std::string& path, const std::string& extension) {
  if (path.size() >= extension.size()) {
    std::string lowered = path.substr(path.size() - extension.size());
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](const unsigned char c) { return std::tolower(c); });
    if (lowered == extension) {
      return path.substr(0, path.size() - extension.size());
    }
  }
  return path;
}

std::vector<Point2D> read_points(std::istream& in, const std::int32_t count) {
  std::vector<Point2D> points;
  points.reserve(static_cast<std::size_t>(count));
  for (std::int32_t i = 0; i < count; ++i) {
    const double x = read_f64le(in);
    const double y = read_f64le(in);
    points.push_back({x, y});
  }
  return points;
}

std::vector<std::vector<Point2D>> read_parts(std::istream& in, const std::int32_t num_parts, const std::int32_t num_points) {
  std::vector<std::int32_t> part_starts(static_cast<std::size_t>(num_parts));
  for (auto& start : part_starts) {
    start = read_i32le(in);
  }
  const auto all_points = read_points(in, num_points);

  std::vector<std::vector<Point2D>> parts(static_cast<std::size_t>(num_parts));
  for (std::int32_t part = 0; part < num_parts; ++part) {
    const auto begin = static_cast<std::size_t>(part_starts[static_cast<std::size_t>(part)]);
    const auto end = part + 1 < num_parts ? static_cast<std::size_t>(part_starts[static_cast<std::size_t>(part) + 1])
                                          : static_cast<std::size_t>(num_points);
    if (begin > all_points.size() || end > all_points.size() || begin > end) {
      throw std::runtime_error("read_shp: malformed part index");
    }
    parts[static_cast<std::size_t>(part)].assign(all_points.begin() + static_cast<std::ptrdiff_t>(begin),
                                                 all_points.begin() + static_cast<std::ptrdiff_t>(end));
  }
  return parts;
}

BoundingBox shape_bounds(const Shape& shape) {
  std::vector<Point2D> all;
  all.insert(all.end(), shape.points.begin(), shape.points.end());
  for (const auto& part : shape.parts) {
    all.insert(all.end(), part.begin(), part.end());
  }
  if (all.empty()) {
    return BoundingBox{};
  }
  return geometry::bounding_box(all);
}

} // namespace

ShapefileData read_shp(const std::string& path) {
  const std::string base = strip_extension(path, ".shp");
  const std::string full_path = base + ".shp";

  std::ifstream in(full_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("read_shp: cannot open '" + full_path + "' for reading");
  }

  const std::int32_t file_code = read_i32be(in);
  if (file_code != 9994) {
    throw std::runtime_error("read_shp: not a shapefile (bad file code) in '" + full_path + "'");
  }
  for (int i = 0; i < 5; ++i) {
    read_i32be(in); // unused
  }
  const std::int32_t file_length_words = read_i32be(in);
  read_i32le(in); // version, unused
  const std::int32_t header_shape_type = read_i32le(in);
  const double bounds_xmin = read_f64le(in);
  const double bounds_ymin = read_f64le(in);
  const double bounds_xmax = read_f64le(in);
  const double bounds_ymax = read_f64le(in);
  for (int i = 0; i < 4; ++i) {
    read_f64le(in); // Zmin/Zmax/Mmin/Mmax, unused
  }
  require_read(in, "read_shp", full_path);

  const auto supported_type = [](const std::int32_t code) -> std::optional<ShapeType> {
    switch (code) {
      case 0:
        return ShapeType::Null;
      case 1:
        return ShapeType::Point;
      case 3:
        return ShapeType::PolyLine;
      case 5:
        return ShapeType::Polygon;
      case 8:
        return ShapeType::MultiPoint;
      default:
        return std::nullopt;
    }
  };

  const auto layer_type = supported_type(header_shape_type);
  if (!layer_type.has_value()) {
    throw std::runtime_error("read_shp: unsupported shape type code " + std::to_string(header_shape_type) +
                             " in '" + full_path + "' (only Point/PolyLine/Polygon/MultiPoint are supported -- " +
                             "Z/M variants and MultiPatch are not)");
  }

  const std::streamoff file_length_bytes = static_cast<std::streamoff>(file_length_words) * 2;

  ShapefileData data;
  data.shape_type = *layer_type;
  data.bounds = BoundingBox{{bounds_xmin, bounds_ymin}, {bounds_xmax, bounds_ymax}};

  while (in.tellg() < file_length_bytes && in.peek() != std::char_traits<char>::eof()) {
    read_i32be(in); // record number, unused
    const std::int32_t content_words = read_i32be(in);
    require_read(in, "read_shp", full_path);
    const std::streamoff record_end = static_cast<std::streamoff>(in.tellg()) + static_cast<std::streamoff>(content_words) * 2;

    const std::int32_t record_type_code = read_i32le(in);
    const auto record_type = supported_type(record_type_code);
    if (!record_type.has_value()) {
      throw std::runtime_error("read_shp: unsupported shape type code " + std::to_string(record_type_code) +
                               " on a record in '" + full_path + "'");
    }

    Shape shape;
    shape.type = *record_type;

    switch (*record_type) {
      case ShapeType::Null:
        break;
      case ShapeType::Point: {
        const double x = read_f64le(in);
        const double y = read_f64le(in);
        shape.points.push_back({x, y});
        break;
      }
      case ShapeType::MultiPoint: {
        for (int i = 0; i < 4; ++i) read_f64le(in); // box, unused (recomputed below)
        const std::int32_t num_points = read_i32le(in);
        shape.points = read_points(in, num_points);
        break;
      }
      case ShapeType::PolyLine:
      case ShapeType::Polygon: {
        for (int i = 0; i < 4; ++i) read_f64le(in); // box, unused (recomputed below)
        const std::int32_t num_parts = read_i32le(in);
        const std::int32_t num_points = read_i32le(in);
        shape.parts = read_parts(in, num_parts, num_points);
        break;
      }
    }
    require_read(in, "read_shp", full_path);

    if (*record_type != ShapeType::Null) {
      shape.bounds = shape_bounds(shape);
      data.shapes.push_back(std::move(shape));
    }

    in.seekg(record_end, std::ios::beg);
  }

  return data;
}

namespace {

struct DbfField {
  std::string name;
  char type{'C'};
  int length{0};
};

} // namespace

dstruct::DataFrame read_dbf(const std::string& path) {
  const std::string base = strip_extension(path, ".dbf");
  const std::string full_path = base + ".dbf";

  std::ifstream in(full_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("read_dbf: cannot open '" + full_path + "' for reading");
  }

  unsigned char version = 0;
  in.read(reinterpret_cast<char*>(&version), 1);
  in.seekg(3, std::ios::cur); // last-update date, unused
  const std::uint32_t num_records = read_u32le(in);
  unsigned char header_size_bytes[2];
  in.read(reinterpret_cast<char*>(header_size_bytes), 2);
  const int header_size = header_size_bytes[0] | (header_size_bytes[1] << 8);
  unsigned char record_size_bytes[2];
  in.read(reinterpret_cast<char*>(record_size_bytes), 2);
  const int record_size = record_size_bytes[0] | (record_size_bytes[1] << 8);
  in.seekg(20, std::ios::cur); // reserved
  require_read(in, "read_dbf", full_path);

  std::vector<DbfField> fields;
  while (in.tellg() < header_size - 1) {
    char first_byte = 0;
    in.read(&first_byte, 1);
    if (first_byte == static_cast<char>(0x0D)) {
      break;
    }

    char name_buf[11] = {};
    name_buf[0] = first_byte;
    in.read(name_buf + 1, 10);
    char type_char = 0;
    in.read(&type_char, 1);
    in.seekg(4, std::ios::cur); // field data address, unused
    unsigned char length_byte = 0;
    in.read(reinterpret_cast<char*>(&length_byte), 1);
    in.seekg(15, std::ios::cur); // decimal count + reserved
    require_read(in, "read_dbf", full_path);

    DbfField field;
    field.name = trim(std::string(name_buf, 11));
    field.type = type_char;
    field.length = length_byte;
    fields.push_back(std::move(field));
  }

  std::vector<std::vector<std::optional<double>>> numeric_columns(fields.size());
  std::vector<std::vector<std::optional<std::string>>> string_columns(fields.size());
  std::vector<bool> is_numeric(fields.size());
  for (std::size_t i = 0; i < fields.size(); ++i) {
    is_numeric[i] = fields[i].type == 'N' || fields[i].type == 'F';
  }

  in.seekg(header_size, std::ios::beg);
  for (std::uint32_t record = 0; record < num_records; ++record) {
    char deletion_flag = 0;
    in.read(&deletion_flag, 1);
    if (!in) {
      break; // fewer records than the header claims -- tolerate a truncated file
    }

    std::vector<char> row_buffer(static_cast<std::size_t>(record_size - 1));
    in.read(row_buffer.data(), static_cast<std::streamsize>(row_buffer.size()));
    require_read(in, "read_dbf", full_path);

    std::size_t offset = 0;
    for (std::size_t field_index = 0; field_index < fields.size(); ++field_index) {
      const auto length = static_cast<std::size_t>(fields[field_index].length);
      const std::string raw = trim(std::string(row_buffer.data() + offset, length));
      offset += length;

      if (is_numeric[field_index]) {
        if (raw.empty()) {
          numeric_columns[field_index].push_back(std::nullopt);
        } else {
          try {
            numeric_columns[field_index].push_back(std::stod(raw));
          } catch (const std::exception&) {
            numeric_columns[field_index].push_back(std::nullopt);
          }
        }
      } else {
        string_columns[field_index].push_back(raw.empty() ? std::optional<std::string>(std::nullopt)
                                                           : std::optional<std::string>(raw));
      }
    }
  }

  dstruct::DataFrame attributes;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    if (is_numeric[i]) {
      attributes.add_column(fields[i].name, std::move(numeric_columns[i]));
    } else {
      attributes.add_column(fields[i].name, std::move(string_columns[i]));
    }
  }
  return attributes;
}

} // namespace datamunge::gis
