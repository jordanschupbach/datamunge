#include <datamunge/gis/shape_layer.hpp>

#include <datamunge/geometry/polygon.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace datamunge::gis {

namespace {

// Strips a trailing ".shp" or ".dbf" (case-insensitive), if present, so a path given with
// either extension resolves both files correctly -- read_shp()/read_dbf() each only strip
// their own extension, so passing e.g. "layer.shp" straight through to read_dbf() would look
// for "layer.shp.dbf" instead of "layer.dbf".
std::string strip_known_extension(const std::string& path) {
  for (const char* extension : {".shp", ".dbf"}) {
    const std::size_t length = std::char_traits<char>::length(extension);
    if (path.size() < length) {
      continue;
    }
    std::string suffix = path.substr(path.size() - length);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](const unsigned char c) { return std::tolower(c); });
    if (suffix == extension) {
      return path.substr(0, path.size() - length);
    }
  }
  return path;
}

} // namespace

ShapeLayer ShapeLayer::read(const std::string& path) {
  const std::string base = strip_known_extension(path);
  auto shp = read_shp(base);
  auto attributes = read_dbf(base);

  if (attributes.nrows() != shp.shapes.size()) {
    throw std::runtime_error("ShapeLayer::read: shape count (" + std::to_string(shp.shapes.size()) +
                             ") does not match attribute row count (" + std::to_string(attributes.nrows()) +
                             ") for '" + path + "'");
  }

  return ShapeLayer(std::move(shp.shapes), shp.shape_type, shp.bounds, std::move(attributes));
}

const Shape& ShapeLayer::shape(const std::size_t index) const {
  if (index >= shapes_.size()) {
    throw std::out_of_range("ShapeLayer::shape index out of range");
  }
  return shapes_[index];
}

namespace {

std::vector<double> xs_of(const std::vector<Point2D>& points) {
  std::vector<double> xs;
  xs.reserve(points.size());
  for (const auto& p : points) xs.push_back(p.x);
  return xs;
}

std::vector<double> ys_of(const std::vector<Point2D>& points) {
  std::vector<double> ys;
  ys.reserve(points.size());
  for (const auto& p : points) ys.push_back(p.y);
  return ys;
}

} // namespace

plot::RPlot ShapeLayer::plot(const plot::RGB fill_color, const plot::RGB border_color, const std::size_t width,
                             const std::size_t height) const {
  auto canvas = plot::RPlot::create();
  canvas.size(width, height);
  canvas.x_label("");
  canvas.y_label("");

  if (!shapes_.empty()) {
    double x_min = bounds_.min.x;
    double x_max = bounds_.max.x;
    double y_min = bounds_.min.y;
    double y_max = bounds_.max.y;
    const double data_w = std::max(x_max - x_min, 1e-9);
    const double data_h = std::max(y_max - y_min, 1e-9);
    const double canvas_aspect = static_cast<double>(width) / static_cast<double>(height);

    // Pad the shorter data-space dimension so the map's aspect ratio matches the canvas's --
    // Plot otherwise scales x and y independently, visibly distorting shapes.
    if (data_w / data_h > canvas_aspect) {
      const double target_h = data_w / canvas_aspect;
      const double pad = (target_h - data_h) / 2.0;
      y_min -= pad;
      y_max += pad;
    } else {
      const double target_w = data_h * canvas_aspect;
      const double pad = (target_w - data_w) / 2.0;
      x_min -= pad;
      x_max += pad;
    }

    // A small margin so shapes don't touch the canvas edge.
    const double margin_x = (x_max - x_min) * 0.05;
    const double margin_y = (y_max - y_min) * 0.05;
    canvas.x_limits(x_min - margin_x, x_max + margin_x);
    canvas.y_limits(y_min - margin_y, y_max + margin_y);
  }

  // Polygon holes: the underlying SVG engine renders one filled <polygon> per ring with no
  // even-odd multi-ring fill rule, so a true hole (e.g. a lake inside a county) can't be cut
  // out directly. Approximate it: per the ESRI winding convention, an outer ring is clockwise
  // (negative signed area) and a hole is counterclockwise (positive); draw every outer ring
  // first, then re-paint every hole in the canvas's own background color on top, which visually
  // punches it out as long as holes don't overlap other shapes drawn afterward.
  std::vector<const std::vector<Point2D>*> outer_rings;
  std::vector<const std::vector<Point2D>*> holes;

  for (const auto& shape : shapes_) {
    switch (shape.type) {
      case ShapeType::Polygon:
        for (const auto& ring : shape.parts) {
          if (ring.size() < 3) {
            continue;
          }
          if (geometry::signed_polygon_area(ring) < 0.0) {
            outer_rings.push_back(&ring);
          } else {
            holes.push_back(&ring);
          }
        }
        break;
      case ShapeType::PolyLine:
        for (const auto& line : shape.parts) {
          canvas.line(xs_of(line), ys_of(line), "", border_color);
        }
        break;
      case ShapeType::Point:
      case ShapeType::MultiPoint:
        canvas.points(xs_of(shape.points), ys_of(shape.points), "", fill_color);
        break;
      case ShapeType::Null:
        break;
    }
  }

  for (const auto* ring : outer_rings) {
    canvas.polygon(xs_of(*ring), ys_of(*ring), fill_color, true);
  }
  for (const auto* ring : holes) {
    canvas.polygon(xs_of(*ring), ys_of(*ring), canvas.background_color(), true);
  }
  for (const auto* ring : outer_rings) {
    canvas.polygon(xs_of(*ring), ys_of(*ring), border_color, false);
  }

  return canvas;
}

} // namespace datamunge::gis
