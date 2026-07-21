#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/gis/shape.hpp>
#include <datamunge/gis/shapefile.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::gis {

/// @brief A shapefile's geometry (".shp") paired row-for-row with its attribute table (".dbf").
class ShapeLayer {
 public:
  /// @brief Reads "<path>.shp" and "<path>.dbf" (`path` may already end in one of those
  ///        extensions, or in neither). Throws std::runtime_error if either file is missing or
  ///        malformed, or if the attribute table's row count doesn't match the shape count.
  [[nodiscard]] static ShapeLayer read(const std::string& path);

  [[nodiscard]] std::size_t size() const { return shapes_.size(); }
  [[nodiscard]] bool empty() const { return shapes_.empty(); }
  [[nodiscard]] ShapeType shape_type() const { return shape_type_; }
  [[nodiscard]] const Shape& shape(std::size_t index) const;
  [[nodiscard]] const BoundingBox& bounds() const { return bounds_; }
  [[nodiscard]] const dstruct::DataFrame& attributes() const { return attributes_; }

  /// @brief Renders every shape as a map: Polygon shapes via one filled RPlot::polygon() call
  ///        per ring (outer rings in `fill_color`, holes approximated by re-filling in the
  ///        canvas's background color -- see the .cpp for why -- since the underlying SVG
  ///        engine has no even-odd multi-ring fill), PolyLine shapes via RPlot::line() per
  ///        part, and Point/MultiPoint shapes via RPlot::points(). The data's x/y range is
  ///        padded so the map's aspect ratio matches the canvas's (`width`/`height`), which
  ///        Plot's default independent x/y autoscaling would otherwise distort.
  [[nodiscard]] plot::RPlot plot(plot::RGB fill_color = {148, 163, 184}, plot::RGB border_color = {51, 65, 85},
                                 std::size_t width = 800, std::size_t height = 800) const;

 private:
  ShapeLayer(std::vector<Shape> shapes, ShapeType shape_type, BoundingBox bounds, dstruct::DataFrame attributes)
      : shapes_(std::move(shapes)), shape_type_(shape_type), bounds_(bounds), attributes_(std::move(attributes)) {}

  std::vector<Shape> shapes_;
  ShapeType shape_type_{ShapeType::Null};
  BoundingBox bounds_{};
  dstruct::DataFrame attributes_;
};

} // namespace datamunge::gis
