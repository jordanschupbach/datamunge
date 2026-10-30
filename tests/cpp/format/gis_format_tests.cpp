// std::format / std::print support for datamunge::gis.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <format>
#include <string>

namespace gis = dm::gis;

static_assert(std::formattable<gis::ShapeType, char>);
static_assert(std::formattable<gis::Shape, char>);
static_assert(std::formattable<gis::ShapefileData, char>);
static_assert(std::formattable<gis::ShapeLayer, char>);

TEST(GisFormat, ValueTypes) {
    EXPECT_EQ(std::format("{}", gis::ShapeType::Polygon), "polygon");
    gis::Shape s;
    s.type = gis::ShapeType::Point;
    s.points = {{1, 2}};
    EXPECT_EQ(std::format("{}", s),
              "Shape(type=point, points=1, parts=0, bounds=BoundingBox(min=Point2D(0, 0), max=Point2D(0, 0)))");
    gis::ShapefileData d;
    d.shape_type = gis::ShapeType::Point;
    d.shapes = {s, s};
    EXPECT_EQ(std::format("{}", d),
              "ShapefileData(type=point, shapes=2, bounds=BoundingBox(min=Point2D(0, 0), max=Point2D(0, 0)))");
}

TEST(GisFormat, ShapeLayer) {
    const auto base = std::filesystem::path(__FILE__).parent_path() / "../../../examples/d/datamunge_gis_ex_counties_d";
    if (!std::filesystem::exists(base.string() + ".shp")) GTEST_SKIP() << "fixture shapefile not found";
    const auto layer = gis::ShapeLayer::read(base.string());
    const std::string text = std::format("{}", layer);
    EXPECT_EQ(text.rfind(std::format("ShapeLayer(type={}, shapes={}, attributes={} x ", layer.shape_type(),
                                     layer.size(), layer.attributes().nrows()),
                         0),
              0u);
    EXPECT_NE(text.find("bounds=BoundingBox("), std::string::npos);
}
