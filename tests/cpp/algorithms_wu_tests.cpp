#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/xiaolin_wu_line.hpp>
#include <limits>
#include <map>
#include <random>
namespace alg = datamunge::algorithms;
namespace {
double wu_test_mass(const std::vector<alg::WuPixel>& pixels) {
  double sum = 0;
  for (const auto& p : pixels)
    sum += p.brightness;
  return sum;
}
std::map<std::pair<int, int>, double> wu_test_map(const std::vector<alg::WuPixel>& pixels) {
  std::map<std::pair<int, int>, double> out;
  for (const auto& p : pixels)
    out[{p.x, p.y}] += p.brightness;
  return out;
}
} // namespace
TEST(WuLine, PreservesFullColumnVariantAndIntegerPairs) {
  const auto pixels = alg::xiaolin_wu_line(0, 0, 12, 5);
  EXPECT_EQ(pixels.size(), 26);
  EXPECT_NEAR(wu_test_mass(pixels), 13, 1e-13);
  EXPECT_EQ(pixels[2].x, 1);
  EXPECT_EQ(pixels[2].y, 0);
  EXPECT_NEAR(pixels[2].brightness, 7.0 / 12, 1e-14);
  EXPECT_NEAR(pixels[3].brightness, 5.0 / 12, 1e-14);
  for (std::size_t i = 0; i < pixels.size(); i += 2)
    EXPECT_NEAR(pixels[i].brightness + pixels[i + 1].brightness, 1, 1e-14);
}
TEST(WuLine, ClippedColumnsConserveProjectedLengthInAllOctants) {
  std::mt19937                           engine(2026);
  std::uniform_real_distribution<double> coord(-20, 20);
  for (int k = 0; k < 1000; ++k) {
    const double x0 = coord(engine), y0 = coord(engine), x1 = coord(engine), y1 = coord(engine);
    const double length = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    const auto   a = alg::xiaolin_wu_line(x0, y0, x1, y1, alg::WuEndpointPolicy::ClippedColumns);
    const auto   b = alg::xiaolin_wu_line(x1, y1, x0, y0, alg::WuEndpointPolicy::ClippedColumns);
    EXPECT_NEAR(wu_test_mass(a), length, 1e-11);
    EXPECT_EQ(wu_test_map(a), wu_test_map(b));
    const auto transposed =
        wu_test_map(alg::xiaolin_wu_line(y0, x0, y1, x1, alg::WuEndpointPolicy::ClippedColumns));
    for (const auto& p : a) {
      EXPECT_GE(p.brightness, 0);
      EXPECT_LE(p.brightness, 1);
      EXPECT_NEAR(wu_test_map(a).at({p.x, p.y}), transposed.at({p.y, p.x}), 1e-12);
    }
  }
}
TEST(WuLine, FractionalEndpointsShortSegmentsAndPoints) {
  const auto half = alg::xiaolin_wu_line(0, 0, 2, 1, alg::WuEndpointPolicy::ClippedColumns);
  EXPECT_DOUBLE_EQ(half[0].brightness + half[1].brightness, 0.5);
  EXPECT_DOUBLE_EQ(half[4].brightness + half[5].brightness, 0.5);
  const auto short_line =
      alg::xiaolin_wu_line(0.1, 0.2, 0.2, 0.25, alg::WuEndpointPolicy::ClippedColumns);
  ASSERT_EQ(short_line.size(), 2);
  EXPECT_NEAR(wu_test_mass(short_line), 0.1, 1e-14);
  const auto tie = alg::xiaolin_wu_line(-0.5, -1, 0.5, 0, alg::WuEndpointPolicy::ClippedColumns);
  EXPECT_DOUBLE_EQ(wu_test_mass(tie), 1);
  for (const auto policy :
       {alg::WuEndpointPolicy::FullColumns, alg::WuEndpointPolicy::ClippedColumns}) {
    const auto p = alg::xiaolin_wu_line(-1.2, 2.25, -1.2, 2.25, policy);
    EXPECT_DOUBLE_EQ(wu_test_mass(p), 1);
    ASSERT_EQ(p.size(), 2);
    EXPECT_EQ(p[0].x, -1);
    EXPECT_EQ(p[0].y, 2);
    EXPECT_DOUBLE_EQ(p[0].brightness, 0.75);
  }
}
TEST(WuLine, RejectsInvalidCoordinatesBeforeCasting) {
  for (double invalid :
       {std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(),
        double(std::numeric_limits<int>::max()),
        double(std::numeric_limits<int>::min())})
    EXPECT_THROW((void)alg::xiaolin_wu_line(0, 0, invalid, 1), std::invalid_argument);
  EXPECT_THROW(
      (void)alg::xiaolin_wu_line(0, 0, 1, 1, static_cast<alg::WuEndpointPolicy>(42)),
      std::invalid_argument);
  const double safe = double(std::numeric_limits<int>::max()) - 2;
  EXPECT_DOUBLE_EQ(wu_test_mass(alg::xiaolin_wu_line(safe, safe, safe, safe)), 1);
}
TEST(WuPolyline, ScalesRowsAndPreservesInput) {
  datamunge::dstruct::DataFrame data;
  data.add_column("x", std::vector<double>{-1, 0, 1});
  data.add_column("y", std::vector<double>{0, 2, 0});
  const auto pixels = alg::wu_polyline(data, "x", "y", 12, 8);
  EXPECT_NEAR(wu_test_mass(pixels), 10, 1e-12);
  EXPECT_EQ(data.double_at("x", 0), -1);
  for (const auto& p : pixels) {
    EXPECT_GE(p.x, 0);
    EXPECT_LT(p.x, 12);
    EXPECT_GE(p.y, 0);
    EXPECT_LT(p.y, 8);
  }
  const auto table = alg::wu_pixel_table(pixels);
  EXPECT_EQ(table.ncols(), 3);
  const auto empty = alg::wu_pixel_table({});
  EXPECT_EQ(empty.nrows(), 0);
  EXPECT_EQ(empty.ncols(), 3);
  const auto merged = alg::wu_pixel_table({{0, 0, 0.75}, {0, 0, 0.5}, {1, 0, 0}});
  EXPECT_EQ(merged.nrows(), 2);
  EXPECT_DOUBLE_EQ(merged.double_at("weight", 0), 1.25);
  EXPECT_DOUBLE_EQ(merged.double_at("weight", 1), 0);
}
TEST(WuPolyline, HandlesEmptySingletonConstantAndExtremeCoordinates) {
  datamunge::dstruct::DataFrame data;
  data.add_column("x", std::vector<double>{});
  data.add_column("y", std::vector<double>{});
  EXPECT_TRUE(alg::wu_polyline(data, "x", "y", 4, 4).empty());
  data = datamunge::dstruct::DataFrame{};
  data.add_column("x", std::vector<double>{0});
  data.add_column("y", std::vector<double>{2});
  EXPECT_DOUBLE_EQ(wu_test_mass(alg::wu_polyline(data, "x", "y", 4, 4)), 1);
  datamunge::dstruct::DataFrame extreme;
  const double                  max = std::numeric_limits<double>::max();
  extreme.add_column("x", std::vector<double>{-max, max});
  extreme.add_column("y", std::vector<double>{2, 2});
  const auto out = alg::wu_polyline(extreme, "x", "y", 12, 8);
  EXPECT_NEAR(wu_test_mass(out), 9, 1e-12);
}
TEST(WuPolyline, ChecksDimensionsSchemaAndBrightness) {
  datamunge::dstruct::DataFrame data;
  data.add_column("x", std::vector<double>{1});
  data.add_column("y", std::vector<double>{2});
  EXPECT_THROW((void)alg::wu_polyline(data, "x", "y", 3, 8), std::invalid_argument);
  EXPECT_THROW((void)alg::wu_polyline(data, "absent", "y", 8, 8), std::invalid_argument);
  data.set_null("x", 0);
  EXPECT_THROW((void)alg::wu_polyline(data, "x", "y", 8, 8), std::invalid_argument);
  data.add_column("text", std::vector<std::string>{"x"});
  EXPECT_THROW((void)alg::wu_polyline(data, "text", "y", 8, 8), std::invalid_argument);
  for (const double invalid : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW((void)alg::wu_pixel_table({{0, 0, invalid}}), std::invalid_argument);
  EXPECT_THROW((void)alg::wu_pixel_plot({}), std::invalid_argument);
}
