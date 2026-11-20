#include <gtest/gtest.h>

#include <datamunge/datamunge.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/plot/figure.hpp>
#include <datamunge/plot/histogram.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/plot/splom.hpp>
#include <datamunge/plot/units.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace units = datamunge::units;
namespace graphics = datamunge::graphics;

TEST(PlotUnits, ConvertAtNinetySixPixelsPerInch) {
  EXPECT_DOUBLE_EQ(units::Length(units::inch{1}).pixels(), 96.0);
  EXPECT_DOUBLE_EQ(units::Length(units::pt{72}).pixels(), 96.0);
  EXPECT_DOUBLE_EQ(units::Length(units::cm{2.54}).pixels(), 96.0);
  EXPECT_DOUBLE_EQ(units::Length(units::mm{25.4}).pixels(), 96.0);
  EXPECT_DOUBLE_EQ(units::Length(units::px{12.5}).pixels(), 12.5);
  EXPECT_EQ(units::Length(units::cm{10}).to_pixels(), 378U);  // 377.95 rounds up
  EXPECT_THROW((void)units::Length(units::mm{0.1}).to_pixels(), std::invalid_argument);
  EXPECT_THROW((void)units::Length(units::inch{-1}).to_pixels(), std::invalid_argument);
}

TEST(PlotUnits, WidthAndHeightSetEachDimensionAndKeepTheOther) {
  auto splom = graphics::Splom::create(datamunge::datasets::iris()).width(units::inch{8});
  EXPECT_EQ(splom.width(), 768U);
  EXPECT_EQ(splom.height(), 960U);
  splom.height(units::inch{4});
  EXPECT_EQ(splom.width(), 768U);
  EXPECT_EQ(splom.height(), 384U);

  graphics::Plot base;
  base.width(units::inch{5}).height(units::inch{4});
  EXPECT_EQ(base.width(), 480U);
  EXPECT_EQ(base.height(), 384U);

  auto hist = datamunge::Histogram({1.0, 2.0, 2.0, 3.0}).width(units::px{500}).height(units::px{300});
  EXPECT_EQ(hist.width(), 500U);
  EXPECT_EQ(hist.height(), 300U);

  auto layout = graphics::RLayout::create(1, 2).width(units::cm{2.54}).height(units::pt{72});
  EXPECT_EQ(layout.width(), 96U);
  EXPECT_EQ(layout.height(), 96U);

  auto fig = graphics::Figure(graphics::Splom::create(datamunge::datasets::iris())).width(units::inch{6}).height(units::inch{4});
  EXPECT_EQ(fig.width(), 576U);
  EXPECT_EQ(fig.height(), 384U);
}

namespace {

/// Width and height from a PNG's IHDR chunk.
std::pair<std::uint32_t, std::uint32_t> png_size(const std::vector<std::uint8_t>& png) {
  const auto be32 = [&](std::size_t at) {
    return (std::uint32_t{png.at(at)} << 24U) | (std::uint32_t{png.at(at + 1)} << 16U) |
           (std::uint32_t{png.at(at + 2)} << 8U) | std::uint32_t{png.at(at + 3)};
  };
  return {be32(16), be32(20)};
}

bool has_chunk(const std::vector<std::uint8_t>& bytes, const std::string& name) {
  return std::search(bytes.begin(), bytes.end(), name.begin(), name.end()) != bytes.end();
}

}  // namespace

TEST(PlotUnits, DpiScalesRasterPixelsButNotTheLayout) {
  auto splom = graphics::Splom::create(datamunge::datasets::iris()).width(units::inch{2}).height(units::inch{2});
  EXPECT_DOUBLE_EQ(splom.dpi(), 96.0);
  const auto plain = splom.layout().to_png();
  EXPECT_EQ(png_size(plain), (std::pair<std::uint32_t, std::uint32_t>{192, 192}));
  EXPECT_FALSE(has_chunk(plain, "pHYs"));

  splom.dpi(300);
  EXPECT_EQ(splom.width(), 192U);  // layout size is unchanged
  const auto sharp = splom.layout().to_png();
  EXPECT_EQ(png_size(sharp), (std::pair<std::uint32_t, std::uint32_t>{600, 600}));
  EXPECT_TRUE(has_chunk(sharp, "pHYs"));

  graphics::Plot base;
  base.size(100, 50).dpi(192);
  EXPECT_EQ(png_size(base.to_png()), (std::pair<std::uint32_t, std::uint32_t>{200, 100}));
  // The PDF page keeps the logical size; only the embedded image gains pixels.
  const auto pdf = base.to_pdf();
  const std::string text(pdf.begin(), pdf.end());
  EXPECT_NE(text.find("/MediaBox [0 0 100 50]"), std::string::npos);
  EXPECT_NE(text.find("/Width 200 /Height 100"), std::string::npos);

  auto hist = datamunge::Histogram({1.0, 2.0, 2.0, 3.0}).size(100, 80).dpi(48);
  EXPECT_DOUBLE_EQ(hist.dpi(), 48.0);
  EXPECT_EQ(png_size(hist.to_png()), (std::pair<std::uint32_t, std::uint32_t>{50, 40}));

  EXPECT_THROW(base.dpi(0), std::invalid_argument);
  EXPECT_THROW(base.dpi(-72), std::invalid_argument);
  EXPECT_THROW(base.dpi(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

TEST(PlotUnits, FigureForwardsDpiAndKeepsItThroughResizing) {
  graphics::Plot base;
  auto fig = graphics::Figure(base).dpi(192);
  EXPECT_DOUBLE_EQ(fig.dpi(), 192.0);
  fig.size(100, 100);  // small: becomes a 1 x 1 grid internally
  EXPECT_DOUBLE_EQ(fig.dpi(), 192.0);
  EXPECT_EQ(png_size(fig.to_png()), (std::pair<std::uint32_t, std::uint32_t>{200, 200}));
  fig.title("t");
  EXPECT_DOUBLE_EQ(fig.dpi(), 192.0);

  auto from_splom = graphics::Figure(graphics::Splom::create(datamunge::datasets::iris()).size(100, 100).dpi(192));
  EXPECT_DOUBLE_EQ(from_splom.dpi(), 192.0);
}
