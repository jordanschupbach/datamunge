#include <gtest/gtest.h>

#include <datamunge/datamunge.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/plot/figure.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using datamunge::graphics::Figure;
using datamunge::graphics::RLayout;
using datamunge::graphics::RPlot;

namespace {

RPlot line_plot(const std::string& title) {
  RPlot plot = RPlot::create();
  plot.lines({1.0, 2.0, 3.0}, {3.0, 1.0, 2.0}).title(title);
  return plot;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool is_png(const std::vector<std::uint8_t>& bytes) {
  return bytes.size() > 8 && bytes[0] == 0x89 && bytes[1] == 'P' && bytes[2] == 'N' && bytes[3] == 'G';
}

}  // namespace

TEST(FigureTest, SinglePlotTitleAndSizeActOnThePlot) {
  Figure fig = line_plot("old");
  fig.title("new").size(640, 480);
  EXPECT_FALSE(fig.is_grid());
  EXPECT_EQ(fig.panel_count(), 1u);
  EXPECT_EQ(fig.title_text(), "new");
  EXPECT_EQ(fig.width(), 640u);
  EXPECT_EQ(fig.height(), 480u);
  EXPECT_EQ(fig.panel(0).width(), 640u);
  EXPECT_NE(fig.to_svg().find(">new</text>"), std::string::npos);
  EXPECT_THROW((void)fig.panel(1), std::out_of_range);
}

TEST(FigureTest, GridSizeRefitsPanelsToCells) {
  Figure fig = RLayout::create(1, 2).add(line_plot("a")).add(line_plot("b"));
  EXPECT_TRUE(fig.is_grid());
  fig.size(1000, 400);
  EXPECT_EQ(fig.panel(0).width(), 500u);
  EXPECT_EQ(fig.panel(1).height(), 400u);
  fig.title("Both");  // the heading band takes height from the cells
  EXPECT_EQ(fig.panel(0).height(), 400u - RLayout::title_band_px);
  EXPECT_NE(fig.to_svg().find(">Both</text>"), std::string::npos);
}

TEST(FigureTest, RasterOutputsAndForcedFormatSaves) {
  Figure fig = line_plot("raster");
  fig.size(200, 150);
  EXPECT_TRUE(is_png(fig.to_png()));
  const auto pdf = fig.to_pdf();
  ASSERT_GT(pdf.size(), 4u);
  EXPECT_EQ(std::string(pdf.begin(), pdf.begin() + 4), "%PDF");

  const auto dir = std::filesystem::temp_directory_path() / "datamunge_figure_tests";
  std::filesystem::create_directories(dir);
  // The save_* forms ignore the extension.
  fig.save_png((dir / "fig.dat").string());
  const auto png_text = read_file(dir / "fig.dat");
  EXPECT_TRUE(is_png({png_text.begin(), png_text.end()}));
  fig.save_svg((dir / "fig.png").string());
  EXPECT_EQ(read_file(dir / "fig.png").rfind("<?xml", 0), 0u);
  fig.save((dir / "fig.pdf").string());
  EXPECT_EQ(read_file(dir / "fig.pdf").rfind("%PDF", 0), 0u);
  EXPECT_THROW(fig.save((dir / "fig.gif").string()), std::invalid_argument);
  std::filesystem::remove_all(dir);
}

TEST(FigureTest, GridPngMatchesLayoutPng) {
  auto layout = RLayout::create(2, 1).add(line_plot("a")).add(line_plot("b")).size(300, 400);
  const Figure fig = layout;
  EXPECT_EQ(fig.to_png(), layout.to_png());
}

TEST(FigureTest, PlotFreeFunctionAcceptsAnyPlotType) {
  const auto fig = datamunge::plot(line_plot("p")).size(320, 240);
  EXPECT_EQ(fig.width(), 320u);
  const auto grid = datamunge::plot(RLayout::create(1, 1).add(line_plot("q")));
  EXPECT_TRUE(grid.is_grid());
}

TEST(FigureTest, LdaWithTwoPredictorsGivesTwoPanels) {
  const datamunge::stats::LDA petals(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  auto fig = datamunge::plot(petals);
  ASSERT_TRUE(fig.is_grid());
  ASSERT_EQ(fig.panel_count(), 2u);
  EXPECT_EQ(fig.panel(0).x_label_text(), "LD1");
  EXPECT_EQ(fig.panel(1).x_label_text(), "Petal.Length");
  fig.title("Petals").size(900, 400);
  EXPECT_EQ(fig.panel(1).width(), 450u);
  EXPECT_NE(fig.to_svg().find(">Petals</text>"), std::string::npos);
}

TEST(FigureTest, LdaWithMorePredictorsGivesDiscriminantPlot) {
  const datamunge::stats::LDA model(datamunge::datasets::iris(),
                                    "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
  const auto fig = datamunge::plot(model);
  EXPECT_FALSE(fig.is_grid());
  EXPECT_EQ(fig.panel(0).y_label_text(), "LD2");
}

TEST(FigureTest, LdaFacadeOverload) {
  const std::unique_ptr<datamunge::DataFrame> iris(datamunge::DataFrame::iris());
  const datamunge::LDA petals(*iris, "Species ~ Petal.Length + Petal.Width");
  EXPECT_EQ(datamunge::plot(petals).panel_count(), 2u);
}

TEST(FigureTest, TinySizesRenderByScalingALargerCanvas) {
  const datamunge::stats::LDA petals(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  auto grid = datamunge::plot(petals).title("t").size(100, 100);
  EXPECT_NO_THROW((void)grid.to_svg());
  EXPECT_TRUE(is_png(grid.to_png()));
  EXPECT_EQ(grid.width(), 100u);
  EXPECT_GE(std::min(grid.panel(0).width(), grid.panel(0).height()), RLayout::min_panel_px);

  Figure single = line_plot("small");
  single.size(120, 90).title("still small");
  EXPECT_FALSE(single.is_grid());
  EXPECT_EQ(single.panel_count(), 1u);
  EXPECT_EQ(single.title_text(), "still small");
  EXPECT_EQ(single.width(), 120u);
  EXPECT_NE(single.to_svg().find("width=\"120\""), std::string::npos);
  EXPECT_TRUE(is_png(single.to_png()));
}
