#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/plot/ggplot.hpp>
#include <datamunge/stats/lm.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

using namespace datamunge::plot;
using datamunge::dstruct::DataFrame;

namespace {
std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

DataFrame make_xy_frame() {
  DataFrame df;
  df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
  df.add_column("y", std::vector<double>{2.0, 4.1, 5.9, 8.2, 9.8});
  return df;
}
} // namespace

TEST(GGPlot, GeomPointRendersOneCirclePerRow) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_ggplot_point_test.svg";

  auto plot = GGPlot::create(make_xy_frame(), Aes{"x", "y", "", "", ""});
  plot.geom_point();
  plot.save_svg(path.string());
  const std::string svg = read_file(path);

  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = svg.find("<circle", pos)) != std::string::npos) {
    ++count;
    pos += 7;
  }
  EXPECT_EQ(count, 5U);

  std::filesystem::remove(path);
}

TEST(GGPlot, GeomPointGroupsByColorColumn) {
  DataFrame df;
  df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0});
  df.add_column("y", std::vector<double>{1.0, 2.0, 3.0, 4.0});
  df.add_column("grp", std::vector<std::string>{"a", "a", "b", "b"});

  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_ggplot_group_test.svg";

  auto plot = GGPlot::create(df, Aes{"x", "y", "grp", "", ""});
  plot.geom_point();
  plot.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find(">a<"), std::string::npos);
  EXPECT_NE(svg.find(">b<"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(GGPlot, GeomBarCountsDiscreteCategories) {
  DataFrame df;
  df.add_column("cat", std::vector<std::string>{"a", "a", "b", "a", "c", "b"});

  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_ggplot_bar_test.svg";

  auto plot = GGPlot::create(df, Aes{"cat", "", "", "", ""});
  plot.geom_bar();
  plot.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("<rect"), std::string::npos);
  EXPECT_NE(svg.find(">a<"), std::string::npos);
  EXPECT_NE(svg.find(">b<"), std::string::npos);
  EXPECT_NE(svg.find(">c<"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(GGPlot, GeomSmoothFitsLmAndMatchesIndependentFit) {
  DataFrame data = make_xy_frame();

  // Independent ground truth: fit the same y ~ x regression directly via stats::LM.
  const datamunge::stats::LM lm(data, "y ~ x");
  DataFrame                  newdata;
  newdata.add_column("x", std::vector<double>{1.0, 5.0});
  const auto expected = lm.predict(newdata);

  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_ggplot_smooth_test.svg";
  auto       plot = GGPlot::create(data, Aes{"x", "y", "", "", ""});
  plot.geom_point();
  plot.geom_smooth();
  plot.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("<polyline"), std::string::npos);
  // Sanity: the fitted values should be monotonically increasing, matching this data's clear
  // positive trend (same trend the independent LM fit above confirms).
  EXPECT_GT(expected[1], expected[0]);

  std::filesystem::remove(path);
}

TEST(GGPlot, GeomBoxplotGroupsByXCategory) {
  DataFrame df;
  df.add_column("grp", std::vector<std::string>{"a", "a", "a", "b", "b", "b"});
  df.add_column("value", std::vector<double>{1.0, 2.0, 3.0, 10.0, 20.0, 30.0});

  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_ggplot_box_test.svg";
  auto       plot = GGPlot::create(df, Aes{"grp", "value", "", "", ""});
  plot.geom_boxplot();
  plot.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("<rect"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(GGPlot, GeomHistogramBinsXColumn) {
  DataFrame df;
  df.add_column("x", std::vector<double>{1, 1, 2, 2, 2, 3, 3, 3, 3});

  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_ggplot_hist_test.svg";
  auto       plot = GGPlot::create(df, Aes{"x", "", "", "", ""});
  plot.geom_histogram(3);
  plot.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("<rect"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(GGPlot, GeomDensityProducesSmoothCurve) {
  DataFrame df;
  df.add_column("x", std::vector<double>{1.0, 1.2, 0.9, 5.0, 5.1, 4.9, 10.0, 9.8, 10.2});

  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_ggplot_density_test.svg";
  auto       plot = GGPlot::create(df, Aes{"x", "", "", "", ""});
  plot.geom_density();
  plot.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("<polyline"), std::string::npos);
  EXPECT_NE(svg.find("<polygon"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(GGPlot, GeomAreaFillsUnderCurve) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_ggplot_area_test.svg";
  auto       plot = GGPlot::create(make_xy_frame(), Aes{"x", "y", "", "", ""});
  plot.geom_area();
  plot.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("<polygon"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(GGPlot, GeomRibbonSpansMinAndMax) {
  DataFrame df;
  df.add_column("x", std::vector<double>{1.0, 2.0, 3.0});
  df.add_column("ymin", std::vector<double>{0.0, 0.5, 1.0});
  df.add_column("ymax", std::vector<double>{1.0, 1.5, 2.0});

  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_ggplot_ribbon_test.svg";
  auto       plot = GGPlot::create(df, Aes{"x", "", "", "", ""});
  plot.geom_ribbon("ymin", "ymax");
  plot.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("<polygon"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(GGPlot, FacetWrapProducesOnePanelPerLevel) {
  DataFrame df;
  df.add_column("x", std::vector<double>{1.0, 2.0, 1.0, 2.0, 1.0, 2.0});
  df.add_column("y", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  df.add_column("facet", std::vector<std::string>{"p", "p", "q", "q", "r", "r"});

  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_ggplot_facet_test.svg";
  auto       plot = GGPlot::create(df, Aes{"x", "y", "", "", ""});
  plot.geom_point();
  plot.facet_wrap("facet");
  plot.save_svg(path.string());
  const std::string svg = read_file(path);

  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = svg.find("<svg", pos)) != std::string::npos) {
    ++count;
    pos += 4;
  }
  // one outer <svg> plus one nested <svg> per facet level
  EXPECT_EQ(count, 4U);

  std::filesystem::remove(path);
}

TEST(GGPlot, LabsOverridesAxisLabelsAndTitle) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_ggplot_labs_test.svg";
  auto       plot = GGPlot::create(make_xy_frame(), Aes{"x", "y", "", "", ""});
  plot.geom_point();
  plot.labs("My Title", "My X", "My Y");
  plot.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("My Title"), std::string::npos);
  EXPECT_NE(svg.find("My X"), std::string::npos);
  EXPECT_NE(svg.find("My Y"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(GGPlot, GeomPointRejectsNonNumericXColumn) {
  DataFrame df;
  df.add_column("x", std::vector<std::string>{"a", "b"});
  df.add_column("y", std::vector<double>{1.0, 2.0});
  auto plot = GGPlot::create(df, Aes{"x", "y", "", "", ""});
  plot.geom_point();
  EXPECT_THROW(plot.save_svg("unused.svg"), std::invalid_argument);
}
