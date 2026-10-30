#include <gtest/gtest.h>

#include <datamunge/datamunge.hpp>
#include <datamunge/plot/plot.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace datamunge::graphics;

namespace {
std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

class SquareCallback final : public datamunge::Callback {
 public:
  double call(double x) override { return x * x; }
};
} // namespace

TEST(Plot, SavesSvgWithExpectedElements) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_test.svg";

  auto plot = RPlot::create();
  plot.points({0.0, 1.0, 2.0}, {1.0, 3.0, 2.0}, "pts", {37, 99, 235}, 5.0)
      .line({0.0, 2.0}, {1.0, 2.0}, "fit", {220, 38, 38}, 2.5)
      .title("Test Scatter")
      .x_label("x axis")
      .y_label("y axis");

  plot.save_svg(path.string());
  const std::string svg = read_file(path);

  EXPECT_NE(svg.find("<svg"), std::string::npos);
  EXPECT_NE(svg.find("Test Scatter"), std::string::npos);
  EXPECT_NE(svg.find("<circle"), std::string::npos);
  EXPECT_NE(svg.find("<polyline"), std::string::npos);
  EXPECT_NE(svg.find("pts"), std::string::npos);
  EXPECT_NE(svg.find("fit"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(Plot, RejectsUnsupportedExtension) {
  auto plot = RPlot::create();
  plot.line({0.0, 1.0}, {0.0, 1.0});
  EXPECT_THROW(plot.save("bad-format.xyz"), std::invalid_argument);
}

TEST(Plot, SavesPngWithExpectedSignature) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_test.png";

  auto plot = RPlot::create();
  plot.line({0.0, 1.0, 2.0}, {1.0, 3.0, 2.0}, "line", {220, 38, 38}, 3.0).title("PNG Test");
  plot.save(path.string());

  std::ifstream in(path, std::ios::binary);
  ASSERT_TRUE(in.good());
  char signature[8] = {};
  in.read(signature, sizeof(signature));
  ASSERT_EQ(in.gcount(), static_cast<std::streamsize>(sizeof(signature)));
  EXPECT_EQ(static_cast<unsigned char>(signature[0]), 0x89U);
  EXPECT_EQ(signature[1], 'P');
  EXPECT_EQ(signature[2], 'N');
  EXPECT_EQ(signature[3], 'G');

  std::filesystem::remove(path);
}

TEST(Plot, SavesPdfWithExpectedHeader) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_test.pdf";

  auto plot = RPlot::create();
  plot.points({0.0, 1.0}, {1.0, 0.0}, "pts", {37, 99, 235}, 5.0).title("PDF Test");
  plot.save(path.string());

  std::ifstream in(path, std::ios::binary);
  ASSERT_TRUE(in.good());
  std::string header(8, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header.size()));
  ASSERT_EQ(in.gcount(), static_cast<std::streamsize>(header.size()));
  EXPECT_EQ(header.substr(0, 5), "%PDF-");

  std::filesystem::remove(path);
}

TEST(Plot, BarSeriesSaveUsesRects) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_bar_test.svg";

  auto plot = RPlot::create();
  plot.bars({1.0, 2.0, 3.0}, {2.0, 4.0, 3.0}, "bars", {22, 163, 74}, 0.6);
  plot.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("<rect"), std::string::npos);

  std::filesystem::remove(path);
}

// --- RPlot factories -------------------------------------------------------

TEST(RPlotFactory, PlotTypePProducesOnlyScatterPoints) {
  auto p = RPlot::plot({0.0, 1.0, 2.0}, {0.0, 1.0, 4.0}, "p");
  ASSERT_EQ(p.series().size(), 1U);
  EXPECT_EQ(p.series()[0].kind, DataSeries::Kind::Scatter);
}

TEST(RPlotFactory, PlotTypeLProducesOnlyLine) {
  auto p = RPlot::plot({0.0, 1.0, 2.0}, {0.0, 1.0, 4.0}, "l");
  ASSERT_EQ(p.series().size(), 1U);
  EXPECT_EQ(p.series()[0].kind, DataSeries::Kind::Line);
}

TEST(RPlotFactory, PlotTypeBProducesBothLineAndPoints) {
  auto p = RPlot::plot({0.0, 1.0, 2.0}, {0.0, 1.0, 4.0}, "b");
  ASSERT_EQ(p.series().size(), 2U);
}

TEST(RPlotFactory, PlotRejectsUnknownType) {
  EXPECT_THROW(RPlot::plot({0.0}, {0.0}, "z"), std::invalid_argument);
}

TEST(RPlotFactory, HistBinCountsMatchManualCounting) {
  // Range is [min, max] = [0.5, 9.9], width 9.4, split into 2 equal-width bins of 4.7:
  // bin 0 = [0.5, 5.2): 0.5, 1.0, 2.0, 4.9, 5.0 -> 5 values.
  // bin 1 = [5.2, 9.9]: 5.5, 6.0, 7.0, 8.0, 9.9 -> 5 values.
  std::vector<double> data = {0.5, 1.0, 2.0, 4.9, 5.0, 5.5, 6.0, 7.0, 8.0, 9.9};
  auto                p = RPlot::hist(data, 2);
  ASSERT_EQ(p.series().size(), 1U);
  const auto& series = p.series()[0];
  EXPECT_EQ(series.kind, DataSeries::Kind::Bar);
  ASSERT_EQ(series.y.size(), 2U);
  EXPECT_DOUBLE_EQ(series.y[0], 5.0);
  EXPECT_DOUBLE_EQ(series.y[1], 5.0);
}

TEST(RPlotFactory, HistRejectsZeroBins) {
  EXPECT_THROW(RPlot::hist({1.0, 2.0}, 0), std::invalid_argument);
}

TEST(RPlotFactory, BarplotUsesSequentialPositionsAndCategoricalLabels) {
  auto p = RPlot::barplot({3.0, 1.0, 4.0}, {"a", "b", "c"});
  ASSERT_EQ(p.series().size(), 1U);
  const auto& series = p.series()[0];
  EXPECT_EQ(series.x, (std::vector<double>{0.0, 1.0, 2.0}));
  ASSERT_EQ(p.x_tick_label_list().size(), 3U);
  EXPECT_EQ(p.x_tick_label_list()[0], "a");
}

TEST(RPlotFactory, BoxplotComputesTypeSevenQuartiles) {
  // Sorted 1..9: type-7 quantiles give Q1=3, median=5, Q3=7.
  std::vector<double> data = {5, 1, 9, 3, 7, 2, 8, 4, 6};
  auto                p = RPlot::boxplot({data});
  ASSERT_EQ(p.series().size(), 1U);
  const auto& series = p.series()[0];
  EXPECT_EQ(series.kind, DataSeries::Kind::Box);
  ASSERT_GE(series.y.size(), 5U);
  EXPECT_DOUBLE_EQ(series.y[1], 3.0); // q1
  EXPECT_DOUBLE_EQ(series.y[2], 5.0); // median
  EXPECT_DOUBLE_EQ(series.y[3], 7.0); // q3
}

TEST(RPlotFactory, BoxplotFlagsExtremeValueAsOutlier) {
  std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 100};
  auto                p = RPlot::boxplot({data});
  const auto&         series = p.series()[0];
  // whisker_hi (index 4) must exclude the extreme outlier; the outlier appears past index 4.
  EXPECT_LT(series.y[4], 100.0);
  const bool has_outlier_beyond_whisker =
      std::any_of(series.y.begin() + 5, series.y.end(), [](double v) { return v == 100.0; });
  EXPECT_TRUE(has_outlier_beyond_whisker);
}

TEST(RPlotFactory, BoxplotRejectsEmptyGroup) {
  EXPECT_THROW(RPlot::boxplot({{}}), std::invalid_argument);
}

TEST(RPlotFactory, PieWedgeAnglesSumToFullCircle) {
  auto        p = RPlot::pie({1.0, 1.0, 2.0}, {"a", "b", "c"});
  std::size_t polygon_count = 0;
  for (const auto& series : p.series()) {
    if (series.kind == DataSeries::Kind::Polygon) {
      ++polygon_count;
    }
  }
  EXPECT_EQ(polygon_count, 3U);
  EXPECT_TRUE(p.axes_hidden());
}

TEST(RPlotFactory, PieRejectsNegativeValue) {
  EXPECT_THROW(RPlot::pie({1.0, -1.0}), std::invalid_argument);
}

TEST(RPlotFactory, PieRejectsAllZero) {
  EXPECT_THROW(RPlot::pie({0.0, 0.0}), std::invalid_argument);
}

TEST(RPlotFactory, CurveSamplesCallbackAcrossRange) {
  SquareCallback f;
  auto           p = RPlot::curve(f, 0.0, 2.0, 3);
  ASSERT_EQ(p.series().size(), 1U);
  const auto& series = p.series()[0];
  ASSERT_EQ(series.x.size(), 3U);
  EXPECT_DOUBLE_EQ(series.x[0], 0.0);
  EXPECT_DOUBLE_EQ(series.x[1], 1.0);
  EXPECT_DOUBLE_EQ(series.x[2], 2.0);
  EXPECT_DOUBLE_EQ(series.y[0], 0.0);
  EXPECT_DOUBLE_EQ(series.y[1], 1.0);
  EXPECT_DOUBLE_EQ(series.y[2], 4.0);
}

TEST(RPlotFactory, CurveRejectsBackwardsRange) {
  SquareCallback f;
  EXPECT_THROW(RPlot::curve(f, 2.0, 0.0), std::invalid_argument);
}

TEST(RPlotFactory, QqnormSortsDataOntoYAxis) {
  auto        p = RPlot::qqnorm({3.0, 1.0, 2.0});
  const auto& series = p.series()[0];
  EXPECT_EQ(series.y, (std::vector<double>{1.0, 2.0, 3.0}));
}

// --- RPlot chainable verbs ---------------------------------------------------

TEST(RPlotVerbs, AblineAddsSlopedReferenceLine) {
  auto p = RPlot::create();
  p.points({0.0, 1.0}, {0.0, 1.0});
  p.abline(0.0, 1.0);
  ASSERT_EQ(p.reference_lines().size(), 1U);
  EXPECT_FALSE(p.reference_lines()[0].vertical);
  EXPECT_DOUBLE_EQ(p.reference_lines()[0].slope, 1.0);
}

TEST(RPlotVerbs, AblineVAddsVerticalReferenceLine) {
  auto p = RPlot::create();
  p.points({0.0, 1.0}, {0.0, 1.0});
  p.abline_v(0.5);
  ASSERT_EQ(p.reference_lines().size(), 1U);
  EXPECT_TRUE(p.reference_lines()[0].vertical);
  EXPECT_DOUBLE_EQ(p.reference_lines()[0].value, 0.5);
}

TEST(RPlotVerbs, QqlineFitsThroughFirstAndThirdQuartiles) {
  auto                 p = RPlot::qqnorm({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  std::vector<double>  data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  p.qqline(data);
  ASSERT_EQ(p.reference_lines().size(), 1U);
  EXPECT_FALSE(p.reference_lines()[0].vertical);
}

TEST(RPlotVerbs, LegendOverridesAutoGeneratedEntries) {
  auto p = RPlot::create();
  p.points({0.0}, {0.0}, "auto-label");
  p.legend({"manual-a", "manual-b"}, {RGB{1, 2, 3}, RGB{4, 5, 6}});
  ASSERT_EQ(p.legend_entries().size(), 2U);
  EXPECT_EQ(p.legend_entries()[0].label, "manual-a");
}

TEST(RPlotVerbs, LegendRejectsMismatchedSizes) {
  auto p = RPlot::create();
  EXPECT_THROW(p.legend({"a", "b"}, {RGB{}}), std::invalid_argument);
}

TEST(RPlotVerbs, TextAddsSingleTextSeries) {
  auto p = RPlot::create();
  p.text(1.0, 2.0, "hello");
  ASSERT_EQ(p.series().size(), 1U);
  EXPECT_EQ(p.series()[0].kind, DataSeries::Kind::Text);
  EXPECT_EQ(p.series()[0].label, "hello");
}

TEST(RPlotVerbs, PolygonStoresVerticesAndFillFlag) {
  auto p = RPlot::create();
  p.polygon({0.0, 1.0, 0.5}, {0.0, 0.0, 1.0}, RGB{1, 2, 3}, true);
  ASSERT_EQ(p.series().size(), 1U);
  EXPECT_EQ(p.series()[0].kind, DataSeries::Kind::Polygon);
  EXPECT_TRUE(p.series()[0].filled);
}

TEST(RPlotVerbs, SegmentsInterleavesEndpointPairs) {
  auto p = RPlot::create();
  p.segments({0.0, 1.0}, {0.0, 1.0}, {1.0, 2.0}, {1.0, 2.0});
  ASSERT_EQ(p.series().size(), 1U);
  const auto& series = p.series()[0];
  EXPECT_EQ(series.kind, DataSeries::Kind::Segment);
  EXPECT_EQ(series.x, (std::vector<double>{0.0, 1.0, 1.0, 2.0}));
}

TEST(RPlotVerbs, SegmentsRejectsMismatchedSizes) {
  auto p = RPlot::create();
  EXPECT_THROW(p.segments({0.0}, {0.0}, {1.0, 2.0}, {1.0}), std::invalid_argument);
}

// --- SVG rendering of new series kinds --------------------------------------

TEST(PlotSvg, BoxplotRendersRectAndMedianLine) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_box_test.svg";

  auto p = RPlot::boxplot({{1, 2, 3, 4, 5}, {2, 4, 6, 8, 10}}, {"g1", "g2"});
  p.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("<rect"), std::string::npos);
  EXPECT_NE(svg.find("g1"), std::string::npos);
  EXPECT_NE(svg.find("g2"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(PlotSvg, PieRendersPolygonsAndHidesAxisLabels) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_pie_test.svg";

  auto p = RPlot::pie({1.0, 2.0, 3.0}, {"x", "y", "z"});
  p.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("<polygon"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(PlotSvg, TextRendersTextElementWithLabel) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_text_test.svg";

  auto p = RPlot::create();
  p.points({0.0, 1.0}, {0.0, 1.0});
  p.text(0.5, 0.5, "annotation");
  p.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("annotation"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(PlotSvg, AblineRendersLineElement) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_abline_test.svg";

  auto p = RPlot::create();
  p.points({0.0, 1.0, 2.0}, {0.0, 1.0, 4.0});
  p.abline(0.0, 1.0);
  p.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("<line"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(PlotSvg, BarplotRendersCategoricalTickLabels) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_barplot_test.svg";

  auto p = RPlot::barplot({1.0, 2.0, 3.0}, {"cat-a", "cat-b", "cat-c"});
  p.save_svg(path.string());
  const std::string svg = read_file(path);
  EXPECT_NE(svg.find("cat-a"), std::string::npos);
  EXPECT_NE(svg.find("cat-b"), std::string::npos);
  EXPECT_NE(svg.find("cat-c"), std::string::npos);

  std::filesystem::remove(path);
}

// --- RLayout -----------------------------------------------------------------

TEST(RLayout, CreateRejectsZeroRowsOrCols) {
  EXPECT_THROW(RLayout::create(0, 2), std::invalid_argument);
  EXPECT_THROW(RLayout::create(2, 0), std::invalid_argument);
}

TEST(RLayout, AddRejectsMoreThanRowsTimesCols) {
  auto layout = RLayout::create(1, 1);
  auto p1 = RPlot::create();
  p1.points({0.0}, {0.0});
  auto p2 = RPlot::create();
  p2.points({0.0}, {0.0});
  layout.add(p1);
  EXPECT_THROW(layout.add(p2), std::invalid_argument);
}

TEST(RLayout, SavesSvgWithOneNestedSvgPerPanel) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_layout_test.svg";

  auto layout = RLayout::create(1, 2);
  auto p1 = RPlot::create();
  p1.points({0.0, 1.0}, {0.0, 1.0}, "left");
  auto p2 = RPlot::create();
  p2.line({0.0, 1.0}, {1.0, 0.0}, "right");
  layout.add(p1).add(p2);
  layout.save_svg(path.string());

  const std::string svg = read_file(path);
  std::size_t       count = 0;
  std::size_t       pos = 0;
  while ((pos = svg.find("<svg", pos)) != std::string::npos) {
    ++count;
    pos += 4;
  }
  // one outer <svg> plus one nested <svg> per panel
  EXPECT_EQ(count, 3U);
  EXPECT_NE(svg.find("left"), std::string::npos);
  EXPECT_NE(svg.find("right"), std::string::npos);

  std::filesystem::remove(path);
}

TEST(PlotSvg, TextColourIsNotOverriddenByStylesheet) {
  // A stylesheet rule "text{fill:...}" beats the fill attribute in every renderer, so the
  // default colour must only target text without its own fill.
  auto p = RPlot::plot({0.0, 1.0}, {0.0, 1.0}, "p", "pts");
  p.text(0.5, 0.5, "white", RGB{255, 255, 255});
  const std::string svg = p.to_svg();
  EXPECT_NE(svg.find("fill=\"rgb(255,255,255)\""), std::string::npos);
  EXPECT_NE(svg.find("text:not([fill]){fill:"), std::string::npos);
  const auto rule = svg.find("text{font-family:");
  ASSERT_NE(rule, std::string::npos);
  EXPECT_EQ(svg.substr(rule, svg.find('}', rule) - rule).find("fill"), std::string::npos);
}

TEST(PlotSvg, ToSvgReturnsSameDocumentAsSaveSvg) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_plot_to_svg_test.svg";

  auto p = RPlot::plot({0.0, 1.0, 2.0}, {0.0, 1.0, 4.0}, "p", "pts");
  p.title("To Svg");
  const std::string svg = p.to_svg();
  EXPECT_EQ(svg.rfind("<?xml", 0), 0U);
  EXPECT_NE(svg.find("<svg"), std::string::npos);
  EXPECT_NE(svg.find("To Svg"), std::string::npos);
  EXPECT_NE(svg.find("</svg>"), std::string::npos);

  p.save_svg(path.string());
  EXPECT_EQ(svg, read_file(path));

  std::filesystem::remove(path);
}

TEST(RLayout, ToSvgReturnsSameDocumentAsSaveSvg) {
  const auto out_dir = std::filesystem::temp_directory_path();
  const auto path = out_dir / "datamunge_layout_to_svg_test.svg";

  auto layout = RLayout::create(1, 2);
  auto p1 = RPlot::create();
  p1.points({0.0, 1.0}, {0.0, 1.0}, "left");
  auto p2 = RPlot::create();
  p2.line({0.0, 1.0}, {1.0, 0.0}, "right");
  layout.add(p1).add(p2);

  const std::string svg = layout.to_svg();
  EXPECT_NE(svg.find("left"), std::string::npos);
  EXPECT_NE(svg.find("right"), std::string::npos);

  layout.save_svg(path.string());
  EXPECT_EQ(svg, read_file(path));

  std::filesystem::remove(path);
}

// --- boxplot colours, raster boxes, ticks and grid ---------------------------

TEST(PlotBoxplot, PerGroupColorsAreRecycled) {
  const std::vector<RGB> colors = {{78, 121, 167}, {242, 142, 43}};
  auto p = RPlot::boxplot({{1, 2, 3}, {2, 3, 4}, {3, 4, 5}}, {"a", "b", "c"}, colors);
  ASSERT_EQ(p.series().size(), 3U);
  EXPECT_EQ(p.series()[0].color.r, 78);
  EXPECT_EQ(p.series()[1].color.r, 242);
  EXPECT_EQ(p.series()[2].color.r, 78);
  EXPECT_THROW(RPlot::boxplot({{1, 2}}, {"a"}, std::vector<RGB>{}), std::invalid_argument);
}

TEST(PlotBoxplot, RasterOutputDrawsBoxes) {
  // If the raster renderer skipped Box series, recolouring the boxes would not change the PNG.
  const auto dir = std::filesystem::temp_directory_path();
  const auto red = dir / "datamunge_box_red.png";
  const auto blue = dir / "datamunge_box_blue.png";
  RPlot::boxplot({{1, 2, 3, 4, 5, 20}}, {"g"}, RGB{220, 38, 38}).save(red.string());
  RPlot::boxplot({{1, 2, 3, 4, 5, 20}}, {"g"}, RGB{37, 99, 235}).save(blue.string());
  std::ifstream a(red, std::ios::binary);
  std::ifstream b(blue, std::ios::binary);
  const std::string red_bytes((std::istreambuf_iterator<char>(a)), std::istreambuf_iterator<char>());
  const std::string blue_bytes((std::istreambuf_iterator<char>(b)), std::istreambuf_iterator<char>());
  EXPECT_FALSE(red_bytes.empty());
  EXPECT_NE(red_bytes, blue_bytes);
  std::filesystem::remove(red);
  std::filesystem::remove(blue);
}

TEST(PlotTicks, PrettyTicksAreTheDefault) {
  auto p = RPlot::create();
  EXPECT_TRUE(p.pretty_ticks_enabled());
  p.line({0.0, 100.0}, {0.0, 1.0});
  const std::string svg = p.to_svg();
  EXPECT_NE(svg.find(">40</text>"), std::string::npos);
  EXPECT_NE(svg.find(">0.4</text>"), std::string::npos);
}

TEST(PlotTicks, GridLinesSitOnTheTicks) {
  auto p = RPlot::create();
  p.line({0.0, 100.0}, {0.0, 1.0});
  const std::string svg = p.to_svg();
  // Every x tick label's x coordinate also starts a vertical grid line.
  const std::string marker = "text-anchor=\"middle\">";
  std::size_t checked = 0;
  for (const std::string label : {"0", "20", "40", "60", "80", "100"}) {
    const auto at = svg.find(marker + label + "</text>");
    ASSERT_NE(at, std::string::npos) << label;
    const auto x_at = svg.rfind("<text x=\"", at);
    const auto x_end = svg.find('"', x_at + 9);
    const std::string x = svg.substr(x_at + 9, x_end - x_at - 9);
    EXPECT_NE(svg.find("<line x1=\"" + x + "\""), std::string::npos) << "no grid line at x=" << x;
    ++checked;
  }
  EXPECT_EQ(checked, 6U);
}

TEST(PlotColors, ViridisEndpointsAndInterpolation) {
  const auto v = viridis_colors(10);
  ASSERT_EQ(v.size(), 10u);
  EXPECT_EQ(v.front().r, 68);
  EXPECT_EQ(v.front().g, 1);
  EXPECT_EQ(v.front().b, 84);
  EXPECT_EQ(v.back().r, 253);
  EXPECT_EQ(v.back().g, 231);
  EXPECT_EQ(v.back().b, 37);
  const auto mid = viridis_colors(3)[1];  // halfway between anchors 5 and 6
  EXPECT_EQ(mid.r, 35);
  EXPECT_EQ(mid.g, 144);
  EXPECT_EQ(mid.b, 140);
  EXPECT_EQ(viridis_colors(1).size(), 1u);
  EXPECT_THROW(viridis_colors(0), std::invalid_argument);
}

TEST(PlotColors, ValueColorsScalesMissingAndConstantValues) {
  const std::vector<RGB> palette{RGB{0, 0, 0}, RGB{100, 100, 100}, RGB{200, 200, 200}};
  const auto c = value_colors({1.0, 2.0, 3.0, std::nan("")}, palette);
  EXPECT_EQ(c[0].r, 0);
  EXPECT_EQ(c[1].r, 100);
  EXPECT_EQ(c[2].r, 200);
  EXPECT_EQ(c[3].r, 190);
  const auto flat = value_colors({4.0, 4.0}, palette);
  EXPECT_EQ(flat[0].r, 100);
  const auto clamped = value_colors({-5.0, 0.5, 9.0}, palette, 0.0, 1.0);
  EXPECT_EQ(clamped[0].r, 0);
  EXPECT_EQ(clamped[1].r, 100);
  EXPECT_EQ(clamped[2].r, 200);
}
