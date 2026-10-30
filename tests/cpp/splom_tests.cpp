#include <gtest/gtest.h>

#include <datamunge/datamunge.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/plot/splom.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <set>
#include <string>
#include <cmath>
#include <vector>

using datamunge::graphics::RLayout;
using datamunge::graphics::RPlot;
using datamunge::graphics::Splom;
using Kind = datamunge::graphics::DataSeries::Kind;

namespace {

std::size_t count_occurrences(const std::string& text, const std::string& needle) {
  std::size_t count = 0;
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

datamunge::dstruct::DataFrame small_frame() {
  datamunge::dstruct::DataFrame df;
  df.add_column("a", std::vector<double>{1.0, 2.0, 3.0, 4.0});
  df.add_column("b", std::vector<double>{4.0, 3.0, 2.0, 1.0});
  df.add_column("label", std::vector<std::string>{"x", "y", "x", "y"});
  return df;
}

} // namespace

namespace {

std::size_t count_kind(const RPlot& p, Kind kind, bool filled_only = false) {
  std::size_t n = 0;
  for (const auto& s : p.series()) {
    if (s.kind == kind && (!filled_only || s.filled)) ++n;
  }
  return n;
}

/// @brief First series of @p kind in @p p; fails the test when there is none.
const datamunge::graphics::DataSeries& first_kind(const RPlot& p, Kind kind) {
  for (const auto& s : p.series()) {
    if (s.kind == kind) return s;
  }
  ADD_FAILURE() << "no series of the requested kind";
  return p.series().front();
}

} // namespace

TEST(Splom, SelectsNumericAndCategoricalColumnsByDefault) {
  const auto matrix = Splom::create(datamunge::datasets::iris());
  const std::vector<std::string> expected{"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"};
  EXPECT_EQ(matrix.variables(), expected);
  EXPECT_EQ(matrix.panel_count(), 5U);
  EXPECT_FALSE(matrix.is_categorical(0));
  EXPECT_TRUE(matrix.is_categorical(4));
  EXPECT_EQ(matrix.levels(4), (std::vector<std::string>{"setosa", "versicolor", "virginica"}));
  EXPECT_TRUE(matrix.levels(0).empty());
  EXPECT_TRUE(matrix.groups().empty());
  EXPECT_THROW(static_cast<void>(matrix.is_categorical(5)), std::out_of_range);
}

TEST(Splom, SkipsHighCardinalityStringColumnsByDefault) {
  datamunge::dstruct::DataFrame df;
  std::vector<double>      x;
  std::vector<std::string> ids;
  for (int i = 0; i < 20; ++i) {
    x.push_back(i);
    ids.push_back("id" + std::to_string(i));
  }
  df.add_column("x", x);
  df.add_column("y", x);
  df.add_column("id", ids);
  EXPECT_EQ(Splom::create(df).variables(), (std::vector<std::string>{"x", "y"}));
  EXPECT_EQ(Splom::create(df, {"x", "id"}).variables(), (std::vector<std::string>{"x", "id"}));
}

TEST(Splom, RejectsBadColumnSelections) {
  const auto df = small_frame();
  EXPECT_THROW(Splom::create(df, {"a", "missing"}), std::invalid_argument);
  EXPECT_NO_THROW(Splom::create(df, {"a", "label"}));
  EXPECT_THROW(Splom::create(df, {"a"}), std::invalid_argument);
  datamunge::dstruct::DataFrame strings_only;
  strings_only.add_column("s", std::vector<std::string>{"p", "q"});
  EXPECT_THROW(Splom::create(strings_only), std::invalid_argument);
}

TEST(Splom, SvgHasOneNestedSvgPerPanelAndTitle) {
  auto matrix = Splom::create(datamunge::datasets::iris());
  matrix.title("Iris pairs");
  const std::string svg = matrix.to_svg();
  EXPECT_EQ(svg.rfind("<?xml", 0), 0U);
  EXPECT_EQ(count_occurrences(svg, "<svg"), 26U); // outer + 5x5 panels
  EXPECT_NE(svg.find("Iris pairs"), std::string::npos);
  for (const auto& name : matrix.variables()) EXPECT_NE(svg.find(name), std::string::npos);
}

TEST(Splom, PanelsFollowPairsConvention) {
  auto matrix = Splom::create(small_frame());
  matrix.diagonal(Splom::Diagonal::Names);
  // panel(row = 0, col = 1): x = variable 1 ("b"), y = variable 0 ("a")
  const RPlot p = matrix.panel(0, 1);
  ASSERT_EQ(p.series().size(), 1U);
  EXPECT_EQ(p.series()[0].x, (std::vector<double>{4.0, 3.0, 2.0, 1.0}));
  EXPECT_EQ(p.series()[0].y, (std::vector<double>{1.0, 2.0, 3.0, 4.0}));
  EXPECT_TRUE(p.has_custom_margins());
  EXPECT_TRUE(p.title_text().empty());

  const RPlot diag = matrix.panel(1, 1);
  ASSERT_EQ(diag.series().size(), 1U);
  EXPECT_EQ(diag.series()[0].kind, datamunge::graphics::DataSeries::Kind::Text);
  EXPECT_EQ(diag.series()[0].label, "b");
  EXPECT_TRUE(diag.axes_hidden());

  EXPECT_THROW(matrix.panel(3, 0), std::out_of_range);
}

TEST(Splom, DiagonalHistogramUsesBars) {
  auto matrix = Splom::create(small_frame());
  matrix.diagonal(Splom::Diagonal::Histogram).bins(2);
  const RPlot diag = matrix.panel(0, 0);
  ASSERT_EQ(diag.series().size(), 2U);
  EXPECT_EQ(diag.series()[0].kind, datamunge::graphics::DataSeries::Kind::Bar);
  EXPECT_EQ(diag.series()[0].x.size(), 2U);
  EXPECT_EQ(diag.series()[1].kind, datamunge::graphics::DataSeries::Kind::Text);
  EXPECT_EQ(diag.series()[1].label, "a");
  EXPECT_THROW(matrix.bins(0), std::invalid_argument);
}

TEST(Splom, GroupByDrawsOnePointLayerPerGroup) {
  auto matrix = Splom::create(small_frame());
  matrix.group_by("label", {{255, 0, 0}, {0, 0, 255}});
  EXPECT_EQ(matrix.groups(), (std::vector<std::string>{"x", "y"}));
  // Upper panel, x = b, y = a; two points per group is too few for a smooth.
  const RPlot p = matrix.panel(0, 1);
  ASSERT_EQ(count_kind(p, Kind::Scatter), 2U);
  EXPECT_EQ(p.series()[0].color.r, 255);
  EXPECT_EQ(p.series()[0].x, (std::vector<double>{4.0, 2.0})); // rows labelled "x"
  EXPECT_EQ(p.series()[1].color.b, 255);

  EXPECT_THROW(matrix.group_by("a"), std::invalid_argument);
  matrix.group_by("");
  EXPECT_TRUE(matrix.groups().empty());
  EXPECT_EQ(count_kind(matrix.panel(0, 1), Kind::Scatter), 1U);
}

TEST(Splom, DropsRowsWithMissingValuesPerPanel) {
  datamunge::dstruct::DataFrame df;
  df.add_column("a", std::vector<std::optional<double>>{1.0, std::nullopt, 3.0});
  df.add_column("b", std::vector<std::optional<double>>{1.0, 2.0, 3.0});
  df.add_column("c", std::vector<std::optional<double>>{1.0, 2.0, 3.0});
  const auto matrix = Splom::create(df);
  EXPECT_EQ(matrix.panel(0, 1).series()[0].x.size(), 2U); // a vs b drops the null row
  EXPECT_EQ(matrix.panel(1, 2).series()[0].x.size(), 3U); // b vs c keeps every row
}

TEST(Splom, SaveSvgMatchesToSvg) {
  const auto path = std::filesystem::temp_directory_path() / "datamunge_splom_test.svg";
  auto matrix = Splom::create(small_frame());
  matrix.size(400, 300).title("tiny");
  matrix.save_svg(path.string());
  std::ifstream in(path);
  const std::string file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_EQ(file, matrix.to_svg());
  EXPECT_NE(file.find("width=\"400\" height=\"300\""), std::string::npos);
  std::filesystem::remove(path);
}

TEST(SplomFacade, WrapsCoreMatrix) {
  datamunge::DataFrame df = *std::unique_ptr<datamunge::DataFrame>(datamunge::DataFrame::empty());
  df.add_numeric_column("a", {1.0, 2.0, 3.0});
  df.add_numeric_column("b", {3.0, 2.0, 1.0});
  df.add_string_column("g", {"u", "v", "u"});
  datamunge::Splom matrix(df);
  matrix.title("facade").group_by("g").diagonal_histogram();
  EXPECT_EQ(matrix.variables(), (std::vector<std::string>{"a", "b", "g"}));
  EXPECT_EQ(matrix.groups(), (std::vector<std::string>{"u", "v"}));
  EXPECT_EQ(count_occurrences(matrix.to_svg(), "<svg"), 10U);

  // The core-frame constructor is what `dm::Splom(dm::datasets::iris())` resolves to.
  datamunge::Splom from_core(datamunge::datasets::iris());
  EXPECT_EQ(from_core.variables().size(), 5U);
  from_core.point_colors({{255, 0, 0}, {0, 0, 255}});
  EXPECT_EQ(first_kind(from_core.panel(0, 1), Kind::Scatter).point_colors.size(), 150U);
}

// --- Panel types by variable kind ----------------------------------------------


TEST(Splom, CategoricalDiagonalIsStackedBar) {
  const auto matrix = Splom::create(datamunge::datasets::iris());
  const RPlot bar = matrix.panel(4, 4);
  // One filled rectangle per level stacked from 0 to 150, plus an outline.
  ASSERT_EQ(count_kind(bar, Kind::Polygon, true), 3U);
  double top = 0.0;
  for (const auto& s : bar.series()) {
    if (s.kind == Kind::Polygon && s.filled) top = std::max(top, *std::max_element(s.y.begin(), s.y.end()));
  }
  EXPECT_DOUBLE_EQ(top, 150.0);
  bool named = false;
  for (const auto& s : bar.series()) named = named || (s.kind == Kind::Text && s.label == "Species");
  EXPECT_TRUE(named);

  const RPlot density = matrix.panel(0, 0);
  EXPECT_EQ(count_kind(density, Kind::Polygon, true), 1U);
  EXPECT_EQ(count_kind(density, Kind::Line), 1U);
}

TEST(Splom, NumericVersusCategoricalIsViolin) {
  const auto matrix = Splom::create(datamunge::datasets::iris());
  // Species (col 4) on x, Sepal.Length (row 0) on y: vertical violins at x = 0, 1, 2.
  const RPlot vertical = matrix.panel(0, 4);
  EXPECT_EQ(count_kind(vertical, Kind::Polygon, true), 3U);
  EXPECT_EQ(vertical.x_tick_label_list(), (std::vector<std::string>{"setosa", "versico", "virgini"}));
  EXPECT_TRUE(vertical.y_tick_label_list().empty());
  for (const auto& s : vertical.series()) {
    if (s.kind != Kind::Polygon || !s.filled) continue;
    const auto [lo, hi] = std::minmax_element(s.x.begin(), s.x.end());
    EXPECT_LT(*hi - *lo, 0.85); // every violin stays inside its own level slot
  }
  // Swapped: Species on y gives horizontal violins labelled on the vertical axis.
  const RPlot horizontal = matrix.panel(4, 0);
  EXPECT_EQ(count_kind(horizontal, Kind::Polygon, true), 3U);
  EXPECT_EQ(horizontal.y_tick_label_list(), (std::vector<std::string>{"setosa", "versicolor", "virginica"}));
  EXPECT_TRUE(horizontal.x_tick_label_list().empty());
}

TEST(Splom, CategoricalPairIsShadedMosaic) {
  datamunge::dstruct::DataFrame df;
  std::vector<std::string> a;
  std::vector<std::string> b;
  for (int i = 0; i < 60; ++i) {
    a.push_back(i % 2 == 0 ? "p" : "q");
    b.push_back(i % 2 == 0 ? "u" : (i % 3 == 0 ? "u" : "v")); // strongly associated with a
  }
  df.add_column("a", a);
  df.add_column("b", b);
  const auto  matrix = Splom::create(df);
  const RPlot mosaic = matrix.panel(1, 0);
  EXPECT_EQ(count_kind(mosaic, Kind::Polygon, true), 3U); // the (p, v) cell is empty
  EXPECT_EQ(mosaic.x_tick_label_list(), (std::vector<std::string>{"p", "q"}));
  EXPECT_EQ(mosaic.y_tick_label_list(), (std::vector<std::string>{"u", "v"}));
  bool shaded = false;
  bool p_value = false;
  for (const auto& s : mosaic.series()) {
    if (s.kind == Kind::Polygon && s.color.r != 226) shaded = true; // |residual| > 2 somewhere
    if (s.kind == Kind::Text && s.label == "chi2 p<0.001") p_value = true;
  }
  EXPECT_TRUE(shaded);
  EXPECT_TRUE(p_value);

  // Independent variables stay neutral gray.
  datamunge::dstruct::DataFrame flat;
  flat.add_column("a", std::vector<std::string>{"p", "p", "q", "q"});
  flat.add_column("b", std::vector<std::string>{"u", "v", "u", "v"});
  for (const auto& s : Splom::create(flat).panel(0, 1).series()) {
    if (s.kind == Kind::Polygon) EXPECT_EQ(s.color.r, 226);
  }
}

TEST(Splom, PointColorsRecycleAndClearGrouping) {
  auto matrix = Splom::create(small_frame());
  matrix.group_by("label").point_colors({{1, 2, 3}, {4, 5, 6}});
  EXPECT_TRUE(matrix.groups().empty());
  const RPlot scatter = matrix.panel(0, 1); // panel() returns by value; keep it alive
  const auto& colors = first_kind(scatter, Kind::Scatter).point_colors;
  ASSERT_EQ(colors.size(), 4U);
  EXPECT_EQ(colors[2].r, 1);
  EXPECT_EQ(colors[3].r, 4);
  matrix.color({9, 9, 9});
  EXPECT_TRUE(first_kind(matrix.panel(0, 1), Kind::Scatter).point_colors.empty());
}

TEST(Plot, CustomTickPositionsReplaceNumericTicks) {
  auto p = RPlot::create();
  p.points({0.0, 1.0}, {0.0, 1.0});
  p.x_ticks({0.25}, {"left"}).y_ticks({0.5, 0.75}, {"mid", "high"});
  const std::string svg = p.to_svg();
  EXPECT_NE(svg.find(">left</text>"), std::string::npos);
  EXPECT_NE(svg.find(">high</text>"), std::string::npos);
  EXPECT_EQ(svg.find(">0.5</text>"), std::string::npos);
  EXPECT_THROW(p.x_ticks({0.0}, {}), std::invalid_argument);
  EXPECT_THROW(p.y_ticks({}, {"a"}), std::invalid_argument);
}

// --- RLayout additions used by Splom ------------------------------------------

TEST(RLayout, TitleReservesBandAndKeepsPanelCopies) {
  auto p1 = RPlot::create();
  p1.points({0.0, 1.0}, {0.0, 1.0});
  auto layout = RLayout::create(1, 2);
  layout.size(400, 240).add(p1);
  p1.title("changed later");
  EXPECT_EQ(layout.panel_count(), 1U);
  EXPECT_NE(layout.panel(0).title_text(), "changed later");
  EXPECT_THROW(layout.panel(1), std::out_of_range);

  const std::string untitled = layout.to_svg();
  EXPECT_NE(untitled.find("y=\"0\""), std::string::npos);
  layout.title("Grid");
  const std::string titled = layout.to_svg();
  EXPECT_NE(titled.find(">Grid</text>"), std::string::npos);
  EXPECT_NE(titled.find("y=\"" + std::to_string(RLayout::title_band_px) + "\""), std::string::npos);
}

TEST(Plot, MarginsAndTickFontChangeSvgLayout) {
  auto p = RPlot::create();
  p.points({0.0, 1.0}, {0.0, 1.0});
  const std::string wide = p.to_svg();
  p.margins(10.0, 5.0, 5.0, 10.0).tick_font_size(9.0);
  const std::string tight = p.to_svg();
  EXPECT_NE(wide, tight);
  EXPECT_NE(tight.find("font-size=\"9\""), std::string::npos);
  EXPECT_EQ(wide.find("font-size=\"9\""), std::string::npos);
  EXPECT_THROW(p.margins(-1.0, 0.0, 0.0, 0.0), std::invalid_argument);
  EXPECT_THROW(p.tick_font_size(0.0), std::invalid_argument);
}

TEST(Plot, PrettyTicksUseRoundValues) {
  auto p = RPlot::create();
  p.points({4.3, 7.9}, {0.0, 1.0}).x_limits(4.12, 8.08).y_limits(0.0, 1.0);
  EXPECT_TRUE(p.pretty_ticks_enabled()); // round ticks are the default
  p.pretty_ticks(false);
  EXPECT_NE(p.to_svg().find(">4.912</text>"), std::string::npos); // off: six even ticks
  p.pretty_ticks(true);
  const std::string svg = p.to_svg();
  EXPECT_EQ(svg.find(">4.912</text>"), std::string::npos);
  for (const char* tick : {">5</text>", ">6</text>", ">7</text>", ">8</text>", ">0.6</text>"}) {
    EXPECT_NE(svg.find(tick), std::string::npos) << tick;
  }
}

TEST(Splom, ColorByFactorUsesPaletteForPointsAndMatchingColumn) {
  const auto iris = datamunge::datasets::iris();
  const auto species = datamunge::factor(datamunge::c(iris.get_column<std::string>("Species").values()));
  const auto palette = datamunge::c(datamunge::RGB{255, 255, 80}, datamunge::RGB{100, 255, 255},
                                    datamunge::RGB{220, 100, 255});
  datamunge::Splom facade(iris);
  facade.set_col(species, palette);
  const RPlot scatter = facade.panel(0, 1); // panel() returns by value; keep it alive
  const auto& colors = first_kind(scatter, Kind::Scatter).point_colors;
  ASSERT_EQ(colors.size(), 150U);
  EXPECT_EQ(colors[0].b, 80);    // setosa
  EXPECT_EQ(colors[149].r, 220); // virginica
  // Species encodes to the same codes, so its violins and bar reuse the palette.
  const RPlot violin = facade.panel(0, 4);
  ASSERT_FALSE(violin.series().empty());
  EXPECT_EQ(violin.series()[0].color.b, 80);
  EXPECT_EQ(facade.panel(4, 4).series()[0].color.g, 255);

  auto core = Splom::create(iris);
  EXPECT_THROW(core.color_by_factor({1, 2}, {{0, 0, 0}}), std::invalid_argument);
  EXPECT_THROW(core.color_by_factor(species.values(), {}), std::invalid_argument);
  core.color_by_factor(species.values(), palette.values()).color({1, 1, 1});
  EXPECT_TRUE(first_kind(core.panel(0, 1), Kind::Scatter).point_colors.empty()); // color() clears the factor
  EXPECT_NE(core.panel(0, 4).series()[0].color.b, 80);
}

// --- Smooths, densities and bivariate densities --------------------------------

namespace {

/// @brief y = 2x + 1 plus deterministic noise of standard deviation about 0.5, x in [0, 10].
datamunge::dstruct::DataFrame linear_frame(std::size_t n) {
  std::vector<double> x;
  std::vector<double> y;
  for (std::size_t i = 0; i < n; ++i) {
    const double xi = 10.0 * static_cast<double>(i) / static_cast<double>(n - 1);
    // A fixed pseudo-random sequence (Weyl), mapped to roughly N(0, 0.5^2) via a sum of uniforms.
    double u = 0.0;
    for (int k = 1; k <= 12; ++k) u += std::fmod(static_cast<double>(i * 12 + k) * 0.6180339887498949, 1.0);
    x.push_back(xi);
    y.push_back(2.0 * xi + 1.0 + 0.5 * (u - 6.0));
  }
  datamunge::dstruct::DataFrame df;
  df.add_column("x", x);
  df.add_column("y", y);
  return df;
}

} // namespace

TEST(Splom, SmoothFollowsTheMeanAndBandCoversPoints) {
  const auto  df = linear_frame(200);
  // Upper panel (row 0, col 1) of a frame ordered (y, x) puts x on the horizontal axis, so its
  // smooth estimates E[y | x].
  const auto  swapped = Splom::create(df, {"y", "x"});
  const RPlot smooth = swapped.panel(0, 1);
  ASSERT_EQ(count_kind(smooth, Kind::Polygon, true), 1U);
  const auto& mean = first_kind(smooth, Kind::Line);
  ASSERT_GT(mean.x.size(), 10U);
  for (std::size_t k = 0; k < mean.x.size(); ++k) {
    EXPECT_NEAR(mean.y[k], 2.0 * mean.x[k] + 1.0, 0.35) << "at x=" << mean.x[k];
  }
  // About 95% of the points fall inside the band (interpolated at each point's x).
  const auto& band = first_kind(smooth, Kind::Polygon);
  const std::size_t g = band.x.size() / 2;
  const auto& points = first_kind(smooth, Kind::Scatter);
  std::size_t inside = 0;
  for (std::size_t i = 0; i < points.x.size(); ++i) {
    std::size_t k = 0;
    while (k + 2 < g && band.x[k + 1] < points.x[i]) ++k;
    const double t = (points.x[i] - band.x[k]) / (band.x[k + 1] - band.x[k]);
    const double upper_y = band.y[k] + t * (band.y[k + 1] - band.y[k]);
    // The lower edge is stored in reverse after the upper edge.
    const double lower_k = band.y[2 * g - 1 - k];
    const double lower_k1 = band.y[2 * g - 2 - k];
    const double lower_y = lower_k + t * (lower_k1 - lower_k);
    if (points.y[i] >= lower_y && points.y[i] <= upper_y) ++inside;
  }
  const double coverage = static_cast<double>(inside) / static_cast<double>(points.x.size());
  EXPECT_GT(coverage, 0.90);
  EXPECT_LT(coverage, 0.995);
}

TEST(Splom, GroupedIrisPanelsHaveOneLayerPerSpecies) {
  auto matrix = Splom::create(datamunge::datasets::iris(), {"Sepal.Length", "Sepal.Width", "Petal.Length"});
  matrix.group_by("Species");
  const RPlot diag = matrix.panel(0, 0);
  EXPECT_EQ(count_kind(diag, Kind::Polygon, true), 3U);
  EXPECT_EQ(count_kind(diag, Kind::Line), 3U);

  const RPlot smooth = matrix.panel(0, 1);
  EXPECT_EQ(count_kind(smooth, Kind::Polygon, true), 3U); // one prediction band per species
  EXPECT_EQ(count_kind(smooth, Kind::Scatter), 3U);
  EXPECT_GE(count_kind(smooth, Kind::Line), 3U);

  const RPlot contours = matrix.panel(1, 0);
  EXPECT_EQ(count_kind(contours, Kind::Scatter), 0U);
  std::set<int> reds;
  for (const auto& s : contours.series()) {
    if (s.kind != Kind::Segment) continue;
    reds.insert(s.color.r);
    for (double x : s.x) {
      EXPECT_GE(x, contours.x_min());
      EXPECT_LE(x, contours.x_max());
    }
  }
  // The innermost contour of each species is drawn in its full palette color.
  for (int red : {78, 242, 225}) EXPECT_TRUE(reds.contains(red)) << red;
}

TEST(Splom, PanelsShareAxisRangesPerVariable) {
  auto matrix = Splom::create(datamunge::datasets::iris(), {"Sepal.Length", "Sepal.Width", "Petal.Length"});
  // Column 0 (Sepal.Length on x): diagonal, bivariate density below.
  EXPECT_DOUBLE_EQ(matrix.panel(0, 0).x_min(), matrix.panel(1, 0).x_min());
  EXPECT_DOUBLE_EQ(matrix.panel(1, 0).x_max(), matrix.panel(2, 0).x_max());
  // Row 0 (Sepal.Length on y): smooths to the right share the y range.
  EXPECT_DOUBLE_EQ(matrix.panel(0, 1).y_min(), matrix.panel(0, 2).y_min());
  EXPECT_DOUBLE_EQ(matrix.panel(0, 1).y_min(), matrix.panel(1, 0).x_min());
}

TEST(Splom, SmootherSettingsAreValidated) {
  auto matrix = Splom::create(small_frame());
  EXPECT_THROW(matrix.span(0.0), std::invalid_argument);
  EXPECT_THROW(matrix.span(1.5), std::invalid_argument);
  EXPECT_THROW(matrix.prediction_level(1.0), std::invalid_argument);
  EXPECT_NO_THROW(matrix.span(0.5).prediction_level(0.8));
}
