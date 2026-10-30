#include <gtest/gtest.h>

#include <datamunge/datamunge.hpp>
#include <datamunge/datasets/datasets.hpp>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dm = datamunge;

TEST(PlotCompose, FluentIrisPlotMatchesGroupedOptions) {
    const auto iris = dm::datasets::iris();
    const auto species = dm::factor(dm::c(iris.get_column<std::string>("Species").values()));
    const auto palette = dm::c(dm::RGB{255, 255, 80}, dm::RGB{100, 255, 255}, dm::RGB{220, 100, 255});
    const auto labels = dm::c("setosa", "versicolor", "virginica");
    const dm::Plot fluent = dm::Plot()
        .set_title("Iris Dataset")
        .set_subtitle("Sepal Width vs Sepal Length")
        .set_xlab("Sepal Width")
        .set_ylab("Sepal Length") |
        dm::points(iris.get_column("Sepal.Width").values(), iris.get_column("Sepal.Length").values())
            .set_col(palette.select(species)) |
        dm::legend().set_labels(labels).set_col(palette);
    const dm::Plot grouped = dm::Plot().with(dm::PlotArgs{
        {"title", "Iris Dataset"}, {"subtitle", "Sepal Width vs Sepal Length"},
        {"xlab", "Sepal Width"}, {"ylab", "Sepal Length"}}) |
        dm::points(iris.get_column("Sepal.Width").values(), iris.get_column("Sepal.Length").values())
            .with(dm::PointsArgs{{"col", palette.select(species)}}) |
        dm::legend().with(dm::LegendArgs{{"labels", labels}, {"col", palette}});
    EXPECT_EQ(fluent.to_svg(), grouped.to_svg());
    ASSERT_EQ(fluent.series().size(), 1u);
    EXPECT_EQ(fluent.series()[0].x.size(), 150u);
    EXPECT_EQ(fluent.legend_entries().size(), 3u);
}

TEST(PlotCompose, FluentSettersMutateAndOwnTheirValues) {
    std::string text = "<Heading &>";
    dm::Plot plot;
    EXPECT_EQ(&plot.set_title(text).set_subtitle(text).set_xlab(text).set_ylab(text), &plot);
    text = "changed";
    EXPECT_EQ(plot.title_text(), "<Heading &>");
    EXPECT_EQ(plot.subtitle_text(), "<Heading &>");
    EXPECT_EQ(plot.x_label_text(), "<Heading &>");
    EXPECT_EQ(plot.y_label_text(), "<Heading &>");

    auto colors = dm::c(dm::RGBA{10, 20, 30, 0.25}, dm::RGB{40, 50, 60});
    auto labels = dm::c("one", "two", "three");
    auto points = dm::points({1, 2, 3}, {2, 4, 6});
    auto legend = dm::legend();
    EXPECT_EQ(&points.set_col(colors), &points);
    EXPECT_EQ(&legend.set_col(colors).set_labels(labels), &legend);
    colors[0] = dm::RGB{255, 255, 255};
    labels[0] = "changed";
    const auto result = plot | points | legend;
    EXPECT_TRUE(plot.series().empty());
    EXPECT_TRUE(plot.legend_entries().empty());
    EXPECT_EQ(result.series()[0].point_colors[0].r, 10);
    ASSERT_EQ(result.legend_entries().size(), 3u);
    EXPECT_EQ(result.legend_entries()[0].label, "one");
    EXPECT_EQ(result.legend_entries()[2].color.r, 10);
    EXPECT_DOUBLE_EQ(result.legend_entries()[2].color.alpha, 0.25);
    const auto expected = dm::Plot().title("<Heading &>").subtitle("<Heading &>")
        .xlab("<Heading &>").ylab("<Heading &>") |
        dm::points({1, 2, 3}, {2, 4, 6}).with(dm::PointsArgs{{"col",
            dm::c(dm::RGBA{10, 20, 30, 0.25}, dm::RGB{40, 50, 60})}}) |
        dm::legend().with(dm::LegendArgs{{"labels", dm::c("one", "two", "three")},
            {"col", dm::c(dm::RGBA{10, 20, 30, 0.25}, dm::RGB{40, 50, 60})}});
    EXPECT_EQ(result.to_svg(), expected.to_svg());
    plot.set_title("").set_subtitle("").set_xlab("").set_ylab("");
    EXPECT_TRUE(plot.title_text().empty());
    EXPECT_TRUE(plot.subtitle_text().empty());
    EXPECT_TRUE(plot.x_label_text().empty());
    EXPECT_TRUE(plot.y_label_text().empty());
}

TEST(PlotCompose, FluentStylesValidateAndMixWithGroupedOptions) {
    auto points = dm::points({1}, {2}).with(dm::PointsArgs{{"col", dm::c(dm::RGB{1, 2, 3})}});
    points.set_col(dm::c(dm::RGB{4, 5, 6}));
    EXPECT_THROW(points.set_col(dm::Vector<dm::RGB>{}), std::invalid_argument);
    EXPECT_EQ((dm::Plot() | points).series()[0].point_colors[0].r, 4);
    const auto default_points = points.with(dm::PointsArgs{});
    EXPECT_TRUE((dm::Plot() | default_points).series()[0].point_colors.empty());

    auto legend = dm::legend().with(dm::LegendArgs{
        {"labels", dm::c("old")}, {"col", dm::c(dm::RGB{1, 2, 3})}});
    legend.set_labels(dm::c("new", "next")).set_col(dm::c(dm::RGB{4, 5, 6}));
    EXPECT_THROW(legend.set_labels(dm::Vector<std::string>{}), std::invalid_argument);
    EXPECT_THROW(legend.set_col(dm::Vector<dm::RGB>{}), std::invalid_argument);
    const auto plot = dm::Plot() | points | legend;
    EXPECT_EQ(plot.legend_entries()[0].label, "new");
    EXPECT_EQ(plot.legend_entries()[1].color.r, 4);
    EXPECT_EQ((plot | legend.with(dm::LegendArgs{})).legend_entries().size(), 2u);
    EXPECT_EQ((dm::Plot() | legend).legend_entries().size(), 2u);
}

TEST(PlotCompose, IncompleteFluentLegendFailsBeforeChangingThePlot) {
    dm::Plot plot;
    plot.points({1}, {2}, "automatic");
    const auto before = plot.to_svg();
    auto labels_only = dm::legend().set_labels(dm::c("one"));
    auto colors_only = dm::legend().set_col(dm::c(dm::RGB{1, 2, 3}));
    EXPECT_THROW(labels_only.apply(plot), std::invalid_argument);
    EXPECT_THROW(colors_only.apply(plot), std::invalid_argument);
    EXPECT_EQ(plot.to_svg(), before);
    EXPECT_NO_THROW(labels_only.set_col(dm::c(dm::RGB{})).apply(plot));
    EXPECT_NO_THROW(colors_only.set_labels(dm::c("two")).apply(plot));
    EXPECT_EQ(plot.legend_entries().size(), 2u);
}

TEST(PlotCompose, IrisPaletteSelectionAndHeadings) {
    const auto iris = dm::datasets::iris();
    const auto palette = dm::c(dm::RGB{255, 255, 80}, dm::RGB{100, 255, 255}, dm::RGB{220, 100, 255});
    const auto colors = palette.select(dm::factor(dm::c(iris.string_column("Species").values())));
    const dm::Plot plot = dm::Plot() |
        dm::points(iris.numeric_column("Sepal.Width").values(), iris.numeric_column("Sepal.Length").values())
            .with(dm::PointsArgs({{"col", colors}})) |
        dm::title("Iris dataset") | dm::subtitle("Sepal Width vs Sepal Length");

    ASSERT_EQ(plot.series().size(), 1u);
    const auto& series = plot.series()[0];
    ASSERT_EQ(series.x.size(), 150u);
    ASSERT_EQ(series.point_colors.size(), 150u);
    EXPECT_EQ(series.point_colors[0].r, 255);
    EXPECT_EQ(series.point_colors[50].r, 100);
    EXPECT_EQ(series.point_colors[100].r, 220);
    EXPECT_EQ(plot.title_text(), "Iris dataset");
    EXPECT_EQ(plot.subtitle_text(), "Sepal Width vs Sepal Length");
    const std::string svg = plot.to_svg();
    EXPECT_NE(svg.find("Sepal Width vs Sepal Length"), std::string::npos);
    EXPECT_NE(svg.find("rgb(255,255,80)"), std::string::npos);
    EXPECT_NE(svg.find("rgb(100,255,255)"), std::string::npos);
    EXPECT_NE(svg.find("rgb(220,100,255)"), std::string::npos);
}

TEST(PlotCompose, RecyclesColorsAndRetainsIndividualAlpha) {
    const auto layer = dm::points(dm::c({1, 2, 3}), dm::c({2, 4, 6})).with(
        dm::PointsArgs({{"col", dm::c({dm::RGBA{10, 20, 30, 0.25}, dm::RGB{40, 50, 60}})}}));
    const dm::Plot original;
    const auto plot = original | layer | dm::title("<Title &>") | dm::subtitle("<Subtitle &>");
    EXPECT_TRUE(original.series().empty());
    const auto svg = plot.to_svg();
    const auto first = svg.find("fill=\"rgb(10,20,30)\" fill-opacity=\"0.25\"");
    const auto second = svg.find("fill=\"rgb(40,50,60)\"");
    const auto third = svg.find("fill=\"rgb(10,20,30)\" fill-opacity=\"0.25\"", first + 1);
    ASSERT_NE(first, std::string::npos);
    ASSERT_NE(second, std::string::npos);
    ASSERT_NE(third, std::string::npos);
    EXPECT_LT(first, second);
    EXPECT_LT(second, third);
    EXPECT_NE(svg.find("&lt;Title &amp;&gt;"), std::string::npos);
    EXPECT_NE(svg.find("&lt;Subtitle &amp;&gt;"), std::string::npos);
    EXPECT_EQ((dm::Plot() | layer).to_svg().find("<g opacity"), std::string::npos);
}

TEST(PlotCompose, OptionsAreValidatedAndLayersAreReusable) {
    const auto base = dm::points({1, 2}, {3, 4});
    const auto styled = base.with(dm::PointsArgs({{"col", dm::c(dm::RGB{200, 50, 10})}}));
    const auto combined = dm::Plot() | base | styled;
    ASSERT_EQ(combined.series().size(), 2u);
    EXPECT_TRUE(combined.series()[0].point_colors.empty());
    EXPECT_EQ(combined.series()[1].point_colors[0].r, 200);
    EXPECT_THROW(dm::points({1}, {1, 2}), std::invalid_argument);
    EXPECT_THROW(dm::points({}, {}), std::invalid_argument);
    EXPECT_THROW((dm::PointsArgs{{"unknown", dm::c(dm::RGB{})}}), std::invalid_argument);
    EXPECT_THROW((dm::PointsArgs{{"col", dm::Vector<dm::RGB>{}}}), std::invalid_argument);
    EXPECT_THROW((dm::PointsArgs{{"col", dm::c(dm::RGB{})}, {"col", dm::c(dm::RGB{})}}), std::invalid_argument);
    dm::Plot direct;
    EXPECT_THROW(direct.points_colored({1}, {2}, {}), std::invalid_argument);
    EXPECT_TRUE(direct.series().empty());
}

TEST(PlotCompose, AxisLabelPipesOwnTextAndLeaveTheInputUnchanged) {
    const auto original = dm::Plot() | dm::points({1, 2}, {3, 4});
    std::string axis = "Width <cm>";
    const auto label = dm::xlab(axis);
    axis.assign("changed");
    const auto plot = original | label | dm::ylab("Length & size");
    EXPECT_EQ(original.x_label_text(), "x");
    EXPECT_EQ(original.y_label_text(), "y");
    EXPECT_EQ(plot.x_label_text(), "Width <cm>");
    EXPECT_EQ(plot.y_label_text(), "Length & size");
    EXPECT_NE(plot.to_svg().find("Width &lt;cm&gt;"), std::string::npos);
    EXPECT_NE(plot.to_svg().find("Length &amp; size"), std::string::npos);
}

TEST(PlotCompose, PlotOptionsOwnTextAndUpdateOnlySuppliedFields) {
    std::string heading = "Iris";
    const dm::PlotArgs args{{"title", heading}, {"subtitle", "Species"},
                            {"xlab", "Width"}, {"ylab", "Length"}};
    heading = "changed";
    auto plot = dm::Plot().with(args) | dm::points({1, 2}, {3, 4});
    EXPECT_EQ(plot.title_text(), "Iris");
    EXPECT_EQ(plot.subtitle_text(), "Species");
    EXPECT_EQ(plot.x_label_text(), "Width");
    EXPECT_EQ(plot.y_label_text(), "Length");
    EXPECT_EQ(&plot.with(dm::PlotArgs{{"title", ""}}), &plot);
    EXPECT_TRUE(plot.title_text().empty());
    EXPECT_EQ(plot.subtitle_text(), "Species");
    EXPECT_EQ(&plot.with(dm::PlotArgs{}), &plot);
    EXPECT_THROW((dm::PlotArgs{{"wrong", "x"}}), std::invalid_argument);
    EXPECT_THROW((dm::PlotArgs{{"title", "first"}, {"title", "second"}}), std::invalid_argument);
}

TEST(PlotCompose, LegendOptionsRecycleOwnedColorsAndValidateTheirTypes) {
    auto names = dm::c({"one", "two", "three"});
    const auto palette = dm::c(dm::RGB{255, 0, 0}, dm::RGBA{0, 0, 255, 0.5});
    const auto layer = dm::legend().with(dm::LegendArgs{{"labels", names}, {"col", palette}});
    names[0] = "changed";
    const auto original = dm::Plot().points({1, 2}, {3, 4}, "automatic");
    const auto plot = original | layer;
    EXPECT_TRUE(original.legend_entries().empty());
    ASSERT_EQ(plot.legend_entries().size(), 3u);
    EXPECT_EQ(plot.legend_entries()[0].label, "one");
    EXPECT_EQ(plot.legend_entries()[0].color.r, 255);
    EXPECT_EQ(plot.legend_entries()[1].color.b, 255);
    EXPECT_EQ(plot.legend_entries()[2].color.r, 255);
    EXPECT_EQ(plot.to_svg().find("automatic"), std::string::npos);
    EXPECT_NE(plot.to_svg().find("fill=\"rgb(0,0,255)\" fill-opacity=\"0.5\""), std::string::npos);
    EXPECT_TRUE((original | dm::legend().with(dm::LegendArgs{})).legend_entries().empty());
    EXPECT_THROW((dm::LegendArgs{{"labels", dm::c({"one"})}}), std::invalid_argument);
    EXPECT_THROW((dm::LegendArgs{{"col", palette}}), std::invalid_argument);
    EXPECT_THROW((dm::LegendArgs{{"labels", palette}, {"col", palette}}), std::invalid_argument);
    EXPECT_THROW((dm::LegendArgs{{"labels", names}, {"col", dm::Vector<dm::RGB>{}}}), std::invalid_argument);
    EXPECT_THROW((dm::LegendArgs{{"labels", names}, {"labels", names}, {"col", palette}}), std::invalid_argument);
    EXPECT_THROW((dm::LegendArgs{{"unknown", names}}), std::invalid_argument);
}

TEST(PlotCompose, ManualLegendAppearsInTheRasterViewerAndExport) {
    auto plot = dm::Plot().points({1, 2}, {3, 4}, "automatic", dm::RGB{0, 255, 0}) |
        dm::legend().with(dm::LegendArgs{
            {"labels", dm::c({"one", "two", "three"})},
            {"col", dm::c(dm::RGB{255, 0, 0}, dm::RGBA{0, 0, 255, 0.5})}});
    plot.size(450, 300).show_grid(false);
    const auto path = std::filesystem::temp_directory_path() / "datamunge_manual_legend_test.png";
    plot.save(path.string());
    const auto image = dm::image::read_png(path.string());
    std::filesystem::remove(path);
    // Raster legend geometry matches the SVG: swatches at x = right - 140, rows 20px apart.
    EXPECT_EQ(image.get_pixel(295, 64), (dm::image::Pixel{255, 0, 0, 255}));
    EXPECT_EQ(image.get_pixel(295, 84), (dm::image::Pixel{128, 128, 255, 255}));
    EXPECT_EQ(image.get_pixel(295, 104), (dm::image::Pixel{255, 0, 0, 255}));
}

TEST(PlotCompose, PngUsesRecycledColorsAndAlpha) {
    auto plot = dm::Plot() |
        dm::points({1, 2, 3}, {2, 2, 2}).with(dm::PointsArgs({
            {"col", dm::c(dm::RGB{255, 0, 0}, dm::RGBA{0, 0, 255, 0.5})}
        })) | dm::subtitle("Palette");
    plot.size(400, 300).x_limits(0, 4).y_limits(0, 4).show_grid(false);
    const auto path = std::filesystem::temp_directory_path() / "datamunge_plot_palette_test.png";
    plot.save(path.string());
    const auto image = dm::image::read_png(path.string());
    std::filesystem::remove(path);
    // Read centers of three separated markers in the chart interior.
    EXPECT_EQ(image.get_pixel(162, 156), (dm::image::Pixel{255, 0, 0, 255}));
    EXPECT_EQ(image.get_pixel(233, 156), (dm::image::Pixel{128, 128, 255, 255}));
    EXPECT_EQ(image.get_pixel(305, 156), (dm::image::Pixel{255, 0, 0, 255}));
}

TEST(PlotCompose, HistogramChainsInheritedSettersWithoutLosingItsType) {
    static_assert(std::is_same_v<decltype(std::declval<dm::Histogram&>().set_col("red")), dm::Histogram&>);
    static_assert(std::is_same_v<decltype(std::declval<dm::Histogram&>().set_col(dm::RGB{1, 2, 3})), dm::Histogram&>);
    static_assert(std::is_same_v<decltype(std::declval<dm::Histogram&>().size(10, 10)), dm::Histogram&>);
    dm::Histogram hist = dm::Histogram({1.0, 2.0, 3.0}).set_title("t").set_col("red").size(400, 300).set_xlab("v");
    EXPECT_EQ(hist.title_text(), "t");
    EXPECT_EQ(hist.width(), 400u);
}

TEST(PlotCompose, PlotConstructorNamesItselfInLengthErrors) {
    try {
        dm::Plot(std::vector<double>{1, 2}, std::vector<double>{1});
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_EQ(std::string(e.what()), "Plot(x, y): x and y must have the same size");
    }
}

TEST(PlotCompose, SvgClampsOutOfRangeColorChannels) {
    auto plot = dm::Plot(dm::c(1, 2), dm::c(1, 2)).set_col(dm::RGB{300, -20, 128});
    const auto svg = plot.to_svg();
    EXPECT_NE(svg.find("rgb(255,0,128)"), std::string::npos);
    EXPECT_EQ(svg.find("rgb(300"), std::string::npos);
}
