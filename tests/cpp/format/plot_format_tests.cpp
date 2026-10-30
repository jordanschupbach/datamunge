// std::format / std::print support for datamunge::graphics (no plot is ever rendered by formatting).
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>
#include <vector>

namespace plt = dm::graphics;

static_assert(std::formattable<plt::RGB, char>);
static_assert(std::formattable<plt::RGBA, char>);
static_assert(std::formattable<plt::DataSeries::Kind, char>);
static_assert(std::formattable<plt::DataSeries, char>);
static_assert(std::formattable<plt::ABLine, char>);
static_assert(std::formattable<plt::LegendEntry, char>);
static_assert(std::formattable<plt::Plot, char>);
static_assert(std::formattable<plt::RPlot, char>);
static_assert(std::formattable<dm::Histogram, char>);
static_assert(std::formattable<plt::RLayout, char>);
static_assert(std::formattable<plt::Aes, char>);
static_assert(std::formattable<plt::GGTheme, char>);
static_assert(std::formattable<plt::GGPlot, char>);
static_assert(std::formattable<plt::Splom::Diagonal, char>);
static_assert(std::formattable<plt::Splom, char>);
static_assert(std::formattable<dm::PlotArgs, char>);
static_assert(std::formattable<dm::PointsArgs, char>);
static_assert(std::formattable<dm::Points, char>);
static_assert(std::formattable<dm::LegendArgs, char>);
static_assert(std::formattable<dm::Legend, char>);
static_assert(std::formattable<dm::PlotLabel::Target, char>);
static_assert(std::formattable<dm::PlotLabel, char>);

TEST(PlotFormat, ColorsAndSmallStructs) {
    EXPECT_EQ(std::format("{}", dm::RGB{255, 0, 0}), "RGB(255, 0, 0)");
    EXPECT_EQ(std::format("{}", dm::RGBA{37, 99, 235, 0.3}), "RGBA(37, 99, 235, 0.3)");
    EXPECT_EQ(std::format("{}", dm::RGB{}), "RGB(31, 41, 55)");
    EXPECT_EQ(std::format("{}", plt::DataSeries::Kind::Box), "Box");
    plt::DataSeries s;
    s.kind = plt::DataSeries::Kind::Scatter;
    s.x = {1, 2};
    s.y = {3, 4};
    s.label = "pts";
    EXPECT_EQ(std::format("{}", s), "DataSeries(kind=Scatter, n=2, label=\"pts\", color=RGB(31, 41, 55))");
    EXPECT_EQ(std::format("{}", plt::ABLine{false, 0.5, 2.0}),
              "ABLine(intercept=0.5, slope=2, color=RGB(0, 0, 0), stroke_width=1)");
    EXPECT_EQ(std::format("{}", plt::ABLine{true, 0.25}), "ABLine(vertical, x=0.25, color=RGB(0, 0, 0), stroke_width=1)");
    EXPECT_EQ(std::format("{}", plt::LegendEntry{"data", {0, 0, 255}}), "LegendEntry(\"data\", RGB(0, 0, 255))");
    plt::Aes aes;
    aes.x = "x";
    aes.y = "y";
    aes.color = "Species";
    EXPECT_EQ(std::format("{}", aes), "Aes(x=\"x\", y=\"y\", color=\"Species\")");
    EXPECT_EQ(std::format("{}", plt::GGTheme::BW), "BW");
    EXPECT_EQ(std::format("{}", plt::Splom::Diagonal::Histogram), "Histogram");
    EXPECT_EQ(std::format("{}", dm::xlab("n")), "PlotLabel(XLabel, \"n\")");
}

TEST(PlotFormat, PlotsDescribeWithoutRendering) {
    auto p = dm::Plot(dm::c(1.0, 2.0, 3.0), dm::c(1.0, 4.0, 9.0)).set_title("Squares");
    EXPECT_EQ(std::format("{}", p), "RPlot 960x640 title=\"Squares\"\n  axes: x=\"x\", y=\"y\"\n  series: 1 [Scatter n=3]");
    p.abline_h(2.0).legend({"a"}, {dm::RGB{0, 0, 255}});
    const std::string text = std::format("{}", p);
    EXPECT_NE(text.find("reference lines: 1"), std::string::npos);
    EXPECT_NE(text.find("legend: [\"a\"]"), std::string::npos);
    const plt::Plot& base = p;
    EXPECT_EQ(std::format("{}", base).rfind("Plot 960x640", 0), 0u);

    const dm::Histogram h(std::vector<double>{1, 2, 2, 3}, 3);
    EXPECT_EQ(std::format("{}", h),
              "Histogram 960x640 title=\"Histogram\"\n  axes: x=\"x\", y=\"Frequency\"\n  series: 1 [Bar n=3]");

    auto layout = plt::RLayout::create(1, 2);
    layout.title("Grid").add(p);
    EXPECT_EQ(std::format("{}", layout), "RLayout 1x2 grid 960x640 title=\"Grid\" panels=1\n  [0] title=\"Squares\" series=1");
}

TEST(PlotFormat, GrammarAndMatrices) {
    const auto iris = dm::datasets::iris();
    plt::Aes aes;
    aes.x = "Sepal.Length";
    aes.y = "Sepal.Width";
    auto gg = plt::GGPlot::create(iris, aes);
    EXPECT_EQ(std::format("{}", gg), "GGPlot rows=150 aes=Aes(x=\"Sepal.Length\", y=\"Sepal.Width\") layers=[] theme=Gray");
    gg.geom_point().geom_smooth().theme_minimal().facet_wrap("Species").labs("Iris", "", "Width");
    EXPECT_EQ(std::format("{}", gg),
              "GGPlot rows=150 aes=Aes(x=\"Sepal.Length\", y=\"Sepal.Width\") layers=[point, smooth] theme=Minimal "
              "facet=\"Species\" title=\"Iris\" ylab=\"Width\"");
    EXPECT_EQ(gg.layer_count(), 2u);
    const auto sp = plt::Splom::create(iris, {"Sepal.Length", "Species"});
    EXPECT_EQ(std::format("{}", sp), "Splom 960x960 variables=[Sepal.Length, Species]");
}

TEST(PlotFormat, ComposeHelpers) {
    EXPECT_EQ(std::format("{}", dm::PlotArgs{{"title", "Squares"}, {"xlab", "n"}}),
              "PlotArgs(title=\"Squares\", xlab=\"n\")");
    EXPECT_EQ(std::format("{}", dm::PlotArgs{}), "PlotArgs()");
    EXPECT_EQ(std::format("{}", dm::PlotArgs{{"ylab", ""}}), "PlotArgs(ylab=\"\")");
    EXPECT_EQ(std::format("{}", dm::PointsArgs{{"col", dm::c(dm::RGB{255, 0, 0})}}), "PointsArgs(col=[RGB(255, 0, 0)])");
    EXPECT_EQ(std::format("{}", dm::points({1, 2}, {3, 4})), "Points(n=2)");
    EXPECT_EQ(std::format("{}", dm::LegendArgs{{"labels", dm::Vector<std::string>{"a", "b"}},
                                               {"col", dm::c(dm::RGB{255, 0, 0})}}),
              "LegendArgs(labels=[\"a\", \"b\"], col=[RGB(255, 0, 0), RGB(255, 0, 0)])");
    EXPECT_EQ(std::format("{}", dm::legend()), "Legend()");
}
