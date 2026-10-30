#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

TEST(DmApi, UmbrellaExposesModulesAndOriginalTypes) {
    static_assert(std::is_same_v<dm::Vector<>, datamunge::Vector<>>);
    static_assert(std::is_same_v<dm::Plot, datamunge::Plot>);
    EXPECT_EQ(dm::datasets::iris().nrows(), 150u);
    EXPECT_EQ(dm::algorithms::z_array("aaaa"), (std::vector<std::size_t>{4, 3, 2, 1}));
    EXPECT_EQ(dm::linalg::DenseMatrix<>(2, 3).cols(), 3u);
    EXPECT_EQ(dm::nn::zeros({2, 3}).numel(), 6);
}

TEST(DmApi, ColumnAccessBorrowsStorageAndPreservesNulls) {
    dm::dstruct::DataFrame frame;
    frame.add_column("x", std::vector<std::optional<double>>{1.0, std::nullopt});
    frame.add_column("label", std::vector<std::optional<std::string>>{"a", std::nullopt});
    const auto& view = frame;
    EXPECT_EQ(&view.get_column("x"), &view.numeric_column("x"));
    EXPECT_EQ(&view.get_column<std::string>("label"), &view.string_column("label"));
    EXPECT_TRUE(view.get_column("x").is_null(1));
    frame.get_column("x").set(1, 3.0);
    frame.get_column<std::string>("label").set(1, std::string("b"));
    EXPECT_DOUBLE_EQ(frame.double_at("x", 1), 3.0);
    EXPECT_EQ(frame.string_at("label", 1), "b");
    frame.get_column("x").set_null(0);
    EXPECT_TRUE(frame.is_null("x", 0));
    EXPECT_THROW(static_cast<void>(view.get_column("missing")), std::out_of_range);
    EXPECT_THROW(static_cast<void>(frame.get_column("label")), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(view.get_column<std::string>("x")), std::invalid_argument);
}

TEST(DmApi, PlaygroundColumnMomentStatistics) {
    const auto other_stats = dm::function([](const auto& x) {
        const auto m = dm::mean(x);
        const auto n = dm::length(x);
        const auto s = dm::sd(x);
        const auto skew = dm::round(dm::sum(dm::pow(x - m, 3) / dm::pow(s, 3)) / n, 3);
        const auto kurtosis = dm::round(dm::sum(dm::pow(x - m, 4) / dm::pow(s, 4)) / n - 3, 3);
        return dm::c(skew, kurtosis);
    });
    // For {1, 2, 3}, mean = 2, sample sd = 1, and excess kurtosis = 2/3 - 3.
    EXPECT_EQ(other_stats(dm::c(1, 2, 3)).values(), (std::vector<double>{0, -2.333}));
    const auto data = dm::datasets::tooth_growth();
    const auto stats = other_stats(dm::c(data.get_column("len").values()));
    EXPECT_EQ(stats.values(), (std::vector<double>{-0.143, -1.043}));
}

TEST(DmApi, QualifiedPlaygroundExpressionRenders) {
    const auto data = dm::datasets::iris();
    const auto species = dm::factor(dm::c(data.get_column<std::string>("Species").values()));
    const dm::Plot plot = dm::Plot() |
        dm::points(data.get_column("Sepal.Width").values(), data.get_column("Sepal.Length").values())
            .with(dm::PointsArgs({{"col", dm::c(dm::RGB{255, 255, 80}, dm::RGB{100, 255, 255},
                                             dm::RGB{220, 100, 255}).select(species)}})) |
        dm::title("Iris dataset") | dm::subtitle("Sepal Width vs Sepal Length") |
        dm::xlab("Sepal Width") | dm::ylab("Sepal Length");
    EXPECT_EQ(plot.series()[0].x.size(), 150u);
    EXPECT_EQ(plot.x_label_text(), "Sepal Width");
    EXPECT_EQ(plot.y_label_text(), "Sepal Length");
    EXPECT_NE(plot.to_svg().find("Sepal Width vs Sepal Length"), std::string::npos);
}

TEST(DmApi, AlgorithmHeadersCoexistWithTheirOwnResultTypes) {
    const std::vector<std::vector<double>> matrix{{3, 0}, {0, 1}};
    const dm::algorithms::EigenPair dominant = dm::algorithms::power_iteration(matrix);
    const dm::algorithms::EigenPair nearest = dm::algorithms::inverse_iteration(matrix, 1.1);
    EXPECT_NEAR(dominant.value, 3.0, 1e-8);
    EXPECT_NEAR(nearest.value, 1.0, 1e-8);

    const dm::algorithms::SearchResult position = dm::algorithms::binary_search({1, 3, 5}, 3);
    const dm::algorithms::InformedSearchResult path = dm::algorithms::a_star_search(
        3, {{0, 1, 2.0}, {1, 2, 3.0}}, 0, 2, {0, 0, 0});
    EXPECT_TRUE(position.found);
    EXPECT_EQ(position.index, 1);
    EXPECT_TRUE(path.found);
    EXPECT_DOUBLE_EQ(path.cost, 5.0);
    EXPECT_EQ(path.path, (std::vector<std::size_t>{0, 1, 2}));

    const std::vector<dm::algorithms::Point2D> polygon{{0, 0}, {2, 0}, {2, 2}, {0, 2}};
    const std::vector<std::array<double, 2>> array_polygon{{0, 0}, {2, 0}, {2, 2}, {0, 2}};
    const auto pixels = dm::algorithms::scanline_fill(polygon);
    EXPECT_EQ(pixels.size(), 4u);
    EXPECT_EQ(pixels, dm::algorithms::scanline_fill(array_polygon));
    EXPECT_EQ(pixels, dm::algorithms::scanline_fill({{0, 0}, {2, 0}, {2, 2}, {0, 2}}));
    dm::algorithms::Point2D point{1, 2};
    point[0] = 3;
    EXPECT_DOUBLE_EQ(point.x, 3);
    EXPECT_DOUBLE_EQ(static_cast<const dm::algorithms::Point2D&>(point)[1], point.y);
}

TEST(DmApi, ScatterConstructorOwnsCoordinatesAndValidatesLengths) {
    auto x = dm::seql(0, 1, 101);
    const auto y = dm::apply(x, [](double value) { return dm::sin(20 / (value + 0.25)); });
    const dm::Plot plot(x, y);
    ASSERT_EQ(plot.series().size(), 1u);
    EXPECT_EQ(plot.series()[0].kind, dm::graphics::DataSeries::Kind::Scatter);
    EXPECT_EQ(plot.series()[0].x, x.values());
    EXPECT_EQ(plot.series()[0].y, y.values());
    x[0] = 10;
    EXPECT_DOUBLE_EQ(plot.series()[0].x[0], 0);
    EXPECT_NE(plot.to_svg().find("<circle"), std::string::npos);
    EXPECT_THROW((dm::Plot({}, {})), std::invalid_argument);
    EXPECT_THROW((dm::Plot({1}, {1, 2})), std::invalid_argument);
    EXPECT_TRUE(dm::Plot().series().empty());
}

TEST(DmApi, PlotColorsReplacePalettesAndValidateBeforeMutation) {
    auto plot = dm::Plot() | dm::points({0, 1}, {1, 2}).set_col(dm::c(dm::RGB{1, 2, 3}));
    plot.lines({0, 1}, {1, 2});
    EXPECT_EQ(&plot.set_col("BlUe"), &plot);
    for (const auto& series : plot.series()) {
        EXPECT_TRUE(series.point_colors.empty());
        EXPECT_EQ(series.color.r, 0);
        EXPECT_EQ(series.color.g, 0);
        EXPECT_EQ(series.color.b, 255);
    }
    EXPECT_NE(plot.to_svg().find("rgb(0,0,255)"), std::string::npos);
    plot.set_col("#1234aB80");
    EXPECT_EQ(plot.series()[0].color.r, 0x12);
    EXPECT_EQ(plot.series()[0].color.g, 0x34);
    EXPECT_EQ(plot.series()[0].color.b, 0xab);
    EXPECT_DOUBLE_EQ(plot.series()[0].color.alpha, 128.0 / 255);
    plot.set_col("#00ff00");
    EXPECT_EQ(plot.series()[0].color.g, 255);
    EXPECT_DOUBLE_EQ(plot.series()[0].color.alpha, 1);
    plot.set_col("transparent");
    EXPECT_DOUBLE_EQ(plot.series()[0].color.alpha, 0);
    plot.set_col(dm::RGBA{10, 20, 30, 0.25});
    const auto before = plot.to_svg();
    for (const auto* invalid : {"unknown", "", "#xyzxyz", "#1234567", "#12345g", "#123456789"}) {
        EXPECT_THROW(plot.set_col(invalid), std::invalid_argument);
        EXPECT_EQ(plot.to_svg(), before);
    }
}

TEST(DmApi, HistogramBinsBoundariesAndSupportsFluentConstruction) {
    dm::Histogram histogram = dm::Histogram(dm::c(0, 1, 2, 3, 4), 4)
        .set_title("Distribution").set_subtitle("Equal bins").set_xlab("Value").set_ylab("Count");
    ASSERT_EQ(histogram.series().size(), 1u);
    const auto& series = histogram.series()[0];
    EXPECT_EQ(series.kind, dm::graphics::DataSeries::Kind::Bar);
    EXPECT_EQ(series.x, (std::vector<double>{0.5, 1.5, 2.5, 3.5}));
    EXPECT_EQ(series.y, (std::vector<double>{1, 1, 1, 2}));
    EXPECT_DOUBLE_EQ(series.bar_width, 1);
    EXPECT_EQ(histogram.title_text(), "Distribution");
    EXPECT_EQ(histogram.subtitle_text(), "Equal bins");
    EXPECT_EQ(histogram.x_label_text(), "Value");
    EXPECT_EQ(histogram.y_label_text(), "Count");
    EXPECT_NE(histogram.to_svg().find("Distribution"), std::string::npos);

    const auto lengths = dm::datasets::iris().get_column("Sepal.Length").values();
    const dm::Histogram iris(lengths);
    EXPECT_EQ(iris.series()[0].y.size(), 10u);
    EXPECT_DOUBLE_EQ(std::accumulate(iris.series()[0].y.begin(), iris.series()[0].y.end(), 0.0), 150);
    EXPECT_EQ(iris.y_label_text(), "Frequency");
    EXPECT_EQ(iris.to_svg(), dm::Plot::hist(lengths).to_svg());
    const dm::Plot composed = iris | dm::title("Iris");
    EXPECT_EQ(composed.title_text(), "Iris");
}

TEST(DmApi, HistogramHandlesConstantDataAndRejectsInvalidInputs) {
    const dm::Histogram constant({2, 2, 2}, 2);
    EXPECT_EQ(constant.series()[0].y, (std::vector<double>{3, 0}));
    EXPECT_DOUBLE_EQ(constant.series()[0].bar_width, 0.5);
    EXPECT_EQ(dm::Histogram({2}, 1).series()[0].y, (std::vector<double>{1}));
    EXPECT_THROW(dm::Histogram({}), std::invalid_argument);
    EXPECT_THROW(dm::Histogram({1}, 0), std::invalid_argument);
    EXPECT_THROW(dm::Histogram({1, std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
    EXPECT_THROW(dm::Histogram({std::numeric_limits<double>::infinity()}), std::invalid_argument);
    const auto maximum = std::numeric_limits<double>::max();
    EXPECT_THROW(dm::Histogram({-maximum, maximum}), std::invalid_argument);
    EXPECT_THROW(dm::Histogram({maximum}), std::invalid_argument);
    EXPECT_THROW(dm::Histogram({0, std::numeric_limits<double>::denorm_min()}), std::invalid_argument);
}
