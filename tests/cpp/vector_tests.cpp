#include <gtest/gtest.h>

#include <datamunge/datamunge.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace dm = datamunge;

TEST(Vector, ElementwiseArithmetic) {
    dm::Vector x = {1.0, 2.0, 4.0};
    dm::Vector y = {0.5, 0.5, 2.0};

    dm::Vector sum = x + y;
    EXPECT_DOUBLE_EQ(sum[0], 1.5);
    EXPECT_DOUBLE_EQ(sum[2], 6.0);

    dm::Vector quotient = 20 / (x + 0.25);
    EXPECT_DOUBLE_EQ(quotient[0], 16.0);
    EXPECT_DOUBLE_EQ(quotient[1], 20.0 / 2.25);

    dm::Vector diff = 1.0 - x * y;
    EXPECT_DOUBLE_EQ(diff[0], 0.5);
    EXPECT_DOUBLE_EQ(diff[2], -7.0);
    EXPECT_DOUBLE_EQ((-x)[1], -2.0);
}

TEST(Vector, LengthMismatchThrows) {
    dm::Vector x = {1.0, 2.0};
    dm::Vector y = {1.0, 2.0, 3.0};
    EXPECT_THROW(x + y, std::invalid_argument);
}

TEST(Vector, ElementwiseMath) {
    dm::Vector x = {0.0, 0.5, 1.0};
    dm::Vector s = dm::sin(x);
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_DOUBLE_EQ(s[i], std::sin(x[i]));
    EXPECT_DOUBLE_EQ(dm::sin(0.5), std::sin(0.5));
    EXPECT_DOUBLE_EQ(dm::pow(x, 2.0)[1], 0.25);
    EXPECT_DOUBLE_EQ(dm::mean(dm::seql(0.0, 1.0, 5)), 0.5);
}

TEST(Vector, RandomDrawsAreSeededAndInRange) {
    dm::set_seed(42);
    dm::Vector u = dm::unif(2.0, 3.0, 1000);
    dm::Vector z = dm::rnorm(5.0, 0.1, 1000);
    ASSERT_EQ(u.size(), 1000u);
    for (double v : u) {
        EXPECT_GE(v, 2.0);
        EXPECT_LT(v, 3.0);
    }
    EXPECT_NEAR(dm::mean(z), 5.0, 0.05);

    dm::set_seed(42);
    EXPECT_DOUBLE_EQ(dm::unif(2.0, 3.0, 1)[0], u[0]);
}

TEST(Vector, PlotsDirectly) {
    dm::Vector x = dm::seql(0.0, 1.0, 10);
    dm::Plot   plot = dm::Plot().points(x, dm::sin(x), "samples", {37, 99, 235}, 6.0);
    ASSERT_EQ(plot.series().size(), 1u);
    EXPECT_EQ(plot.series()[0].x, x.values());
    EXPECT_NE(plot.to_svg().find("<circle"), std::string::npos);
}

TEST(Vector, SeqlSpansEndpoints) {
    dm::Vector x = dm::seql(0.0, 1.0, 5);
    ASSERT_EQ(x.size(), 5u);
    EXPECT_DOUBLE_EQ(x[0], 0.0);
    EXPECT_DOUBLE_EQ(x[1], 0.25);
    EXPECT_DOUBLE_EQ(x[4], 1.0);
    EXPECT_EQ(dm::seql(3.0, 7.0, 1).values(), std::vector<double>{3.0});
    EXPECT_TRUE(dm::seql(0.0, 1.0, 0).empty());
}

TEST(Vector, UnlabeledColorOverloads) {
    dm::Vector x = dm::seql(0.0, 1.0, 10);
    dm::Plot   plot = dm::Plot()
                        .points(x, dm::sin(x), {37, 99, 235}, 6.0)
                        .lines(x, dm::cos(x), {20, 200, 44}, 2.0);
    ASSERT_EQ(plot.series().size(), 2u);
    EXPECT_EQ(plot.series()[0].kind, dm::plot::DataSeries::Kind::Scatter);
    EXPECT_DOUBLE_EQ(plot.series()[0].marker_size, 6.0);
    EXPECT_EQ(plot.series()[1].kind, dm::plot::DataSeries::Kind::Line);
    EXPECT_EQ(plot.series()[1].color.g, 200);
    EXPECT_TRUE(plot.series()[1].label.empty());
}

TEST(Vector, PlotChainsSettersAndAlpha) {
    dm::Vector x = dm::seql(0.0, 1.0, 5);
    dm::Plot   plot = dm::Plot()
                        .points(x, x, "pts", dm::RGBA{37, 99, 235, 0.2}, 6.0)
                        .lines(x, x, "line", dm::RGB{200, 10, 44}, 3.0)
                        .title("T")
                        .xlab("X")
                        .ylab("Y");
    EXPECT_EQ(plot.title_text(), "T");
    EXPECT_EQ(plot.x_label_text(), "X");
    EXPECT_EQ(plot.y_label_text(), "Y");
    EXPECT_DOUBLE_EQ(plot.series()[0].color.alpha, 0.2);
    EXPECT_DOUBLE_EQ(plot.series()[1].color.alpha, 1.0);

    const std::string svg = plot.to_svg();
    EXPECT_NE(svg.find("fill-opacity=\"0.2\""), std::string::npos);
    EXPECT_EQ(svg.find("<g opacity"), std::string::npos);  // opaque line isn't wrapped

    dm::Plot faint = dm::Plot().lines(x, x, "l", dm::RGBA{0, 0, 0, 0.5});
    EXPECT_NE(faint.to_svg().find("<g opacity=\"0.5\">"), std::string::npos);
}
