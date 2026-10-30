#include <gtest/gtest.h>

#include <datamunge/datamunge.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
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

TEST(Vector, RecyclesUnequalLengths) {
    dm::Vector x = {1.0, 2.0};
    dm::Vector y = {1.0, 2.0, 3.0};
    EXPECT_EQ((x + y).values(), (std::vector<double>{2.0, 4.0, 4.0}));
    EXPECT_EQ((x - y).values(), (std::vector<double>{0.0, 0.0, -2.0}));
    EXPECT_EQ((x * y).values(), (std::vector<double>{1.0, 4.0, 3.0}));
    EXPECT_EQ((y / x).values(), (std::vector<double>{1.0, 1.0, 3.0}));
    x += y;
    EXPECT_EQ(x.values(), (std::vector<double>{2.0, 4.0, 4.0}));
    x += x;
    EXPECT_EQ(x.values(), (std::vector<double>{4.0, 8.0, 8.0}));
    EXPECT_TRUE((x + dm::Vector<>()).empty());
    EXPECT_TRUE((dm::Vector<>() / y).empty());
}

TEST(Vector, CombinesAndCoercesOwnedValues) {
    const auto mixed = dm::c({1, "asdf", 21.1});
    EXPECT_EQ(mixed.values(), (std::vector<std::string>{"1", "asdf", "21.1"}));
    EXPECT_EQ(mixed.select(dm::c({2, 1})).values(), (std::vector<std::string>{"asdf", "1"}));
    EXPECT_EQ(dm::c(1, "asdf", 21.1).values(), mixed.values());
    EXPECT_EQ(dm::c({1, 2.5}).values(), (std::vector<double>{1.0, 2.5}));
    EXPECT_EQ(dm::c(1, 2.5).values(), (std::vector<double>{1.0, 2.5}));
    EXPECT_EQ(dm::c({true, "x", false}).values(), (std::vector<std::string>{"TRUE", "x", "FALSE"}));
    EXPECT_EQ(dm::c({-0.0, "x", std::numeric_limits<double>::infinity()}).values(),
              (std::vector<std::string>{"0", "x", "Inf"}));
    EXPECT_TRUE(dm::c().empty());
    EXPECT_TRUE(dm::c({}).empty());
    char text[] = "owned";
    const auto strings = dm::c({text, text});
    text[0] = 'X';
    EXPECT_EQ(strings[0], "owned");
    dm::Vector<std::string> typed(std::vector<std::string>{"a", "b"});
    EXPECT_EQ(typed.select({2, 1})[0], "b");
    struct Record { int id; };
    const auto records = dm::c({Record{10}, Record{20}});
    EXPECT_EQ(records.select({2, 2})[1].id, 20);
    std::ostringstream stream;
    stream << mixed.select({2, 1});
    EXPECT_EQ(stream.str(), "[asdf, 1]");
}

TEST(Vector, SelectsOneBasedIndicesAndExclusions) {
    const auto x = dm::c({10, 20, 30, 40});
    EXPECT_EQ(x.select(dm::c({4, 1, 1, 0})).values(), (std::vector<double>{40, 10, 10}));
    EXPECT_EQ(x.select(std::vector<int>{-2, -2, 0, -99}).values(), (std::vector<double>{10, 30, 40}));
    EXPECT_EQ(x.select(dm::c({2.9, 0.5, 1.1})).values(), (std::vector<double>{20, 10}));
    EXPECT_TRUE(x.select({0, 0}).empty());
    EXPECT_TRUE(x.select({}).empty());
    EXPECT_THROW(x.select({-1, 2}), std::invalid_argument);
    EXPECT_THROW(x.select({5}), std::out_of_range);
    EXPECT_THROW(x.select(dm::c({std::numeric_limits<double>::quiet_NaN()})), std::invalid_argument);
    EXPECT_THROW(x.select(dm::c({std::numeric_limits<double>::infinity()})), std::invalid_argument);
    EXPECT_THROW(x.select(dm::c({1e100})), std::out_of_range);
    EXPECT_EQ(x[0], 10);  // Direct C++ storage indexing stays zero-based.
    EXPECT_EQ(x.size(), 4u);
}

TEST(Vector, RecyclesLogicalMasksAndValues) {
    const auto x = dm::c({10, 20, 30, 40, 50});
    EXPECT_EQ(x.select(dm::c({true, false})).values(), (std::vector<double>{10, 30, 50}));
    EXPECT_EQ(x.select(std::vector<bool>{false, true}).values(), (std::vector<double>{20, 40}));
    EXPECT_TRUE(x.select(std::vector<bool>{}).empty());
    EXPECT_TRUE(x.select(dm::c({false})).empty());
    dm::Vector mask = {true, false};
    EXPECT_EQ(x.select(mask).values(), (std::vector<double>{10, 30, 50}));
    EXPECT_THROW(x.select(std::vector<bool>{true, false, false, false, false, true}), std::out_of_range);
    EXPECT_EQ(dm::c({7, 8}).recycled(5).values(), (std::vector<double>{7, 8, 7, 8, 7}));
    EXPECT_TRUE(dm::c().recycled(0).empty());
    EXPECT_THROW(dm::c().recycled(1), std::invalid_argument);
}

TEST(Vector, FactorsUseSortedOneBasedLevels) {
    const auto codes = dm::factor(dm::c({"virginica", "setosa", "versicolor", "setosa"}));
    EXPECT_EQ(codes.values(), (std::vector<int>{3, 1, 2, 1}));
    EXPECT_EQ(dm::c({"yellow", "cyan", "purple"}).select(codes).values(),
              (std::vector<std::string>{"purple", "yellow", "cyan", "yellow"}));
    EXPECT_TRUE(dm::factor(dm::Vector<std::string>{}).empty());
}

TEST(Vector, ElementwiseMath) {
    dm::Vector x = {0.0, 0.5, 1.0};
    dm::Vector s = dm::sin(x);
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_DOUBLE_EQ(s[i], std::sin(x[i]));
    EXPECT_DOUBLE_EQ(dm::sin(0.5), std::sin(0.5));
    EXPECT_DOUBLE_EQ(dm::pow(x, 2.0)[1], 0.25);
    EXPECT_DOUBLE_EQ(dm::mean(dm::seql(0.0, 1.0, 5)), 0.5);
}

TEST(Vector, LengthAndSampleStandardDeviation) {
    EXPECT_EQ(dm::length(dm::c()), 0u);
    EXPECT_EQ(dm::length(dm::c("a", "b")), 2u);
    EXPECT_EQ(dm::length(dm::c(true, false)), 2u);
    const auto x = dm::c(2, 4, 4, 4, 5, 5, 7, 9);
    EXPECT_NEAR(dm::sd(x), std::sqrt(32.0 / 7), 1e-14);
    EXPECT_NEAR(dm::sd(dm::c(1e12 + 1, 1e12 + 2, 1e12 + 3)), 1, 1e-14);
    EXPECT_DOUBLE_EQ(dm::sd(dm::c(7, 7, 7)), 0);
    EXPECT_TRUE(std::isnan(dm::sd(dm::c())));
    EXPECT_TRUE(std::isnan(dm::sd(dm::c(1))));
    EXPECT_TRUE(std::isnan(dm::sd(dm::c(1, std::numeric_limits<double>::quiet_NaN()))));
    EXPECT_TRUE(std::isnan(dm::sd(dm::c(1, std::numeric_limits<double>::infinity()))));
    EXPECT_EQ(x.values(), (std::vector<double>{2, 4, 4, 4, 5, 5, 7, 9}));
}

TEST(Vector, RoundsScalarsAndVectorsToDecimalPlaces) {
    EXPECT_DOUBLE_EQ(dm::round(1.23456, 3), 1.235);
    EXPECT_DOUBLE_EQ(dm::round(-1.23456, 3), -1.235);
    // Expected values come from R 4.5.2's round(): ties go to the even digit, and the nearer
    // candidate is judged against the binary value actually stored.
    EXPECT_DOUBLE_EQ(dm::round(1.25, 1), 1.2);
    EXPECT_DOUBLE_EQ(dm::round(-1.25, 1), -1.2);
    EXPECT_DOUBLE_EQ(dm::round(125.0, -1), 120);
    EXPECT_DOUBLE_EQ(dm::round(-125.0, -1), -120);
    EXPECT_DOUBLE_EQ(dm::round(2.5), 2);
    EXPECT_DOUBLE_EQ(dm::round(-2.5), -2);
    EXPECT_DOUBLE_EQ(dm::round(5.5), 6);
    EXPECT_DOUBLE_EQ(dm::round(6.5), 6);
    EXPECT_DOUBLE_EQ(dm::round(0.125, 2), 0.12);
    EXPECT_DOUBLE_EQ(dm::round(0.15, 1), 0.1);
    EXPECT_DOUBLE_EQ(dm::round(2.675, 2), 2.67);
    EXPECT_DOUBLE_EQ(dm::round(1.005, 2), 1.0);
    EXPECT_DOUBLE_EQ(dm::round(0.285, 2), 0.28);
    EXPECT_DOUBLE_EQ(dm::round(1250.0, -2), 1200);
    EXPECT_TRUE(std::signbit(dm::round(-0.5)));
    EXPECT_DOUBLE_EQ(dm::round(2), 2);
    const auto x = dm::c(1.25, -1.25, 2.01);
    EXPECT_EQ(dm::round(x, 1).values(), (std::vector<double>{1.2, -1.2, 2.0}));
    EXPECT_EQ(dm::round(x).values(), (std::vector<double>{1, -1, 2}));
    EXPECT_EQ(x.values(), (std::vector<double>{1.25, -1.25, 2.01}));
    EXPECT_TRUE(dm::round(dm::c(), 3).empty());
    EXPECT_TRUE(std::isnan(dm::round(std::numeric_limits<double>::quiet_NaN(), 3)));
    EXPECT_EQ(dm::round(std::numeric_limits<double>::infinity(), 3),
              std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(dm::round(1.25, std::numeric_limits<int>::max()), 1.25);
    EXPECT_DOUBLE_EQ(dm::round(-1.25, std::numeric_limits<int>::min()), 0);
    EXPECT_TRUE(std::signbit(dm::round(-1.25, std::numeric_limits<int>::min())));
    EXPECT_DOUBLE_EQ(dm::round(1e308, 3), 1e308);
}

TEST(Vector, AppliesPlaygroundFunctionWithoutChangingInput) {
    const auto x = dm::seql(0, 1, 101);
    const auto y = dm::apply(x, dm::function([](auto value) {
        return dm::sin(20 / (value + 0.25));
    }));
    ASSERT_EQ(y.size(), x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_DOUBLE_EQ(x[i], static_cast<double>(i) / 100);
        EXPECT_DOUBLE_EQ(y[i], std::sin(20 / (x[i] + 0.25)));
    }
}

TEST(Vector, ApplySupportsTypedResultsMoveOnlyCallablesAndEmptyInputs) {
    auto counter = dm::function([count = std::make_unique<int>(0)](double) mutable {
        return std::to_string(++*count);
    });
    const auto empty = dm::apply(dm::Vector<>{}, counter);
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(dm::apply(dm::c(3, 2, 1), counter).values(),
              (std::vector<std::string>{"1", "2", "3"}));
    EXPECT_EQ(dm::apply(dm::c(true, false), [](bool value) { return !value; }).values(),
              (std::vector<bool>{false, true}));

    auto offset = dm::function([value = 4](double x) { return x + value; });
    auto owned_copy = dm::function(offset);
    EXPECT_EQ(dm::apply(dm::c(1, 2), owned_copy).values(), (std::vector<double>{5, 6}));

    const dm::Vector<std::string> strings{"first", "second"};
    auto copied = dm::apply(strings, [](const std::string& value) -> const std::string& { return value; });
    copied[0] = "changed";
    EXPECT_EQ(strings[0], "first");

    const auto input = dm::c(1, 2);
    EXPECT_THROW(dm::apply(input, [](double) -> double { throw std::runtime_error("failure"); }),
                 std::runtime_error);
    EXPECT_EQ(input.values(), (std::vector<double>{1, 2}));
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

TEST(Vector, RunifSharesUnifValidationAndSeededSequence) {
    dm::set_seed(42);
    const auto expected = dm::unif(0, 1, 1000);
    dm::set_seed(42);
    const auto actual = dm::runif(0, 1, 1000);
    EXPECT_EQ(actual.values(), expected.values());
    for (const auto value : actual) {
        EXPECT_GE(value, 0);
        EXPECT_LT(value, 1);
    }
    EXPECT_TRUE(dm::runif(0, 1, 0).empty());
    EXPECT_THROW(dm::runif(1, 0, 1), std::invalid_argument);
    EXPECT_THROW(dm::runif(0, std::numeric_limits<double>::infinity(), 1), std::invalid_argument);
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
    EXPECT_EQ(plot.series()[0].kind, dm::graphics::DataSeries::Kind::Scatter);
    EXPECT_DOUBLE_EQ(plot.series()[0].marker_size, 6.0);
    EXPECT_EQ(plot.series()[1].kind, dm::graphics::DataSeries::Kind::Line);
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
