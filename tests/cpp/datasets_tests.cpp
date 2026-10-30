#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>

#include <cmath>
#include <stdexcept>
#include <numbers>
#include <string>

TEST(Datasets, IrisShapeAndBalance) {
    const auto df = datamunge::datasets::iris();
    EXPECT_EQ(df.nrows(), 150u);
    EXPECT_EQ(df.ncols(), 5u);

    const auto counts = df.value_counts("Species");
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_EQ(counts.at("setosa"), 50u);
    EXPECT_EQ(counts.at("versicolor"), 50u);
    EXPECT_EQ(counts.at("virginica"), 50u);

    EXPECT_DOUBLE_EQ(df.double_at("Sepal.Length", 0), 5.1);
    EXPECT_DOUBLE_EQ(df.double_at("Petal.Width", 149), 1.8);
    EXPECT_EQ(df.string_at("Species", 149), "virginica");
}

TEST(Datasets, PenguinsShapeNullsAndBalance) {
    const auto df = datamunge::datasets::penguins();
    EXPECT_EQ(df.nrows(), 344u);
    EXPECT_EQ(df.ncols(), 8u);

    EXPECT_EQ(df.null_count("bill_length_mm"), 2u);
    EXPECT_EQ(df.null_count("bill_depth_mm"), 2u);
    EXPECT_EQ(df.null_count("flipper_length_mm"), 2u);
    EXPECT_EQ(df.null_count("body_mass_g"), 2u);
    EXPECT_EQ(df.null_count("sex"), 11u);
    EXPECT_EQ(df.null_count("species"), 0u);
    EXPECT_EQ(df.null_count("island"), 0u);

    const auto counts = df.value_counts("species");
    ASSERT_EQ(counts.size(), 3u);
    EXPECT_EQ(counts.at("Adelie"), 152u);
    EXPECT_EQ(counts.at("Chinstrap"), 68u);
    EXPECT_EQ(counts.at("Gentoo"), 124u);

    EXPECT_TRUE(df.is_null("bill_length_mm", 3));
    EXPECT_EQ(df.string_at("species", 0), "Adelie");
    EXPECT_EQ(df.string_at("island", 0), "Torgersen");
    EXPECT_DOUBLE_EQ(df.double_at("bill_length_mm", 0), 39.1);
}

TEST(Datasets, ToothGrowthShapeAndBalance) {
    const auto df = datamunge::datasets::tooth_growth();
    EXPECT_EQ(df.nrows(), 60u);
    EXPECT_EQ(df.ncols(), 3u);

    const auto supp = df.value_counts("supp");
    ASSERT_EQ(supp.size(), 2u);
    EXPECT_EQ(supp.at("VC"), 30u);
    EXPECT_EQ(supp.at("OJ"), 30u);

    const auto len = df.get_column("len").values();
    double total = 0.0;
    for (const double v : len) total += v;
    EXPECT_NEAR(total / 60.0, 18.8133333333, 1e-9);  // mean(ToothGrowth$len) in R

    EXPECT_DOUBLE_EQ(df.double_at("len", 0), 4.2);
    EXPECT_EQ(df.string_at("supp", 0), "VC");
    EXPECT_DOUBLE_EQ(df.double_at("dose", 0), 0.5);
    EXPECT_DOUBLE_EQ(df.double_at("len", 59), 23.0);
    EXPECT_EQ(df.string_at("supp", 59), "OJ");
    EXPECT_DOUBLE_EQ(df.double_at("dose", 59), 2.0);
}

TEST(Datasets, SwissRollLiesOnTheSpiral) {
    const auto roll = datamunge::datasets::swiss_roll(500, 0.0, 7);
    ASSERT_EQ(roll.nrows(), 500u);
    ASSERT_EQ(roll.ncols(), 6u);
    const double pi = 3.14159265358979323846;
    for (std::size_t i = 0; i < roll.nrows(); ++i) {
        const double t = roll.double_at("t", i);
        const double h = roll.double_at("height", i);
        EXPECT_GE(t, 1.5 * pi);
        EXPECT_LE(t, 4.5 * pi);
        EXPECT_GE(h, 0.0);
        EXPECT_LE(h, 21.0);
        EXPECT_DOUBLE_EQ(roll.double_at("x", i), t * std::cos(t));
        EXPECT_DOUBLE_EQ(roll.double_at("y", i), h);
        EXPECT_DOUBLE_EQ(roll.double_at("z", i), t * std::sin(t));
        const double s = roll.double_at("arc_length", i);
        EXPECT_NEAR(s, 0.5 * (t * std::sqrt(1 + t * t) + std::asinh(t)), 1e-12);
    }
    // s(t) is the integral of sqrt(1 + t^2): Simpson's rule from 1.5 pi to 4.5 pi.
    const double a = 1.5 * pi, b = 4.5 * pi;
    double integral = 0.0;
    const int m = 2000;
    for (int j = 0; j <= m; ++j) {
        const double tau = a + (b - a) * j / m;
        const double wgt = (j == 0 || j == m) ? 1.0 : (j % 2 ? 4.0 : 2.0);
        integral += wgt * std::sqrt(1 + tau * tau);
    }
    integral *= (b - a) / (3.0 * m);
    auto arc = [](double t) { return 0.5 * (t * std::sqrt(1 + t * t) + std::asinh(t)); };
    EXPECT_NEAR(arc(b) - arc(a), integral, 1e-9);
}

TEST(Datasets, SwissRollIsReproducibleAndNoiseOnlyMovesCoordinates) {
    const auto a = datamunge::datasets::swiss_roll(50, 0.5, 3);
    const auto b = datamunge::datasets::swiss_roll(50, 0.5, 3);
    const auto clean = datamunge::datasets::swiss_roll(50, 0.0, 3);
    bool moved = false;
    for (std::size_t i = 0; i < 50; ++i) {
        EXPECT_DOUBLE_EQ(a.double_at("x", i), b.double_at("x", i));
        EXPECT_DOUBLE_EQ(a.double_at("t", i), clean.double_at("t", i));
        EXPECT_DOUBLE_EQ(a.double_at("height", i), clean.double_at("height", i));
        moved = moved || a.double_at("x", i) != clean.double_at("x", i);
    }
    EXPECT_TRUE(moved);
    EXPECT_THROW((void)datamunge::datasets::swiss_roll(10, -1.0), std::invalid_argument);
}

TEST(DatasetsGenerators, BreimanBenchmarksHaveTheirKnownBayesErrors) {
    // twonorm: the Bayes rule is the sign of the coordinate sum, with error Phi(-2) = 0.02275.
    const auto  two = datamunge::datasets::twonorm(20000, 20, 1);
    std::size_t wrong = 0, ones = 0;
    for (std::size_t i = 0; i < two.nrows(); ++i) {
        double sum = 0.0;
        for (int j = 1; j <= 20; ++j) sum += two.double_at("x" + std::to_string(j), i);
        const bool one = two.string_at("class", i) == "1";
        ones += one;
        wrong += (sum > 0.0) != one;
    }
    EXPECT_NEAR(static_cast<double>(wrong) / 20000.0, 0.02275, 0.004);
    EXPECT_NEAR(static_cast<double>(ones) / 20000.0, 0.5, 0.015);
    EXPECT_EQ(two.ncols(), 21u);

    // circle: the ball fills half the cube in 2 and 3 dimensions; flipped labels give the Bayes error.
    for (const std::size_t d : {2u, 3u}) {
        const auto  disc  = datamunge::datasets::circle(20000, d, 0.0, 2);
        std::size_t inside = 0;
        for (std::size_t i = 0; i < disc.nrows(); ++i) inside += disc.string_at("class", i) == "1";
        EXPECT_NEAR(static_cast<double>(inside) / 20000.0, 0.5, 0.015);
    }
    const auto  noisy = datamunge::datasets::circle(20000, 2, 0.15, 3);
    std::size_t bayes_wrong = 0;
    for (std::size_t i = 0; i < noisy.nrows(); ++i) {
        const double a = noisy.double_at("x1", i), b = noisy.double_at("x2", i);
        bayes_wrong += (a * a + b * b < 2.0 / std::numbers::pi) != (noisy.string_at("class", i) == "1");
    }
    EXPECT_NEAR(static_cast<double>(bayes_wrong) / 20000.0, 0.15, 0.01);

    // ringnorm class 1 has variance 4; threenorm is balanced.
    const auto ring = datamunge::datasets::ringnorm(20000, 3, 4);
    double     ss = 0.0, count = 0.0;
    for (std::size_t i = 0; i < ring.nrows(); ++i)
        if (ring.string_at("class", i) == "1") ss += ring.double_at("x1", i) * ring.double_at("x1", i), count += 1.0;
    EXPECT_NEAR(ss / count, 4.0, 0.2);
    EXPECT_EQ(datamunge::datasets::threenorm(10, 4, 5).ncols(), 5u);
    EXPECT_THROW((void)datamunge::datasets::circle(5, 0), std::invalid_argument);
    EXPECT_THROW((void)datamunge::datasets::twonorm(5, 0), std::invalid_argument);
}
