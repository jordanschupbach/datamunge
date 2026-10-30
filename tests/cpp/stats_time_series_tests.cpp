#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/time_series.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace st = datamunge::stats;

namespace {
// diff(diff(log(AirPassengers)), lag = 12), the series behind R's airline-model plots.
std::vector<double> airline_w() {
    auto y = datamunge::datasets::air_passengers().get_column<double>("passengers").values();
    for (double& v : y) v = std::log(v);
    return st::difference(st::difference(y, 1), 12);
}
}  // namespace

TEST(TimeSeries, DifferenceMatchesDefinition) {
    const std::vector<double> x = {1, 4, 9, 16, 25};
    EXPECT_EQ(st::difference(x), (std::vector<double>{3, 5, 7, 9}));
    EXPECT_EQ(st::difference(x, 1, 2), (std::vector<double>{2, 2, 2}));
    EXPECT_EQ(st::difference(x, 2), (std::vector<double>{8, 12, 16}));
    EXPECT_THROW(st::difference(x, 0), std::invalid_argument);
    EXPECT_THROW(st::difference(x, 3, 2), std::invalid_argument);
}

TEST(TimeSeries, AcfAndPacfMatchR) {
    const auto w = airline_w();
    ASSERT_EQ(w.size(), 131u);
    // R: acf(w, lag.max = 13)$acf and pacf(w, lag.max = 13)$acf, rounded to 4 places.
    const std::vector<double> r_acf  = {-0.3411, 0.1050, -0.2021, 0.0214, 0.0557, 0.0308, -0.0556,
                                        -0.0008, 0.1764, -0.0764, 0.0644, -0.3866, 0.1516};
    const std::vector<double> r_pacf = {-0.3411, -0.0128, -0.1927, -0.1250, 0.0331, 0.0347, -0.0602,
                                        -0.0202, 0.2256, 0.0431, 0.0466, -0.3387, -0.1092};
    const auto a = st::acf(w, 13);
    const auto p = st::pacf(w, 13);
    EXPECT_DOUBLE_EQ(a[0], 1.0);
    for (std::size_t k = 0; k < 13; ++k) {
        EXPECT_NEAR(a[k + 1], r_acf[k], 6e-5) << k + 1;
        EXPECT_NEAR(p[k], r_pacf[k], 6e-5) << k + 1;
    }
}

TEST(TimeSeries, PacfOfAr1VanishesAfterLagOne) {
    std::vector<double> x(4000);
    unsigned s = 1;
    double prev = 0;
    for (double& v : x) {  // AR(1), phi = 0.6, with a small LCG for uniform noise
        s = s * 1103515245u + 12345u;
        const double e = (static_cast<double>(s >> 8) / 16777216.0) - 0.5;
        v = 0.6 * prev + e;
        prev = v;
    }
    const auto p = st::pacf(x, 5);
    EXPECT_NEAR(p[0], 0.6, 0.05);
    for (std::size_t k = 1; k < 5; ++k) EXPECT_LT(std::fabs(p[k]), 0.05);
}

TEST(TimeSeries, LjungBoxMatchesHandComputation) {
    const std::vector<double> x = {1, -2, 3, 0.5, -1, 2, -3, 1.5, 0, -0.5};
    const auto r = st::acf(x, 3);
    const double n = 10;
    const double q = n * (n + 2) * (r[1] * r[1] / 9 + r[2] * r[2] / 8 + r[3] * r[3] / 7);
    const auto t = st::ljung_box_test(x, 3, 1);
    EXPECT_NEAR(t.statistic, q, 1e-12);
    EXPECT_DOUBLE_EQ(t.parameter1, 2.0);
    EXPECT_NEAR(t.p_value, std::exp(-q / 2), 1e-10);  // chi-squared with 2 df
    EXPECT_THROW(st::ljung_box_test(x, 1, 1), std::invalid_argument);
}

TEST(TimeSeries, ArmaAcfClosedForms) {
    // AR(1): rho_k = phi^k.
    const auto a = st::arma_acf({0.6}, {}, 4);
    for (int k = 0; k <= 4; ++k) EXPECT_NEAR(a[k], std::pow(0.6, k), 1e-12);
    // Airline MA: (1 + theta B)(1 + Theta B^12).
    const double th = -0.4, Th = -0.6;
    std::vector<double> ma(13, 0.0);
    ma[0] = th; ma[11] = Th; ma[12] = th * Th;
    const auto m = st::arma_acf({}, ma, 14);
    EXPECT_NEAR(m[1], th / (1 + th * th), 1e-12);
    EXPECT_NEAR(m[12], Th / (1 + Th * Th), 1e-12);
    EXPECT_NEAR(m[11], m[1] * m[12], 1e-12);
    EXPECT_NEAR(m[13], m[1] * m[12], 1e-12);
    EXPECT_NEAR(m[5], 0.0, 1e-12);
    EXPECT_NEAR(m[14], 0.0, 1e-12);
    EXPECT_THROW(st::arma_acf({1.01}, {}, 3), std::invalid_argument);
}
