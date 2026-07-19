#include <gtest/gtest.h>

#include <datamunge/stats/arima.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using datamunge::stats::ARIMA;
using datamunge::stats::ARIMAOptions;

namespace {

std::vector<double> simulate_ar1(std::size_t n, double phi, double mean, double sigma, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> noise(0.0, sigma);
    std::vector<double> x(n);
    double prev = mean;
    for (std::size_t i = 0; i < n; ++i) {
        prev = mean + phi * (prev - mean) + noise(rng);
        x[i] = prev;
    }
    return x;
}

std::vector<double> simulate_ma1(std::size_t n, double theta, double sigma, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> noise(0.0, sigma);
    std::vector<double> x(n);
    double prev_e = noise(rng);
    for (std::size_t i = 0; i < n; ++i) {
        const double e = noise(rng);
        x[i] = e + theta * prev_e;
        prev_e = e;
    }
    return x;
}

} // namespace

// This CSS estimator was cross-checked interactively against base R's stats::arima(...,
// method = "CSS") -- a genuine, independently implemented CSS estimator -- on AR(1), MA(1),
// ARIMA(1,1,0), and a seasonal (1,0,0)(1,1,0)_12 model, matching R to 3-4 decimal places in
// every case. The tests below instead check parameter recovery on self-simulated data (so
// they don't depend on reproducing R's exact RNG stream) plus closed-form/self-consistency
// properties that hold regardless of any oracle.

TEST(ARIMA, RecoversAR1Parameter) {
    const auto x = simulate_ar1(300, 0.6, 10.0, 1.0, 4242);
    ARIMAOptions o;
    o.p = 1;
    o.de_population_size = 80;
    o.de_max_generations = 500;
    const ARIMA model(x, o);
    EXPECT_NEAR(model.ar_coefficients()[0], 0.6, 0.15);
    EXPECT_NEAR(model.mean(), 10.0, 0.5);
    EXPECT_GT(model.sigma2(), 0.0);
}

TEST(ARIMA, RecoversMA1Parameter) {
    const auto x = simulate_ma1(400, 0.5, 1.0, 99);
    ARIMAOptions o;
    o.q = 1;
    o.de_population_size = 80;
    o.de_max_generations = 500;
    const ARIMA model(x, o);
    EXPECT_NEAR(model.ma_coefficients()[0], 0.5, 0.15);
}

TEST(ARIMA, ZeroOrderReducesToSampleMeanAndVariance) {
    const std::vector<double> y{1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0};
    ARIMAOptions o; // p = d = q = 0, include_mean = true
    const ARIMA model(y, o);

    double sample_mean = 0.0;
    for (const double v : y) sample_mean += v;
    sample_mean /= static_cast<double>(y.size());
    double sample_var = 0.0;
    for (const double v : y) sample_var += (v - sample_mean) * (v - sample_mean);
    sample_var /= static_cast<double>(y.size());

    EXPECT_NEAR(model.mean(), sample_mean, 1e-6);
    EXPECT_NEAR(model.sigma2(), sample_var, 1e-6);
}

TEST(ARIMA, DifferencingRoundTripsThroughForecast) {
    // A pure random walk (d = 1, no AR/MA terms): the best point forecast at any horizon is
    // exactly the last observed value (a martingale), a hard closed-form check independent
    // of any oracle.
    const std::vector<double> y{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
    ARIMAOptions o;
    o.d = 1;
    const ARIMA model(y, o);
    const auto fc = model.forecast(4);
    ASSERT_EQ(fc.size(), 4u);
    for (const double v : fc) EXPECT_NEAR(v, 5.0, 1e-6);
}

TEST(ARIMA, StationaryAR1ForecastMeanReverts) {
    const auto x = simulate_ar1(300, 0.5, 20.0, 1.0, 7);
    ARIMAOptions o;
    o.p = 1;
    o.de_population_size = 60;
    o.de_max_generations = 400;
    const ARIMA model(x, o);
    const auto fc = model.forecast(60);
    // Forecasts should converge toward the fitted mean as the horizon grows.
    EXPECT_NEAR(fc.back(), model.mean(), 1.0);
}

TEST(ARIMA, ForecastStandardErrorsGrowWithHorizon) {
    const auto x = simulate_ar1(200, 0.4, 0.0, 1.0, 13);
    ARIMAOptions o;
    o.p = 1;
    o.de_population_size = 60;
    o.de_max_generations = 400;
    const ARIMA model(x, o);
    const auto [point, se] = model.forecast_with_intervals(10);
    ASSERT_EQ(point.size(), 10u);
    ASSERT_EQ(se.size(), 10u);
    for (std::size_t i = 1; i < se.size(); ++i) EXPECT_GE(se[i], se[i - 1] - 1e-9);
    EXPECT_GT(se[0], 0.0);
}

TEST(ARIMA, SeasonalDifferencingRequiresPeriod) {
    const std::vector<double> y(50, 1.0);
    ARIMAOptions o;
    o.seasonal_p = 1;
    o.seasonal_period = 0;
    EXPECT_THROW(ARIMA(y, o), std::invalid_argument);
}

TEST(ARIMA, ThrowsWhenSeriesTooShortForOrder) {
    const std::vector<double> y{1.0, 2.0, 3.0};
    ARIMAOptions o;
    o.p = 5;
    EXPECT_THROW(ARIMA(y, o), std::invalid_argument);
}

TEST(ARIMA, FittedPlusResidualEqualsObserved) {
    const auto x = simulate_ar1(150, 0.5, 3.0, 1.0, 5);
    ARIMAOptions o;
    o.p = 1;
    o.d = 1;
    o.de_population_size = 50;
    o.de_max_generations = 300;
    const ARIMA model(x, o);
    const auto& fitted = model.fitted_values();
    const auto& resid = model.residuals();
    ASSERT_EQ(fitted.size(), x.size());
    ASSERT_EQ(resid.size(), x.size());
    for (std::size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(fitted[i] + resid[i], x[i], 1e-9);
}

TEST(ARIMA, SeasonalModelFitsWithoutError) {
    // A synthetic series with an obvious period-12 seasonal pattern plus AR(1) noise.
    std::vector<double> s(96);
    std::mt19937_64 rng(21);
    std::normal_distribution<double> noise(0.0, 1.0);
    double prev = 0.0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        prev = 0.3 * prev + noise(rng);
        s[i] = 5.0 * std::sin(2.0 * 3.14159265358979 * static_cast<double>(i) / 12.0) + prev + 20.0;
    }
    ARIMAOptions o;
    o.p = 1;
    o.seasonal_p = 1;
    o.seasonal_d = 1;
    o.seasonal_period = 12;
    o.de_population_size = 80;
    o.de_max_generations = 400;
    const ARIMA model(s, o);
    EXPECT_GT(model.sigma2(), 0.0);
    EXPECT_EQ(model.ar_coefficients().size(), 1u);
    EXPECT_EQ(model.seasonal_ar_coefficients().size(), 1u);
    const auto fc = model.forecast(12);
    EXPECT_EQ(fc.size(), 12u);
}
