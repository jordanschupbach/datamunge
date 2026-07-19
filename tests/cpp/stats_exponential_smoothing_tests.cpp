#include <gtest/gtest.h>

#include <datamunge/stats/exponential_smoothing.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using datamunge::stats::ExponentialSmoothing;
using datamunge::stats::ExponentialSmoothingOptions;
using datamunge::stats::SeasonalType;
using datamunge::stats::TrendType;

// This module's Holt-Winters implementation was cross-checked interactively against base R's
// HoltWinters() on a synthetic seasonal series with trend: fitted alpha/gamma and 5-step
// forecasts matched R within a few percent (a looser match than ARIMA's, expected since the
// two implementations use different initial-state heuristics and only the smoothing
// parameters are jointly optimized). The tests below check closed-form/self-consistency
// properties instead, which hold regardless of initialization differences.

TEST(ExponentialSmoothing, AlphaOneReducesToNaiveForecast) {
    const std::vector<double> y{4.0, 7.0, 2.0, 9.0, 5.0, 8.0, 1.0, 6.0};
    ExponentialSmoothingOptions o;
    o.alpha = 1.0; // level jumps to the latest observation every step
    const ExponentialSmoothing model(y, o);

    const auto& fitted = model.fitted_values();
    // With alpha == 1, the one-step-ahead fitted value at time t is exactly y[t-1].
    for (std::size_t t = 1; t < y.size(); ++t) EXPECT_NEAR(fitted[t], y[t - 1], 1e-9);

    const auto fc = model.forecast(3);
    for (const double v : fc) EXPECT_NEAR(v, y.back(), 1e-9);
}

TEST(ExponentialSmoothing, ConstantSeriesIsFittedExactlyRegardlessOfAlpha) {
    const std::vector<double> y(20, 7.0);
    ExponentialSmoothingOptions o;
    const ExponentialSmoothing model(y, o);
    for (const double v : model.forecast(5)) EXPECT_NEAR(v, 7.0, 1e-6);
    EXPECT_NEAR(model.sse(), 0.0, 1e-6);
}

TEST(ExponentialSmoothing, LinearTrendIsExtrapolatedForward) {
    std::vector<double> y(30);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = 2.0 + 1.5 * static_cast<double>(i);
    ExponentialSmoothingOptions o;
    o.trend = TrendType::Additive;
    o.de_population_size = 40;
    o.de_max_generations = 250;
    const ExponentialSmoothing model(y, o);

    const auto fc = model.forecast(5);
    ASSERT_EQ(fc.size(), 5u);
    // A noiseless linear series should extrapolate very close to the true continuation.
    for (std::size_t h = 0; h < fc.size(); ++h) {
        const double expected = 2.0 + 1.5 * static_cast<double>(y.size() + h);
        EXPECT_NEAR(fc[h], expected, 1.0);
    }
}

TEST(ExponentialSmoothing, SeasonalForecastRepeatsThePattern) {
    // A clean period-4 seasonal pattern with no trend or noise.
    const std::vector<double> pattern{10.0, 20.0, 15.0, 5.0};
    std::vector<double> y;
    for (int cycle = 0; cycle < 6; ++cycle)
        for (const double v : pattern) y.push_back(v);

    ExponentialSmoothingOptions o;
    o.seasonal = SeasonalType::Additive;
    o.seasonal_period = 4;
    o.de_population_size = 40;
    o.de_max_generations = 250;
    const ExponentialSmoothing model(y, o);

    const auto fc = model.forecast(8); // two full cycles ahead
    ASSERT_EQ(fc.size(), 8u);
    for (std::size_t h = 0; h < fc.size(); ++h) EXPECT_NEAR(fc[h], pattern[h % 4], 1.0);
}

TEST(ExponentialSmoothing, FittedPlusResidualEqualsObserved) {
    std::vector<double> y{3.0, 5.0, 4.0, 8.0, 7.0, 9.0, 6.0, 10.0, 9.0, 12.0};
    ExponentialSmoothingOptions o;
    o.trend = TrendType::Additive;
    const ExponentialSmoothing model(y, o);
    const auto& fitted = model.fitted_values();
    const auto& resid = model.residuals();
    ASSERT_EQ(fitted.size(), y.size());
    ASSERT_EQ(resid.size(), y.size());
    for (std::size_t i = 0; i < y.size(); ++i) EXPECT_NEAR(fitted[i] + resid[i], y[i], 1e-9);
}

TEST(ExponentialSmoothing, RejectsSeasonalPeriodLessThanTwo) {
    const std::vector<double> y(20, 1.0);
    ExponentialSmoothingOptions o;
    o.seasonal = SeasonalType::Additive;
    o.seasonal_period = 1;
    EXPECT_THROW(ExponentialSmoothing(y, o), std::invalid_argument);
}

TEST(ExponentialSmoothing, ThrowsWhenSeriesTooShortForSeasonalPeriod) {
    const std::vector<double> y{1.0, 2.0, 3.0};
    ExponentialSmoothingOptions o;
    o.seasonal = SeasonalType::Additive;
    o.seasonal_period = 12;
    EXPECT_THROW(ExponentialSmoothing(y, o), std::invalid_argument);
}

TEST(ExponentialSmoothing, DampedTrendFlattensFasterThanUndamped) {
    std::vector<double> y(30);
    for (std::size_t i = 0; i < y.size(); ++i) y[i] = 2.0 + 1.0 * static_cast<double>(i);

    ExponentialSmoothingOptions undamped;
    undamped.trend = TrendType::Additive;
    undamped.de_population_size = 40;
    undamped.de_max_generations = 250;
    const ExponentialSmoothing undamped_model(y, undamped);

    ExponentialSmoothingOptions damped;
    damped.trend = TrendType::AdditiveDamped;
    damped.phi = 0.7; // pin damping so this checks the forecast shape, not a fitted phi
    damped.de_population_size = 40;
    damped.de_max_generations = 250;
    const ExponentialSmoothing damped_model(y, damped);

    const auto undamped_fc = undamped_model.forecast(20);
    const auto damped_fc = damped_model.forecast(20);
    // The damped forecast's total rise over the horizon should be smaller than the
    // undamped (linear) forecast's, since the damped trend contribution saturates.
    const double undamped_rise = undamped_fc.back() - undamped_fc.front();
    const double damped_rise = damped_fc.back() - damped_fc.front();
    EXPECT_LT(damped_rise, undamped_rise);
}

TEST(ExponentialSmoothing, PinnedSmoothingParametersAreRespected) {
    const std::vector<double> y{4.0, 7.0, 2.0, 9.0, 5.0, 8.0, 1.0, 6.0, 3.0, 7.0};
    ExponentialSmoothingOptions o;
    o.alpha = 0.3;
    const ExponentialSmoothing model(y, o);
    EXPECT_DOUBLE_EQ(model.alpha(), 0.3);
}
