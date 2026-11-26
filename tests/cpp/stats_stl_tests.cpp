#include <datamunge/stats/stl.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_stl_data.inc"

void expect_close(const std::vector<double>& got, const std::vector<double>& want, double tol) {
    ASSERT_EQ(got.size(),want.size());
    for (std::size_t i = 0; i < got.size(); ++i)
        EXPECT_NEAR(got[i],want[i],tol) << "index " << i;
}

// Reference values: R 4.5.3 stats::stl.
TEST(Stl, PeriodicMatchesR) {
    const auto d = stl_decomposition(nottem_series,{.period=12});
    expect_close(d.seasonal,nottem_periodic_seasonal,1e-10);
    expect_close(d.trend,nottem_periodic_trend,1e-10);
    EXPECT_EQ(d.seasonal_window,10*nottem_series.size()+1);
    for (std::size_t i = 0; i < d.remainder.size(); ++i)
        EXPECT_NEAR(d.seasonal[i]+d.trend[i]+d.remainder[i],nottem_series[i],1e-12);
}

TEST(Stl, SeasonalWindowMatchesR) {
    const auto d = stl_decomposition(nottem_series,{.period=12,.seasonal_window=7});
    EXPECT_EQ(d.trend_window,23U);
    EXPECT_EQ(d.lowpass_window,13U);
    expect_close(d.seasonal,nottem_s7_seasonal,1e-10);
    expect_close(d.trend,nottem_s7_trend,1e-10);
}

// Reference: statsmodels STL(period=12, seasonal=13, trend=21, low_pass=13, seasonal_deg=0, robust=True,
// seasonal_jump=2, trend_jump=3, low_pass_jump=2).fit(inner_iter=1, outer_iter=15).
TEST(Stl, RobustMatchesStatsmodels) {
    const auto d = stl_decomposition(log_deaths,{.period=12,.seasonal_window=13,.robust=true});
    EXPECT_EQ(d.trend_window,21U);
    expect_close(d.seasonal,deaths_robust_seasonal,1e-10);
    expect_close(d.trend,deaths_robust_trend,1e-10);
    expect_close(d.weights,deaths_robust_weights,1e-10);
}

TEST(Stl, CustomSettingsMatchR) {
    const auto d = stl_decomposition(log_deaths,{.period=12,.seasonal_window=9,.seasonal_degree=1,.trend_window=25,
                                                 .lowpass_window=13,.seasonal_jump=1,.trend_jump=1,.lowpass_jump=1});
    expect_close(d.seasonal,deaths_custom_seasonal,1e-10);
    expect_close(d.trend,deaths_custom_trend,1e-10);
}

TEST(Stl, Validation) {
    EXPECT_THROW((void)stl_decomposition(std::vector<double>(24,1.0),{.period=12}),std::invalid_argument);
    EXPECT_THROW((void)stl_decomposition(nottem_series,{.period=1}),std::invalid_argument);
    EXPECT_THROW((void)stl_decomposition(nottem_series,{.period=12,.seasonal_window=7,.seasonal_degree=2}),
                 std::invalid_argument);
    auto bad = nottem_series;
    bad[5] = NAN;
    EXPECT_THROW((void)stl_decomposition(bad,{.period=12}),std::invalid_argument);
}

} // namespace
