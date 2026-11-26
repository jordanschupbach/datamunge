#include <datamunge/stats/unit_root.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_unit_root_data.inc"

// Reference values: statsmodels 0.14.6 adfuller(x, maxlag = 3, regression, autolag = None) and
// adfuller(x, regression, autolag = "AIC" / "BIC").
TEST(UnitRoot, AdfFixedLagsMatchStatsmodels) {
    const AdfOptions fixed{.selection=LagSelection::fixed,.lags=3};
    auto o = fixed;
    o.trend = UnitRootTrend::none;
    auto r = augmented_dickey_fuller(random_walk,o);
    EXPECT_NEAR(r.statistic,-.4993451403565382,1e-10);
    EXPECT_NEAR(r.p_value,.496393187875898,1e-10);
    EXPECT_EQ(r.lags,3U);
    EXPECT_EQ(r.observations,146U);
    EXPECT_NEAR(r.critical.one_percent,-2.5812238525051603,1e-12);
    o.trend = UnitRootTrend::constant;
    r = augmented_dickey_fuller(random_walk,o);
    EXPECT_NEAR(r.statistic,-1.7782922027662516,1e-10);
    EXPECT_NEAR(r.p_value,.391291016055911,1e-10);
    EXPECT_NEAR(r.critical.five_percent,-2.881548071241103,1e-12);
    EXPECT_NEAR(r.critical.ten_percent,-2.577438765246763,1e-12);
    o.trend = UnitRootTrend::linear;
    r = augmented_dickey_fuller(random_walk,o);
    EXPECT_NEAR(r.statistic,-1.6690563876174116,1e-10);
    EXPECT_NEAR(r.p_value,.7642376498030773,1e-10);
    EXPECT_NEAR(r.critical.one_percent,-4.022154287383327,1e-12);
    r = augmented_dickey_fuller(stationary_ar,{.trend=UnitRootTrend::none,.selection=LagSelection::fixed,.lags=3});
    EXPECT_NEAR(r.statistic,-3.5993890018700734,1e-10);
    EXPECT_NEAR(r.p_value,.0003370297173496464,1e-12);
    r = augmented_dickey_fuller(trend_stationary,{.trend=UnitRootTrend::linear,.selection=LagSelection::fixed,.lags=3});
    EXPECT_NEAR(r.statistic,-3.6713395703455314,1e-10);
    EXPECT_NEAR(r.p_value,.024315356970402175,1e-10);
}

TEST(UnitRoot, AdfLagSelectionMatchesStatsmodels) {
    auto r = augmented_dickey_fuller(random_walk);
    EXPECT_EQ(r.lags,0U);
    EXPECT_EQ(r.observations,149U);
    EXPECT_NEAR(r.statistic,-2.432431616575272,1e-10);
    EXPECT_NEAR(r.p_value,.13279123045186558,1e-10);
    r = augmented_dickey_fuller(trend_stationary);
    EXPECT_EQ(r.lags,3U);
    EXPECT_NEAR(r.statistic,-1.871738155698232,1e-10);
    r = augmented_dickey_fuller(trend_stationary,{.trend=UnitRootTrend::linear,.selection=LagSelection::bic});
    EXPECT_EQ(r.lags,0U);
    EXPECT_NEAR(r.statistic,-4.913496846175772,1e-10);
    EXPECT_NEAR(r.p_value,.0002948451656759007,1e-12);
}

// Reference values: arch 8.0.0 PhillipsPerron(x, lags = 4, trend, test_type = "tau" / "rho").
TEST(UnitRoot, PhillipsPerronMatchesArch) {
    auto r = phillips_perron(random_walk,{.trend=UnitRootTrend::none,.lags=4});
    EXPECT_NEAR(r.statistic,-.8633952180075517,1e-10);
    EXPECT_NEAR(r.p_value,.34401766087267904,1e-10);
    EXPECT_NEAR(r.z_alpha,-1.249823081535781,1e-10);
    r = phillips_perron(random_walk,{.lags=4});
    EXPECT_NEAR(r.statistic,-2.386291155586568,1e-10);
    EXPECT_NEAR(r.p_value,.1456430703029239,1e-10);
    EXPECT_NEAR(r.z_alpha,-10.492228618312836,1e-9);
    EXPECT_EQ(r.observations,149U);
    r = phillips_perron(stationary_ar,{.trend=UnitRootTrend::linear,.lags=4});
    EXPECT_NEAR(r.statistic,-4.928140571832421,1e-10);
    EXPECT_NEAR(r.p_value,.00027753918108273197,1e-12);
    EXPECT_NEAR(r.z_alpha,-42.52326846732475,1e-9);
    // Default Newey-West lags: floor(4 (150/100)^(1/4)) = 4.
    EXPECT_EQ(phillips_perron(random_walk).lags,4U);
}

TEST(UnitRoot, TablesAndValidation) {
    EXPECT_EQ(unit_root_p_value(5,UnitRootTrend::constant),1.0);
    EXPECT_EQ(unit_root_p_value(-25,UnitRootTrend::linear),0.0);
    EXPECT_NEAR(unit_root_critical_values(UnitRootTrend::constant,1000000).five_percent,-2.86154,1e-5);
    EXPECT_THROW((void)augmented_dickey_fuller({1,2,3}),std::invalid_argument);
    EXPECT_THROW((void)augmented_dickey_fuller(std::vector<double>(50,1.0),{.selection=LagSelection::fixed,.lags=1}),
                 std::invalid_argument);
    EXPECT_THROW((void)augmented_dickey_fuller(random_walk,{.selection=LagSelection::fixed,.lags=148}),
                 std::invalid_argument);
    EXPECT_THROW((void)phillips_perron({1,2,NAN,4,5}),std::invalid_argument);
    EXPECT_THROW((void)unit_root_critical_values(UnitRootTrend::none,0),std::invalid_argument);
}

} // namespace

namespace {
using namespace datamunge::stats;

// Reference values: statsmodels 0.14.6 kpss(x, regression, nlags = "auto" / 4 / 14 (legacy)).
TEST(Kpss, MatchesStatsmodels) {
    auto r = kpss_test(random_walk);
    EXPECT_EQ(r.lags,8U);
    EXPECT_NEAR(r.statistic,.31666237168748335,1e-12);
    EXPECT_EQ(r.p_value,.1);
    EXPECT_TRUE(r.p_value_truncated);
    r = kpss_test(random_walk,{.lags=4});
    EXPECT_NEAR(r.statistic,.5222813100937338,1e-12);
    EXPECT_NEAR(r.p_value,.03664835358249239,1e-14);
    EXPECT_FALSE(r.p_value_truncated);
    r = kpss_test(random_walk,{.trend=UnitRootTrend::linear,.lags=14});
    EXPECT_NEAR(r.statistic,.18501604909856956,1e-12);
    EXPECT_NEAR(r.p_value,.021618981588036414,1e-14);
    r = kpss_test(stationary_ar,{.lags=4});
    EXPECT_NEAR(r.statistic,.4172863109754308,1e-12);
    EXPECT_NEAR(r.p_value,.06970417630369363,1e-14);
    r = kpss_test(trend_stationary,{.trend=UnitRootTrend::linear});
    EXPECT_EQ(r.lags,7U);
    EXPECT_NEAR(r.statistic,.07611538969565278,1e-12);
    r = kpss_test(trend_stationary);
    EXPECT_NEAR(r.statistic,1.5201007612587658,1e-11);
    EXPECT_EQ(r.p_value,.01);
    EXPECT_TRUE(r.p_value_truncated);
    EXPECT_EQ(r.critical.two_and_a_half_percent,.574);
    EXPECT_THROW((void)kpss_test(random_walk,{.trend=UnitRootTrend::none}),std::invalid_argument);
    EXPECT_THROW((void)kpss_test(std::vector<double>(20,3.0)),std::invalid_argument);
}

} // namespace
