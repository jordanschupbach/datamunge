#include <datamunge/stats/diebold_mariano.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_diebold_mariano_data.inc"

void expect_matches(const DieboldMarianoTest& r, const std::vector<double>& ref) {
    EXPECT_NEAR(r.statistic,ref[0],1e-12*std::abs(ref[0]));
    EXPECT_NEAR(r.p_value,ref[1],1e-11*ref[1]);
}

// References: R forecast 9.0.1 dm.test.
TEST(DieboldMariano, MatchesRForecast) {
    expect_matches(diebold_mariano_test(e1,e2),dm_h1);
    expect_matches(diebold_mariano_test(e1,e2,{.horizon=3}),dm_h3_acf);
    expect_matches(diebold_mariano_test(e1,e2,{.horizon=3,.variance=DmVariance::bartlett}),dm_h3_bartlett);
    expect_matches(diebold_mariano_test(e1,e2,{.horizon=2,.power=1,.alternative=Alternative::Less}),dm_power1_less);
    expect_matches(diebold_mariano_test(e1,e2,{.alternative=Alternative::Greater}),dm_greater);
    const auto r = diebold_mariano_test(e1,e2,{.horizon=3});
    EXPECT_EQ(r.horizon_used,3u);
    EXPECT_NEAR(r.correction,std::sqrt((81.0-6+3.0/80*2)/80),1e-15);
}

TEST(DieboldMariano, NegativeVarianceFallsBackToHorizonOne) {
    const auto r = diebold_mariano_test(e3,e4,{.horizon=2});
    EXPECT_EQ(r.horizon_used,1u);
    expect_matches(r,dm_fallback);
    // The same differentials through the general-loss entry point.
    expect_matches(diebold_mariano_test_differentials(d_alt,{.horizon=2}),dm_fallback);
    // Bartlett weights keep the variance positive, so no fallback.
    EXPECT_EQ(diebold_mariano_test_differentials(d_alt,{.horizon=2,.variance=DmVariance::bartlett}).horizon_used,2u);
}

TEST(DieboldMariano, Validation) {
    EXPECT_THROW((void)diebold_mariano_test({1,2,3},{1,2}),std::invalid_argument);
    EXPECT_THROW((void)diebold_mariano_test({1,2},{1,2}),std::invalid_argument);
    EXPECT_THROW((void)diebold_mariano_test(e1,e2,{.horizon=0}),std::invalid_argument);
    EXPECT_THROW((void)diebold_mariano_test(e1,e2,{.horizon=81}),std::invalid_argument);
    EXPECT_THROW((void)diebold_mariano_test(e1,e2,{.power=0}),std::invalid_argument);
    EXPECT_THROW((void)diebold_mariano_test(e1,e1),std::invalid_argument);
    EXPECT_THROW((void)diebold_mariano_test_differentials({1,NAN,2}),std::invalid_argument);
}

} // namespace
