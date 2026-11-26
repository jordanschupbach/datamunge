#include <datamunge/stats/equivalence.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

const std::vector<double> x{5.1,4.8,5.3,5.0,4.9,5.4,4.7};
const std::vector<double> y{5.0,5.2,4.7,5.1,4.9,4.6};

// Reference values: TOSTER t_TOST(x, y, eqb = c(-0.4, 0.4)) in R 4.5.3 (Welch and var.equal = TRUE).
TEST(Equivalence, TwoSampleTostMatchesTOSTER) {
    const auto welch = tost_mean_difference(x,y,{.lower_margin=-.4,.upper_margin=.4});
    EXPECT_NEAR(welch.estimate,.111904761904762,1e-12);
    EXPECT_NEAR(welch.std_error,.135396453227126,1e-12);
    EXPECT_NEAR(welch.df,10.9508351466215,1e-10);
    EXPECT_NEAR(welch.t_lower,3.780784132107567,1e-10);
    EXPECT_NEAR(welch.t_upper,-2.127790139465193,1e-10);
    EXPECT_NEAR(welch.p_lower,.0015333679705376,1e-8);
    EXPECT_NEAR(welch.p_upper,.0284532598069308,1e-8);
    EXPECT_NEAR(welch.p_value,.0284532598069308,1e-8);
    EXPECT_NEAR(welch.conf_low,-.131351609761035,1e-8);
    EXPECT_NEAR(welch.conf_high,.355161133570558,1e-8);
    EXPECT_TRUE(welch.equivalent);

    const auto pooled = tost_mean_difference(x,y,{.lower_margin=-.4,.upper_margin=.4,.equal_variance=true});
    EXPECT_DOUBLE_EQ(pooled.df,11);
    EXPECT_NEAR(pooled.t_lower,3.74892302362059,1e-10);
    EXPECT_NEAR(pooled.p_upper,.02929558221904261,1e-8);
    // A narrower margin is not established.
    EXPECT_FALSE(tost_mean_difference(x,y,{.lower_margin=-.25,.upper_margin=.25}).equivalent);
}

TEST(Equivalence, OneSampleTostMatchesTOSTER) {
    const auto r = tost_mean({.2,-.1,.4,.1,0,.3,-.2,.1},{.lower_margin=-.5,.upper_margin=.5});
    EXPECT_NEAR(r.t_lower,8.48528137423857,1e-10);
    EXPECT_NEAR(r.t_upper,-5.65685424949238,1e-10);
    EXPECT_NEAR(r.p_lower,3.12321544709271e-05,1e-10);
    EXPECT_NEAR(r.p_value,3.84490774505392e-04,1e-10);
    EXPECT_NEAR(r.conf_low,-.0339669379150093,1e-9);
    EXPECT_NEAR(r.conf_high,.233966937915009,1e-9);
}

// Reference values: restricted maximum likelihood by optimize() on the binomial likelihood along
// p1 - p2 = delta, and the score interval by uniroot(), in R 4.5.3.
TEST(Equivalence, FarringtonManningMatchesDirectMaximization) {
    const auto r = noninferiority_risk_difference(85,100,88,100,.1);
    EXPECT_NEAR(r.restricted_p1,.8047901254475002,1e-7);
    EXPECT_NEAR(r.restricted_p2,.9047901254475001,1e-7);
    EXPECT_NEAR(r.statistic,1.4192975727819779,1e-6);
    EXPECT_NEAR(r.p_value,.0779061397355622,1e-6);
    EXPECT_FALSE(r.noninferior);

    const auto s = noninferiority_risk_difference(170,200,150,180,.08);
    EXPECT_NEAR(s.restricted_p1,.7946519174528147,1e-7);
    EXPECT_NEAR(s.statistic,2.5607717241913703,1e-6);
    EXPECT_NEAR(s.p_value,.0052219977670831,1e-7);
    EXPECT_TRUE(s.noninferior);
    EXPECT_NEAR(s.conf_low,-.0569684570483820,1e-7);
    EXPECT_NEAR(s.conf_high,.0919036058803268,1e-7);
    EXPECT_GT(s.conf_low,-.08); // Rejection agrees with the interval.
}

TEST(Equivalence, WaldRiskDifferenceAndEdgeCases) {
    const auto w = noninferiority_risk_difference(85,100,88,100,.1,.025,RiskDifferenceMethod::wald);
    const double se = std::sqrt(.85*.15/100+.88*.12/100);
    EXPECT_NEAR(w.statistic,(-.03+.1)/se,1e-12);
    EXPECT_NEAR(w.conf_low,-.03-1.959963984540054*se,1e-9);
    // All successes in both groups: the score test still works, the Wald variance is zero.
    const auto all = noninferiority_risk_difference(50,50,50,50,.1);
    EXPECT_GT(all.statistic,0);
    EXPECT_LE(all.conf_high,1.0);
    EXPECT_THROW((void)noninferiority_risk_difference(50,50,50,50,.1,.025,RiskDifferenceMethod::wald),
                 std::invalid_argument);
    EXPECT_THROW((void)noninferiority_risk_difference(5,4,1,4,.1),std::invalid_argument);
    EXPECT_THROW((void)noninferiority_risk_difference(1,4,1,4,1.0),std::invalid_argument);
    EXPECT_THROW((void)tost_mean({1,2,3},{.lower_margin=1,.upper_margin=-1}),std::invalid_argument);
    EXPECT_THROW((void)tost_mean({1,1,1},{.lower_margin=-1,.upper_margin=1}),std::invalid_argument);
    EXPECT_THROW((void)tost_mean({1,2},{.lower_margin=-1,.upper_margin=1,.alpha=.5}),std::invalid_argument);
}

} // namespace
