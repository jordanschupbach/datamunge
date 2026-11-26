#include <datamunge/stats/survival_tests.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_survival_tests_data.inc"

std::vector<SurvivalObservation> records() {
    std::vector<SurvivalObservation> r;
    for (std::size_t i = 0; i < time.size(); ++i)
        r.push_back({time[i],status[i] == 1});
    return r;
}

std::vector<int> labels(const std::vector<double>& v) { return {v.begin(),v.end()}; }

void expect_matches(const SurvivalGroupTest& t, const std::vector<double>& obs, const std::vector<double>& exp,
                    const std::vector<double>& var, double chisq) {
    ASSERT_EQ(t.observed.size(),obs.size());
    for (std::size_t k = 0; k < obs.size(); ++k) {
        EXPECT_NEAR(t.observed[k],obs[k],1e-10);
        EXPECT_NEAR(t.expected[k],exp[k],1e-10);
        for (std::size_t j = 0; j < obs.size(); ++j)
            EXPECT_NEAR(t.variance(j,k),var[k*obs.size()+j],1e-10);
    }
    EXPECT_NEAR(t.chi_square,chisq,1e-9);
}

// References: R survival 3.8.6 survdiff on lung (ph.ecog 0-2).
TEST(SurvivalGroupTest, MatchesSurvdiffForRho) {
    const auto obs = records();
    const auto ecog_groups = labels(ecog);
    const auto lr = survival_group_test(obs,ecog_groups);
    expect_matches(lr,r0_obs,r0_exp,r0_var,r0_chisq[0]);
    EXPECT_EQ(lr.degrees_of_freedom,2u);
    EXPECT_NEAR(lr.p_value,r0_p[0],1e-12);
    expect_matches(survival_group_test(obs,ecog_groups,{.rho=1}),r1_obs,r1_exp,r1_var,r1_chisq[0]);
    expect_matches(survival_group_test(obs,ecog_groups,{.rho=.5}),r05_obs,r05_exp,r05_var,r05_chisq[0]);
    EXPECT_NEAR(survival_group_test(obs,labels(sex)).chi_square,two_chisq[0],1e-9);
}

TEST(SurvivalGroupTest, StratifiedMatchesSurvdiff) {
    const auto t = survival_group_test(records(),labels(ecog),{.strata=labels(sex)});
    expect_matches(t,st_obs,st_exp,st_var,st_chisq[0]);
}

// Fleming-Harrington G(rho, gamma), compared with direct two-group sums computed in R.
TEST(SurvivalGroupTest, FlemingHarringtonGamma) {
    const auto obs = records();
    const struct { double rho, gamma, u, v, chisq; } cases[]{
        {0,0,19.690212103043695,39.903864274271363,9.7159625944504562},
        {0,1,5.4372166764713761,8.9782884147395983,3.2927573520989273},
        {1,1,3.1879306787626382,1.4000733913563144,7.2588351977397076},
        {.5,2,1.6600557527775375,1.0435695438540609,2.6407297132803085}};
    for (const auto& c : cases) {
        const auto t = survival_group_test(obs,labels(sex),{.rho=c.rho,.gamma=c.gamma});
        EXPECT_NEAR(t.observed[0]-t.expected[0],c.u,1e-10);
        EXPECT_NEAR(t.variance(0,0),c.v,1e-10);
        EXPECT_NEAR(t.chi_square,c.chisq,1e-9);
    }
}

TEST(SurvivalGroupTest, Validation) {
    const std::vector<SurvivalObservation> obs{{1,true},{2,true},{3,false}};
    EXPECT_THROW((void)survival_group_test(obs,{0,0,0}),std::invalid_argument);
    EXPECT_THROW((void)survival_group_test(obs,{0,1}),std::invalid_argument);
    EXPECT_THROW((void)survival_group_test(obs,{0,1,1},{.rho=-1}),std::invalid_argument);
    EXPECT_THROW((void)survival_group_test(obs,{0,1,1},{.strata={0}}),std::invalid_argument);
    const auto none = survival_group_test({{1,false},{2,false}},{0,1});
    EXPECT_EQ(none.degrees_of_freedom,0u);
    EXPECT_EQ(none.p_value,1.0);
}

} // namespace
