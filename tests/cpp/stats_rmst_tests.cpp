#include <datamunge/stats/rmst.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_rmst_data.inc"

std::vector<SurvivalObservation> records(const std::vector<double>& t, const std::vector<double>& s) {
    std::vector<SurvivalObservation> r;
    for (std::size_t i = 0; i < t.size(); ++i)
        r.push_back({t[i],s[i] == 1});
    return r;
}

void expect_arm(const RmstEstimate& e, const std::vector<double>& ref) {
    EXPECT_NEAR(e.rmst,ref[0],1e-9);
    EXPECT_NEAR(e.standard_error,ref[1],1e-9);
    EXPECT_NEAR(e.lower,ref[2],1e-8);
    EXPECT_NEAR(e.upper,ref[3],1e-8);
}

void expect_contrasts(const RmstComparison& c, const std::vector<double>& u) {
    const RmstContrast* rows[]{&c.difference,&c.ratio,&c.time_lost_ratio};
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(rows[k]->estimate,u[4*k],1e-9);
        EXPECT_NEAR(rows[k]->lower,u[4*k+1],1e-8);
        EXPECT_NEAR(rows[k]->upper,u[4*k+2],1e-8);
        EXPECT_NEAR(rows[k]->p_value,u[4*k+3],1e-10);
    }
}

std::vector<int> labels(const std::vector<double>& v) { return {v.begin(),v.end()}; }

// References: survRM2 1.0.4 rmst2 (lung by sex at 90%, veteran by treatment at 95%).
TEST(Rmst, MatchesSurvRM2OnLung) {
    const auto obs = records(lung_time,lung_status);
    const auto c365 = rmst_compare(obs,labels(lung_sex),365,.9);
    expect_arm(c365.arms[0],lung365_arm0);
    expect_arm(c365.arms[1],lung365_arm1);
    EXPECT_NEAR(c365.arms[0].rmtl,lung365_rmtl0[0],1e-9);
    expect_contrasts(c365,lung365_unadj);
    const auto c700 = rmst_compare(obs,labels(lung_sex),700,.9);
    expect_arm(c700.arms[0],lung700_arm0);
    expect_arm(c700.arms[1],lung700_arm1);
    expect_contrasts(c700,lung700_unadj);
    EXPECT_EQ(c700.groups[0],1);
    EXPECT_EQ(c700.groups[1],2);
}

TEST(Rmst, MatchesSurvRM2OnVeteran) {
    const auto c = rmst_compare(records(vet_time,vet_status),labels(vet_trt),200);
    expect_arm(c.arms[0],vet_arm0);
    expect_arm(c.arms[1],vet_arm1);
    expect_contrasts(c,vet_unadj);
}

TEST(Rmst, SmallExampleAndValidation) {
    // S = 1 on [0, 1) and 3/4 on [1, 3), so RMST(3) = 1 + 0.75 * 2.
    const auto e = rmst({{1,true},{2,false},{3,true},{4,true}},3);
    EXPECT_NEAR(e.rmst,1+0.75*2,1e-14);
    EXPECT_NEAR(e.rmtl,0.5,1e-14);
    EXPECT_THROW((void)rmst({},1),std::invalid_argument);
    EXPECT_THROW((void)rmst({{1,true}},2),std::invalid_argument);
    EXPECT_THROW((void)rmst({{1,true}},0),std::invalid_argument);
    EXPECT_THROW((void)rmst({{1,true}},1,1.5),std::invalid_argument);
    EXPECT_THROW((void)rmst_compare({{1,true},{2,true}},{0,0},1),std::invalid_argument);
    EXPECT_THROW((void)rmst_compare({{1,true},{2,true},{3,true}},{0,1,2},1),std::invalid_argument);
    EXPECT_THROW((void)rmst_compare({{1,true},{5,true}},{0,1},3),std::invalid_argument);
}

} // namespace
