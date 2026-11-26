#include <datamunge/stats/multistate.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_gray_data.inc"

std::vector<CompetingRiskObservation> records() {
    std::vector<CompetingRiskObservation> r;
    for (std::size_t i = 0; i < g_time.size(); ++i)
        r.push_back({g_time[i],static_cast<int>(g_cause[i])});
    return r;
}

std::vector<int> labels(const std::vector<double>& v) { return {v.begin(),v.end()}; }

// reference rows are (stat, p) per cause.
void expect_tests(const std::vector<GrayTest>& t, const std::vector<double>& ref) {
    ASSERT_EQ(t.size()*2,ref.size());
    for (std::size_t k = 0; k < t.size(); ++k) {
        EXPECT_EQ(t[k].cause,static_cast<int>(k)+1);
        EXPECT_NEAR(t[k].statistic,ref[2*k],1e-9*(1+ref[2*k]));
        EXPECT_NEAR(t[k].p_value,ref[2*k+1],1e-12);
    }
}

// References: cmprsk 2.2.12 cuminc on mgus2 (progression vs death).
TEST(GrayTest, MatchesCuminc) {
    const auto obs = records();
    expect_tests(gray_test(obs,labels(g_sex)),sex0);
    expect_tests(gray_test(obs,labels(g_sex),{.rho=1}),sex1);
    expect_tests(gray_test(obs,labels(g_age)),age0);
    expect_tests(gray_test(obs,labels(g_age),{.strata=labels(g_anemia)}),age_strat);
    expect_tests(gray_test(obs,labels(g_age),{.rho=.5,.strata=labels(g_anemia)}),age_rho);
    EXPECT_EQ(gray_test(obs,labels(g_age))[0].degrees_of_freedom,2u);
}

TEST(GrayTest, Validation) {
    const std::vector<CompetingRiskObservation> obs{{1,1},{2,2},{3,0}};
    EXPECT_THROW((void)gray_test(obs,{0,0,0}),std::invalid_argument);
    EXPECT_THROW((void)gray_test(obs,{0,1}),std::invalid_argument);
    EXPECT_THROW((void)gray_test(obs,{0,1,1},{.rho=-1}),std::invalid_argument);
    EXPECT_THROW((void)gray_test({{1,0},{2,0}},{0,1}),std::invalid_argument);
}

} // namespace
