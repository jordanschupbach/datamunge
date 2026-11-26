#include <datamunge/stats/multistate.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_multistate_data.inc"

void expect_states(const AalenJohansenEstimate& e, const std::vector<double>& times, const std::vector<double>& p,
                   const std::vector<double>& se) {
    const std::size_t K = e.states;
    ASSERT_EQ(e.times.size(),times.size());
    for (std::size_t r = 0; r < times.size(); ++r) {
        EXPECT_EQ(e.times[r],times[r]);
        for (std::size_t s = 0; s < K; ++s) {
            EXPECT_NEAR(e.probabilities(r,s),p[r*K+s],1e-12);
            EXPECT_NEAR(e.standard_errors(r,s),se[r*K+s],1e-10);
        }
    }
}

// References: R survival 3.8.6 survfit with a factor status.
TEST(AalenJohansen, CompetingRisksMatchSurvfitOnMgus2) {
    std::vector<CompetingRiskObservation> obs;
    for (std::size_t i = 0; i < g_time.size(); ++i)
        obs.push_back({g_time[i],static_cast<int>(g_cause[i])});
    const auto e = cumulative_incidence(obs);
    EXPECT_EQ(e.states,3u);
    expect_states(e,g_times,g_p,g_se);
    for (std::size_t r = 0; r < g_risk.size(); ++r)
        EXPECT_EQ(e.at_risk(r,0),g_risk[r]);
}

TEST(AalenJohansen, IllnessDeathMatchesSurvfitWithIds) {
    std::vector<MultiStateRecord> records;
    for (std::size_t i = 0; i < i_id.size(); ++i)
        records.push_back({i_start[i],i_stop[i],static_cast<int>(i_from[i]),static_cast<int>(i_to[i]),
                           static_cast<int>(i_id[i])});
    const auto e = aalen_johansen(records,3);
    expect_states(e,i_times,i_p,i_se);
    // Chapman–Kolmogorov: P(0, t) = P(0, s) P(s, t), and p(t) = p(0) P(0, t).
    const auto a = e.transition_matrix(0,3), b = e.transition_matrix(3,6), c = e.transition_matrix(0,6);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double ab = 0;
            for (std::size_t k = 0; k < 3; ++k)
                ab += a(i,k)*b(k,j);
            EXPECT_NEAR(ab,c(i,j),1e-13);
        }
    const auto p6 = e.state_probabilities(6);
    for (std::size_t j = 0; j < 3; ++j)
        EXPECT_NEAR(p6[j],c(0,j),1e-13);
}

TEST(AalenJohansen, Validation) {
    EXPECT_THROW((void)aalen_johansen({}),std::invalid_argument);
    EXPECT_THROW((void)aalen_johansen({{1,1,0,1}}),std::invalid_argument);
    EXPECT_THROW((void)aalen_johansen({{0,1,0,-1}}),std::invalid_argument);
    EXPECT_THROW((void)aalen_johansen({{0,1,0,3}},2),std::invalid_argument);
    EXPECT_THROW((void)cumulative_incidence({{0,1}}),std::invalid_argument);
    const auto e = cumulative_incidence({{1,1},{2,0}});
    EXPECT_THROW((void)e.transition_matrix(2,1),std::invalid_argument);
}

} // namespace
