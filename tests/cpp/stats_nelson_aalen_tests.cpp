#include <datamunge/stats/nelson_aalen.hpp>

#include <datamunge/random/distributions.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_nelson_aalen_data.inc"

std::vector<SurvivalObservation> records(const std::vector<double>& t, const std::vector<double>& s) {
    std::vector<SurvivalObservation> r;
    for (std::size_t i = 0; i < t.size(); ++i)
        r.push_back({t[i],s[i] == 1});
    return r;
}

// References: R survival 3.8.6 survfit on lung and a simulated delayed-entry sample.
TEST(NelsonAalen, MatchesSurvfitOnLung) {
    const auto obs = records(lung_time,lung_status);
    const auto na = nelson_aalen(obs);
    ASSERT_EQ(na.size(),a_time.size());
    for (std::size_t i = 0; i < na.size(); ++i) {
        EXPECT_EQ(na[i].time,a_time[i]);
        EXPECT_EQ(double(na[i].at_risk),a_risk[i]);
        EXPECT_EQ(double(na[i].events),a_event[i]);
        EXPECT_EQ(double(na[i].censored),a_censor[i]);
        EXPECT_NEAR(na[i].cumulative_hazard,a_cumhaz[i],1e-13);
        EXPECT_NEAR(na[i].standard_error,a_se[i],1e-13);
        // Log intervals by their formula.
        const double z = datamunge::random::normal_quantile(.975), h = na[i].cumulative_hazard;
        EXPECT_NEAR(na[i].upper,h*std::exp(z*a_se[i]/h),1e-12);
    }
    const auto fh = nelson_aalen(obs,{.ties=HazardTies::fleming_harrington});
    for (std::size_t i = 0; i < fh.size(); ++i) {
        EXPECT_NEAR(fh[i].cumulative_hazard,b_cumhaz[i],1e-13);
        EXPECT_NEAR(fh[i].standard_error,b_se[i],1e-13);
    }
    // Plain intervals are R's log-scale intervals for exp(-H): S exp(-+z se).
    const auto plain = nelson_aalen(obs,{.interval=HazardInterval::plain,.confidence=.9});
    for (std::size_t i = 0; i < plain.size(); ++i) {
        EXPECT_NEAR(plain[i].survival,s_surv[i],1e-14);
        EXPECT_NEAR(std::exp(-plain[i].upper),s_lower[i],1e-13);
        EXPECT_NEAR(std::min(1.0,std::exp(-plain[i].lower)),s_upper[i],1e-13);
    }
}

TEST(NelsonAalen, DelayedEntryMatchesSurvfit) {
    const auto obs = records(e_time,e_status);
    const auto na = nelson_aalen(obs,{.entry=e_entry});
    ASSERT_EQ(na.size(),e_times.size());
    for (std::size_t i = 0; i < na.size(); ++i) {
        EXPECT_EQ(na[i].time,e_times[i]);
        EXPECT_EQ(double(na[i].at_risk),e_risk[i]);
        EXPECT_NEAR(na[i].cumulative_hazard,e_cumhaz[i],1e-13);
        EXPECT_NEAR(na[i].standard_error,e_se[i],1e-13);
    }
    const auto fh = nelson_aalen(obs,{.ties=HazardTies::fleming_harrington,.entry=e_entry});
    for (std::size_t i = 0; i < fh.size(); ++i) {
        EXPECT_NEAR(fh[i].cumulative_hazard,e2_cumhaz[i],1e-13);
        EXPECT_NEAR(fh[i].standard_error,e2_se[i],1e-13);
    }
}

TEST(NelsonAalen, Validation) {
    EXPECT_THROW((void)nelson_aalen({}),std::invalid_argument);
    EXPECT_THROW((void)nelson_aalen({{-1,true}}),std::invalid_argument);
    EXPECT_THROW((void)nelson_aalen({{1,true},{2,false}},{.entry={0}}),std::invalid_argument);
    EXPECT_THROW((void)nelson_aalen({{1,true},{2,false}},{.entry={1,0}}),std::invalid_argument);
    EXPECT_THROW((void)nelson_aalen({{1,true}},{.confidence=1}),std::invalid_argument);
    const auto none = nelson_aalen({{1,false},{2,false}});
    EXPECT_EQ(none.back().cumulative_hazard,0.0);
}

} // namespace
