#include <datamunge/stats/survival.hpp>
#include <datamunge/stats/turnbull.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_turnbull_data.inc"

std::vector<CensoredTime> records(const std::vector<double>& left, const std::vector<double>& right,
                                  const std::vector<double>& keep = {}, double label = 0) {
    std::vector<CensoredTime> t;
    for (std::size_t i = 0; i < left.size(); ++i)
        if (keep.empty() || keep[i] == label)
            t.push_back({left[i],right[i]});
    return t;
}

void expect_matches(const TurnbullEstimate& e, const std::vector<double>& lo, const std::vector<double>& hi,
                    const std::vector<double>& closed, const std::vector<double>& p) {
    EXPECT_TRUE(e.converged);
    ASSERT_EQ(e.intervals.size(),lo.size());
    for (std::size_t j = 0; j < lo.size(); ++j) {
        EXPECT_EQ(e.intervals[j].lower,lo[j]);
        EXPECT_EQ(e.intervals[j].upper,hi[j]);
        EXPECT_EQ(e.intervals[j].lower_closed,closed[j] == 1);
        EXPECT_NEAR(e.intervals[j].mass,p[j],1e-7);
    }
}

// References: interval 1.1.1.0 icfit with epsilon 1e-12 on the breast cosmesis data.
TEST(Turnbull, MatchesIcfitOnBreastCosmesis) {
    expect_matches(turnbull(records(b_left,b_right)),all_lo,all_hi,all_lclosed,all_p);
    expect_matches(turnbull(records(b_left,b_right,b_rad,1)),rad_lo,rad_hi,rad_lclosed,rad_p);
    expect_matches(turnbull(records(b_left,b_right,b_rad,0)),rc_lo,rc_hi,rc_lclosed,rc_p);
}

TEST(Turnbull, MatchesIcfitWithExactTimes) {
    const auto e = turnbull(records(x_left,x_right));
    expect_matches(e,x_lo,x_hi,x_lclosed,x_p);
    EXPECT_LE(e.kkt_violation,1e-10);
}

TEST(Turnbull, ReducesToKaplanMeier) {
    const std::vector<SurvivalObservation> obs{{1,true},{2,false},{2,true},{3,true},{4,false},{5,true},{6,false}};
    std::vector<CensoredTime> t;
    for (const auto& o : obs)
        t.push_back(o.event ? CensoredTime::exact(o.time) : CensoredTime::right(o.time));
    const auto e = turnbull(t);
    const auto km = kaplan_meier(obs);
    for (const auto& point : km.points)
        EXPECT_NEAR(e.survival(point.time),point.survival,1e-9) << point.time;
}

TEST(Turnbull, Validation) {
    EXPECT_THROW((void)turnbull({}),std::invalid_argument);
    EXPECT_THROW((void)turnbull({CensoredTime::interval(3,2)}),std::invalid_argument);
    EXPECT_THROW((void)turnbull({{-1,2}}),std::invalid_argument);
}

} // namespace
