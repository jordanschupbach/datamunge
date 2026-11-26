#include <datamunge/causal/staggered_did.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using datamunge::causal::AggregatedEffect;
using datamunge::causal::DidBasePeriod;
using datamunge::causal::DidControlGroup;
using datamunge::causal::staggered_did;
using datamunge::causal::StaggeredDid;
using datamunge::causal::StaggeredDidOptions;
const double na = std::numeric_limits<double>::quiet_NaN();
#include "causal_did_data.inc"

void expect_value(double got, double want, const std::string& what) {
    if (std::isnan(want))
        EXPECT_TRUE(std::isnan(got)) << what;
    else
        EXPECT_NEAR(got, want, 1e-9 * (1 + std::abs(want))) << what;
}

void expect_did(const StaggeredDid& r, const std::vector<double>& gt, const std::vector<double>& dyn,
                const std::vector<double>& dyn_overall, const std::vector<double>& grp,
                const std::vector<double>& grp_overall, const std::vector<double>& simple) {
    ASSERT_EQ(r.group_time.size() * 4, gt.size());
    for (std::size_t k = 0; k < r.group_time.size(); ++k) {
        const auto& e = r.group_time[k];
        const auto label = "ATT(" + std::to_string(e.group) + ", " + std::to_string(e.time) + ")";
        EXPECT_EQ(e.group, gt[4 * k]) << label;
        EXPECT_EQ(e.time, gt[4 * k + 1]) << label;
        expect_value(e.att, gt[4 * k + 2], label);
        expect_value(e.standard_error, gt[4 * k + 3], label + " se");
    }
    const auto check = [](const std::vector<AggregatedEffect>& got, const std::vector<double>& want,
                          const std::string& what) {
        ASSERT_EQ(got.size() * 3, want.size()) << what;
        for (std::size_t k = 0; k < got.size(); ++k) {
            EXPECT_EQ(got[k].key, want[3 * k]) << what;
            expect_value(got[k].att, want[3 * k + 1], what + " att");
            expect_value(got[k].standard_error, want[3 * k + 2], what + " se");
        }
    };
    check(r.event_study, dyn, "event study");
    check(r.by_group, grp, "group");
    expect_value(r.dynamic_overall.att, dyn_overall[0], "dynamic overall");
    expect_value(r.dynamic_overall.standard_error, dyn_overall[1], "dynamic overall se");
    expect_value(r.group_overall.att, grp_overall[0], "group overall");
    expect_value(r.group_overall.standard_error, grp_overall[1], "group overall se");
    expect_value(r.simple.att, simple[0], "simple");
    expect_value(r.simple.standard_error, simple[1], "simple se");
}
} // namespace

TEST(CausalStaggeredDid, MatchesDidPackage) {
    expect_did(staggered_did(m_unit, m_year, m_cohort, m_lemp), d_never_gt, d_never_dyn, d_never_dyn_overall,
               d_never_grp, d_never_grp_overall, d_never_simple);
    expect_did(staggered_did(m_unit, m_year, m_cohort, m_lemp, {.control_group = DidControlGroup::not_yet_treated}),
               d_notyet_gt, d_notyet_dyn, d_notyet_dyn_overall, d_notyet_grp, d_notyet_grp_overall, d_notyet_simple);
    expect_did(staggered_did(m_unit, m_year, m_cohort, m_lemp, {.base_period = DidBasePeriod::universal}),
               d_univ_gt, d_univ_dyn, d_univ_dyn_overall, d_univ_grp, d_univ_grp_overall, d_univ_simple);
    expect_did(staggered_did(m_unit, m_year, m_cohort, m_lemp, {.anticipation = 1}), d_antic_gt, d_antic_dyn,
               d_antic_dyn_overall, d_antic_grp, d_antic_grp_overall, d_antic_simple);
}

TEST(CausalStaggeredDid, ValidatesPanel) {
    EXPECT_THROW((void)staggered_did({1, 1, 2}, {1, 2, 1}, {2, 2, 0}, {0, 1, 0}), std::invalid_argument);
    EXPECT_THROW((void)staggered_did({1, 1, 2, 2}, {1, 2, 1, 2}, {2, 0, 0, 0}, {0, 1, 0, 1}), std::invalid_argument);
    EXPECT_THROW((void)staggered_did({1, 1, 2, 2}, {1, 2, 1, 2}, {0, 0, 0, 0}, {0, 1, 0, 1}), std::invalid_argument);
}
