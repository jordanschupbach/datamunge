#include <datamunge/stats/gage_rr.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_gage_data.inc"
} // namespace

TEST(GageRR, FullModelMatchesAnovaAndLme4) {
    const auto g = gage_rr(g1_y, 10, 3, 3, {.tolerance = 12});
    ASSERT_FALSE(g.interaction_pooled);
    ASSERT_EQ(g.anova.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(g.anova[i].df, g1_full[i]);
        EXPECT_NEAR(g.anova[i].sum_of_squares, g1_full[4 + i], 1e-9);
    }
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(g.anova[i].f, g1_full[8 + i], 1e-9 * g1_full[8 + i]);
        EXPECT_NEAR(g.anova[i].p_value, g1_full[11 + i], 1e-12);
    }
    // Components: Total GRR, Repeatability, Reproducibility, Operator, OxP, Part, Total.
    const auto& c = g.components;
    EXPECT_NEAR(c[1].variance, g1_vc_full[0], 1e-12);
    EXPECT_NEAR(c[4].variance, g1_vc_full[1], 1e-12);
    EXPECT_NEAR(c[3].variance, g1_vc_full[2], 1e-12);
    EXPECT_NEAR(c[5].variance, g1_vc_full[3], 1e-12);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(g1_vc_full[i], g1_lmer[i], 1e-4 * std::max(1.0, g1_lmer[i])); // ANOVA = REML (lme4 tolerance)
    EXPECT_NEAR(c[2].variance, c[3].variance + c[4].variance, 1e-15);
    EXPECT_NEAR(c[0].variance, c[1].variance + c[2].variance, 1e-15);
    EXPECT_NEAR(c[6].variance, c[0].variance + c[5].variance, 1e-15);
    EXPECT_NEAR(c[0].percent_contribution + c[5].percent_contribution, 100, 1e-12);
    EXPECT_NEAR(c[0].study_variation, 6 * std::sqrt(c[0].variance), 1e-12);
    EXPECT_NEAR(c[0].percent_tolerance, 100 * c[0].study_variation / 12, 1e-12);
    EXPECT_EQ(g.distinct_categories, std::floor(1.41 * c[5].sd / c[0].sd));
}

TEST(GageRR, PoolsNonsignificantInteraction) {
    const auto g = gage_rr(g2_y, 10, 3, 2);
    ASSERT_TRUE(g.interaction_pooled);
    ASSERT_EQ(g.anova.size(), 3u);
    const auto& c = g.components;
    EXPECT_NEAR(c[1].variance, g2_vc_pooled[0], 1e-12);
    EXPECT_NEAR(c[3].variance, g2_vc_pooled[1], 1e-12);
    EXPECT_NEAR(c[5].variance, g2_vc_pooled[2], 1e-12);
    EXPECT_EQ(c[4].variance, 0.0);
    EXPECT_NEAR(g.anova[0].f, g2_vc_pooled[3], 1e-9 * g2_vc_pooled[3]);
    EXPECT_NEAR(g.anova[1].f, g2_vc_pooled[4], 1e-9 * g2_vc_pooled[4]);
    EXPECT_NEAR(g.anova[1].p_value, g2_vc_pooled[6], 1e-12);
    EXPECT_NEAR(c[1].variance, g2_lmer[0], 1e-4);
    EXPECT_NEAR(c[3].variance, g2_lmer[2], 1e-4);
    EXPECT_TRUE(std::isnan(c[0].percent_tolerance));
    // Never pooling gives the full model with the truncated interaction.
    const auto f = gage_rr(g2_y, 10, 3, 2, {.pool_interaction_above = 1});
    EXPECT_FALSE(f.interaction_pooled);
    EXPECT_NEAR(f.components[1].variance, g2_vc_full[0], 1e-12);
    EXPECT_EQ(f.components[4].variance, 0.0);
    EXPECT_NEAR(f.components[3].variance, g2_vc_full[2], 1e-12);
    EXPECT_THROW(gage_rr({1, 2, 3}, 1, 1, 3), std::invalid_argument);
    EXPECT_THROW(gage_rr(g2_y, 10, 3, 3), std::invalid_argument);
    EXPECT_THROW(gage_rr(g2_y, 10, 3, 2, {.tolerance = -1}), std::invalid_argument);
}
