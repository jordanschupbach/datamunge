#include <datamunge/stats/generalizability.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_generalizability_data.inc"
} // namespace

TEST(GStudy, MatchesAnovaEmsAndLme4) {
    const auto g = g_study(g3_y, {20, 4, 3});
    ASSERT_EQ(g.components.size(), 7u);
    for (std::size_t i = 0; i < 7; ++i) {
        EXPECT_NEAR(g.components[i].mean_square, g3_ms[i], 1e-10) << i;
        EXPECT_EQ(g.components[i].df, g3_df[i]) << i;
        EXPECT_NEAR(g.components[i].raw_variance, g3_sig[i], 1e-10) << i;
        EXPECT_NEAR(g.components[i].standard_error, g3_se[i], 1e-10) << i;
        EXPECT_NEAR(g.components[i].variance, g3_lmer[i], 1e-4) << i; // REML optimiser tolerance
    }
    EXPECT_EQ(g.components[3].factors, (std::vector<std::size_t>{0, 1}));
    double pct = 0;
    for (const auto& c : g.components)
        pct += c.percent;
    EXPECT_NEAR(pct, 100.0, 1e-10);
    const std::vector<std::vector<double>> sizes{{4, 3}, {2, 1}, {8, 6}};
    for (std::size_t s = 0; s < 3; ++s) {
        const auto d = d_study(g, sizes[s]);
        EXPECT_NEAR(d.generalizability, g3_d[4 * s], 1e-12) << s;
        EXPECT_NEAR(d.dependability, g3_d[4 * s + 1], 1e-12) << s;
        EXPECT_NEAR(d.relative_error, g3_d[4 * s + 2], 1e-12) << s;
        EXPECT_NEAR(d.absolute_error, g3_d[4 * s + 3], 1e-12) << s;
    }
}

TEST(GStudy, FourFactorsWithNegativeEstimates) {
    const auto g = g_study(g4_y, {6, 3, 2, 2});
    ASSERT_EQ(g.components.size(), 15u);
    bool negative = false;
    for (std::size_t i = 0; i < 15; ++i) {
        EXPECT_NEAR(g.components[i].mean_square, g4_ms[i], 1e-10) << i;
        EXPECT_NEAR(g.components[i].raw_variance, g4_sig[i], 1e-10) << i;
        EXPECT_NEAR(g.components[i].standard_error, g4_se[i], 1e-10) << i;
        EXPECT_NEAR(g.components[i].variance, std::max(0.0, g4_sig[i]), 1e-10) << i;
        negative = negative || g4_sig[i] < 0;
    }
    EXPECT_TRUE(negative);
}

TEST(GStudy, OneFacetEqualsCronbachAlpha) {
    const auto g = g_study(g1_y, {30, 6});
    EXPECT_NEAR(d_study(g, {6}).generalizability, g1_alpha[0], 1e-12);
    // More items raise reliability (Spearman-Brown).
    EXPECT_GT(d_study(g, {12}).generalizability, d_study(g, {6}).generalizability);
    EXPECT_THROW(g_study({1, 2, 3}, {3}), std::invalid_argument);
    EXPECT_THROW(g_study({1, 2, 3, 4}, {2, 3}), std::invalid_argument);
    EXPECT_THROW(d_study(g, {6, 2}), std::invalid_argument);
    EXPECT_THROW(d_study(g, {0}), std::invalid_argument);
}
