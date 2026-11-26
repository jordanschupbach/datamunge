#include <datamunge/stats/gev.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_gev_data.inc"
} // namespace

TEST(Gev, DistributionFunctions) {
    EXPECT_NEAR(gev_cdf(3.9, 3.87, 0.198, -0.05), gv_dist[0], 1e-14);
    EXPECT_NEAR(gev_cdf(4.6, 3.87, 0.198, -0.05), gv_dist[1], 1e-14);
    EXPECT_NEAR(gev_quantile(0.9, 3.87, 0.198, -0.05), gv_dist[2], 1e-13);
    EXPECT_NEAR(gev_quantile(0.999, 3.87, 0.198, -0.05), gv_dist[3], 1e-13);
    EXPECT_NEAR(gev_log_density(4.1, 3.87, 0.198, -0.05), gv_dist[4], 1e-13);
    EXPECT_NEAR(gev_cdf(4, 3.87, 0.198, 0), gv_dist[5], 1e-14);
    EXPECT_NEAR(gev_quantile(0.99, 3.87, 0.198, 0.3), gv_dist[6], 1e-13);
    // Continuity at xi = 0 and support limits.
    EXPECT_NEAR(gev_cdf(4.2, 3.87, 0.198, 1e-12), gev_cdf(4.2, 3.87, 0.198, 0), 1e-12);
    EXPECT_EQ(gev_cdf(10, 0, 1, -0.5), 1.0); // above the upper endpoint 2
    EXPECT_EQ(gev_cdf(-10, 0, 1, 0.5), 0.0); // below the lower endpoint -2
    EXPECT_EQ(gev_log_density(10, 0, 1, -0.5), -INFINITY);
    EXPECT_THROW(gev_cdf(0, 0, 0, 0), std::invalid_argument);
}

TEST(Gev, MaximumLikelihoodMatchesReferences) {
    for (const auto& [data, ref, gum] : {std::tuple{&gv_portpirie, &gv_portpirie_gev, &gv_portpirie_gumbel},
                                         std::tuple{&gv_fremantle, &gv_fremantle_gev, &gv_fremantle_gumbel}}) {
        const auto f = fit_gev(*data);
        ASSERT_TRUE(f.converged);
        EXPECT_NEAR(f.location, (*ref)[0], 1e-6);
        EXPECT_NEAR(f.scale, (*ref)[1], 1e-6);
        EXPECT_NEAR(f.shape, (*ref)[2], 1e-5);
        EXPECT_GE(f.log_likelihood, (*ref)[6] - 1e-9);
        EXPECT_NEAR(f.log_likelihood, (*ref)[6], 1e-8);
        if (!std::isnan((*ref)[3])) {
            EXPECT_NEAR(f.location_se / (*ref)[3], 1.0, 1e-4);
            EXPECT_NEAR(f.scale_se / (*ref)[4], 1.0, 1e-4);
        }
        EXPECT_NEAR(f.shape_se / (*ref)[5], 1.0, 1e-3);
        const auto g = fit_gev(*data, {.gumbel = true});
        EXPECT_NEAR(g.location, (*gum)[0], 1e-7);
        EXPECT_NEAR(g.scale, (*gum)[1], 1e-7);
        EXPECT_NEAR(g.location_se / (*gum)[2], 1.0, 1e-5);
        EXPECT_NEAR(g.scale_se / (*gum)[3], 1.0, 1e-5);
        EXPECT_NEAR(g.log_likelihood, (*gum)[4], 1e-9);
        EXPECT_EQ(g.shape, 0.0);
        EXPECT_TRUE(std::isnan(g.shape_se));
    }
    // ismev's Nelder-Mead agrees loosely.
    const auto f = fit_gev(gv_portpirie);
    EXPECT_NEAR(f.shape, gv_ismev[2], 1e-4);
    EXPECT_NEAR(f.log_likelihood, gv_ismev[6], 1e-7);
}

TEST(Gev, LMomentsAndGumbelTest) {
    for (const auto& [data, ref] : {std::pair{&gv_portpirie, &gv_portpirie_lmom}, std::pair{&gv_fremantle, &gv_fremantle_lmom}}) {
        const auto f = fit_gev(*data, {.method = GevMethod::l_moments});
        EXPECT_NEAR(f.location, (*ref)[0], 1e-12);
        EXPECT_NEAR(f.scale, (*ref)[1], 1e-12);
        EXPECT_NEAR(f.shape, (*ref)[2], 1e-12);
    }
    const auto t = gumbel_test(gv_portpirie);
    EXPECT_NEAR(t.statistic, 2 * (gv_portpirie_gev[6] - gv_portpirie_gumbel[4]), 1e-7);
    EXPECT_GT(t.p_value, 0.5);
    const auto u = gumbel_test(gv_fremantle);
    EXPECT_NEAR(u.statistic, 2 * (gv_fremantle_gev[6] - gv_fremantle_gumbel[4]), 1e-7);
    EXPECT_LT(u.p_value, 0.01);
}

TEST(Gev, ReturnLevels) {
    for (const auto& [data, ref, refg] :
         {std::tuple{&gv_portpirie, &gv_portpirie_rl100, &gv_portpirie_rl10_gumbel},
          std::tuple{&gv_fremantle, &gv_fremantle_rl100, &gv_fremantle_rl10_gumbel}}) {
        const auto f = fit_gev(*data);
        const auto r = gev_return_level(f, *data, 100);
        EXPECT_NEAR(r.level, (*ref)[0], 1e-5);
        EXPECT_NEAR(r.standard_error / (*ref)[1], 1.0, 1e-3);
        EXPECT_NEAR(r.profile_lower, (*ref)[2], 1e-5);
        EXPECT_NEAR(r.profile_upper, (*ref)[3], 1e-5);
        const auto g = fit_gev(*data, {.gumbel = true});
        const auto rg = gev_return_level(g, *data, 10);
        EXPECT_NEAR(rg.level, (*refg)[0], 1e-7);
        EXPECT_NEAR(rg.standard_error / (*refg)[1], 1.0, 1e-5);
        EXPECT_NEAR(rg.profile_lower, (*refg)[2], 1e-7);
        EXPECT_NEAR(rg.profile_upper, (*refg)[3], 1e-7);
    }
    const auto lm = fit_gev(gv_portpirie, {.method = GevMethod::l_moments});
    EXPECT_THROW(gev_return_level(lm, gv_portpirie, 100), std::invalid_argument);
    EXPECT_THROW(gev_return_level(fit_gev(gv_portpirie), gv_portpirie, 1), std::invalid_argument);
    EXPECT_THROW(fit_gev({1, 2}), std::invalid_argument);
    EXPECT_THROW(fit_gev({2, 2, 2}), std::invalid_argument);
}
