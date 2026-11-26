#include <datamunge/stats/gpd.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::stats;
#include "stats_gpd_data.inc"
} // namespace

TEST(Gpd, DistributionFunctions) {
    EXPECT_NEAR(gpd_cdf(2, 7.4, 0.18), gp_dist[0], 1e-14);
    EXPECT_NEAR(gpd_cdf(10, 7.4, 0.18), gp_dist[1], 1e-14);
    EXPECT_NEAR(gpd_quantile(0.5, 7.4, 0.18), gp_dist[2], 1e-12);
    EXPECT_NEAR(gpd_quantile(0.999, 7.4, 0.18), gp_dist[3], 1e-10);
    EXPECT_NEAR(gpd_log_density(5, 7.4, 0.18), gp_dist[4], 1e-13);
    EXPECT_NEAR(gpd_quantile(0.9, 2, -0.3), gp_dist[5], 1e-13);
    EXPECT_EQ(gpd_cdf(-1, 1, 0.2), 0.0);
    EXPECT_EQ(gpd_cdf(10, 2, -0.3), 1.0); // above the endpoint 2 / 0.3
    EXPECT_NEAR(gpd_cdf(3, 2, 1e-12), gpd_cdf(3, 2, 0), 1e-12);
    EXPECT_THROW(gpd_quantile(1, 1, 0), std::invalid_argument);
}

TEST(Gpd, FitMatchesReferences) {
    const auto f = fit_gpd(gp_rain, 30);
    ASSERT_TRUE(f.converged);
    EXPECT_NEAR(f.scale, gp_fit[0], 1e-6);
    EXPECT_NEAR(f.shape, gp_fit[1], 1e-7);
    EXPECT_NEAR(f.scale_se / gp_fit[2], 1.0, 1e-5);
    EXPECT_NEAR(f.shape_se / gp_fit[3], 1.0, 1e-5);
    EXPECT_NEAR(f.covariance(0, 1) / gp_fit[4], 1.0, 1e-5);
    EXPECT_NEAR(f.log_likelihood, gp_fit[5], 1e-9);
    EXPECT_EQ(f.exceedances, static_cast<std::size_t>(gp_fit[6]));
    EXPECT_NEAR(f.exceedance_rate, gp_fit[7], 1e-15);
    EXPECT_GE(f.log_likelihood, gp_ismev[4]); // ismev's Nelder-Mead stops slightly short
    EXPECT_NEAR(f.log_likelihood, gp_ismev[4], 1e-5);
    const auto e = fit_gpd(gp_rain, 30, {.exponential = true});
    EXPECT_NEAR(e.scale, gp_mrl[3], 1e-8); // exponential MLE = mean excess exactly
    EXPECT_NEAR(e.scale, gp_exp[0], 2e-7);  // R's optimize is less precise
    EXPECT_NEAR(e.scale_se / gp_exp[1], 1.0, 1e-6);
    EXPECT_NEAR(e.log_likelihood, gp_exp[2], 1e-9);
    EXPECT_TRUE(std::isnan(e.shape_se));
}

TEST(Gpd, ReturnLevels) {
    const auto f = fit_gpd(gp_rain, 30);
    const auto r = gpd_return_level(f, gp_rain, 100, 365);
    EXPECT_NEAR(r.level, gp_rl100[0], 1e-5);
    EXPECT_NEAR(r.standard_error / gp_rl100[1], 1.0, 1e-5);
    EXPECT_NEAR(r.profile_lower, gp_rl100[2], 1e-5);
    EXPECT_NEAR(r.profile_upper, gp_rl100[3], 1e-5);
    const auto e = fit_gpd(gp_rain, 30, {.exponential = true});
    const auto re = gpd_return_level(e, gp_rain, 10, 365);
    EXPECT_NEAR(re.level, 30 + gp_mrl[3] * std::log(3650 * gp_fit[7]), 1e-7);
    EXPECT_NEAR(re.level, gp_rl10_exp[0], 1e-6);
    EXPECT_NEAR(re.standard_error / gp_rl10_exp[1], 1.0, 1e-6);
    EXPECT_NEAR(re.profile_lower, gp_rl10_exp[2], 1e-6);
    EXPECT_NEAR(re.profile_upper, gp_rl10_exp[3], 1e-6);
    EXPECT_THROW(gpd_return_level(f, gp_rain, 0.1, 365), std::invalid_argument); // m zeta < 1
    // Cluster-maxima form: the level at theta equals the formula with m zeta theta.
    const auto rt = gpd_return_level(f, gp_rain, 100, 365, 0.95, 0.6);
    const double mzt = 100 * 365 * f.exceedance_rate * 0.6;
    EXPECT_NEAR(rt.level, 30 + f.scale * (std::pow(mzt, f.shape) - 1) / f.shape, 1e-10);
    EXPECT_LT(rt.level, r.level);
    EXPECT_LT(rt.profile_lower, rt.level);
    EXPECT_GT(rt.profile_upper, rt.level);
    EXPECT_THROW(gpd_return_level(f, gp_rain, 100, 365, 0.95, 1.2), std::invalid_argument);
}

TEST(Gpd, ThresholdDiagnostics) {
    const auto d = threshold_diagnostics(gp_rain, {10, 30, 50, 200});
    ASSERT_EQ(d.size(), 3u); // no exceedances of 200
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(d[i].mean_excess, gp_mrl[3 * i], 1e-12);
        EXPECT_NEAR(d[i].mean_excess_lower, gp_mrl[3 * i + 1], 1e-12);
        EXPECT_EQ(d[i].exceedances, static_cast<std::size_t>(gp_mrl[3 * i + 2]));
    }
    EXPECT_NEAR(d[1].shape, gp_fit[1], 1e-7);
    EXPECT_NEAR(d[1].modified_scale, gp_fit[0] - 30 * gp_fit[1], 1e-5);
    EXPECT_TRUE(d[1].shape_lower < d[1].shape && d[1].shape < d[1].shape_upper);
    EXPECT_THROW(fit_gpd(gp_rain, 1000), std::invalid_argument);
}
