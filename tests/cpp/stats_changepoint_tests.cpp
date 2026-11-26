#include <datamunge/stats/changepoint.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

#include "stats_changepoint_data.inc"

void expect_matches(const ChangepointResult& r, const std::vector<double>& cpts, const std::vector<double>& pen,
                    const std::vector<double>& means, const std::vector<double>& vars) {
    ASSERT_EQ(r.changepoints.size(),cpts.size());
    for (std::size_t i = 0; i < cpts.size(); ++i)
        EXPECT_EQ(double(r.changepoints[i]),cpts[i]) << i;
    EXPECT_NEAR(r.penalty,pen[0],1e-12*pen[0]);
    ASSERT_EQ(r.means.size(),means.size());
    for (std::size_t i = 0; i < means.size(); ++i) {
        EXPECT_NEAR(r.means[i],means[i],1e-12*std::max(1.0,std::abs(means[i])));
        EXPECT_NEAR(r.variances[i],vars[i],1e-11*std::max(1.0,vars[i]));
    }
}

// References: R changepoint 2.3 with method = "PELT".
TEST(Pelt, NormalMeanMatchesR) {
    expect_matches(pelt_changepoints(mean_x),mean_mbic_cpts,mean_mbic_pen,mean_mbic_means,mean_mbic_vars);
    expect_matches(pelt_changepoints(mean_x,{.penalty=ChangePenalty::bic}),mean_bic_cpts,mean_bic_pen,mean_bic_means,
                   mean_bic_vars);
    expect_matches(pelt_changepoints(mean_x,{.penalty=ChangePenalty::aic}),mean_aic_cpts,mean_aic_pen,mean_aic_means,
                   mean_aic_vars);
    expect_matches(pelt_changepoints(mean_x,{.penalty=ChangePenalty::manual,.penalty_value=25,.min_segment_length=10}),
                   mean_manual_cpts,mean_manual_pen,mean_manual_means,mean_manual_vars);
    // Dividing the data by 0.8 is the same as declaring noise variance 0.64.
    const auto scaled = pelt_changepoints(mean_x,{.noise_variance=.64});
    ASSERT_EQ(scaled.changepoints.size(),mean_scaled_cpts.size());
    for (std::size_t i = 0; i < mean_scaled_cpts.size(); ++i)
        EXPECT_EQ(double(scaled.changepoints[i]),mean_scaled_cpts[i]);
}

TEST(Pelt, VarianceAndMeanVarianceMatchR) {
    expect_matches(pelt_changepoints(var_x,{.cost=ChangeCost::normal_variance}),var_mbic_cpts,var_mbic_pen,var_mbic_means,
                   var_mbic_vars);
    expect_matches(pelt_changepoints(var_x,{.cost=ChangeCost::normal_variance,.penalty=ChangePenalty::bic,.known_mean=0.0}),
                   var_known_cpts,var_known_pen,var_known_means,var_known_vars);
    expect_matches(pelt_changepoints(meanvar_x,{.cost=ChangeCost::normal_mean_variance}),meanvar_mbic_cpts,
                   meanvar_mbic_pen,meanvar_mbic_means,meanvar_mbic_vars);
    expect_matches(pelt_changepoints(meanvar_x,{.cost=ChangeCost::normal_mean_variance,
                                                .penalty=ChangePenalty::hannan_quinn,.min_segment_length=5}),
                   meanvar_hq_cpts,meanvar_hq_pen,meanvar_hq_means,meanvar_hq_vars);
}

TEST(Pelt, PoissonAndExponentialMatchR) {
    expect_matches(pelt_changepoints(pois_x,{.cost=ChangeCost::poisson}),pois_mbic_cpts,pois_mbic_pen,pois_mbic_means,
                   pois_mbic_vars);
    expect_matches(pelt_changepoints(exp_x,{.cost=ChangeCost::exponential}),exp_mbic_cpts,exp_mbic_pen,exp_mbic_means,
                   exp_mbic_vars);
}

// Pruning must not change the optimum: compare with unpruned optimal partitioning.
TEST(Pelt, AgreesWithExhaustiveOptimalPartitioning) {
    const std::vector<double> x(mean_x.begin(),mean_x.begin()+120);
    for (const double beta : {2.0, 6.0, 15.0}) {
        const auto r = pelt_changepoints(x,{.penalty=ChangePenalty::manual,.penalty_value=beta,.min_segment_length=3});
        const std::size_t n = x.size(), m = 3;
        std::vector<double> f(n+1,INFINITY);
        f[0] = -beta;
        for (std::size_t t = m; t <= n; ++t)
            for (std::size_t s = 0; s+m <= t; ++s) {
                if (s != 0 && s < m)
                    continue;
                double sum = 0, sq = 0;
                for (std::size_t i = s; i < t; ++i) {
                    sum += x[i];
                    sq += x[i]*x[i];
                }
                f[t] = std::min(f[t],f[s]+sq-sum*sum/double(t-s)+beta);
            }
        EXPECT_NEAR(r.cost,f[n],1e-9*std::abs(f[n])) << beta;
    }
}

TEST(Pelt, Validation) {
    EXPECT_THROW((void)pelt_changepoints({}),std::invalid_argument);
    EXPECT_THROW((void)pelt_changepoints({1,NAN,2}),std::invalid_argument);
    EXPECT_THROW((void)pelt_changepoints({1,2.5,3},{.cost=ChangeCost::poisson}),std::invalid_argument);
    EXPECT_THROW((void)pelt_changepoints({1,0,3},{.cost=ChangeCost::exponential}),std::invalid_argument);
    EXPECT_THROW((void)pelt_changepoints(var_x,{.cost=ChangeCost::normal_variance,.min_segment_length=1}),std::invalid_argument);
    EXPECT_THROW((void)pelt_changepoints(mean_x,{.penalty=ChangePenalty::manual,.penalty_value=-1}),std::invalid_argument);
    EXPECT_THROW((void)pelt_changepoints(mean_x,{.noise_variance=0}),std::invalid_argument);
    // Too short to split: one segment.
    const auto r = pelt_changepoints({1,5,9},{.cost=ChangeCost::normal_mean_variance});
    EXPECT_TRUE(r.changepoints.empty());
    EXPECT_DOUBLE_EQ(r.means[0],5.0);
}

} // namespace
