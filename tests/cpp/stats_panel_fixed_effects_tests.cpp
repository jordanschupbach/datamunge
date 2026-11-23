#include <datamunge/stats/panel_fixed_effects.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fit_two_way_fixed_effects;
using datamunge::stats::PanelClustering;

TEST(PanelFixedEffects, UnbalancedDisconnectedPanelMatchesExplicitDummyFitAndClusterCovariance) {
    std::vector<std::int64_t> unit, time;
    std::vector<double> y, predictor;
    for (int s = 1; s <= 10; ++s) {
        for (int v = 1; v <= 4; ++v) {
            if ((s + 2*v) % 5 == 0)
                continue;
            const int replicates = (s + v) % 3 == 0 ? 2 : 1;
            for (int r = 0; r < replicates; ++r) {
                const auto x = std::sin(.63*s*v + .8*r) + .25*v - .2*s;
                const auto z = std::cos(.47*s + 1.4*v - r) + .1*s*v;
                unit.push_back(5*s - 31);
                time.push_back(2*v + (s <= 5 ? -10 : 17));
                predictor.insert(predictor.end(), {x,z});
                y.push_back(4 + .85*x - 1.3*z + .9*std::sin(s) + .5*v*v +
                    (.2 + .11*s)*std::cos(s*v + 2*r) + .27*std::sin(2*s - v + 3*r));
            }
        }
    }
    const DenseMatrix<double> design(y.size(),2,predictor);
    const std::array modes{PanelClustering::Unit,PanelClustering::Time,PanelClustering::UnitAndTime};
    // Base R lm with explicit dummy columns, and group sums of its influence scores.
    const std::array<double,3> covariance[2][3]{
        {{.031291792772328,-.010186285246270,.023613297186654},
         {.036895392731206,-.003978057304508,.019405782543159},
         {.039533039285768,-.016482410787744,.026848140125789}},
        {{.060467232410296,-.019683642987962,.045629559780974},
         {.073332457602398,-.007906697748090,.038570499464664},
         {.082358866508148,-.031751809660760,.055169480433366}}};
    for (const bool adjusted : {false, true}) {
        for (std::size_t k = 0; k < modes.size(); ++k) {
            SCOPED_TRACE(k);
            const auto fit = fit_two_way_fixed_effects(design,y,unit,time,
                {.clustering=modes[k],.finite_sample_adjustment=adjusted});
            EXPECT_EQ(fit.connected_components,2u);
            EXPECT_EQ(fit.fixed_effect_rank,16u);
            EXPECT_EQ(fit.residual_df,23u);
            EXPECT_EQ(fit.cluster_df,k == 0 ? 9u : 7u);
            EXPECT_NEAR(fit.coefficients[0],.57575144392746,3e-12);
            EXPECT_NEAR(fit.coefficients[1],-1.81903260284238,3e-12);
            EXPECT_NEAR(fit.residuals[0],.59908301837603,3e-12);
            EXPECT_DOUBLE_EQ(fit.time_effects[0],0);
            EXPECT_DOUBLE_EQ(fit.time_effects[4],0);
            double rss = 0;
            for (std::size_t i = 0; i < y.size(); ++i) {
                rss += fit.residuals[i] * fit.residuals[i];
                const auto s = std::ranges::lower_bound(fit.unit_levels,unit[i]) - fit.unit_levels.begin();
                const auto t = std::ranges::lower_bound(fit.time_levels,time[i]) - fit.time_levels.begin();
                const auto mean = fit.unit_effects[s] + fit.time_effects[t] +
                    design(i,0) * fit.coefficients[0] + design(i,1) * fit.coefficients[1];
                EXPECT_NEAR(fit.fitted[i], mean, 2e-11);
            }
            EXPECT_NEAR(rss,7.69578455212089,3e-12);
            EXPECT_NEAR(fit.covariance(0,0),covariance[adjusted][k][0],3e-12);
            EXPECT_NEAR(fit.covariance(0,1),covariance[adjusted][k][1],3e-12);
            EXPECT_NEAR(fit.covariance(1,1),covariance[adjusted][k][2],3e-12);
            if (adjusted && k == 2) {
                const auto test = fit.test_contrast({1,-.3},.25);
                EXPECT_NEAR(test.estimate,1.121461224780177,3e-12);
                EXPECT_NEAR(test.standard_error,.326152120250056,3e-12);
                EXPECT_NEAR(test.t_statistic,2.671947139610926,3e-12);
                EXPECT_NEAR(test.p_value,.031909491045123,3e-12);
            }
        }
    }
    EXPECT_THROW((void)fit_two_way_fixed_effects(design,y,unit,time,{.max_iterations=1}), std::runtime_error);
}

TEST(PanelFixedEffects, AbsorbsLongSparsePanelsWithLargeUnitOffsets) {
    const std::int64_t units = 220;
    std::vector<std::int64_t> unit,time;
    std::vector<double> x,y;
    for (std::int64_t u = units; u > 0; --u) {
        for (std::int64_t t : {u+1,u}) {
            for (int r = 0; r < 3; ++r) {
                const auto observed = 2048.0*u*u + 17.0*t + r;
                x.push_back(observed);
                y.push_back(37.0*u + .25*t*t + 1.5*observed + (r == 1 ? -1.0 : .5));
                unit.push_back(u);
                time.push_back(t);
            }
        }
    }
    const auto fit = fit_two_way_fixed_effects(DenseMatrix<double>(x.size(),1,x),y,unit,time);
    EXPECT_EQ(fit.connected_components,1u);
    EXPECT_EQ(fit.fixed_effect_rank,440u);
    EXPECT_EQ(fit.residual_df,879u);
    EXPECT_NEAR(fit.coefficients[0],1.5,1e-10);
    EXPECT_NEAR(fit.residuals[0],.5,2e-10);
    EXPECT_NEAR(fit.residuals[1],-1,2e-10);
    EXPECT_NEAR(fit.time_effects.back(), .25*(221*221-1),2e-7);
}

TEST(PanelFixedEffects, ValidatesAliasClusterCountAndContrastVariance) {
    const std::vector<std::int64_t> u{1,1,1,2,2,2,3,3,3}, t{1,2,3,1,2,3,1,2,3};
    const std::vector<double> y{1,4,3,5,2,8,4,7,6};
    const DenseMatrix<double> x(9,1,{0,1,0,1,0,2,0,2,1}), constant(9,1,1), period(9,1,{1,2,3,1,2,3,1,2,3});
    EXPECT_THROW((void)fit_two_way_fixed_effects(constant,y,u,t), std::invalid_argument);
    EXPECT_THROW((void)fit_two_way_fixed_effects(period,y,u,t), std::invalid_argument);
    EXPECT_THROW((void)fit_two_way_fixed_effects(x,y,u,std::vector<std::int64_t>(9,1),
        {.clustering=PanelClustering::Time}), std::invalid_argument);
    auto invalid = y;
    invalid[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)fit_two_way_fixed_effects(x,invalid,u,t), std::invalid_argument);
    const auto fit = fit_two_way_fixed_effects(x,y,u,t);
    EXPECT_THROW((void)fit.test_contrast({0}), std::domain_error);
    EXPECT_THROW((void)fit.test_contrast({1,1}), std::invalid_argument);
}
} // namespace
