#include <datamunge/stats/silhouette.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::silhouette_analysis;

TEST(Silhouette, ReportsNegativeAndSingletonWidthsAgainstIndependentRCalculation) {
    const std::array x{0.,1.,5.,7.,8.,20.};
    DenseMatrix<double> d(x.size(),x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        for (std::size_t j = 0; j < x.size(); ++j)
            d(i,j) = std::abs(x[i]-x[j]);
    const auto fit = silhouette_analysis(d,{10,10,10,20,20,99});
    const std::array expected{.6,.615384615384615,-.444444444444444,.8,.833333333333333,0.};
    const std::array<std::size_t,6> neighbors{20,20,20,10,10,20};
    ASSERT_EQ(fit.observations.size(),x.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(fit.observations[i].width,expected[i],1e-14);
        EXPECT_EQ(fit.observations[i].neighbor,neighbors[i]);
    }
    EXPECT_DOUBLE_EQ(fit.observations[2].within_mean,4.5);
    EXPECT_DOUBLE_EQ(fit.observations[2].neighbor_mean,2.5);
    EXPECT_DOUBLE_EQ(fit.observations[5].within_mean,0);
    EXPECT_DOUBLE_EQ(fit.observations[5].neighbor_mean,12.5);
    ASSERT_EQ(fit.clusters.size(),3u);
    EXPECT_EQ(fit.clusters[0].label,10u);
    EXPECT_EQ(fit.clusters[0].count,3u);
    EXPECT_NEAR(fit.clusters[0].mean_width,.256980056980057,1e-14);
    EXPECT_NEAR(fit.clusters[1].mean_width,.816666666666667,1e-14);
    EXPECT_NEAR(fit.mean_width,.400712250712251,1e-14);
}

TEST(Silhouette, ResolvesTiesAndAllZeroDistancesDeterministically) {
    const DenseMatrix<double> zeros(4,4,0);
    const auto fit = silhouette_analysis(zeros,{90,90,44,3});
    EXPECT_DOUBLE_EQ(fit.mean_width,0);
    EXPECT_EQ(fit.observations[0].neighbor,3u);
    EXPECT_EQ(fit.observations[3].neighbor,44u);
    const DenseMatrix<double> ties(3,3,{0,1,1,1,0,1,1,1,0});
    const auto all_singletons = silhouette_analysis(ties,{5,14,7});
    EXPECT_EQ(all_singletons.observations[0].neighbor,7u);
    EXPECT_EQ(all_singletons.observations[1].neighbor,5u);
    EXPECT_DOUBLE_EQ(all_singletons.mean_width,0);
}

TEST(Silhouette, RejectsMalformedMatrixAndOneCluster) {
    DenseMatrix<double> d(3,3,{0,1,4,1,0,3,4,3,0});
    EXPECT_THROW((void)silhouette_analysis(d,{2,2,2}),std::invalid_argument);
    EXPECT_THROW((void)silhouette_analysis(d,{2,3}),std::invalid_argument);
    d(0,0) = 1;
    EXPECT_THROW((void)silhouette_analysis(d,{2,2,8}),std::invalid_argument);
    d(0,0) = 0;
    d(0,1) = 4;
    EXPECT_THROW((void)silhouette_analysis(d,{2,2,8}),std::invalid_argument);
    d(0,1) = d(1,0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)silhouette_analysis(d,{2,2,8}),std::invalid_argument);
}
}
