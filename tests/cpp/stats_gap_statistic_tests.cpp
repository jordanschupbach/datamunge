#include <datamunge/stats/gap_statistic.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::GapPartitioner;
using datamunge::stats::GapReference;
using datamunge::stats::gap_statistic;

TEST(GapStatistic, SelectsSameTwoGroupsAsRClusGap) {
    std::vector<double> values;
    for (const int sign : {-1,1})
        for (const double v : {-.3,-.1,.1,.3})
            for (const double u : {-.5,-.3,-.1,.1,.3,.5}) {
                values.push_back(4*sign+u);
                values.push_back(sign+.45*u+v);
            }
    const DenseMatrix<double> data(48,2,std::move(values));
    const auto fit = gap_statistic(data,{.max_clusters=4,.reference_samples=80,.seed=114},
        {.n_init=9,.seed=27});
    ASSERT_EQ(fit.entries.size(),4u);
    EXPECT_EQ(fit.selected_clusters,2u);
    EXPECT_NEAR(fit.entries[0].log_dispersion,6.7155457973877,1e-12);
    EXPECT_NEAR(fit.entries[1].log_dispersion,2.2120037147628,1e-12);
    EXPECT_NEAR(fit.entries[0].gap,-.85138735631878,.10);
    EXPECT_NEAR(fit.entries[1].gap,2.24548991483996,.13);
    EXPECT_NEAR(fit.entries[1].simulation_error,.14274846739484,.08);
}

TEST(GapStatistic, ReproducesDirectPairwiseReferenceDispersionAndMonteCarloError) {
    const DenseMatrix<double> data(4,1,{-4,-3,2,4});
    std::vector<double> logs;
    std::size_t calls = 0;
    const GapPartitioner split = [&](const auto& x,std::size_t k) {
        EXPECT_EQ(k,2u);
        if (calls++ != 0) {
            for (const double v : x.data()) {
                EXPECT_GE(v,-4);
                EXPECT_LE(v,4);
            }
            const double a = x(0,0)-x(1,0), b = x(2,0)-x(3,0);
            logs.push_back(std::log((a*a+b*b)/2));
        }
        return std::vector<std::size_t>{22,22,5,5};
    };
    const auto controls = datamunge::stats::GapStatisticOptions{
        .max_clusters=2,.reference_samples=12,.seed=12,.reference=GapReference::AxisAlignedBox};
    const auto fit = gap_statistic(data,split,controls);
    ASSERT_EQ(calls,13u);
    ASSERT_EQ(logs.size(),12u);
    const double average = std::accumulate(logs.begin(),logs.end(),0.0)/logs.size();
    double squares = 0;
    for (const auto value : logs)
        squares += (value-average)*(value-average);
    EXPECT_NEAR(fit.entries[1].log_dispersion,std::log(2.5),1e-14);
    EXPECT_NEAR(fit.entries[1].reference_mean_log_dispersion,average,1e-14);
    EXPECT_NEAR(fit.entries[1].gap,average-std::log(2.5),1e-14);
    EXPECT_NEAR(fit.entries[1].simulation_error,
        std::sqrt(squares/(logs.size()-1)*(1+1.0/logs.size())),1e-14);
    const GapPartitioner without_capture = [](const auto&,std::size_t) {
        return std::vector<std::size_t>{22,22,5,5};
    };
    const auto repeated = gap_statistic(data,without_capture,controls);
    EXPECT_DOUBLE_EQ(fit.entries[1].gap,repeated.entries[1].gap);
}

TEST(GapStatistic, SamplesOnPrincipalSubspaceAndRejectsUndefinedPartitions) {
    const DenseMatrix<double> line(6,3,{
        -5,-9,3,-4,-7,3,-3,-5,3,3,7,3,4,9,3,5,11,3
    });
    const GapPartitioner split = [](const auto& x,std::size_t) {
        for (std::size_t i = 0; i < x.rows(); ++i) {
            EXPECT_NEAR(x(i,1),2*x(i,0)+1,1e-13);
            EXPECT_DOUBLE_EQ(x(i,2),3);
        }
        return std::vector<std::size_t>{0,0,0,1,1,1};
    };
    const auto fit = gap_statistic(line,split,{.max_clusters=2,.reference_samples=10});
    EXPECT_EQ(fit.entries.size(),2u);
    const auto empty = [](const auto& x,std::size_t) {
        return std::vector<std::size_t>(x.rows());
    };
    EXPECT_THROW((void)gap_statistic(line,empty,{.max_clusters=2,.reference_samples=10}),std::domain_error);
    EXPECT_THROW((void)gap_statistic(DenseMatrix<double>(3,2,1),
        {.max_clusters=1,.reference_samples=10}),std::domain_error);
    EXPECT_THROW((void)gap_statistic(line,{.max_clusters=2,.reference_samples=1}),std::invalid_argument);
}
}
