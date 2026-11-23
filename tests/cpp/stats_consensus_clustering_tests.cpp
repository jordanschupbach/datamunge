#include <datamunge/stats/consensus_clustering.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::ConsensusPartitioner;
using datamunge::stats::consensus_clustering;

TEST(ConsensusClustering, CountsCoAssignmentsOnlyOnAvailablePairs) {
    const DenseMatrix<double> observations(8,1,{0,1,2,3,4,5,6,7});
    std::vector<std::vector<std::size_t>> sampled;
    const ConsensusPartitioner split = [&](const auto& rows,std::size_t k) {
        EXPECT_EQ(k,2u);
        EXPECT_EQ(rows.rows(),3u);
        auto& indices = sampled.emplace_back();
        for (const auto value : rows.data())
            indices.push_back(static_cast<std::size_t>(value));
        EXPECT_TRUE(std::ranges::is_sorted(indices));
        EXPECT_TRUE(std::adjacent_find(indices.begin(),indices.end()) == indices.end());
        return sampled.size()%2 ? std::vector<std::size_t>{99,23,23}
                                : std::vector<std::size_t>{23,99,99};
    };
    const auto fit = consensus_clustering(observations,split,
        {.subsamples=7,.sample_fraction=.4,.seed=79});
    ASSERT_EQ(sampled.size(),7u);
    ASSERT_EQ(fit.sample_size,3u);
    DenseMatrix<std::uint64_t> exposed(8,8), assigned(8,8);
    for (const auto& indices : sampled)
        for (std::size_t i = 0; i < indices.size(); ++i)
            for (std::size_t j = 0; j < indices.size(); ++j) {
                ++exposed(indices[i],indices[j]);
                assigned(indices[i],indices[j]) += ((i == 0) == (j == 0));
            }
    EXPECT_EQ(fit.co_sampled.data(),exposed.data());
    EXPECT_EQ(fit.co_assigned.data(),assigned.data());
    std::size_t uncovered = 0;
    for (std::size_t i = 0; i < exposed.rows(); ++i)
        for (std::size_t j = 0; j < exposed.cols(); ++j) {
            if (exposed(i,j) == 0) {
                EXPECT_TRUE(std::isnan(fit.probability(i,j)));
                ++uncovered;
            } else {
                EXPECT_DOUBLE_EQ(fit.probability(i,j),static_cast<double>(assigned(i,j))/exposed(i,j));
            }
        }
    EXPECT_GT(uncovered,0u);
}

TEST(ConsensusClustering, KMeansRecoversStableGroupsAndReproducibleCounts) {
    const DenseMatrix<double> x(10,1,{-5.1,-5,-4.9,-4.8,-4.7,4.7,4.8,4.9,5,5.1});
    const auto options = datamunge::stats::ConsensusClusteringOptions{
        .subsamples=32,.sample_fraction=.8,.seed=121};
    const auto first = consensus_clustering(x,options);
    const auto second = consensus_clustering(x,options);
    ASSERT_EQ(first.sample_size,8u);
    EXPECT_EQ(first.co_sampled.data(),second.co_sampled.data());
    EXPECT_EQ(first.co_assigned.data(),second.co_assigned.data());
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < x.rows(); ++j) {
            EXPECT_GT(first.co_sampled(i,j),0u);
            EXPECT_DOUBLE_EQ(first.probability(i,j),i/5 == j/5 ? 1 : 0);
        }
}

TEST(ConsensusClustering, RejectsInvalidPartitionsAndSubsampleControls) {
    const DenseMatrix<double> observations(4,1,{0,1,5,6});
    const auto wrong_size = [](const auto&,std::size_t) { return std::vector<std::size_t>{0}; };
    const auto wrong_groups = [](const auto& rows,std::size_t) {
        return std::vector<std::size_t>(rows.rows(),9);
    };
    EXPECT_THROW((void)consensus_clustering(observations,{.subsamples=0}),std::invalid_argument);
    EXPECT_THROW((void)consensus_clustering(observations,{.clusters=4,.sample_fraction=.5}),
        std::invalid_argument);
    EXPECT_THROW((void)consensus_clustering(observations,wrong_size,
        {.subsamples=2,.sample_fraction=1}),std::invalid_argument);
    EXPECT_THROW((void)consensus_clustering(observations,wrong_groups,
        {.subsamples=2,.sample_fraction=1}),std::domain_error);
    const DenseMatrix<double> invalid(2,1,std::numeric_limits<double>::infinity());
    EXPECT_THROW((void)consensus_clustering(invalid,{.sample_fraction=1}),std::invalid_argument);
}
}
