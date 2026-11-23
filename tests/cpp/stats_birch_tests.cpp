#include <datamunge/stats/birch.hpp>

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::Birch;

TEST(Birch, RepeatedBatchesBuildTheSameBalancedSummariesAsOneBatch) {
    const DenseMatrix<double> first(6,1,{0,.125,3,3.125,6,6.125});
    const DenseMatrix<double> second(8,1,{9,9.125,12,12.125,15,15.125,18,18.125});
    std::vector<double> all = first.data();
    all.insert(all.end(),second.data().begin(),second.data().end());
    Birch streamed(1,{.threshold=.08,.branching_factor=2});
    streamed.partial_fit(first);
    const auto saved = streamed;
    streamed.partial_fit(second);
    Birch once(1,{.threshold=.08,.branching_factor=2});
    once.partial_fit(DenseMatrix<double>(14,1,std::move(all)));

    EXPECT_EQ(streamed.observations(),14u);
    EXPECT_EQ(saved.observations(),6u);
    EXPECT_GE(streamed.height(),3u);
    EXPECT_EQ(streamed.height(),once.height());
    const auto leaves = streamed.leaf_summaries();
    const auto expected = once.leaf_summaries();
    ASSERT_EQ(leaves.size(),7u);
    ASSERT_EQ(expected.size(),leaves.size());
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        EXPECT_EQ(leaves[i].count,2u);
        EXPECT_DOUBLE_EQ(leaves[i].radius,.0625);
        EXPECT_EQ(leaves[i].centroid,expected[i].centroid);
    }
    EXPECT_EQ(saved.leaf_summaries().size(),3u);
}

TEST(Birch, WardMergeUsesObservationCountsInsteadOfUnweightedCentroids) {
    std::vector<double> points(100,0);
    points.insert(points.end(),100,2);
    points.push_back(5);
    Birch tree(1,{.threshold=0});
    tree.partial_fit(DenseMatrix<double>(201,1,std::move(points)));
    const auto partition = tree.cluster_leaves(2);
    EXPECT_EQ(partition.predict(DenseMatrix<double>(3,1,{0,2,5})),
        (std::vector<std::size_t>{0,1,1}));
    EXPECT_EQ(partition.clusters[0].count,100u);
    EXPECT_EQ(partition.clusters[1].count,101u);
    EXPECT_NEAR(partition.clusters[1].centroid[0],205.0/101,1e-14);
    EXPECT_NEAR(partition.clusters[1].radius,30.0/101,1e-14);
}

TEST(Birch, WeightedWardPartitionAndSummaryMatchIndependentRCalculation) {
    const DenseMatrix<double> seeds(6,2,{-3,0,-2.9,.7,-1.8,.2,.9,.2,2.2,0,2.9,.6});
    const std::array<std::size_t,6> counts{7,2,9,1,6,8};
    Birch tree(2,{.threshold=0,.branching_factor=3});
    for (std::size_t i = 0; i < seeds.rows(); ++i) {
        DenseMatrix<double> batch(counts[i],2);
        for (std::size_t j = 0; j < batch.rows(); ++j) {
            batch(j,0) = seeds(i,0);
            batch(j,1) = seeds(i,1);
        }
        tree.partial_fit(batch);
    }
    const auto partition = tree.cluster_leaves(2);
    ASSERT_EQ(partition.leaves.size(),6u);
    const auto labels = partition.predict(seeds);
    const auto left = labels[0], right = labels[3];
    ASSERT_NE(left,right);
    EXPECT_EQ(labels,(std::vector<std::size_t>{left,left,left,right,right,right}));
    EXPECT_EQ(partition.clusters[left].count,18u);
    EXPECT_NEAR(partition.clusters[left].centroid[0],-2.388888888888889,1e-14);
    EXPECT_NEAR(partition.clusters[left].radius,.624895052917283,1e-14);
    EXPECT_EQ(partition.clusters[right].count,15u);
    EXPECT_NEAR(partition.clusters[right].centroid[0],2.486666666666666,1e-14);
    EXPECT_NEAR(partition.clusters[right].radius,.612680812314028,1e-14);
}

TEST(Birch, InvalidBatchPreservesTreeAndEmptyGeometryIsConsistent) {
    Birch tree(1);
    EXPECT_EQ(tree.height(),0u);
    tree.partial_fit(DenseMatrix<double>(0,1));
    EXPECT_EQ(tree.observations(),0u);
    EXPECT_THROW((void)tree.cluster_leaves(1),std::invalid_argument);
    tree.partial_fit(DenseMatrix<double>(1,1,{0}));
    EXPECT_THROW(tree.partial_fit(DenseMatrix<double>(2,1,{2,std::numeric_limits<double>::infinity()})),
        std::invalid_argument);
    EXPECT_EQ(tree.observations(),1u);
    EXPECT_EQ(tree.leaf_summaries().size(),1u);
    EXPECT_THROW((void)tree.cluster_leaves(2),std::invalid_argument);
    EXPECT_THROW((void)tree.cluster_leaves(1).predict(DenseMatrix<double>(1,2,{0,0})),
        std::invalid_argument);
    EXPECT_THROW((void)Birch(3,{.branching_factor=1}),std::invalid_argument);
}
}
