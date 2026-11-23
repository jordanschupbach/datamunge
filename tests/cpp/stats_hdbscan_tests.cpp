#include <datamunge/stats/hdbscan.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fit_hdbscan;

TEST(Hdbscan, EuclideanCoreDistancesHierarchyAndStabilityMatchR) {
    const DenseMatrix<double> x(15,2,{
        -3.03,.01,-3.1,-.10,-2.92,.04,-3.12,.03,-2.96,-.08,-3.02,.13,
        2.72,.16,3.23,-.21,3.18,.31,2.83,-.28,2.94,.23,3.22,.05,
        0,5.3,-8,-3,10,2
    });
    const auto fit = fit_hdbscan(x,{.min_cluster_size=3});
    const std::array expected_core{.114017542509914,.131529464379659,.126491106406735,
        .131529464379659,.126491106406735,.134536240470737,.453541618817943,
        .406078810084939,.263058928759318,.453541618817943,.252982212813471,
        .263058928759318,5.860759336468270,5.744153549479679,7.054849395982878};
    const std::array expected_mst{.126491106406735,.126491106406735,.131529464379659,
        .131529464379659,.134536240470737,.263058928759318,.263058928759318,
        .406078810084939,.453541618817943,.453541618817943,5.641276451300716,
        5.744153549479679,5.860759336468270,7.054849395982878};
    ASSERT_EQ(fit.core_distances.size(),expected_core.size());
    ASSERT_EQ(fit.minimum_spanning_tree.size(),expected_mst.size());
    for (std::size_t i = 0; i < expected_core.size(); ++i)
        EXPECT_NEAR(fit.core_distances[i],expected_core[i],2e-14);
    for (std::size_t i = 0; i < expected_mst.size(); ++i)
        EXPECT_NEAR(fit.minimum_spanning_tree[i].distance,expected_mst[i],2e-14);
    ASSERT_EQ(fit.clusters.size(),3u);
    EXPECT_FALSE(fit.clusters[0].selected);
    EXPECT_TRUE(fit.clusters[1].selected);
    EXPECT_TRUE(fit.clusters[2].selected);
    EXPECT_EQ(fit.clusters[1].birth_size,6u);
    EXPECT_EQ(fit.clusters[2].birth_size,6u);
    EXPECT_NEAR(fit.clusters[1].stability,45.2921532645998,1e-12);
    EXPECT_NEAR(fit.clusters[2].stability,17.2130145494303,1e-12);
    EXPECT_EQ(fit.labels,(std::vector<std::int64_t>{0,0,0,0,0,0,1,1,1,1,1,1,-1,-1,-1}));
    for (const auto& cluster : fit.clusters) {
        std::size_t accounted = 0;
        for (const auto& edge : fit.condensed_tree)
            if (edge.parent == cluster.id)
                accounted += edge.child_size;
        EXPECT_EQ(accounted,cluster.birth_size);
    }
}

TEST(Hdbscan, ExcessOfMassChoosesParentOrChildrenOnDifferentBranches) {
    const DenseMatrix<double> x(9,1,{0,.01,1,1.01,10,10.8,11.9,12.7,50});
    const auto fit = fit_hdbscan(x,{.min_cluster_size=2,.min_samples=1});
    EXPECT_EQ(fit.core_distances,std::vector<double>(9));
    EXPECT_EQ(fit.labels[0],fit.labels[1]);
    EXPECT_EQ(fit.labels[2],fit.labels[3]);
    EXPECT_NE(fit.labels[0],fit.labels[2]);
    EXPECT_NE(fit.labels[0],fit.labels[4]);
    EXPECT_NE(fit.labels[2],fit.labels[4]);
    for (std::size_t i = 4; i < 8; ++i)
        EXPECT_EQ(fit.labels[i],fit.labels[4]);
    EXPECT_EQ(fit.labels[8],-1);
    std::size_t selected = 0;
    for (const auto& cluster : fit.clusters)
        selected += cluster.selected;
    EXPECT_EQ(selected,3u);
}

TEST(Hdbscan, TiedLevelsAndDuplicatePointsDoNotCreateSpuriousClusters) {
    const DenseMatrix<double> x(7,1,{0,0,0,5,5,5,10});
    const auto fit = fit_hdbscan(x,{.min_cluster_size=3});
    ASSERT_EQ(fit.clusters.size(),3u);
    EXPECT_EQ(fit.labels,(std::vector<std::int64_t>{0,0,0,1,1,1,-1}));
    EXPECT_TRUE(std::isinf(fit.clusters[1].stability));
    EXPECT_TRUE(std::isinf(fit.clusters[2].stability));
    const auto flat = fit_hdbscan(DenseMatrix<double>(5,1,2),{.min_cluster_size=3});
    EXPECT_EQ(flat.labels,std::vector<std::int64_t>(5,-1));
    const auto root = fit_hdbscan(DenseMatrix<double>(5,1,2),
        {.min_cluster_size=3,.allow_single_cluster=true});
    EXPECT_EQ(root.labels,std::vector<std::int64_t>(5,0));
    EXPECT_EQ(root.clusters.size(),1u);
}

TEST(Hdbscan, RejectsInvalidOptionsAndObservations) {
    DenseMatrix<double> x(3,1,{0,1,2});
    EXPECT_THROW((void)fit_hdbscan(x,{.min_cluster_size=1}),std::invalid_argument);
    EXPECT_THROW((void)fit_hdbscan(x,{.min_cluster_size=2,.min_samples=4}),std::invalid_argument);
    x(0,0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)fit_hdbscan(x,{.min_cluster_size=2}),std::invalid_argument);
    x(0,0) = std::numeric_limits<double>::max();
    x(1,0) = -x(0,0);
    EXPECT_THROW((void)fit_hdbscan(x,{.min_cluster_size=2}),std::overflow_error);
}
}
