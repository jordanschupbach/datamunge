#include <datamunge/stats/spectral_clustering.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fit_spectral_clustering;
using datamunge::stats::fit_spectral_clustering_knn;

DenseMatrix<double> weak_bridge() {
    return DenseMatrix<double>(6,6,{
        0,1,.9,0,0,0,
        1,0,.8,0,0,0,
        .9,.8,0,.08,0,0,
        0,0,.08,0,1.1,.9,
        0,0,0,1.1,0,.7,
        0,0,0,.9,.7,0
    });
}

TEST(SpectralClustering, ConnectedPartitionAndEmbeddingMatchIndependentRCalculation) {
    const auto w = weak_bridge();
    const auto fit = fit_spectral_clustering(w,{.n_clusters=2,.seed=121});
    ASSERT_EQ(fit.connected_components,1u);
    EXPECT_EQ(fit.labels,(std::vector<std::size_t>{0,0,0,1,1,1}));
    ASSERT_EQ(fit.eigenvalues.size(),2u);
    EXPECT_DOUBLE_EQ(fit.eigenvalues[0],0);
    EXPECT_NEAR(fit.eigenvalues[1],.0281393473981193,1e-13);
    const auto angles = fit.embedding*fit.embedding.transpose();
    for (std::size_t i = 0; i < w.rows(); ++i)
        EXPECT_NEAR(angles(i,i),1,2e-16);
    EXPECT_NEAR(angles(0,3),.00594943092783036,1e-13);
    EXPECT_NEAR(angles(2,3),.0372553430296460,1e-13);
    EXPECT_NEAR(angles(3,5),.9996882759482905,1e-13);
    for (const double multiplier : {1e-300, 1e308}) {
        const auto scaled = fit_spectral_clustering(w*multiplier,{.n_clusters=2});
        EXPECT_EQ(scaled.labels,fit.labels);
        EXPECT_NEAR(scaled.eigenvalues[1],fit.eigenvalues[1],1e-14);
    }
}

TEST(SpectralClustering, ExactComponentsIncludeIsolatedVertices) {
    const DenseMatrix<double> w(5,5,{
        0,1,0,0,0,
        1,0,0,0,0,
        0,0,0,2,0,
        0,0,2,0,0,
        0,0,0,0,0
    });
    const auto fit = fit_spectral_clustering(w,{.n_clusters=3});
    EXPECT_EQ(fit.connected_components,3u);
    EXPECT_EQ(fit.labels,(std::vector<std::size_t>{0,0,1,1,2}));
    EXPECT_EQ(fit.eigenvalues,(std::vector<double>(3,0)));
    EXPECT_DOUBLE_EQ(fit.embedding(4,2),1);
    const auto each = fit_spectral_clustering(w,{.n_clusters=5});
    EXPECT_EQ(each.labels,(std::vector<std::size_t>{0,1,2,3,4}));
    EXPECT_NEAR(each.eigenvalues[4],2,1e-15);
    EXPECT_THROW((void)fit_spectral_clustering(w,{.n_clusters=2}),std::invalid_argument);
}

TEST(SpectralClustering, KnnGraphAndMalformedInputs) {
    const DenseMatrix<double> x(6,2,{
        0,0,0,1,1,0,10,10,11,10,10,11
    });
    const auto fit = fit_spectral_clustering_knn(x,2,{.n_clusters=2});
    EXPECT_EQ(fit.labels,(std::vector<std::size_t>{0,0,0,1,1,1}));
    EXPECT_THROW((void)fit_spectral_clustering_knn(x,0,{.n_clusters=2}),std::invalid_argument);
    EXPECT_THROW((void)fit_spectral_clustering(weak_bridge(),{.n_clusters=2,.n_init=0}),
        std::invalid_argument);
    auto w = weak_bridge();
    w(0,1) = .5;
    EXPECT_THROW((void)fit_spectral_clustering(w,{.n_clusters=2}),std::invalid_argument);
    w(0,1) = w(1,0) = -1;
    EXPECT_THROW((void)fit_spectral_clustering(w,{.n_clusters=2}),std::invalid_argument);
    w(0,1) = w(1,0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)fit_spectral_clustering(w,{.n_clusters=2}),std::invalid_argument);
}
}
