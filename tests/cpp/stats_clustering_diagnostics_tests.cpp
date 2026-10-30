#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/stats/clustering_diagnostics.hpp>
#include <datamunge/stats/dbscan.hpp>
#include <datamunge/stats/kmeans.hpp>
#include <limits>
namespace s = datamunge::stats;
using datamunge::dstruct::DataFrame;
TEST(ClusteringDiagnostics, AgreementIsPermutationInvariantAndChanceAdjusted) {
  const auto a = s::cluster_agreement({7, 7, 42, 42}, {"a", "a", "b", "b"});
  EXPECT_EQ(a.clusters, 2);
  EXPECT_DOUBLE_EQ(a.purity, 1);
  EXPECT_DOUBLE_EQ(a.adjusted_rand, 1);
  const auto b = s::cluster_agreement({3, 8, 3, 8}, {"a", "a", "b", "b"});
  EXPECT_DOUBLE_EQ(b.purity, .5);
  EXPECT_DOUBLE_EQ(b.adjusted_rand, -.5);
  const auto singleton = s::cluster_agreement({0, 1, 2, 3}, {"a", "a", "b", "b"});
  EXPECT_DOUBLE_EQ(singleton.purity, 1);
  EXPECT_DOUBLE_EQ(singleton.adjusted_rand, 0);
  EXPECT_DOUBLE_EQ(s::cluster_agreement({8}, {""}).adjusted_rand, 1);
  EXPECT_DOUBLE_EQ(s::cluster_agreement({7, 8}, {"a", "b"}).adjusted_rand, 1);
}
TEST(ClusteringDiagnostics, NoiseCoverageAndAllNoiseAreExplicit) {
  const auto a = s::cluster_agreement({0, 0, -1, -1}, {"a", "a", "b", "b"});
  EXPECT_EQ(a.observations, 4);
  EXPECT_EQ(a.noise, 2);
  EXPECT_EQ(a.clusters, 1);
  EXPECT_DOUBLE_EQ(a.coverage, .5);
  EXPECT_DOUBLE_EQ(a.purity, 1);
  EXPECT_DOUBLE_EQ(a.adjusted_rand, 1);
  const auto b = s::cluster_agreement({-1, -1, -1, -1}, {"a", "a", "b", "b"});
  EXPECT_EQ(b.clusters, 0);
  EXPECT_DOUBLE_EQ(b.coverage, 0);
  EXPECT_TRUE(std::isnan(b.purity));
  EXPECT_DOUBLE_EQ(b.adjusted_rand, 0);
  EXPECT_DOUBLE_EQ(s::cluster_agreement({-1, -1}, {"a", "a"}).adjusted_rand, 1);
}
TEST(ClusteringDiagnostics, CompositionHasNoiseAndDeterministicMajorityTies) {
  const auto t = s::cluster_composition({42, 42, -1}, {"b", "a", "z"});
  ASSERT_EQ(t.nrows(), 2);
  EXPECT_EQ(t.double_at("cluster", 0), -1);
  EXPECT_EQ(t.double_at("cluster", 1), 42);
  EXPECT_EQ(t.string_at("majority_class", 1), "a");
  EXPECT_EQ(t.double_at("majority_count", 1), 1);
  EXPECT_EQ(t.double_at("purity", 1), .5);
}
TEST(ClusteringDiagnostics, RejectsUnalignedAndInvalidLabels) {
  EXPECT_THROW((void)s::cluster_agreement({}, {}), std::invalid_argument);
  EXPECT_THROW((void)s::cluster_agreement({0}, {"a", "b"}), std::invalid_argument);
  EXPECT_THROW((void)s::cluster_composition({-2}, {"a"}), std::invalid_argument);
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 1});
  d.add_column("y", std::vector<double>{0, 1});
  EXPECT_THROW((void)s::plot_clusters(d, "x", "y", {0}), std::invalid_argument);
  EXPECT_THROW((void)s::plot_clusters(d, "missing", "y", {0, 0}), std::invalid_argument);
  EXPECT_THROW((void)s::plot_clusters(d, "x", "y", {0, -2}), std::invalid_argument);
  DataFrame bad;
  bad.add_column("x", std::vector<double>{0, std::numeric_limits<double>::infinity()});
  EXPECT_THROW((void)s::plot_clusters(bad, "x", "x", {0, 1}), std::invalid_argument);
  const auto svg = s::plot_clusters(d, "x", "y", {-1, 42}, "projection").to_svg();
  EXPECT_NE(svg.find("noise"), std::string::npos);
  EXPECT_NE(svg.find("cluster 42"), std::string::npos);
}
TEST(ClusteringDiagnostics, FourPointCentroidsInertiaAndWardHeightConvention) {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 2, 8, 10});
  s::KMeansOptions k;
  k.n_clusters = 2;
  const s::KMeans fit(d, {"x"}, k);
  EXPECT_NEAR(fit.inertia(), 4, 1e-12);
  auto a = fit.cluster_centers()(0, 0), b = fit.cluster_centers()(1, 0);
  if (a > b)
    std::swap(a, b);
  EXPECT_EQ(a, 1);
  EXPECT_EQ(b, 9);
  const s::AgglomerativeClustering ward(d, {"x"});
  ASSERT_EQ(ward.merge_history().size(), 3);
  double total = 0;
  for (const auto& m : ward.merge_history())
    total += m.distance * m.distance / 2;
  EXPECT_NEAR(total, 68, 1e-12);
  EXPECT_NEAR(ward.merge_history().back().distance, std::sqrt(128.), 1e-12);
  EXPECT_EQ(ward.cut(2), ward.labels());
  EXPECT_NE(s::plot_dendrogram(ward, "tiny tree").to_svg().find("tiny tree"), std::string::npos);
}
TEST(ClusteringDiagnostics, DensityChainReclaimsBorderAndCountsSelf) {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, .1, .2, .3, 1.5, 1.6, 1.7, 1.8, 3});
  s::DBSCANOptions o;
  o.eps         = .11;
  o.min_samples = 3;
  const s::DBSCAN fit(d, {"x"}, o);
  ASSERT_EQ(fit.n_clusters(), 2);
  EXPECT_EQ(fit.n_noise(), 1);
  EXPECT_EQ(fit.labels()[0], fit.labels()[1]);
  EXPECT_EQ(fit.labels()[0], fit.labels()[3]);
  EXPECT_NE(fit.labels()[0], fit.labels()[4]);
  EXPECT_EQ(fit.labels()[8], -1);
  o.eps = .05;
  EXPECT_EQ(s::DBSCAN(d, {"x"}, o).n_clusters(), 0);
  o.eps = 1.21;
  EXPECT_EQ(s::DBSCAN(d, {"x"}, o).n_clusters(), 1);
}

TEST(ClusteringDiagnostics, ProjectionHandlesConstantCoordinatesAndRejectsOverflowingRanges) {
  DataFrame d;
  d.add_column("x", std::vector<double>{1e10, 1e10});
  EXPECT_FALSE(s::plot_clusters(d, "x", "x", {0, 0}).to_svg().empty());
  DataFrame extreme;
  extreme.add_column("x", std::vector<double>{-1e308, 1e308});
  EXPECT_THROW((void)s::plot_clusters(extreme, "x", "x", {0, 1}), std::invalid_argument);
}
