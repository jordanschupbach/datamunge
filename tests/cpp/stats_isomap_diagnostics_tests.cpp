#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/stats/isomap.hpp>
#include <datamunge/stats/mds.hpp>
#include <limits>
using datamunge::dstruct::DataFrame;
using namespace datamunge::stats;
namespace {
DataFrame line() {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 2, 4});
  d.add_column("y", std::vector<double>{0, 0, 0, 0});
  return d;
}
DataFrame square() {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 1, 0});
  d.add_column("y", std::vector<double>{0, 0, 1, 1});
  return d;
}
} // namespace
TEST(IsomapDiagnostics, LineShortestPathsAndSpectrumHaveClosedForms) {
  const Isomap              m(line(), {"x", "y"}, {1, 2});
  const std::vector<double> x = {0, 1, 2, 4};
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_NEAR(m.geodesic_distances()(i, j), std::abs(x[i] - x[j]), 1e-12);
      EXPECT_NEAR(m.gram_matrix()(i, j), (x[i] - 1.75) * (x[j] - 1.75), 1e-12);
      EXPECT_NEAR(
          std::abs(m.embedding()(i, 0) - m.embedding()(j, 0)), std::abs(x[i] - x[j]), 1e-12);
    }
  EXPECT_NEAR(m.eigenvalues()[0], 8.75, 1e-12);
  EXPECT_DOUBLE_EQ(m.reference_diagnostics(line(), {"x", "y"}).double_at("max_edge_stretch", 0), 1);
  EXPECT_NEAR(m.dimension_diagnostics().double_at("relative_geodesic_error", 0), 0, 1e-12);
}
TEST(IsomapDiagnostics, SquareCycleExposesPerfectInertiaWithDistanceError) {
  const Isomap              m(square(), {"x", "y"}, {2, 2});
  const std::vector<double> eigen = {2, 2, 0, -1};
  for (std::size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(m.eigenvalues()[i], eigen[i], 1e-12);
  EXPECT_NEAR(m.geodesic_distances()(0, 2), 2, 1e-12);
  EXPECT_NEAR(m.goodness_of_fit(), 1, 1e-12);
  const auto d = m.dimension_diagnostics();
  EXPECT_NEAR(d.double_at("negative_mass", 1), .2, 1e-12);
  EXPECT_NEAR(d.double_at("relative_strain", 1), 1.0 / 3, 1e-12);
  EXPECT_NEAR(
      d.double_at("relative_geodesic_error", 1), (std::sqrt(2.0) - 1) / std::sqrt(3.0), 1e-12);
  EXPECT_NEAR(d.double_at("residual_variance", 1), 0, 1e-12);
  const auto e = m.embedding() * m.embedding().transpose();
  for (std::size_t i = 0; i < 4; ++i) {
    double rowsum = 0;
    for (std::size_t j = 0; j < 4; ++j)
      rowsum += e(i, j);
    EXPECT_NEAR(rowsum, 0, 1e-12);
  }
}
TEST(IsomapDiagnostics, NeighborTiesUseIndicesAndGraphUsesUnion) {
  const auto g = isomap_graph(line(), {"x", "y"}, 2);
  EXPECT_EQ(g.neighbors[2], (std::vector<std::size_t>{1, 0}));
  EXPECT_TRUE(std::isfinite(g.adjacency(1, 3)));
  EXPECT_TRUE(std::isinf(g.adjacency(0, 3)));
  EXPECT_EQ(g.edge_table().nrows(), 5u);
  EXPECT_DOUBLE_EQ(g.diagnostics().double_at("edges", 0), 5);
  EXPECT_DOUBLE_EQ(g.diagnostics().double_at("mean_degree", 0), 2.5);
  const auto edges = g.edge_table();
  bool       found = false;
  for (std::size_t i = 0; i < edges.nrows(); ++i)
    if (edges.double_at("row_i", i) == 1 && edges.double_at("row_j", i) == 3) {
      found = true;
      EXPECT_DOUBLE_EQ(edges.double_at("mutual", i), 0);
    }
  EXPECT_TRUE(found);
}
TEST(IsomapDiagnostics, DisconnectedComponentsCanBeInspectedBeforeFitting) {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 0, 10, 11, 10});
  d.add_column("y", std::vector<double>{0, 0, 1, 0, 0, 1});
  const auto g = isomap_graph(d, {"x", "y"}, 2);
  EXPECT_EQ(g.component_sizes, (std::vector<std::size_t>{3, 3}));
  EXPECT_EQ(g.component_labels, (std::vector<std::size_t>{0, 0, 0, 1, 1, 1}));
  EXPECT_EQ(g.edge_table().nrows(), 6u);
  EXPECT_THROW(Isomap(d, {"x", "y"}, {2, 2}), std::invalid_argument);
  const auto connected = isomap_graph(d, {"x", "y"}, 3);
  EXPECT_EQ(connected.component_sizes.size(), 1u);
}
TEST(IsomapDiagnostics, DuplicatePointsKeepZeroWeightEdges) {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 0, 1, 2});
  const Isomap m(d, {"x"}, {1, 2});
  EXPECT_DOUBLE_EQ(m.neighborhood_graph().adjacency(0, 1), 0);
  EXPECT_DOUBLE_EQ(m.geodesic_distances()(0, 1), 0);
  EXPECT_NEAR(m.embedding()(0, 0), m.embedding()(1, 0), 1e-12);
  EXPECT_EQ(m.neighborhood_graph().component_sizes.size(), 1u);
}
TEST(IsomapDiagnostics, CompleteGraphReducesToClassicalMdsForBothMetrics) {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 2, 4, 3});
  d.add_column("y", std::vector<double>{0, 2, -1, 1, 3});
  for (auto metric : {DistanceMetric::Euclidean, DistanceMetric::Manhattan}) {
    const Isomap iso(d, {"x", "y"}, {4, 4, metric});
    const MDS    mds(d, {"x", "y"}, {4, metric});
    const auto   a = iso.embedding() * iso.embedding().transpose(),
               b   = mds.embedding() * mds.embedding().transpose();
    for (std::size_t i = 0; i < 5; ++i)
      for (std::size_t j = 0; j < 5; ++j) {
        EXPECT_NEAR(iso.gram_matrix()(i, j), mds.gram_matrix()(i, j), 1e-11);
        EXPECT_NEAR(a(i, j), b(i, j), 1e-10);
      }
  }
}
TEST(IsomapDiagnostics, IncreasingKOnlyAddsEdgesAndShortensDistances) {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 2, 3, 4});
  d.add_column("y", std::vector<double>{0, .2, -.1, .3, 0});
  const Isomap a(d, {"x", "y"}, {2, 2}), b(d, {"x", "y"}, {2, 3});
  for (std::size_t i = 0; i < 5; ++i)
    for (std::size_t j = 0; j < 5; ++j) {
      EXPECT_LE(b.geodesic_distances()(i, j), a.geodesic_distances()(i, j) + 1e-12);
      EXPECT_GE(a.geodesic_distances()(i, j) + 1e-12, a.neighborhood_graph().distances(i, j));
      if (std::isfinite(a.neighborhood_graph().adjacency(i, j)))
        EXPECT_TRUE(std::isfinite(b.neighborhood_graph().adjacency(i, j)));
    }
}
TEST(IsomapDiagnostics, UniformRescalingPreservesGraphAndScalesGeometry) {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 1.1, 2.5, 4.3});
  d.add_column("y", std::vector<double>{0, .1, -.2, .3});
  const Isomap base(d, {"x", "y"}, {2, 2});
  for (double scale : {1e-100, 1e100}) {
    auto         changed = d.mutate_with("x", [&](const auto& r) {
                      return r.get_double("x") * scale;
                    }).mutate_with("y", [&](const auto& r) { return r.get_double("y") * scale; });
    const Isomap m(changed, {"x", "y"}, {2, 2});
    EXPECT_EQ(m.neighborhood_graph().neighbors, base.neighborhood_graph().neighbors);
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = 0; j < 4; ++j) {
        EXPECT_NEAR(m.geodesic_distances()(i, j) / scale, base.geodesic_distances()(i, j), 1e-11);
        EXPECT_NEAR(m.gram_matrix()(i, j) / (scale * scale), base.gram_matrix()(i, j), 1e-10);
      }
    EXPECT_NEAR(m.goodness_of_fit(), base.goodness_of_fit(), 1e-12);
  }
}
TEST(IsomapDiagnostics, JointOmissionAndReferencesUseOriginalRowIndices) {
  DataFrame d;
  d.add_column("x", std::vector<std::optional<double>>{0, 1, std::nullopt, 2, 4});
  d.add_column("y", std::vector<double>{0, 0, 0, 0, 0});
  const Isomap m(d, {"x", "y"}, {1, 2});
  EXPECT_EQ(m.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4}));
  const auto pairs = m.pair_distances();
  EXPECT_DOUBLE_EQ(pairs.double_at("row_i", 5), 3);
  EXPECT_DOUBLE_EQ(pairs.double_at("row_j", 5), 4);
  EXPECT_NEAR(
      m.reference_diagnostics(d, {"x", "y"}).double_at("geodesic_relative_error", 0), 0, 1e-12);
  EXPECT_THROW((void)m.reference_diagnostics(d.head(4), {"x"}), std::invalid_argument);
  EXPECT_THROW((void)m.reference_diagnostics(d, {"not_a_column"}), std::invalid_argument);
}
TEST(IsomapDiagnostics, ConstantGeometryDoesNotInventAResidualCorrelation) {
  DataFrame d;
  d.add_column("x", std::vector<double>{1, 1, 1, 1});
  const Isomap m(d, {"x"}, {2, 2});
  EXPECT_DOUBLE_EQ(m.goodness_of_fit(), 0);
  EXPECT_DOUBLE_EQ(m.reference_diagnostics(d, {"x"}).double_at("max_edge_stretch", 0), 0);
  EXPECT_TRUE(std::isinf(m.reference_diagnostics(line(), {"x"}).double_at("max_edge_stretch", 0)));
  EXPECT_DOUBLE_EQ(m.dimension_diagnostics().double_at("relative_geodesic_error", 1), 0);
  EXPECT_TRUE(std::isnan(m.dimension_diagnostics().double_at("residual_variance", 1)));
  DataFrame ref;
  ref.add_column("x", std::vector<double>{0, 0, 0, 0});
  const Isomap nonconstant(line(), {"x"}, {1, 2});
  EXPECT_TRUE(
      std::isinf(
          nonconstant.reference_diagnostics(ref, {"x"}).double_at("geodesic_relative_error", 0)));
}
TEST(IsomapDiagnostics, InvalidFiniteDomainsAndOversizedKAreRejected) {
  auto d = line();
  d.remove_column("x");
  d.add_column("x", std::vector<double>{0, 1, 2, std::numeric_limits<double>::quiet_NaN()});
  EXPECT_THROW((void)isomap_graph(d, {"x"}, 2), std::invalid_argument);
  EXPECT_THROW(
      (void)isomap_graph(line(), {"x"}, std::numeric_limits<std::size_t>::max()),
      std::invalid_argument);
  EXPECT_THROW(
      (void)isomap_graph(line(), {"x"}, 2, static_cast<DistanceMetric>(99)), std::invalid_argument);
  d.remove_column("x");
  d.add_column("x", std::vector<double>{0, 1e200, 2e200, 4e200});
  EXPECT_THROW(Isomap(d, {"x"}, {1, 2}), std::invalid_argument);
}
TEST(IsomapDiagnostics, PlotAndPrefixChecksPreserveAlignmentContracts) {
  const Isomap m(square(), {"x", "y"}, {2, 2});
  EXPECT_EQ(m.plot_shepard().series()[0].x.size(), 6u);
  EXPECT_THROW((void)m.dimension(2), std::out_of_range);
  EXPECT_THROW((void)m.pair_distances(3), std::out_of_range);
  EXPECT_THROW((void)m.neighborhood_graph().plot(0, 2), std::out_of_range);
}
