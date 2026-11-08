#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <datamunge/algorithms/fuzzy_c_means.hpp>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
long double oracle_objective(
    const FuzzyCMeansMatrix& x,
    const FuzzyCMeansMatrix& v,
    const FuzzyCMeansMatrix& u,
    long double              m) {
  long double J = 0;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < v.size(); ++j) {
      long double r = 0;
      for (std::size_t d = 0; d < x[i].size(); ++d) {
        const long double z = static_cast<long double>(x[i][d]) - v[j][d];
        r += z * z;
      }
      J += std::pow(static_cast<long double>(u[i][j]), m) * r;
    }
  return J;
}
const FuzzyCMeansMatrix points{{0}, {5}, {10}}, initial{{.8, .2}, {.5, .5}, {.2, .8}};
} // namespace
TEST(FuzzyCMeansChecked, TinyAnalyticCenterSumsAndCoordinateDescent) {
  const auto t = fuzzy_c_means_trace_from(points, initial, 2, 1, 0, 48);
  ASSERT_EQ(t.frames.size(), 1u);
  const auto& f = t.frames[0];
  EXPECT_NEAR(f.centers[0][0], 55. / 31., 1e-14);
  EXPECT_NEAR(f.centers[1][0], 255. / 31., 1e-14);
  EXPECT_NEAR(f.denominators[0], .93, 1e-15);
  EXPECT_NEAR(f.numerators[0][0], 1.65, 1e-15);
  EXPECT_NEAR(f.numerators[1][0], 7.65, 2e-15);
  EXPECT_NEAR(f.objective_before, 80.5, 1e-13);
  EXPECT_NEAR(f.objective_centers, 454. / 31., 1e-13);
  EXPECT_LE(f.objective_centers, f.objective_before);
  EXPECT_LE(f.objective_membership, f.objective_centers);
  EXPECT_NEAR(f.objective_centers, oracle_objective(points, f.centers, initial, 2), 1e-13);
  EXPECT_NEAR(f.objective_membership, oracle_objective(points, f.centers, f.membership, 2), 1e-13);
  for (std::size_t i = 0; i < 3; ++i) {
    const auto        d0 = f.distances[i][0], d1 = f.distances[i][1];
    const long double expected =
        static_cast<long double>(d1) * d1 /
        (static_cast<long double>(d0) * d0 + static_cast<long double>(d1) * d1);
    EXPECT_NEAR(f.membership[i][0], expected, 1e-15);
  }
  EXPECT_EQ(t.result.centers, f.centers);
  EXPECT_EQ(t.result.membership, f.membership);
  // The returned centers are deliberately not refreshed from the last updated memberships.
  long double den = 0, num = 0;
  for (std::size_t i = 0; i < 3; ++i) {
    const long double u = f.membership[i][0];
    den += u * u;
    num += u * u * points[i][0];
  }
  EXPECT_GT(std::abs(static_cast<double>(num / den) - f.centers[0][0]), 1e-3);
}
TEST(FuzzyCMeansChecked, IndependentInversePowerQueryOracleAndOptimality) {
  std::mt19937 rng(123);
  for (int test = 0; test < 400; ++test) {
    const auto        c = 1 + rng() % 5, d = 1 + rng() % 4;
    FuzzyCMeansMatrix v(c, std::vector<double>(d)), x(1, std::vector<double>(d));
    for (auto& row : v)
      for (auto& z : row)
        z = (int(rng() % 41) - 20) / 4.;
    for (auto& z : x[0])
      z = 20. + (rng() % 20) / 4.;
    for (double m : {1.5, 2., 3., 5.}) {
      const auto               u = fuzzy_c_means_membership(x, v, m);
      std::vector<long double> weights(c);
      long double              sum = 0;
      for (std::size_t j = 0; j < c; ++j) {
        long double dist2 = 0;
        for (std::size_t k = 0; k < d; ++k) {
          const long double z = static_cast<long double>(x[0][k]) - v[j][k];
          dist2 += z * z;
        }
        weights[j] = std::pow(dist2, -1.L / (static_cast<long double>(m) - 1));
        sum += weights[j];
      }
      for (std::size_t j = 0; j < c; ++j)
        EXPECT_NEAR(u[0][j], weights[j] / sum, 2e-14);
    }
  }
  const auto u = fuzzy_c_means_membership({{2}}, {{0}, {10}}, 2);
  EXPECT_NEAR(u[0][0], 16. / 17., 1e-15);
  const double optimum = fuzzy_c_means_objective({{2}}, {{0}, {10}}, u);
  for (int i = 0; i <= 100; ++i) {
    const double q = i / 100.;
    EXPECT_GE(fuzzy_c_means_objective({{2}}, {{0}, {10}}, {{q, 1 - q}}) + 1e-14, optimum);
  }
  EXPECT_DOUBLE_EQ(fuzzy_c_means_membership({{5}}, {{0}, {10}}, 5)[0][0], .5);
}
TEST(FuzzyCMeansChecked, EveryActualPhaseHasIndependentCertificates) {
  const auto t        = fuzzy_c_means_trace_from(points, initial, 2, 8, 0, 0);
  auto       incoming = initial;
  auto       centers  = t.initial_centers;
  double     previous = t.initial_objective;
  for (const auto& f : t.frames) {
    EXPECT_DOUBLE_EQ(f.objective_before, previous);
    double change = 0;
    for (std::size_t j = 0; j < 2; ++j) {
      long double den = 0, num = 0;
      for (std::size_t i = 0; i < 3; ++i) {
        const long double weight = static_cast<long double>(incoming[i][j]) * incoming[i][j];
        den += weight;
        num += weight * points[i][0];
      }
      EXPECT_NEAR(f.denominators[j], den, 3e-15);
      EXPECT_NEAR(f.numerators[j][0], num, 3e-14);
      EXPECT_NEAR(f.centers[j][0], num / den, 3e-14);
      for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(f.distances[i][j], std::fabs(points[i][0] - f.centers[j][0]), 1e-14);
        change = std::max(change, std::abs(f.membership[i][j] - incoming[i][j]));
      }
    }
    EXPECT_DOUBLE_EQ(f.max_change, change);
    EXPECT_NEAR(f.objective_centers, oracle_objective(points, f.centers, incoming, 2), 1e-12);
    EXPECT_NEAR(
        f.objective_membership, oracle_objective(points, f.centers, f.membership, 2), 1e-12);
    EXPECT_LE(f.objective_centers, f.objective_before + 1e-12);
    EXPECT_LE(f.objective_membership, f.objective_centers + 1e-12);
    incoming = f.membership;
    centers  = f.centers;
    previous = f.objective_membership;
  }
  EXPECT_EQ(t.work.initial_draws, 0u);
  EXPECT_EQ(t.work.center_weights, 48u);
  EXPECT_EQ(t.work.weighted_coordinates, 48u);
  EXPECT_EQ(t.work.center_divisions, 16u);
  EXPECT_EQ(t.work.membership_distances, 48u);
  EXPECT_EQ(t.work.ratio_powers, 96u);
  EXPECT_EQ(t.work.membership_updates, 48u);
  EXPECT_EQ(t.work.final_objective_pairs, 6u);
  EXPECT_EQ(t.work.diagnostic_objective_pairs, 102u);
}
TEST(FuzzyCMeansChecked, HistoricalTiesDeadClustersZeroDimensionsAndZeroIterations) {
  const auto dup = fuzzy_c_means_membership({{0}}, {{0}, {0}});
  EXPECT_EQ(dup, (FuzzyCMeansMatrix{{0, 1}}));
  const auto near = fuzzy_c_means_membership({{0}}, {{1e-13}, {9e-13}});
  EXPECT_EQ(near, (FuzzyCMeansMatrix{{0, 1}}));
  const auto dead = fuzzy_c_means_trace_from({{1}, {3}}, {{1, 0}, {1, 0}}, 2, 1);
  EXPECT_EQ(dead.frames[0].denominators[1], 0);
  EXPECT_EQ(dead.frames[0].centers[1][0], 0);
  const auto z = fuzzy_c_means({{1}, {3}}, 1, 2, 0);
  EXPECT_EQ(z.centers, (FuzzyCMeansMatrix{{0}}));
  EXPECT_EQ(z.membership, (FuzzyCMeansMatrix{{1}, {1}}));
  EXPECT_EQ(z.objective, 10);
  EXPECT_EQ(z.iterations, 0u);
  const auto dim0 = fuzzy_c_means_trace({{}, {}, {}}, 2, 2, 3, 0);
  EXPECT_EQ(dim0.frames.size(), 3u);
  EXPECT_EQ(dim0.result.objective, 0);
  EXPECT_EQ(dim0.result.membership, (FuzzyCMeansMatrix{{0, 1}, {0, 1}, {0, 1}}));
  EXPECT_EQ(dim0.work.ratio_powers, 0u);
  EXPECT_EQ(fuzzy_c_means_work({{0}, {0}}, 1, 2, 5, 1e-5).result.iterations, 1u);
  const auto noearly = fuzzy_c_means_work({{0}, {0}}, 1, 2, 5, 0);
  EXPECT_EQ(noearly.result.iterations, 5u);
  EXPECT_FALSE(noearly.stopped_by_tolerance);
  EXPECT_TRUE(fuzzy_c_means_membership({}, {{1}}).empty());
  EXPECT_EQ(fuzzy_c_means_objective({}, {{1}}, {}), 0);
}
TEST(FuzzyCMeansChecked, CompleteDomainChecksPrecedeZeroIterationsAndQuotas) {
  EXPECT_THROW(fuzzy_c_means({}, 0), std::invalid_argument);
  EXPECT_THROW(fuzzy_c_means({{1}}, 0), std::invalid_argument);
  EXPECT_THROW(fuzzy_c_means({{1}}, 2), std::invalid_argument);
  EXPECT_THROW(fuzzy_c_means_trace({{1, 2}, {3}}, 1, 2, 0, 0, 0, 1), std::invalid_argument);
  for (double bad : {NAN, INFINITY, -INFINITY}) {
    EXPECT_THROW(fuzzy_c_means_trace({{1}, {bad}}, 1, 2, 0, 0, 0, 1), std::invalid_argument);
    EXPECT_THROW(fuzzy_c_means(points, 2, bad, 0), std::invalid_argument);
    EXPECT_THROW(fuzzy_c_means(points, 2, 2, 0, bad), std::invalid_argument);
    EXPECT_THROW(fuzzy_c_means_membership({}, {{bad}}), std::invalid_argument);
  }
  EXPECT_THROW(fuzzy_c_means(points, 2, 1), std::invalid_argument);
  EXPECT_THROW(fuzzy_c_means(points, 2, 2, 0, -1), std::invalid_argument);
  for (const auto& u : std::vector<FuzzyCMeansMatrix>{
           {{1, 0}},
           {{.5, .5}, {.5}, {.5, .5}},
           {{.5, .5}, {.5, .5}, {.5, .6}},
           {{-1, 2}, {.5, .5}, {.5, .5}},
           {{NAN, 0}, {.5, .5}, {.5, .5}}})
    EXPECT_THROW(fuzzy_c_means_trace_from(points, u, 2, 0, 0, 1), std::invalid_argument);
  EXPECT_THROW(fuzzy_c_means_membership({{1, 2}}, {{0}}), std::invalid_argument);
  EXPECT_THROW(fuzzy_c_means_membership({}, {}), std::invalid_argument);
  EXPECT_THROW(fuzzy_c_means_objective(points, {{1}, {2}}, {{1}, {1}, {1}}), std::invalid_argument);
  EXPECT_THROW(fuzzy_c_means_partition({}), std::invalid_argument);
  EXPECT_THROW(fuzzy_c_means_partition({{}}), std::invalid_argument);
}
TEST(FuzzyCMeansChecked, ActualNumericalOverflowUnderflowAndTraceExtraWork) {
  const double huge = std::numeric_limits<double>::max();
  EXPECT_THROW(fuzzy_c_means_membership({{huge}}, {{0}}), std::overflow_error);
  EXPECT_THROW(fuzzy_c_means_membership({{1}}, {{0}, {1000}}, 1.000001), std::overflow_error);
  EXPECT_THROW(fuzzy_c_means_objective({{huge}}, {{0}}, {{1}}), std::overflow_error);
  EXPECT_THROW(fuzzy_c_means({{huge}, {huge}}, 1), std::overflow_error);
  EXPECT_THROW(fuzzy_c_means({{1}, {2}}, 2, 10000, 1), std::underflow_error);
  const auto ordinary = fuzzy_c_means({{huge}}, 1);
  EXPECT_EQ(ordinary.objective, 0);
  EXPECT_EQ(ordinary.centers[0][0], huge);
  EXPECT_THROW(fuzzy_c_means_trace({{huge}}, 1), std::overflow_error);
}
TEST(FuzzyCMeansChecked, WholeAdmissionVirtualOverflowAndOwnedHistory) {
  EXPECT_EQ(fuzzy_c_means_trace_entries(3, 1, 2, 1), 48u);
  EXPECT_EQ(fuzzy_c_means_trace_entries(3, 1, 2, 0), 19u);
  EXPECT_EQ(fuzzy_c_means_trace_entries(160, 1, 2, 200), 195004u);
  EXPECT_THROW(fuzzy_c_means_trace_from(points, initial, 2, 1, 0, 47), std::length_error);
  auto x = points, u = initial;
  auto t  = fuzzy_c_means_trace_from(x, u, 2, 1, 0, 48);
  x[0][0] = 99;
  u[0][0] = .1;
  EXPECT_EQ(t.data, points);
  EXPECT_EQ(t.initial_membership, initial);
  t.frames[0].membership[0][0] = 7;
  EXPECT_EQ(t.initial_membership, initial);
  EXPECT_NE(t.result.membership[0][0], 7);
  EXPECT_THROW(fuzzy_c_means_trace_from(points, initial, 2, 100, 100, 48), std::length_error);
  EXPECT_NO_THROW(fuzzy_c_means_trace_from(points, initial, 2, 2, 0, 0));
  EXPECT_THROW(fuzzy_c_means_trace_entries(SIZE_MAX, 1, 1, 1), std::length_error);
  EXPECT_THROW(fuzzy_c_means_trace_entries(1, SIZE_MAX, 1, 0), std::length_error);
  EXPECT_THROW(fuzzy_c_means_trace_entries(1, 1, 1, SIZE_MAX), std::length_error);
}
TEST(FuzzyCMeansChecked, PartitionIdentitiesAndGeometricTransformations) {
  const auto uniform = fuzzy_c_means_partition({{.5, .5}, {.5, .5}});
  EXPECT_EQ(uniform.coefficient, .5);
  EXPECT_NEAR(uniform.entropy_nats, std::log(2.), 1e-15);
  const auto hard = fuzzy_c_means_partition({{1, 0}, {0, 1}});
  EXPECT_EQ(hard.coefficient, 1);
  EXPECT_EQ(hard.entropy_nats, 0);
  const auto        base        = fuzzy_c_means_trace_from(points, initial, 2, 2, 0);
  FuzzyCMeansMatrix transformed = points;
  for (auto& row : transformed)
    row[0] = 10 - 3 * row[0];
  const auto shifted = fuzzy_c_means_trace_from(transformed, initial, 2, 2, 0);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR(base.result.membership[i][j], shifted.result.membership[i][j], 2e-14);
  for (std::size_t j = 0; j < 2; ++j)
    EXPECT_NEAR(10 - 3 * base.result.centers[j][0], shifted.result.centers[j][0], 2e-13);
  EXPECT_NEAR(9 * base.result.objective, shifted.result.objective, 1e-11);
}
TEST(FuzzyCMeansChecked, OriginalFixtureAndNativePredictionPreserveReport) {
  const auto data = fuzzy_c_means_example_data();
  ASSERT_EQ(data.size(), 160u);
  std::mt19937_64                  rng(21);
  std::normal_distribution<double> g0(0, 2.2), g1(10, 2.2);
  for (std::size_t i = 0; i < 160; ++i)
    EXPECT_DOUBLE_EQ(data[i][0], i < 80 ? g0(rng) : g1(rng));
  const auto t = fuzzy_c_means_trace(data, 2, 2, 200, 1e-6, 1);
  EXPECT_EQ(t.result.iterations, 11u);
  EXPECT_NEAR(t.result.objective, 623.804, 5e-4);
  EXPECT_NEAR(t.result.centers[0][0], -.0614927, 5e-8);
  EXPECT_NEAR(t.result.centers[1][0], 10.4761, 5e-5);
  EXPECT_EQ(fuzzy_c_means_membership(data, t.result.centers), t.result.membership);
  const auto query =
      fuzzy_c_means_membership({{0}, {2.5}, {5}, {7.5}, {10}, {-1e6}, {1e6}}, t.result.centers);
  EXPECT_NEAR(query[0][0], .999966, 5e-7);
  EXPECT_NEAR(query[2][0], .539282, 5e-7);
  EXPECT_NEAR(query[4][0], .00223376, 5e-9);
  EXPECT_NEAR(query[5][0], .5, 1e-5);
  EXPECT_NEAR(query[6][0], .5, 1e-5);
  const auto& w = t.work;
  EXPECT_EQ(w.initial_draws, 320u);
  EXPECT_EQ(w.center_weights, 3520u);
  EXPECT_EQ(w.membership_distances, 3520u);
  EXPECT_EQ(w.ratio_powers, 7040u);
  EXPECT_EQ(w.final_objective_pairs, 320u);
  EXPECT_EQ(w.diagnostic_objective_pairs, 7360u);
  EXPECT_EQ(804 + 971 * t.frames.size(), 11485u);
}
