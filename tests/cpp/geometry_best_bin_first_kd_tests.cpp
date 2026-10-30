#include <datamunge/geometry/best_bin_first.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

namespace geo = datamunge::geometry;

namespace {

std::vector<std::vector<double>> cloud(std::size_t n, std::size_t d, unsigned seed, bool integer) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> z;
  std::uniform_int_distribution<int> u(0, 4);
  std::vector<std::vector<double>> pts(n, std::vector<double>(d));
  for (auto& p : pts)
    for (auto& v : p) v = integer ? u(rng) : z(rng);
  return pts;
}

double dist(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(s);
}

}  // namespace

TEST(BestBinFirstKd, UnboundedSearchIsExactForBothPriorities) {
  for (bool integer : {false, true})  // integer grids create many ties and duplicates
    for (auto priority : {geo::BinPriority::Cell, geo::BinPriority::Hyperplane})
      for (std::size_t d : {2u, 5u, 16u}) {
        const auto pts = cloud(600, d, 7, integer);
        const geo::BestBinFirstKd tree(pts, 4, priority);
        const auto queries = cloud(40, d, 8, integer);
        for (const auto& q : queries) {
          const auto hit = tree.knn(q, 5);
          const auto truth = geo::knn_brute_force(pts, q, 5);
          ASSERT_TRUE(hit.exact);
          ASSERT_EQ(hit.indices.size(), 5u);
          for (int i = 0; i < 5; ++i) EXPECT_NEAR(hit.distances[i], dist(pts[truth[i]], q), 1e-12);
          for (int i = 0; i < 5; ++i) EXPECT_NEAR(hit.distances[i], dist(pts[hit.indices[i]], q), 1e-12);
        }
      }
}

TEST(BestBinFirstKd, BudgetCapsWorkAndCellKeyNeedsNoMoreWork) {
  const auto pts = cloud(2000, 12, 1, false);
  const geo::BestBinFirstKd cell(pts, 8, geo::BinPriority::Cell), plane(pts, 8, geo::BinPriority::Hyperplane);
  const auto queries = cloud(50, 12, 2, false);
  long long cell_work = 0, plane_work = 0;
  for (const auto& q : queries) {
    const auto capped = cell.knn(q, 1, 30);
    EXPECT_LE(capped.examined, 30);
    EXPECT_EQ(capped.indices.size(), 1u);
    cell_work += cell.knn(q, 1).examined;
    plane_work += plane.knn(q, 1).examined;
  }
  EXPECT_LE(cell_work, plane_work);
  EXPECT_EQ(cell.size(), 2000u);
  EXPECT_EQ(cell.dimension(), 12u);
  EXPECT_GE(cell.leaf_count(), 2000u / 8);
  EXPECT_THROW(geo::BestBinFirstKd({{1.0, 2.0}, {1.0}}), std::invalid_argument);
  EXPECT_THROW((void)cell.knn(queries[0], 0), std::invalid_argument);
}
