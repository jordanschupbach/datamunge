#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/optics.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(OpticsTrace, IndependentNeighborhoodCoreDistancesEveryUpdateSelectionAndQuota) {
  std::mt19937_64 rng(264273);
  for (std::size_t n = 0; n <= 25; ++n)
    for (std::size_t minimum : {1, 2, 5}) {
      std::vector<std::vector<double>> data(n, std::vector<double>(2));
      for (auto& row : data)
        for (auto& v : row)
          v = double(int(rng() % 15) - 7) / 2;
      auto t = optics_trace(data, 2., minimum);
      EXPECT_EQ(t.run.work.neighborhood_queries, n);
      EXPECT_EQ(t.run.work.distance_calls, n * n);
      EXPECT_EQ(t.run.work.coordinate_differences, 2 * n * n);
      ASSERT_EQ(t.neighborhoods.size(), n);
      auto order = t.run.result.ordering;
      std::sort(order.begin(), order.end());
      for (std::size_t i = 0; i < n; ++i)
        EXPECT_EQ(order[i], i);
      std::size_t units = 2 + 2 * n + 3 * n + t.updates.size() + t.visits.size();
      for (auto& q : t.neighborhoods) {
        std::vector<std::pair<std::size_t, double>> expected;
        std::vector<double>                         distances;
        for (std::size_t j = 0; j < n; ++j) {
          auto value = std::hypot(data[q.point][0] - data[j][0], data[q.point][1] - data[j][1]);
          if (value <= 2.) {
            expected.push_back({j, value});
            distances.push_back(value);
          }
        }
        ASSERT_EQ(q.neighbors.size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
          EXPECT_EQ(q.neighbors[i].first, expected[i].first);
          EXPECT_NEAR(q.neighbors[i].second, expected[i].second, 1e-14);
        }
        std::sort(distances.begin(), distances.end());
        if (distances.size() < minimum)
          EXPECT_TRUE(std::isinf(q.core_distance));
        else
          EXPECT_NEAR(q.core_distance, distances[minimum - 1], 1e-14);
        units += 1 + 2 * q.neighbors.size();
      }
      std::size_t scans = 0;
      for (auto& s : t.selections) {
        ASSERT_EQ(s.seeds.size(), s.reachabilities.size());
        auto best = std::min_element(s.reachabilities.begin(), s.reachabilities.end()) -
                    s.reachabilities.begin();
        EXPECT_EQ(s.winning_position, best);
        EXPECT_EQ(s.point, s.seeds[best]);
        units += 1 + 2 * s.seeds.size();
        scans += s.seeds.size();
      }
      EXPECT_EQ(t.run.work.seed_scan_entries, scans);
      EXPECT_EQ(t.run.work.seed_extractions, t.selections.size());
      std::size_t insertions = 0, decreases = 0;
      for (auto& u : t.updates) {
        if (u.processed) {
          EXPECT_FALSE(u.inserted);
          EXPECT_FALSE(u.decreased);
          EXPECT_EQ(u.before, u.after);
          EXPECT_EQ(u.proposed, 0);
        } else {
          EXPECT_EQ(u.proposed, std::max(u.core, u.distance));
          if (std::isinf(u.before)) {
            EXPECT_TRUE(u.inserted);
            EXPECT_EQ(u.after, u.proposed);
          } else if (u.proposed < u.before) {
            EXPECT_TRUE(u.decreased);
            EXPECT_EQ(u.after, u.proposed);
          } else
            EXPECT_EQ(u.after, u.before);
        }
        insertions += u.inserted;
        decreases += u.decreased;
      }
      EXPECT_EQ(t.run.work.seed_insertions, insertions);
      EXPECT_EQ(t.run.work.decreases, decreases);
      EXPECT_EQ(t.run.work.update_candidates, t.updates.size());
      EXPECT_EQ(t.captured_entries, units);
      EXPECT_EQ(optics(data, 2., minimum).ordering, t.run.result.ordering);
      EXPECT_NO_THROW(optics_trace(data, 2., minimum, units));
      EXPECT_THROW(optics_trace(data, 2., minimum, units - 1), std::length_error);
    }
}
TEST(OpticsTrace, TinyCutDuplicateZeroDimensionsFullScaleAndCompleteGuards) {
  auto t = optics_trace({{0.}, {.5}, {1.}, {4.}}, .75, 2);
  EXPECT_EQ(t.run.result.ordering, (std::vector<std::size_t>{0, 1, 2, 3}));
  EXPECT_EQ(optics_extract_dbscan(t.run, .75), (std::vector<int>{0, 0, 0, -1}));
  auto summary = optics_cluster_summary(optics_extract_dbscan(t.run, .75));
  EXPECT_EQ(summary.sizes, (std::vector<std::size_t>{3}));
  EXPECT_EQ(summary.noise, 1);
  auto profile = optics_reachability_profile(t.run.result);
  EXPECT_EQ(profile.undefined, (std::vector<char>{1, 0, 0, 1}));
  EXPECT_EQ(profile.displayed[0], profile.undefined_height);
  EXPECT_EQ(optics({{}, {}}, 0, 2).core_distance, (std::vector<double>{0, 0}));
  EXPECT_EQ(optics({{0}, {0}}, 0, 2).reachability[1], 0);
  auto wide = optics_run({{0.}, {1e200}}, 1e201, 2);
  EXPECT_GE(wide.work.widened_distances, 2);
  EXPECT_EQ(wide.result.core_distance[0], 1e200);
  EXPECT_THROW(
      optics({{std::numeric_limits<double>::max()}, {-std::numeric_limits<double>::max()}}, 0, 1),
      std::overflow_error);
  EXPECT_THROW(optics({{0.}}, 1, 0), std::invalid_argument);
  EXPECT_THROW(optics({{0.}, {1., 2.}}, 1, 1), std::invalid_argument);
  EXPECT_THROW(optics({{NAN}}, 1, 1), std::invalid_argument);
  EXPECT_THROW(optics({}, INFINITY, 1), std::invalid_argument);
  EXPECT_THROW(optics_extract_dbscan(t.run, 1.), std::invalid_argument);
  EXPECT_THROW(optics_cluster_summary({1}), std::invalid_argument);
  auto broken        = t.run.result;
  broken.ordering[0] = 1;
  EXPECT_THROW(optics_extract_dbscan(broken, .5), std::invalid_argument);
  EXPECT_EQ(optics_demo_data().size(), 132);
}
