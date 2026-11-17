#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/nearest_neighbor_interpolation.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NearestNeighborTrace, IndependentFullScanEveryQueryAndActualSearchBounds) {
  for (int n = 0; n <= 20; ++n) {
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
      x[i] = i * i + i;
      y[i] = i % 3 - 1;
    }
    NearestNeighborInterpolant model(x, y);
    for (int k = -2; k <= 2 * (n * n + n) + 2; ++k) {
      double q = k / 2.;
      auto   t = model.trace(q);
      if (!n) {
        EXPECT_EQ(t.value, 0);
        EXPECT_EQ(t.route, 0);
        continue;
      }
      std::size_t best = 0;
      for (std::size_t i = 1; i < x.size(); ++i)
        if (std::abs(static_cast<long double>(q) - x[i]) <
            std::abs(static_cast<long double>(q) - x[best]))
          best = i;
      EXPECT_EQ(t.selected, best);
      EXPECT_EQ(t.value, y[best]);
      EXPECT_EQ(model.evaluate(q), t.value);
      std::size_t lo = 0, hi = n - 1;
      for (auto& s : t.steps) {
        EXPECT_EQ(s.lo, lo);
        EXPECT_EQ(s.hi, hi);
        EXPECT_EQ(s.mid, lo + (hi - lo) / 2);
        EXPECT_EQ(s.abscissa, x[s.mid]);
        EXPECT_EQ(s.go_right, x[s.mid] < q);
        if (s.go_right)
          lo = s.mid + 1;
        else
          hi = s.mid;
      }
      EXPECT_EQ(t.captured_entries, 2 * n + 2 + t.steps.size());
      EXPECT_NO_THROW(model.trace(q, t.captured_entries));
      EXPECT_THROW(model.trace(q, t.captured_entries - 1), std::length_error);
    }
  }
}
TEST(NearestNeighborTrace, CompleteGuardsWideDistancesOwnershipAndOriginalStudy) {
  EXPECT_THROW(NearestNeighborInterpolant({0}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(NearestNeighborInterpolant({0, 0}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(NearestNeighborInterpolant({1, 0}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(NearestNeighborInterpolant({0, std::nan("")}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(NearestNeighborInterpolant({0, 1}, {1, std::nan("")}), std::invalid_argument);
  EXPECT_THROW(NearestNeighborInterpolant({0, 1}, {1, 2}, 1), std::length_error);
  NearestNeighborInterpolant empty({}, {});
  EXPECT_THROW(empty.evaluate(std::nan("")), std::invalid_argument);
  auto                       m = std::numeric_limits<double>::max();
  NearestNeighborInterpolant wide({-m, m}, {10, 20});
  EXPECT_EQ(wide.evaluate(0), 10);
  EXPECT_EQ(wide.evaluate(m / 2), 20);
  EXPECT_EQ(wide.trace(0).left_distance, wide.trace(0).right_distance);
  auto                       fixture = nearest_neighbor_demo();
  NearestNeighborInterpolant model(fixture.xs, fixture.ys);
  fixture.xs.clear();
  fixture.ys.clear();
  EXPECT_EQ(model.evaluate(0), 0);
  auto study = nearest_neighbor_demo_study();
  EXPECT_EQ(study.queries.size(), 321);
  for (std::size_t i = 0; i < study.values.size(); ++i) {
    EXPECT_EQ(study.values[i], model.evaluate(study.queries[i]));
    EXPECT_LE(study.absolute_error[i], study.error_reference + 1e-12);
  }
  EXPECT_THROW(nearest_neighbor_demo_study(1), std::invalid_argument);
}
