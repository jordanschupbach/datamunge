#include <datamunge/optim/annealing_search.hpp>

#include <gtest/gtest.h>

namespace o = datamunge::optim;

namespace {

/// @brief Record bounded annealing trials on a sphere.
struct Sphere final : o::ArbitraryFunction {
  std::vector<std::vector<double>> history; ///< Callback coordinates.
  /// @brief Evaluate the sum of the squared coordinates.
  /// @param x Feasible trial.
  /// @return Squared distance from the origin.
  double evaluate(const std::vector<double>& x) override {
    history.push_back(x);
    return x[0] * x[0] + x[1] * x[1];
  }
};

} // namespace

TEST(AnnealingSearch, AllVariantsReplayAndTrackTheBestAcrossBudgetBoundaries) {
  for (const auto method : {o::AnnealingMethod::reheating, o::AnnealingMethod::tunneling,
                            o::AnnealingMethod::tsallis, o::AnnealingMethod::vfsa}) {
    for (const auto budget : {1u, 3u, 67u}) {
      Sphere f, g;
      o::AnnealingSearchOptions options{budget, 17, method};
      options.patience = 1;
      options.restart_every = 1;
      options.reanneal_every = 1;
      std::vector<double> x{2, 2}, y = x;
      const o::AnnealingSearch search(options);
      const auto value = search.optimize(f, x, {0, 0}, {1, 1});
      EXPECT_EQ(f.history.size(), budget);
      EXPECT_EQ(f.history.front(), (std::vector<double>{1, 1}));
      EXPECT_EQ(value, x[0] * x[0] + x[1] * x[1]);
      EXPECT_EQ(value, search.optimize(g, y, {0, 0}, {1, 1}));
      EXPECT_EQ(f.history, g.history);
      EXPECT_EQ(x, y);
      for (const auto& point : f.history) {
        EXPECT_LE(value, point[0] * point[0] + point[1] * point[1]);
        EXPECT_LE(0, point[0]);
        EXPECT_GE(1, point[0]);
        EXPECT_LE(0, point[1]);
        EXPECT_GE(1, point[1]);
      }
    }
  }
}

TEST(AnnealingSearch, ReannealingCountsAProbeInAPartialSweep) {
  Sphere f;
  o::AnnealingSearchOptions options{3, 7, o::AnnealingMethod::vfsa};
  options.reanneal_every = 1;
  options.probe_distance = 0.25;
  std::vector<double> x{0, 0};
  EXPECT_EQ(o::AnnealingSearch(options).optimize(f, x, {0, 0}, {1, 1}), 0);
  ASSERT_EQ(f.history.size(), 3);
  EXPECT_EQ(f.history.back(), (std::vector<double>{0.25, 0}));
}
