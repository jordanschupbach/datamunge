#include <datamunge/optim/cma_es_variants.hpp>

#include <gtest/gtest.h>

namespace o = datamunge::optim;

namespace {

/// @brief Records trial points for a two-dimensional sphere.
struct Sphere : o::ArbitraryFunction {
  std::vector<std::vector<double>> points; ///< Observed trials.
  /// @brief Score a feasible trial.
  /// @param x Trial coordinates.
  /// @return Squared distance from zero.
  double evaluate(const std::vector<double>& x) override {
    points.push_back(x);
    EXPECT_LE(0, x[0]);
    EXPECT_GE(1, x[0]);
    EXPECT_LE(0, x[1]);
    EXPECT_GE(1, x[1]);
    return x[0] * x[0] + x[1] * x[1];
  }
};

} // namespace

TEST(CMASearch, ReplaysAllModesAndCountsRestartsAndPartialPopulations) {
  for (const auto method : {o::CMASearchMethod::separable, o::CMASearchMethod::ipop, o::CMASearchMethod::bipop}) {
    for (const auto budget : {1u, 4u, 73u}) {
      o::CMASearchOptions options{budget, 42, method, 6};
      options.generations_per_restart = 2;
      const o::CMASearch search(options);
      Sphere f, replay;
      std::vector<double> x{2, 2}, y = x;
      const auto loss = search.optimize(f, x, {0, 0}, {1, 1});
      ASSERT_EQ(f.points.size(), budget);
      EXPECT_EQ(f.points.front(), (std::vector<double>{1, 1}));
      EXPECT_EQ(loss, x[0] * x[0] + x[1] * x[1]);
      EXPECT_EQ(search.optimize(replay, y, {0, 0}, {1, 1}), loss);
      EXPECT_EQ(f.points, replay.points);
      EXPECT_EQ(x, y);
      for (const auto& point : f.points) EXPECT_LE(loss, point[0]*point[0] + point[1]*point[1]);
    }
  }
}
