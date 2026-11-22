#include <datamunge/optim/evolutionary_programming.hpp>
#include <datamunge/optim/natural_evolution_search.hpp>

#include <gtest/gtest.h>

#include <set>

namespace o = datamunge::optim;

namespace {

/// @brief Record trials on a bounded quadratic with an off-axis optimum.
struct Bowl : o::ArbitraryFunction {
  std::vector<std::vector<double>> points; ///< Observed trials.
  /// @brief Score one bounded trial.
  /// @param x Feasible point.
  /// @return Weighted squared error.
  double evaluate(const std::vector<double>& x) override {
    points.push_back(x);
    EXPECT_GE(x[0], 0);
    EXPECT_LE(x[0], 1);
    EXPECT_GE(x[1], 0);
    EXPECT_LE(x[1], 1);
    return x[0] * x[0] + (1 - x[1]) * (1 - x[1]);
  }
};

/// @brief Verify the budget, best observed point and deterministic callback replay.
template <class Optimizer>
auto run(const Optimizer& optimizer, std::size_t budget) {
  Bowl f, g;
  std::vector<double> x{2, -2}, y = x;
  const auto score = optimizer.optimize(f, x, {0, 0}, {1, 1});
  EXPECT_EQ(f.points.size(), budget);
  EXPECT_EQ(f.points.front(), (std::vector<double>{1, 0}));
  EXPECT_EQ(score, x[0] * x[0] + (1 - x[1]) * (1 - x[1]));
  EXPECT_EQ(optimizer.optimize(g, y, {0, 0}, {1, 1}), score);
  EXPECT_EQ(f.points, g.points);
  for (const auto& q : f.points) EXPECT_LE(score, q[0]*q[0] + (1-q[1])*(1-q[1]));
  return f.points;
}

} // namespace

TEST(NaturalEvolutionSearch, EachVariantReplaysPartialPopulationAndPairBudgets) {
  std::set<std::vector<std::vector<double>>> histories;
  for (const auto method : {o::NaturalSearchMethod::xnes, o::NaturalSearchMethod::snes,
                            o::NaturalSearchMethod::openai_es, o::NaturalSearchMethod::ars}) {
    for (const auto budget : {1u, 2u, 86u}) {
      o::NaturalSearchOptions options{budget, 7, method, 6};
      options.retained_directions = 3;
      const auto history = run(o::NaturalEvolutionSearch(options), budget);
      if (budget == 86) histories.insert(history);
    }
  }
  EXPECT_EQ(histories.size(), 4);
}

TEST(EvolutionaryProgramming, UsesExactBudgetThroughInitialAndSurvivorGenerations) {
  for (const auto budget : {1u, 3u, 75u}) run(o::EvolutionaryProgramming({budget, 7, 6, 5, 0.2}), budget);
}
