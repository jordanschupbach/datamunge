#include <datamunge/optim/pareto_search.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace o = datamunge::optim;

namespace {

/// @brief Record a known biobjective trade-off with a dominated free coordinate.
struct Tradeoff final : o::MultiObjectiveFunction {
  std::vector<std::vector<double>> trials; ///< Actual objective call history.

  /// @brief Evaluate a curved Pareto trade-off with a common nonnegative penalty.
  /// @param x Feasible coordinates.
  /// @return Two simultaneous minimization losses.
  std::vector<double> evaluate(const std::vector<double>& x) override {
    EXPECT_LE(0, x[0]);
    EXPECT_GE(1, x[0]);
    EXPECT_LE(0, x[1]);
    EXPECT_GE(1, x[1]);
    trials.push_back(x);
    return {x[0] * x[0] + x[1], (1 - x[0]) * (1 - x[0]) + x[1]};
  }
};

} // namespace

TEST(ParetoSearch, AllModesReplayAndReturnMutuallyNondominatedEvaluatedPoints) {
  for (const auto method : {o::ParetoSearchMethod::paes, o::ParetoSearchMethod::pesa2,
                            o::ParetoSearchMethod::mopso, o::ParetoSearchMethod::gde3,
                            o::ParetoSearchMethod::nsga3, o::ParetoSearchMethod::ibea,
                            o::ParetoSearchMethod::hype, o::ParetoSearchMethod::sms_emoa}) {
    for (const auto budget : {1u, 5u, 79u}) {
      Tradeoff f, replay;
      o::ParetoSearchOptions options{budget, 7, method, 8, 9, 4};
      options.hypervolume_samples = 37;
      const auto result = o::pareto_search(f, {0, 0}, {1, 1}, options);
      const auto second = o::pareto_search(replay, {0, 0}, {1, 1}, options);
      EXPECT_EQ(result.evaluations, budget);
      EXPECT_EQ(f.trials.size(), budget);
      EXPECT_EQ(f.trials, replay.trials);
      ASSERT_FALSE(result.front.empty());
      EXPECT_LE(result.front.size(), method == o::ParetoSearchMethod::paes || method == o::ParetoSearchMethod::pesa2 ||
                method == o::ParetoSearchMethod::mopso ? options.capacity : options.population);
      ASSERT_EQ(result.front.size(), second.front.size());
      for (std::size_t i = 0; i < result.front.size(); ++i) {
        const auto& point = result.front[i];
        EXPECT_EQ(point.coordinates, second.front[i].coordinates);
        EXPECT_EQ(point.objectives, second.front[i].objectives);
        EXPECT_NE(std::ranges::find(f.trials, point.coordinates), f.trials.end());
        const auto x = point.coordinates[0], y = point.coordinates[1];
        EXPECT_EQ(point.objectives, (std::vector<double>{x * x + y, (1 - x) * (1 - x) + y}));
        for (std::size_t j = i + 1; j < result.front.size(); ++j) {
          EXPECT_NE(point.objectives, result.front[j].objectives);
          EXPECT_FALSE(o::dominates(point.objectives, result.front[j].objectives));
          EXPECT_FALSE(o::dominates(result.front[j].objectives, point.objectives));
        }
      }
    }
  }
}

TEST(ParetoSearch, RejectsMalformedObjectiveShapeAndNonfiniteValues) {
  struct Invalid : o::MultiObjectiveFunction {
    std::vector<double> values;
    std::vector<double> evaluate(const std::vector<double>&) override { return values; }
  } f;
  f.values = {0};
  EXPECT_THROW(o::pareto_search(f, {0}, {1}, {1}), std::invalid_argument);
  f.values = {0, std::numeric_limits<double>::infinity()};
  EXPECT_THROW(o::pareto_search(f, {0}, {1}, {1}), std::domain_error);
}

TEST(ParetoSearch, Nsga3UsesValidatedThreeObjectiveDirections) {
  struct Tradeoff3 : o::MultiObjectiveFunction {
    std::vector<double> evaluate(const std::vector<double>& x) override {
      return {x[0], (1 - x[0]) * x[1], (1 - x[0]) * (1 - x[1])};
    }
  } f;
  o::ParetoSearchOptions options;
  options.evaluations = 35;
  options.population = 8;
  options.method = o::ParetoSearchMethod::nsga3;
  options.reference_directions = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
  const auto result = o::pareto_search(f, {0, 0}, {1, 1}, options);
  EXPECT_EQ(result.front.size(), 8);
  EXPECT_EQ(result.evaluations, 35);
  options.reference_directions.front().pop_back();
  EXPECT_THROW(o::pareto_search(f, {0, 0}, {1, 1}, options), std::invalid_argument);
  options.method = o::ParetoSearchMethod::sms_emoa;
  EXPECT_THROW(o::pareto_search(f, {0, 0}, {1, 1}, options), std::invalid_argument);
}
