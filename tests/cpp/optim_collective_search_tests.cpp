#include <datamunge/optim/collective_search.hpp>
#include <datamunge/optim/equi_energy_search.hpp>

#include <gtest/gtest.h>

#include <set>

namespace o = datamunge::optim;

namespace {

/// @brief Record feasible trial points and evaluate a shifted bowl.
struct Bowl : o::ArbitraryFunction {
  std::vector<std::vector<double>> trials; ///< Observed evaluation order.
  /// @brief Score one box-constrained point.
  /// @param x Feasible candidate.
  /// @return Sum of squared distances from (0, 0.5).
  double evaluate(const std::vector<double>& x) override {
    EXPECT_GE(x[0], 0);
    EXPECT_LE(x[0], 1);
    EXPECT_GE(x[1], 0);
    EXPECT_LE(x[1], 1);
    trials.push_back(x);
    return x[0] * x[0] + (x[1] - 0.5) * (x[1] - 0.5);
  }
};

} // namespace

TEST(CollectiveSearch, EveryMechanismRespectsTheBudgetAndSeededReplay) {
  std::set<std::vector<std::vector<double>>> histories;
  for (const auto method : {
      o::CollectiveSearchMethod::teaching_learning, o::CollectiveSearchMethod::salp_swarm,
      o::CollectiveSearchMethod::flower_pollination, o::CollectiveSearchMethod::invasive_weed,
      o::CollectiveSearchMethod::bat, o::CollectiveSearchMethod::biogeography,
      o::CollectiveSearchMethod::bacterial_chemotaxis, o::CollectiveSearchMethod::bacterial_foraging,
      o::CollectiveSearchMethod::artificial_fish, o::CollectiveSearchMethod::gravitational,
      o::CollectiveSearchMethod::krill_herd, o::CollectiveSearchMethod::shuffled_frog,
      o::CollectiveSearchMethod::imperialist, o::CollectiveSearchMethod::clonal_selection,
      o::CollectiveSearchMethod::artificial_immune_network, o::CollectiveSearchMethod::stochastic_fractal,
      o::CollectiveSearchMethod::charged_system}) {
    for (const auto budget : {1u, 3u, 135u}) {
      Bowl f, replay;
      std::vector<double> x{2, -2}, y = x;
      const o::CollectiveSearch search({budget, 7, method, 8});
      const auto value = search.optimize(f, x, {0, 0}, {1, 1});
      EXPECT_EQ(value, search.optimize(replay, y, {0, 0}, {1, 1}));
      EXPECT_EQ(f.trials.size(), budget);
      EXPECT_EQ(f.trials, replay.trials);
      EXPECT_EQ(x, y);
      EXPECT_EQ(value, x[0]*x[0] + (x[1]-0.5)*(x[1]-0.5));
      for (const auto& t : f.trials) EXPECT_LE(value, t[0]*t[0] + (t[1]-0.5)*(t[1]-0.5));
      if (budget == 135) histories.insert(f.trials);
    }
  }
  EXPECT_EQ(histories.size(), 17);
}

TEST(EquiEnergySearch, CachedJumpsConsumeNoAdditionalObjectiveCalls) {
  Bowl f, replay;
  o::EquiEnergyOptions options{67, 7};
  options.ring_width = 100;
  options.jump_probability = 1;
  const o::EquiEnergySearch search(options);
  const auto result = search.run(f, {1, 1}, {0, 0}, {1, 1});
  const auto second = search.run(replay, {1, 1}, {0, 0}, {1, 1});
  EXPECT_EQ(result.evaluations, 67);
  EXPECT_EQ(f.trials.size(), result.evaluations);
  EXPECT_GT(result.accepted_jumps, 0);
  EXPECT_LE(result.accepted_jumps, result.jumps);
  EXPECT_EQ(f.trials, replay.trials);
  EXPECT_EQ(result.coordinates, second.coordinates);
  EXPECT_EQ(result.value, second.value);
}
