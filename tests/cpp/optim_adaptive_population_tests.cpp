#include <datamunge/optim/adaptive_differential_evolution.hpp>
#include <datamunge/optim/particle_swarm_variants.hpp>

#include <gtest/gtest.h>

#include <set>

namespace o = datamunge::optim;

namespace {

/// @brief Record quadratic trial coordinates to verify budget and historical best.
struct PopulationObjective : o::ArbitraryFunction {
  std::vector<std::vector<double>> trials; ///< Observed objective calls.

  /// @brief Score one point and record it.
  /// @param x Feasible coordinates.
  /// @return Quadratic loss.
  double evaluate(const std::vector<double>& x) override {
    trials.push_back(x);
    return x[0] * x[0] + 2 * x[1] * x[1];
  }
};

/// @brief Verify native boxed entry point, exact work and deterministic replay.
/// @tparam Optimizer Type supporting the boxed optimizer signature.
/// @param optimizer Configured optimizer.
/// @param budget Required number of callbacks.
/// @return Actual trial history.
template <class Optimizer>
std::vector<std::vector<double>> check_population(const Optimizer& optimizer, std::size_t budget) {
  PopulationObjective f, replay;
  std::vector<double> x{4, -4};
  auto y = x;
  const std::vector<double> lower{-2, -2}, upper{2, 2};
  const auto best = optimizer.optimize(f, x, lower, upper);
  EXPECT_EQ(f.trials.size(), budget);
  EXPECT_EQ(f.trials.front(), (std::vector<double>{2, -2}));
  EXPECT_EQ(best, x[0] * x[0] + 2 * x[1] * x[1]);
  EXPECT_EQ(optimizer.optimize(replay, y, lower, upper), best);
  EXPECT_EQ(y, x);
  EXPECT_EQ(f.trials, replay.trials);
  for (const auto& point : f.trials) {
    EXPECT_LE(best, point[0] * point[0] + 2 * point[1] * point[1]);
    EXPECT_GE(point[0], -2);
    EXPECT_LE(point[0], 2);
    EXPECT_GE(point[1], -2);
    EXPECT_LE(point[1], 2);
  }
  return f.trials;
}

} // namespace

TEST(ParticleSwarmVariants, AllRulesSupportPartialSweepsAndReplayTheirOwnTrajectories) {
  std::set<std::vector<std::vector<double>>> histories;
  for (const auto rule : {o::ParticleSwarmRule::bare_bones, o::ParticleSwarmRule::fully_informed,
                          o::ParticleSwarmRule::comprehensive_learning}) {
    for (const auto budget : {1u, 5u, 75u}) {
      o::ParticleSwarmSearchOptions options{budget, 7, rule, 8};
      options.refresh_after = 2;
      const auto history = check_population(o::ParticleSwarmSearch(options), budget);
      if (budget == 75) histories.insert(history);
    }
  }
  EXPECT_EQ(histories.size(), 3);
}

TEST(AdaptiveDifferentialEvolution, AllVariantsUseExactCapsAndDistinctUpdateRules) {
  std::set<std::vector<std::vector<double>>> histories;
  for (const auto method : {o::AdaptiveDEMethod::jde, o::AdaptiveDEMethod::sade, o::AdaptiveDEMethod::jade,
                            o::AdaptiveDEMethod::shade, o::AdaptiveDEMethod::lshade}) {
    for (const auto budget : {1u, 5u, 83u}) {
      o::AdaptiveDEOptions options{budget, 7, method, 8};
      options.learning_period = 3;
      options.history_size = 3;
      const auto history = check_population(o::AdaptiveDifferentialEvolution(options), budget);
      if (budget == 83) histories.insert(history);
    }
  }
  EXPECT_EQ(histories.size(), 5);
}

TEST(BinaryParticleSwarm, GeneratesOnlyBitsAndReturnsBestScoredPoint) {
  std::vector<o::CategoricalPoint> history;
  constexpr std::size_t dimension = 4;
  const auto loss = [&](const o::CategoricalPoint& point) {
    EXPECT_EQ(point.size(), dimension);
    double sum = 0;
    for (const auto bit : point) {
      EXPECT_LE(bit, 1);
      sum += 1 - bit;
    }
    history.push_back(point);
    return sum;
  };
  const auto run = o::binary_particle_swarm(loss, dimension, {53, 42, 8});
  EXPECT_EQ(history.size(), 53);
  EXPECT_EQ(run.evaluations, 53);
  EXPECT_EQ(run.value, 0);
  EXPECT_EQ(run.point, o::CategoricalPoint(dimension, 1));
  EXPECT_THROW(o::binary_particle_swarm(loss, 0), std::invalid_argument);
}
