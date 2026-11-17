#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/optim/moead.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::optim;
namespace {
struct Observed final : MultiObjectiveFunction {
  std::vector<std::vector<double>> inputs, outputs;
  std::vector<double>              evaluate(const std::vector<double>& x) override {
    inputs.push_back(x);
    auto f = ZDT1Objective(x.size()).evaluate(x);
    outputs.push_back(f);
    return f;
  }
};
} // namespace
TEST(MOEADTrace, ActualCallbacksSequentialPopulationAndScalarReplacementReplay) {
  for (auto decomposition : {"tchebycheff", "weighted_sum"})
    for (std::uint64_t seed : {0ULL, 1ULL, 42ULL}) {
      MOEADOptions o;
      o.population_size   = 7;
      o.neighborhood_size = 3;
      o.max_generations   = 4;
      o.seed              = seed;
      o.decomposition     = decomposition;
      Observed f;
      auto     t = MOEAD(o).trace(f, {0, 0, 0}, {1, 1, 1});
      ASSERT_EQ(t.weights.size(), 7);
      EXPECT_EQ(t.evaluations, 36);
      EXPECT_EQ(f.inputs.size(), 36);
      EXPECT_EQ(t.probe.coordinates, f.inputs[0]);
      EXPECT_EQ(t.probe.objectives, f.outputs[0]);
      auto pop   = t.initial_population;
      auto ideal = t.initial_ideal;
      for (std::size_t i = 0; i < pop.size(); ++i) {
        EXPECT_EQ(pop[i].coordinates, f.inputs[i + 1]);
        EXPECT_EQ(pop[i].objectives, f.outputs[i + 1]);
        EXPECT_NEAR(t.weights[i][0] + t.weights[i][1], 1, 1e-15);
        EXPECT_EQ(t.neighbors[i].size(), 3);
      }
      std::size_t eval = 1 + pop.size();
      for (auto& s : t.steps) {
        EXPECT_EQ(s.child.coordinates, f.inputs[eval]);
        EXPECT_EQ(s.child.objectives, f.outputs[eval++]);
        EXPECT_NE(
            std::find(
                t.neighbors[s.subproblem].begin(), t.neighbors[s.subproblem].end(), s.parent_a),
            t.neighbors[s.subproblem].end());
        EXPECT_NE(
            std::find(
                t.neighbors[s.subproblem].begin(), t.neighbors[s.subproblem].end(), s.parent_b),
            t.neighbors[s.subproblem].end());
        for (std::size_t k = 0; k < 2; ++k)
          ideal[k] = std::min(ideal[k], s.child.objectives[k]);
        EXPECT_EQ(s.ideal, ideal);
        auto score = [&](const std::vector<double>& v, const std::vector<double>& w) {
          if (o.decomposition == "weighted_sum")
            return w[0] * (v[0] - ideal[0]) + w[1] * (v[1] - ideal[1]);
          return std::max(
              std::max(w[0], 1e-6) * std::abs(v[0] - ideal[0]),
              std::max(w[1], 1e-6) * std::abs(v[1] - ideal[1]));
        };
        ASSERT_EQ(s.replacements.size(), t.neighbors[s.subproblem].size());
        for (std::size_t j = 0; j < s.replacements.size(); ++j) {
          auto& r = s.replacements[j];
          EXPECT_EQ(r.neighbor, t.neighbors[s.subproblem][j]);
          auto before = score(pop[r.neighbor].objectives, t.weights[r.neighbor]),
               after  = score(s.child.objectives, t.weights[r.neighbor]);
          EXPECT_DOUBLE_EQ(r.incumbent_score, before);
          EXPECT_DOUBLE_EQ(r.child_score, after);
          EXPECT_EQ(r.replaced, after <= before);
          if (r.replaced)
            pop[r.neighbor] = s.child;
        }
      }
      EXPECT_EQ(eval, t.evaluations);
      EXPECT_EQ(t.final_ideal, ideal);
      for (std::size_t i = 0; i < pop.size(); ++i) {
        EXPECT_EQ(pop[i].coordinates, t.final_population[i].coordinates);
        EXPECT_EQ(pop[i].objectives, t.final_population[i].objectives);
      }
      auto front = non_dominated_front(pop);
      ASSERT_EQ(front.size(), t.front.size());
      for (std::size_t i = 0; i < front.size(); ++i)
        EXPECT_EQ(front[i].objectives, t.front[i].objectives);
      Observed ordinary;
      auto     result = MOEAD(o).optimize(ordinary, {0, 0, 0}, {1, 1, 1});
      EXPECT_EQ(ordinary.inputs, f.inputs);
      ASSERT_EQ(result.size(), t.front.size());
      for (std::size_t i = 0; i < result.size(); ++i)
        EXPECT_EQ(result[i].objectives, t.front[i].objectives);
      Observed exact;
      EXPECT_NO_THROW(MOEAD(o).trace(exact, {0, 0, 0}, {1, 1, 1}, t.captured_entries));
      Observed shortcap;
      EXPECT_THROW(
          MOEAD(o).trace(shortcap, {0, 0, 0}, {1, 1, 1}, t.captured_entries - 1),
          std::length_error);
    }
}
TEST(MOEADTrace, BoundsOptionsDimensionGuardsAndNativeReferences) {
  Observed     f;
  MOEADOptions o;
  o.population_size = 0;
  EXPECT_THROW(MOEAD(o).optimize(f, {0, 0}, {1, 1}), std::invalid_argument);
  o.population_size = 3;
  o.crossover_rate  = std::nan("");
  EXPECT_THROW(MOEAD(o).optimize(f, {0, 0}, {1, 1}), std::invalid_argument);
  EXPECT_THROW(
      MOEAD().optimize(f, {0, 0}, {1, std::numeric_limits<double>::infinity()}),
      std::invalid_argument);
  EXPECT_TRUE(f.inputs.empty());
  struct Invalid final : MultiObjectiveFunction {
    std::vector<double> evaluate(const std::vector<double>&) override { return {1}; }
  } invalid;
  EXPECT_THROW(MOEAD().optimize(invalid, {0}, {1}), std::invalid_argument);
  struct NotFinite final : MultiObjectiveFunction {
    std::vector<double> evaluate(const std::vector<double>&) override { return {1, std::nan("")}; }
  } nonfinite;
  EXPECT_THROW(MOEAD().optimize(nonfinite, {0}, {1}), std::domain_error);
  struct Change final : MultiObjectiveFunction {
    int                 calls = 0;
    std::vector<double> evaluate(const std::vector<double>&) override {
      return ++calls == 1 ? std::vector<double>{1, 2} : std::vector<double>{1, 2, 3};
    }
  } change;
  EXPECT_THROW(MOEAD().optimize(change, {0}, {1}), std::invalid_argument);
  EXPECT_THROW(ZDT1Objective(1), std::invalid_argument);
  ZDT1Objective z(2);
  EXPECT_EQ(z.evaluate({.25, 0}), (std::vector<double>{.25, .5}));
  EXPECT_THROW(z.evaluate({1}), std::invalid_argument);
  EXPECT_THROW(z.evaluate({-1, 0}), std::invalid_argument);
  auto front = zdt1_front(101);
  auto s     = zdt1_front_summary(front);
  EXPECT_EQ(s.points, 101);
  EXPECT_EQ(s.min_f1, 0);
  EXPECT_EQ(s.max_f1, 1);
  EXPECT_NEAR(s.mean_gap, 0, 1e-15);
  EXPECT_THROW(zdt1_front_summary({}), std::invalid_argument);
  EXPECT_EQ(moead_original_choices().size(), 5);
}
