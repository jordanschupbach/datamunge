#include <datamunge/optim/mixed_search.hpp>

#include <gtest/gtest.h>

#include <stdexcept>

namespace o = datamunge::optim;

namespace {

/// @brief Mixed domain with one active numeric branch for each category.
const std::vector<o::MixedParameter> domain{
    {o::ParameterKind::categorical, 0, 1, 2},
    {o::ParameterKind::continuous, -1, 1, 2, o::ParameterCondition{0, 0}},
    {o::ParameterKind::integer, -3, 3, 2, o::ParameterCondition{0, 1}},
};

/// @brief Check feasibility and score the active choice.
/// @param point Typed conditional candidate.
/// @return Known mixed-domain loss.
double score(const o::MixedConfiguration& point) {
  const auto branch = std::get<std::size_t>(point[0]);
  EXPECT_LT(branch, 2);
  if (branch == 0) {
    EXPECT_TRUE(std::holds_alternative<std::monostate>(point[2]));
    const auto x = std::get<double>(point[1]);
    EXPECT_GE(x, -1);
    EXPECT_LE(x, 1);
    return x*x;
  }
  EXPECT_TRUE(std::holds_alternative<std::monostate>(point[1]));
  const auto x = std::get<std::int64_t>(point[2]);
  EXPECT_GE(x, -3);
  EXPECT_LE(x, 3);
  return 1 + static_cast<double>(x*x);
}

} // namespace

TEST(MixedSearch, FeasibleConditionalValuesAndExactSeparateModelCounts) {
  for (const auto method : {o::MixedSearchMethod::tpe, o::MixedSearchMethod::smac,
                            o::MixedSearchMethod::surrogate_evolution}) {
    std::vector<o::MixedConfiguration> observed;
    o::MixedSearchOptions options{17, 7, method, 4, 8, 0.25, 0, 4, 2};
    const auto objective = [&](const auto& point) { observed.push_back(point); return score(point); };
    const auto result = o::mixed_search(objective, domain, options);
    EXPECT_EQ(result.evaluations, 17);
    EXPECT_EQ(observed.size(), 17);
    EXPECT_EQ(result.value, score(result.point));
    EXPECT_EQ(result.model_evaluations, (17 - 4) * (method == o::MixedSearchMethod::surrogate_evolution ? 16 : 8));
    const auto history = observed;
    observed.clear();
    const auto again = o::mixed_search(objective, domain, options);
    EXPECT_EQ(again.point, result.point);
    EXPECT_EQ(history, observed);
  }
}

TEST(MixedSearch, BohbUsesFidelityModelsAndIteratedRacingSharesWorkCaps) {
  o::ResourceSearchOptions resource;
  resource.resource_budget = 64;
  resource.trials = 12;
  resource.maximum_resource = 4;
  o::MixedSearchOptions model;
  model.startup = 2;
  model.candidates = 4;
  model.random_fraction = 0;
  std::vector<o::MixedConfiguration> actual;
  const auto run = o::bohb_search([&](std::size_t id, const auto& config, std::uint64_t) {
    EXPECT_EQ(id, actual.size());
    actual.push_back(config);
  }, [&](std::size_t id, std::size_t, std::size_t fidelity) {
    return score(actual[id]) + 1.0 / fidelity;
  }, domain, resource, model);
  EXPECT_EQ(run.configurations, actual);
  EXPECT_LE(run.schedule.consumed, 64);
  EXPECT_GT(run.model_starts, 0);
  ASSERT_TRUE(run.schedule.best);
  EXPECT_EQ(run.schedule.best->resource, 4);

  std::size_t calls = 0;
  const auto races = o::iterated_racing([&](const auto& point, std::uint64_t) {
    ++calls;
    return score(point);
  }, domain, {53, 7, 5, 13});
  EXPECT_EQ(calls, 53);
  EXPECT_EQ(races.evaluations, 53);
  EXPECT_EQ(score(races.point), races.value);
}

TEST(MixedSearch, RejectsInvalidParentBeforeCallingObjective) {
  auto invalid = domain;
  invalid.front().condition = o::ParameterCondition{1, 0};
  EXPECT_THROW(o::mixed_search([](const auto&) { ADD_FAILURE(); return 0; }, invalid),
               std::invalid_argument);
}
