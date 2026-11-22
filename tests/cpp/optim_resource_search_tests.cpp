#include <datamunge/optim/population_based_training.hpp>
#include <datamunge/optim/resource_search.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <stdexcept>

namespace o = datamunge::optim;

TEST(ResourceSearch, TransfersTrialStateAndChargesOnlyAddedFidelity) {
  for (const auto method : {o::ResourceSearchMethod::successive_halving, o::ResourceSearchMethod::hyperband,
                            o::ResourceSearchMethod::asha}) {
    o::ResourceSearchOptions options;
    options.resource_budget = 29;
    options.trials = 8;
    options.maximum_resource = 4;
    options.trace_limit = 29;
    options.method = method;
    std::vector<std::size_t> resources;
    const auto start = [&](std::size_t trial, std::uint64_t) {
      EXPECT_EQ(resources.size(), trial);
      resources.push_back(0);
    };
    const auto advance = [&](std::size_t trial, std::size_t previous, std::size_t next) {
      EXPECT_EQ(previous, resources.at(trial));
      EXPECT_LT(previous, next);
      resources[trial] = next;
      return static_cast<double>(trial) - next;
    };
    const auto duration = [](std::size_t trial, std::size_t added) { return (trial % 3 + 1) * static_cast<double>(added); };
    const auto run = o::resource_search(start, advance, options, duration);
    ASSERT_TRUE(run.best);
    EXPECT_LE(run.consumed, options.resource_budget);
    EXPECT_EQ(run.consumed, std::accumulate(resources.begin(), resources.end(), std::size_t{0}));
    EXPECT_EQ(run.starts, resources.size());
    EXPECT_EQ(run.trace.size(), run.evaluations);
    EXPECT_EQ(run.best->resource, *std::ranges::max_element(resources));
    double previous = 0;
    for (const auto& event : run.trace) {
      if (method == o::ResourceSearchMethod::asha) {
        EXPECT_GE(event.completion, previous);
        previous = event.completion;
      } else {
        EXPECT_EQ(event.completion, -1);
      }
    }
  }
}

TEST(ResourceSearch, RespectsBudgetBeforeStartingAWorker) {
  o::ResourceSearchOptions options;
  options.method = o::ResourceSearchMethod::asha;
  options.resource_budget = 1;
  options.minimum_resource = 2;
  auto start = [](std::size_t, std::uint64_t) { ADD_FAILURE(); };
  auto advance = [](std::size_t, std::size_t, std::size_t) { ADD_FAILURE(); return 0; };
  const auto run = o::resource_search(start, advance, options);
  EXPECT_EQ(run.starts, 0);
  EXPECT_EQ(run.consumed, 0);
  EXPECT_FALSE(run.best);
}

TEST(PopulationBasedTraining, CopiesTrainedCheckpointsAndStopsAtPartialRound) {
  struct Worker {
    int progress;
    int rate;
  };
  const auto advance = [](Worker& worker, std::size_t before, std::size_t after) {
    worker.progress += static_cast<int>(after - before) * worker.rate;
    return -worker.progress;
  };
  const auto explore = [](Worker& worker, auto&) { worker.rate += 1; };
  const auto run = o::population_based_training(std::vector<Worker>{{0, 4}, {0, 1}, {0, 2}},
                                                advance, explore, {10, 7, 2, 8, 1});
  ASSERT_TRUE(run.best);
  EXPECT_EQ(run.consumed, 10);
  EXPECT_EQ(run.evaluations, 5);
  EXPECT_EQ(run.replacements, 1);
  EXPECT_EQ(run.workers[1].progress, 18);
  EXPECT_EQ(run.best->worker, 1);
  EXPECT_EQ(run.best->resource, 4);
  EXPECT_EQ(run.best->loss, -18);
}
