#include <datamunge/optim/discrete_genetic_search.hpp>

#include <gtest/gtest.h>

#include <algorithm>

namespace o = datamunge::optim;

TEST(DiscreteGeneticSearch, OrderCrossoverAndMutationPreservePermutationsAcrossIslands) {
  const o::CategoricalPoint ordered{0, 1, 2, 3, 4, 5, 6};
  for (const bool steady : {false, true}) {
    for (const auto budget : {1u, 9u, 150u}) {
      o::DiscreteGeneticOptions options{budget, 42, o::GeneticEncoding::permutation, 8};
      options.steady_state = steady;
      options.islands = 3;
      options.crossover = 1;
      options.mutation = 1;
      options.migration_interval = 2;
      std::vector<o::CategoricalPoint> seen;
      const auto score = [&](const o::CategoricalPoint& x) {
        auto sorted = x;
        std::ranges::sort(sorted);
        EXPECT_EQ(sorted, ordered);
        seen.push_back(x);
        double value = 0;
        for (std::size_t j = 0; j < x.size(); ++j) value += x[j] == j ? 0 : 1;
        return value;
      };
      const auto first = o::discrete_genetic_search(score, ordered.size(), options);
      EXPECT_EQ(first.evaluations, budget);
      EXPECT_EQ(seen.size(), budget);
      const auto prior = seen;
      seen.clear();
      const auto again = o::discrete_genetic_search(score, ordered.size(), options);
      EXPECT_EQ(again.point, first.point);
      EXPECT_EQ(again.value, first.value);
      EXPECT_EQ(seen, prior);
    }
  }
}

TEST(DiscreteGeneticSearch, BinaryMatchesKnownAnswerAndCountsDuplicateChildren) {
  auto calls = 0u;
  const auto result = o::discrete_genetic_search([&](const auto& genes) {
    ++calls;
    EXPECT_LE(genes.front(), 1);
    return static_cast<double>(genes.front());
  }, 1, {47, 123});
  EXPECT_EQ(calls, 47);
  EXPECT_EQ(result.evaluations, 47);
  EXPECT_EQ(result.point, o::CategoricalPoint{0});
  EXPECT_EQ(result.value, 0);
}
