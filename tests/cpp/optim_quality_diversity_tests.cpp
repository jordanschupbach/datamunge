#include <datamunge/optim/quality_diversity_search.hpp>

#include <gtest/gtest.h>

namespace o = datamunge::optim;

TEST(MapElitesSearch, PreservesQualityAndEarliestEliteOfEachCell) {
  const auto generate = [](auto&) { return 4; };
  const auto mutate = [](int& x, auto&) { x = std::max(x - 1, 0); };
  const auto loss = [](int x) { return x * x; };
  const auto cell = [](int x) { return std::size_t(x / 3); };
  const auto result = o::map_elites_search(generate, mutate, loss, cell, {20, 7, 2, 1});
  EXPECT_EQ(result.evaluations, 20);
  ASSERT_TRUE(result.cells[0]);
  ASSERT_TRUE(result.cells[1]);
  EXPECT_EQ(result.cells[0]->state, 0);
  EXPECT_EQ(result.cells[1]->state, 3);
  EXPECT_EQ(result.best_cell, 0);
  EXPECT_EQ(result.cells[1]->evaluation, 1);
}

TEST(NoveltySearch, ReplaysBothQualityAndNoveltyModesAndBoundsTheArchive) {
  auto generate = [](auto& rng) { return std::uniform_int_distribution<int>(0, 6)(rng); };
  auto mutate = [](int& x, auto& rng) { x = (x + std::uniform_int_distribution<int>(1, 6)(rng)) % 7; };
  auto describe = [](int x) { return x % 3; };
  auto distance = [](int a, int b) { return std::abs(a - b); };
  for (const bool competition : {false, true}) {
    o::NoveltyOptions options{31, 42, 4, 3, 8, 1, competition};
    std::vector<int> history;
    const auto score = [&](int x) {
      history.push_back(x);
      return x*x;
    };
    const auto result = o::novelty_search(generate, mutate, score, describe, distance, options);
    EXPECT_EQ(result.evaluations, 31);
    EXPECT_EQ(result.archive_size, 8);
    EXPECT_EQ(result.loss, result.best*result.best);
    const auto prior = history;
    history.clear();
    const auto second = o::novelty_search(generate, mutate, score, describe, distance, options);
    EXPECT_EQ(second.best, result.best);
    EXPECT_EQ(history, prior);
  }
}
