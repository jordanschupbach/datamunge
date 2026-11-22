#include <datamunge/optim/grasp.hpp>
#include <datamunge/optim/path_relink.hpp>
#include <datamunge/optim/scatter_search.hpp>

#include <gtest/gtest.h>

#include <numeric>

namespace o = datamunge::optim;

TEST(Grasp, RestrictedListAndReactiveReplay) {
  std::mt19937_64 rng(42);
  EXPECT_EQ(o::grasp_choose(std::vector{std::pair{3, 4.0}, std::pair{9, 8.0}}, 0, rng), 3);
  EXPECT_FALSE(o::grasp_choose(std::vector<std::pair<int, double>>{}, 0.5, rng));
  const auto score = [](int x) { return x; };
  const auto construct = [](double alpha, auto&) -> std::optional<int> { return alpha == 0 ? 0 : 10; };
  const auto enumerate = [](int, std::size_t, auto&) { return std::vector<int>{}; };
  const auto apply = [](int& x, int m) { x += m; };
  o::GraspOptions options{{100, 7}, {0, 1}, 4};
  const auto result = o::grasp_search(20, score, construct, enumerate, apply, options);
  const auto replay = o::grasp_search(20, score, construct, enumerate, apply, options);
  EXPECT_EQ(result.search.best, 0);
  EXPECT_EQ(result.search.evaluations, 100);
  EXPECT_EQ(result.constructions, 99);
  EXPECT_EQ(result.alpha_probabilities, replay.alpha_probabilities);
  ASSERT_EQ(result.alpha_probabilities.size(), 2);
  EXPECT_GT(result.alpha_probabilities[0], result.alpha_probabilities[1]);
  EXPECT_NEAR(std::accumulate(result.alpha_probabilities.begin(), result.alpha_probabilities.end(), 0.0), 1, 1e-15);
}

TEST(PathRelink, ScoresOnlyForwardMovesAndKeepsBestIntermediateState) {
  const auto score = [](int x) { return x == 1 ? 0 : 4; };
  const auto distance = [](int x, int y) { return std::size_t(std::abs(x - y)); };
  const auto enumerate = [](int, int, auto&) { return std::vector{1, -1}; };
  const auto apply = [](int& x, int move) { x += move; };
  const auto result = o::path_relink(2, 0, score, distance, enumerate, apply, 8);
  EXPECT_EQ(result.search.evaluations, 3);
  EXPECT_EQ(result.search.accepted, 2);
  EXPECT_EQ(result.search.best, 1);
  EXPECT_EQ(result.search.value, 0);
  EXPECT_TRUE(result.reached_guide);
  const auto truncated = o::path_relink(2, 0, score, distance, enumerate, apply, 2);
  EXPECT_EQ(truncated.search.evaluations, 2);
  EXPECT_EQ(truncated.search.best, 1);
  EXPECT_FALSE(truncated.reached_guide);
}

TEST(GraspPathRelink, SharesBudgetAndCanFindBestIntermediateState) {
  const auto score = [](int x) { return (x - 1) * (x - 1); };
  const auto construct = [](double, auto&) -> std::optional<int> { return 2; };
  const auto local = [](int, std::size_t, auto&) { return std::vector<int>{}; };
  const auto distance = [](int x, int y) { return std::size_t(std::abs(x - y)); };
  const auto forward = [](int, int, auto&) { return std::vector{-1}; };
  const auto apply = [](int& x, int move) { x += move; };
  const auto result = o::grasp_path_relink(0, score, construct, local, apply, distance, forward, apply, {{3}});
  EXPECT_EQ(result.search.best, 1);
  EXPECT_EQ(result.search.value, 0);
  EXPECT_EQ(result.search.evaluations, 3);
  EXPECT_EQ(result.constructions, 1);
}


TEST(ScatterSearch, RelinksDiverseReferencesUnderOneBudget) {
  const auto score = [](int x) { return x == 1 ? 0 : 5 + x; };
  const auto diversify = [](auto&) -> std::optional<int> { return 2; };
  const auto distance = [](int a, int b) { return std::size_t(std::abs(a - b)); };
  const auto forward = [](int, int, auto&) { return std::vector{-1}; };
  const auto apply = [](int& x, int move) { x += move; };
  const auto result = o::scatter_search(0, score, diversify, distance, forward, apply, {4, 7, 2, 1, 1});
  EXPECT_EQ(result.best, 1);
  EXPECT_EQ(result.value, 0);
  EXPECT_EQ(result.evaluations, 4);
  EXPECT_EQ(result.accepted, 2);
}
