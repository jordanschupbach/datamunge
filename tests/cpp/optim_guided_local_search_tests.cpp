#include <datamunge/optim/guided_local_search.hpp>

#include <gtest/gtest.h>

namespace o = datamunge::optim;

TEST(GuidedLocalSearch, PenalizesAFeatureAndEscapesLocalMinimum) {
  const auto score = [](int x) { return std::vector{4, 5, 1}[x]; };
  const auto features = [](int x) { return std::vector<std::size_t>{static_cast<std::size_t>(x)}; };
  const auto moves = [](int x, auto&) { return x < 2 ? std::vector{1} : std::vector<int>{}; };
  const auto apply = [](int& x, int move) { x += move; };
  const auto result = o::guided_local_search(0, score, {5, 1, 0}, features, moves, apply, {8, 7, 2});
  EXPECT_EQ(result.search.evaluations, 4);
  EXPECT_EQ(result.search.accepted, 2);
  EXPECT_EQ(result.search.best, 2);
  EXPECT_EQ(result.search.value, 1);
  EXPECT_EQ(result.penalties, (std::vector<std::size_t>{1, 0, 0}));
}

TEST(GuidedLocalSearch, RejectsMalformedFeatures) {
  const auto score = [](int x) { return x; };
  const auto features = [](int) { return std::vector<std::size_t>{0, 0}; };
  const auto moves = [](int, auto&) { return std::vector<int>{}; };
  const auto apply = [](int& x, int move) { x += move; };
  EXPECT_THROW(o::guided_local_search(0, score, {5}, features, moves, apply), std::invalid_argument);
}
