#include <datamunge/optim/variable_neighborhood_search.hpp>

#include <gtest/gtest.h>

namespace o = datamunge::optim;

TEST(VariableNeighborhoodSearch, DescentRestartsAtFirstNeighborhood) {
  const auto score = [](int x) { return x * x; };
  std::vector<std::pair<int, std::size_t>> scans;
  const auto enumerate = [&](int x, std::size_t k, auto&) {
    scans.emplace_back(x, k);
    return k == 0 && x == 1 ? std::vector{-1} : k == 1 && x == 3 ? std::vector{-2} : std::vector<int>{};
  };
  const auto shake = [](int, std::size_t, auto&) -> std::optional<int> { return {}; };
  const auto apply = [](int& x, int move) { x += move; };
  const auto result = o::variable_neighborhood_search(3, score, enumerate, shake, apply,
      o::VariableNeighborhoodMethod::descent, {20, 7, 2});
  EXPECT_EQ(result.best, 0);
  EXPECT_EQ(result.evaluations, 3);
  EXPECT_EQ(result.stop, o::NeighborhoodStop::no_move);
  EXPECT_EQ(scans, (std::vector<std::pair<int, std::size_t>>{{3, 0}, {3, 1}, {1, 0}, {0, 0}, {0, 1}}));
}

TEST(VariableNeighborhoodSearch, BasicAndIteratedDescentCanEscapeThroughWorseState) {
  const auto score = [](int x) { return std::vector{3, 5, 0}[x]; };
  const auto enumerate = [](int x, std::size_t, auto&) {
    return x == 1 ? std::vector{2} : std::vector<int>{};
  };
  const auto apply = [](int& x, int move) { x = move; };
  const auto shake = [](int, std::size_t, auto&) -> std::optional<int> { return 1; };
  const auto perturb = [](int, auto&) -> std::optional<int> { return 1; };
  const auto reduced = o::variable_neighborhood_search(0, score, enumerate, shake, apply,
      o::VariableNeighborhoodMethod::reduced, {3});
  const auto basic = o::variable_neighborhood_search(0, score, enumerate, shake, apply,
      o::VariableNeighborhoodMethod::basic, {3});
  const auto iterated = o::iterated_local_search(0, score, enumerate, perturb, apply, {3});
  EXPECT_EQ(reduced.best, 0);
  EXPECT_EQ(reduced.evaluations, 3);
  EXPECT_EQ(basic.best, 2);
  EXPECT_EQ(basic.evaluations, 3);
  EXPECT_EQ(iterated.best, 2);
  EXPECT_EQ(iterated.evaluations, 3);
}
