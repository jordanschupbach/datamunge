#include <datamunge/optim/discrete_replica_exchange.hpp>
#include <datamunge/optim/tabu_search.hpp>

#include <gtest/gtest.h>

namespace o = datamunge::optim;

TEST(TabuSearch, AspirationAllowsImprovingTabuMove) {
  for (const double final : {3.0, 7.0}) {
    const auto loss = [final](int state) { return std::vector{5.0, 6.0, final}[state]; };
    const auto moves = [](int state, auto&) {
      return state < 2 ? std::vector{1} : std::vector<int>{};
    };
    const auto apply = [](int& state, int move) { state += move; };
    const auto keys = [](int, int move) { return std::pair{move, move}; };
    const auto result = o::tabu_search(0, loss, moves, apply, keys, {10, 42, 2});
    EXPECT_EQ(result.evaluations, 3);
    EXPECT_EQ(result.stop, o::NeighborhoodStop::no_move);
    EXPECT_EQ(result.best, final < 5 ? 2 : 0);
    EXPECT_EQ(result.accepted, final < 5 ? 2 : 1);
  }
}

TEST(TabuSearch, ConsumesAvailablePrefixOfNeighborhood) {
  const auto result = o::tabu_search(
      5, [](int x) { return x; },
      [](int, auto&) { return std::vector{-1, -2, -3}; },
      [](int& x, int move) { x += move; },
      [](int, int move) { return std::pair{move, -move}; }, {3, 7, 4});
  EXPECT_EQ(result.best, 3);
  EXPECT_EQ(result.value, 3);
  EXPECT_EQ(result.evaluations, 3);
  EXPECT_EQ(result.accepted, 1);
  EXPECT_EQ(result.stop, o::NeighborhoodStop::budget);
}

TEST(DiscreteReplicaExchange, BudgetCanEndWhileInitializingAndDuringSweep) {
  const auto loss = [](int x) { return x; };
  std::vector<int> proposed_from;
  const auto propose = [&](int x, auto&) -> std::optional<int> {
    proposed_from.push_back(x);
    return 0;
  };
  const auto apply = [](int&, int) {};
  const auto partial = o::discrete_replica_exchange(std::vector{10, 0}, loss, propose, apply,
                                                    {1, 42, {1, 4}});
  EXPECT_EQ(partial.search.best, 10);
  EXPECT_EQ(partial.search.evaluations, 1);
  EXPECT_EQ(partial.exchanges, 0);
  EXPECT_TRUE(proposed_from.empty());

  const auto exchanged = o::discrete_replica_exchange(std::vector{10, 0}, loss, propose, apply,
                                                      {5, 42, {1, 4}});
  EXPECT_EQ(exchanged.search.best, 0);
  EXPECT_EQ(exchanged.search.evaluations, 5);
  EXPECT_EQ(exchanged.search.accepted, 3);
  EXPECT_EQ(exchanged.exchanges, 1);
  EXPECT_EQ(exchanged.accepted_exchanges, 1);
  EXPECT_EQ(proposed_from, (std::vector{10, 0, 0}));
}

TEST(DiscreteReplicaExchange, ValidatesLadderAndStopsWhenNoReplicaHasMoves) {
  const auto propose = [](int, auto&) -> std::optional<int> { return {}; };
  const auto apply = [](int&, int) {};
  const auto loss = [](int x) { return x * x; };
  EXPECT_THROW(o::discrete_replica_exchange(std::vector{1, 3}, loss, propose, apply, {5, 1, {1, 1}}),
               std::invalid_argument);
  const auto result = o::discrete_replica_exchange(std::vector{1, 3}, loss, propose, apply,
                                                   {5, 1, {1, 3}});
  EXPECT_EQ(result.search.evaluations, 2);
  EXPECT_EQ(result.search.stop, o::NeighborhoodStop::no_move);
  EXPECT_EQ(result.exchanges, 0);
}

TEST(ReactiveTabuSearch, DuplicateStatesCanTriggerAnEscape) {
  std::vector<int> visited;
  const auto score = [](int x) { return x; };
  const auto enumerate = [&](int state, auto&) {
    visited.push_back(state);
    return std::vector{0, 1};
  };
  const auto apply = [](int& x, int move) { x = move; };
  const auto keys = [](int, int move) { return std::pair{move, move}; };
  const auto state_key = [](int state) { return state; };
  o::ReactiveTabuOptions options;
  options.tabu = {41, 7, 0};
  options.minimum_tenure = 0;
  options.maximum_tenure = 0;
  options.repeats_before_escape = 1;
  const auto result = o::reactive_tabu_search(0, score, enumerate, apply, keys, state_key, options);
  EXPECT_EQ(result.evaluations, 41);
  EXPECT_EQ(result.best, 0);
  EXPECT_NE(std::ranges::find(visited, 1), visited.end());
}
