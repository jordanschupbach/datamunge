#include <datamunge/optim/neighborhood_search.hpp>

#include <gtest/gtest.h>

#include <limits>

namespace o = datamunge::optim;

TEST(NeighborhoodSearch, TypedMovesTerminateOnNoMove) {
  enum class Move { decrement };
  const auto result = o::neighborhood_search(
      4, [](int x) { return x * x; },
      [](int x, auto&) -> std::optional<Move> {
        if (x > 0) return Move::decrement;
        return {};
      },
      [](int& x, Move) { --x; });
  EXPECT_EQ(result.best, 0);
  EXPECT_EQ(result.value, 0);
  EXPECT_EQ(result.evaluations, 5);
  EXPECT_EQ(result.accepted, 4);
  EXPECT_EQ(result.stop, o::NeighborhoodStop::no_move);
}

TEST(NeighborhoodSearch, AcceptanceChangesTrajectoryAndKeepsBest) {
  const auto step = [](int x, auto&) -> std::optional<int> { return x + 1; };
  const auto apply = [](int& x, int move) { x = move; };
  const auto loss = [](int x) { return x == 1 ? 2 : -x; };
  o::NeighborhoodSearchOptions options;
  options.evaluations = 3;

  const auto greedy = o::neighborhood_search(0, loss, step, apply, options);
  EXPECT_EQ(greedy.best, 0);
  EXPECT_EQ(greedy.accepted, 0);
  EXPECT_EQ(greedy.stop, o::NeighborhoodStop::budget);

  for (const auto policy : {o::NeighborhoodAcceptance::threshold, o::NeighborhoodAcceptance::record,
                            o::NeighborhoodAcceptance::deluge, o::NeighborhoodAcceptance::annealing}) {
    options.acceptance = policy;
    options.tolerance = 2;
    options.initial_level = 2;
    options.initial_temperature = 1e12;
    const auto result = o::neighborhood_search(0, loss, step, apply, options);
    EXPECT_EQ(result.best, 2);
    EXPECT_EQ(result.value, -2);
    EXPECT_EQ(result.accepted, 2);
    EXPECT_EQ(result.evaluations, 3);
  }
}

TEST(NeighborhoodSearch, LateAcceptanceUsesTheOlderIncumbent) {
  const auto step = [](int x, auto&) -> std::optional<int> { return x + 1; };
  const auto apply = [](int& x, int move) { x = move; };
  const auto loss = [](int x) { return std::vector{5, 1, 3, 10}[x]; };
  o::NeighborhoodSearchOptions options;
  options.acceptance = o::NeighborhoodAcceptance::late;
  options.evaluations = 4;
  options.history_length = 2;
  const auto result = o::neighborhood_search(0, loss, step, apply, options);
  EXPECT_EQ(result.best, 1);
  EXPECT_EQ(result.value, 1);
  EXPECT_EQ(result.accepted, 2);
  EXPECT_EQ(result.evaluations, 4);
}

TEST(NeighborhoodSearch, BudgetOneSkipsProposalAndValidationPrecedesScoring) {
  bool called = false;
  const auto step = [&](int, auto&) -> std::optional<int> { called = true; return {}; };
  const auto apply = [](int& x, int move) { x = move; };
  const auto loss = [](int x) { return x; };
  o::NeighborhoodSearchOptions options;
  options.evaluations = 1;
  const auto result = o::neighborhood_search(7, loss, step, apply, options);
  EXPECT_FALSE(called);
  EXPECT_EQ(result.value, 7);
  EXPECT_EQ(result.evaluations, 1);
  options.evaluations = 0;
  EXPECT_THROW(o::neighborhood_search(7, loss, step, apply, options), std::invalid_argument);
  options.evaluations = 1;
  const auto nonfinite = [](int) { return std::numeric_limits<double>::infinity(); };
  EXPECT_THROW(o::neighborhood_search(7, nonfinite, step, apply, options), std::domain_error);
}

TEST(NeighborhoodSearch, ExtendedDelugeSchedulesTargetAndAcceptsStrictImprovement) {
  const auto propose = [](int x, auto&) -> std::optional<int> { return x + 1; };
  const auto apply = [](int& x, int move) { x = move; };
  const auto score = [](int x) { return std::vector{5, 4, 3}[x]; };
  o::NeighborhoodSearchOptions options;
  options.evaluations = 3;
  options.acceptance = o::NeighborhoodAcceptance::extended_deluge;
  options.target_level = 0;
  const auto result = o::neighborhood_search(0, score, propose, apply, options);
  EXPECT_EQ(result.best, 2);
  EXPECT_EQ(result.accepted, 2);
  options.target_level = 6;
  EXPECT_THROW(o::neighborhood_search(0, score, propose, apply, options), std::invalid_argument);
}
