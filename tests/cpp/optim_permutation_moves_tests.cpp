#include <datamunge/optim/permutation_moves.hpp>

#include <gtest/gtest.h>

#include <set>
#include <utility>

namespace o = datamunge::optim;

namespace {

/// @brief Undirected edges of a closed tour.
auto edges(const std::vector<std::size_t>& tour) {
  std::set<std::pair<std::size_t, std::size_t>> result;
  for (std::size_t i = 0; i < tour.size(); ++i) {
    result.insert(std::minmax(tour[i], tour[(i + 1) % tour.size()]));
  }
  return result;
}

} // namespace

TEST(PermutationMoves, TwoAndThreeOptPreservePermutationAndReplaceExactEdges) {
  const std::vector<std::size_t> tour{0, 1, 2, 3, 4, 5, 6, 7};
  const auto original = edges(tour);
  std::mt19937_64 rng(17);
  for (int i = 0; i < 100; ++i) {
    auto two = tour, three = tour;
    o::apply_two_opt(two, *o::sample_two_opt(tour.size(), rng));
    o::apply_three_opt(three, *o::sample_three_opt(tour.size(), rng));
    const auto check = [&](std::vector<std::size_t> changed, std::size_t removed) {
      EXPECT_EQ(changed.front(), 0);
      const auto updated = edges(changed);
      const auto preserved = std::ranges::count_if(original, [&](const auto& e) {
        return updated.contains(e);
      });
      EXPECT_EQ(static_cast<std::size_t>(preserved), original.size() - removed);
      std::ranges::sort(changed);
      EXPECT_EQ(changed, tour);
    };
    check(two, 2);
    check(three, 3);
  }
}

TEST(PermutationMoves, RejectsAdjacentCutsWithoutMutation) {
  std::mt19937_64 rng(7);
  EXPECT_FALSE(o::sample_two_opt(3, rng));
  EXPECT_FALSE(o::sample_three_opt(5, rng));
  const std::vector<std::size_t> initial{0, 1, 2, 3, 4, 5};
  auto tour = initial;
  EXPECT_THROW(o::apply_two_opt(tour, {1, 6}), std::invalid_argument);
  EXPECT_THROW(o::apply_three_opt(tour, {1, 2, 4, o::ThreeOptPattern::exchange}), std::invalid_argument);
  EXPECT_EQ(tour, initial);
}
