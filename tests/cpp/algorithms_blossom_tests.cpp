#include <gtest/gtest.h>

#include <datamunge/algorithms/blossom.hpp>

using datamunge::algorithms::blossom_maximum_matching;
using datamunge::algorithms::kGeneralMatchingUnmatched;

TEST(Blossom, ContractsOddCycleToFindMaximumMatching) {
  // The triangle 0-1-2 is a blossom. The stem and two outside vertices permit
  // three pairs, but a search that never contracts the triangle can get stuck at two.
  const auto result = blossom_maximum_matching(
      6, {{0, 1}, {1, 2}, {2, 0}, {1, 3}, {3, 4}, {2, 5}});
  EXPECT_EQ(result.size, 3U);
  for (std::size_t v = 0; v < result.mate.size(); ++v) {
    ASSERT_NE(result.mate[v], kGeneralMatchingUnmatched);
    EXPECT_EQ(result.mate[result.mate[v]], v);
  }
}

TEST(Blossom, HandlesDisconnectedAndUnmatchedVertices) {
  const auto result = blossom_maximum_matching(7, {{0, 1}, {1, 2}, {2, 0}, {3, 4}});
  EXPECT_EQ(result.size, 2U);
  EXPECT_EQ(result.mate[5], kGeneralMatchingUnmatched);
  EXPECT_EQ(result.mate[6], kGeneralMatchingUnmatched);
}

TEST(Blossom, RejectsInvalidEndpointsAndIgnoresLoops) {
  EXPECT_THROW((void)blossom_maximum_matching(3, {{0, 3}}), std::invalid_argument);
  EXPECT_EQ(blossom_maximum_matching(2, {{0, 0}, {0, 1}, {0, 1}}).size, 1U);
}
