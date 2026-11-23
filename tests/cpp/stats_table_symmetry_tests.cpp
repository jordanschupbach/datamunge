#include <datamunge/stats/table_symmetry.hpp>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using namespace datamunge::stats;

TEST(TableSymmetry, PairedTestsMatchRChiSquareCalculations) {
  const std::vector<std::vector<std::size_t>> counts{{30, 8, 5}, {2, 24, 10}, {7, 3, 18}};
  const auto marginal = stuart_maxwell_test(counts);
  EXPECT_DOUBLE_EQ(marginal.parameter1, 2);
  EXPECT_NEAR(marginal.statistic, 1.1576354679802956, 1e-14);
  EXPECT_NEAR(marginal.p_value, .5605607068272043, 1e-14);
  const auto symmetry = bowker_test(counts);
  EXPECT_DOUBLE_EQ(symmetry.parameter1, 3);
  EXPECT_NEAR(symmetry.statistic, 7.7025641025641027, 1e-14);
  EXPECT_NEAR(symmetry.p_value, .0525758968886514, 1e-14);
}

TEST(TableSymmetry, SparseTransitionsUseTheirEffectiveDegreesOfFreedom) {
  const auto large = std::numeric_limits<std::size_t>::max();
  const std::vector<std::vector<std::size_t>> counts{
      {1, large, 0, 0}, {large - 1, 1, 1, 0}, {0, 0, 1, 0}, {0, 0, 0, 4}};
  const auto marginal = stuart_maxwell_test(counts);
  const auto symmetry = bowker_test(counts);
  EXPECT_DOUBLE_EQ(marginal.parameter1, 2);
  EXPECT_DOUBLE_EQ(symmetry.parameter1, 2);
  EXPECT_NEAR(marginal.statistic, 1, 1e-9);
  EXPECT_DOUBLE_EQ(marginal.statistic, symmetry.statistic);

  const auto cycle = std::vector<std::vector<std::size_t>>{{2, 9, 0}, {0, 2, 9}, {9, 0, 2}};
  EXPECT_DOUBLE_EQ(stuart_maxwell_test(cycle).statistic, 0);
  EXPECT_DOUBLE_EQ(bowker_test(cycle).statistic, 27);
  for (const auto test : {stuart_maxwell_test, bowker_test}) {
    const auto diagonal = test({{4, 0}, {0, 1}});
    EXPECT_DOUBLE_EQ(diagonal.statistic, 0);
    EXPECT_DOUBLE_EQ(diagonal.parameter1, 0);
    EXPECT_DOUBLE_EQ(diagonal.p_value, 1);
    EXPECT_THROW(test({{0, 0}, {0, 0}}), std::invalid_argument);
    EXPECT_THROW(test({{1, 0}, {0}}), std::invalid_argument);
  }
}

TEST(TableSymmetry, BreslowDayMatchesFixedMarginFitsInR) {
  std::vector<std::array<std::size_t, 4>> counts{{8, 2, 4, 6}, {5, 3, 2, 6},
                                                {13, 2, 3, 17}, {0, 9, 4, 8}};
  const auto adjusted = breslow_day_test(counts);
  const auto raw = breslow_day_test(counts, false);
  EXPECT_DOUBLE_EQ(adjusted.parameter1, 3);
  EXPECT_NEAR(adjusted.estimate1, 3.979865771812080, 1e-14);
  EXPECT_NEAR(raw.statistic, 17.04756910285942, 1e-12);
  EXPECT_NEAR(raw.p_value, .0006909987854626963, 1e-14);
  EXPECT_NEAR(adjusted.statistic, 16.76936934961211, 1e-12);
  EXPECT_NEAR(adjusted.p_value, .0007882732733776489, 1e-14);

  counts.push_back({5, 0, 4, 0});
  EXPECT_DOUBLE_EQ(breslow_day_test(counts).statistic, adjusted.statistic);
  for (auto& [a, b, c, d] : counts) {
    std::swap(a, b);
    std::swap(c, d);
  }
  const auto swapped = breslow_day_test(counts);
  EXPECT_NEAR(swapped.statistic, adjusted.statistic, 1e-12);
  EXPECT_NEAR(swapped.estimate1, 1 / adjusted.estimate1, 1e-14);
  EXPECT_THROW(breslow_day_test({{1, 0, 0, 3}, {4, 0, 0, 2}}), std::invalid_argument);
  EXPECT_THROW(breslow_day_test({{1, 1, 1, 1}, {5, 0, 4, 0}}), std::invalid_argument);
}

} // namespace
