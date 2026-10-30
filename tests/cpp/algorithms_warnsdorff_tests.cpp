#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/warnsdorff.hpp>
#include <limits>
#include <set>
namespace alg = datamunge::algorithms;
using DF      = datamunge::dstruct::DataFrame;
namespace {
bool independent_knight_edge(std::pair<int, int> a, std::pair<int, int> b) {
  const int dr = a.first - b.first, dc = a.second - b.second;
  return dr * dr + dc * dc == 5;
}
DF ordered_grids(std::vector<std::string>& names) {
  DF data;
  for (int j = 0; j < 64; ++j) {
    const auto name = "p" + std::to_string(j);
    names.push_back(name);
    data.add_column(name, std::vector<double>{double(j), double(2 * j + 1)});
  }
  return data;
}
} // namespace
TEST(WarnsdorffDiagnostics, DefaultPathTraceAndCountersDescribeTheActualChoice) {
  const auto result = alg::warnsdorff_attempt(8, 8);
  ASSERT_TRUE(result.completed);
  EXPECT_FALSE(result.closed);
  ASSERT_EQ(result.path.size(), 64);
  EXPECT_EQ(result.path, alg::warnsdorff_knight_tour(8, 8));
  EXPECT_EQ((std::set<std::pair<int, int>>(result.path.begin(), result.path.end()).size()), 64);
  for (std::size_t i = 1; i < result.path.size(); ++i)
    EXPECT_TRUE(independent_knight_edge(result.path[i - 1], result.path[i]));
  ASSERT_EQ(result.trace.nrows(), 63);
  EXPECT_EQ(result.candidate_probes, 8 * result.trace.nrows());
  EXPECT_EQ(result.trace.double_at("minimum_degree", 0), 5);
  EXPECT_EQ(result.trace.double_at("ties", 0), 2);
  EXPECT_EQ(result.trace.double_at("selected_row", 0), 1);
  EXPECT_EQ(result.trace.double_at("selected_column", 0), 2);
  double legal = 0;
  for (std::size_t i = 0; i < result.trace.nrows(); ++i)
    legal += result.trace.double_at("legal_moves", i);
  EXPECT_EQ(result.degree_probes, 8 * legal);
  EXPECT_LE(result.degree_probes, 64 * result.trace.nrows());
  for (std::size_t i = 0; i < 64; ++i)
    EXPECT_EQ(
        result.path_table.double_at("row_major_index", i),
        result.path[i].first * 8 + result.path[i].second);
}
TEST(WarnsdorffDiagnostics, FailedPrefixRemainsLegalWhileLegacyReturnsEmpty) {
  const auto result = alg::warnsdorff_attempt(8, 8, 3, 3);
  EXPECT_FALSE(result.completed);
  EXPECT_FALSE(result.closed);
  ASSERT_EQ(result.path.size(), 62);
  EXPECT_TRUE(alg::warnsdorff_knight_tour(8, 8, 3, 3).empty());
  ASSERT_EQ(result.trace.nrows(), 62);
  EXPECT_EQ(result.trace.string_at("action", 61), "stuck");
  EXPECT_EQ(result.trace.double_at("legal_moves", 61), 0);
  EXPECT_EQ(result.trace.double_at("minimum_degree", 61), -1);
  for (std::size_t i = 1; i < result.path.size(); ++i)
    EXPECT_TRUE(independent_knight_edge(result.path[i - 1], result.path[i]));
}
TEST(WarnsdorffDiagnostics, ProfilesRespectColourConstraintsAndClosedTourRotations) {
  const auto small = alg::warnsdorff_start_profile(5, 5);
  double     count = 0;
  for (std::size_t i = 0; i < small.nrows(); ++i) {
    const auto complete = small.double_at("completed", i);
    count += complete;
    if (small.double_at("colour", i) == 1)
      EXPECT_EQ(complete, 0);
  }
  EXPECT_EQ(count, 10);
  const auto chess     = alg::warnsdorff_start_profile(8, 8);
  double     completed = 0, closed = 0;
  for (std::size_t i = 0; i < chess.nrows(); ++i) {
    completed += chess.double_at("completed", i);
    closed += chess.double_at("closed", i);
  }
  EXPECT_EQ(completed, 61);
  EXPECT_EQ(closed, 6);
  const auto cycle = alg::warnsdorff_attempt(8, 8, 4, 5);
  ASSERT_TRUE(cycle.closed);
  for (std::size_t offset = 0; offset < 64; ++offset) {
    auto rotated = cycle.path;
    std::rotate(rotated.begin(), rotated.begin() + offset, rotated.end());
    for (std::size_t j = 1; j < rotated.size(); ++j)
      ASSERT_TRUE(independent_knight_edge(rotated[j - 1], rotated[j]));
  }
}
TEST(WarnsdorffDiagnostics, ChecksLimitsBeforeAllocationAndHandlesTinyBoards) {
  const auto one = alg::warnsdorff_attempt(1, 1);
  EXPECT_TRUE(one.completed);
  EXPECT_FALSE(one.closed);
  EXPECT_EQ(one.trace.nrows(), 0);
  const auto two = alg::warnsdorff_attempt(1, 2);
  EXPECT_FALSE(two.completed);
  EXPECT_EQ(two.path.size(), 1);
  EXPECT_THROW((void)alg::warnsdorff_attempt(0, 8), std::invalid_argument);
  EXPECT_THROW((void)alg::warnsdorff_attempt(8, 8, -1, 0), std::invalid_argument);
  EXPECT_THROW((void)alg::warnsdorff_attempt(8, 8, 0, 8), std::invalid_argument);
  EXPECT_THROW((void)alg::warnsdorff_attempt(8, 8, 0, 0, 0), std::invalid_argument);
  EXPECT_THROW((void)alg::warnsdorff_attempt(8, 8, 0, 0, 63), std::length_error);
  EXPECT_THROW(
      (void)alg::warnsdorff_attempt(std::numeric_limits<int>::max(), 3, 0, 0, 100),
      std::length_error);
  EXPECT_THROW((void)alg::warnsdorff_start_profile(17, 17), std::length_error);
  EXPECT_THROW((void)alg::warnsdorff_knight_tour(-1, 8), std::invalid_argument);
}
TEST(KnightGrid, FullPathIsAnInvertiblePermutationAndConservesIntegerSums) {
  std::vector<std::string> names;
  const auto               data    = ordered_grids(names);
  auto                     attempt = alg::warnsdorff_attempt(8, 8);
  attempt.completed = false; // Consumers validate actual coverage, not a caller-editable flag.
  const auto values = alg::knight_tour_grid_values(data, 0, names, attempt);
  for (std::size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(values.double_at("row_major_value", i), i);
    EXPECT_EQ(
        values.double_at("tour_value", i), attempt.path[i].first * 8 + attempt.path[i].second);
  }
  std::vector<double> reconstructed(64), original(64);
  for (std::size_t i = 0; i < 64; ++i) {
    const auto index = static_cast<std::size_t>(
        values.double_at("tour_row", i) * 8 + values.double_at("tour_column", i));
    reconstructed[index] = values.double_at("tour_value", i);
    original[i]          = i;
  }
  EXPECT_EQ(reconstructed, original);
}
TEST(KnightGrid, SummaryAndValidationDoNotAcceptBadPathsOrMissingValues) {
  std::vector<std::string> names;
  auto                     data    = ordered_grids(names);
  const auto               attempt = alg::warnsdorff_attempt(8, 8);
  const auto               summary = alg::knight_tour_grid_summary(data, names, attempt);
  EXPECT_EQ(summary.double_at("value_sum", 0), 2016);
  EXPECT_EQ(summary.double_at("value_sum", 1), 4096);
  EXPECT_EQ(summary.double_at("row_major_variation", 0), 63);
  EXPECT_EQ(summary.double_at("row_major_variation", 1), 126);
  EXPECT_EQ(summary.double_at("sum_difference", 0), 0);
  EXPECT_EQ(summary.double_at("sum_difference", 1), 0);
  EXPECT_EQ(alg::knight_tour_grid_summary(data.head(0), names, attempt).nrows(), 0);
  EXPECT_THROW((void)alg::knight_tour_grid_values(data, 2, names, attempt), std::out_of_range);
  auto bad        = attempt;
  bad.path.back() = bad.path.front();
  EXPECT_THROW((void)alg::knight_tour_grid_summary(data, names, bad), std::invalid_argument);
  EXPECT_THROW(
      (void)alg::knight_tour_grid_summary(data, names, alg::warnsdorff_attempt(8, 8, 3, 3)),
      std::invalid_argument);
  auto duplicate = names;
  duplicate[0]   = duplicate[1];
  EXPECT_THROW(
      (void)alg::knight_tour_grid_summary(data, duplicate, attempt), std::invalid_argument);
  data.set_null(names[0], 0);
  EXPECT_THROW((void)alg::knight_tour_grid_summary(data, names, attempt), std::invalid_argument);
  data.replace_column(
      names[0],
      DF::numeric_column_type(std::vector<double>{std::numeric_limits<double>::infinity(), 0}));
  EXPECT_THROW((void)alg::knight_tour_grid_summary(data, names, attempt), std::invalid_argument);
  std::vector<std::string> huge_names;
  auto                     huge = ordered_grids(huge_names);
  for (const auto& name : huge_names)
    huge.replace_column(
        name, DF::numeric_column_type(std::vector<double>(2, std::numeric_limits<double>::max())));
  EXPECT_THROW((void)alg::knight_tour_grid_summary(huge, huge_names, attempt), std::overflow_error);
  EXPECT_THROW((void)alg::plot_knight_tour(alg::WarnsdorffAttempt{}), std::invalid_argument);
  EXPECT_THROW((void)alg::plot_knight_tour(attempt, true, 63), std::length_error);
}
