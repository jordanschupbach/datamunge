#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/marching_squares.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(MarchingSquaresTrace, AllSixteenActualCasesAndBothSaddleRules) {
  const std::vector<std::vector<std::array<unsigned, 2>>> pairs{
      {},
      {{3, 0}},
      {{0, 1}},
      {{3, 1}},
      {{1, 2}},
      {{3, 0}, {1, 2}},
      {{0, 2}},
      {{3, 2}},
      {{3, 2}},
      {{0, 2}},
      {{0, 1}, {3, 2}},
      {{1, 2}},
      {{3, 1}},
      {{0, 1}},
      {{3, 0}},
      {}};
  for (unsigned mask = 0; mask < 16; ++mask)
    for (auto rule :
         {MarchingSaddleConnectivity::AboveDisconnected,
          MarchingSaddleConnectivity::AboveConnected}) {
      std::array<double, 4> v{};
      for (unsigned k = 0; k < 4; ++k)
        v[k] = (mask & (1u << k)) ? 1 : 0;
      std::vector<std::vector<double>> grid{{v[0], v[1]}, {v[3], v[2]}};
      auto                             t = marching_squares_trace(grid, .5, rule);
      ASSERT_EQ(t.cells.size(), 1);
      EXPECT_EQ(t.cells[0].case_index, mask);
      auto expected = pairs[mask];
      if (rule == MarchingSaddleConnectivity::AboveConnected && (mask == 5 || mask == 10))
        expected = pairs[mask == 5 ? 10 : 5];
      ASSERT_EQ(t.analysis.segments.size(), expected.size());
      const std::array<std::array<double, 2>, 4> points{{{.5, 0}, {1, .5}, {.5, 1}, {0, .5}}};
      for (std::size_t i = 0; i < expected.size(); ++i) {
        auto a = points[expected[i][0]], b = points[expected[i][1]];
        EXPECT_EQ(t.analysis.segments[i], (ContourSegment{a[0], a[1], b[0], b[1]}));
      }
      EXPECT_EQ(t.analysis.work.cells, 1);
      EXPECT_EQ(t.analysis.work.classifications, 4);
      EXPECT_EQ(t.analysis.work.interpolations, t.crossings.size());
      EXPECT_EQ(t.analysis.work.segments, expected.size());
      EXPECT_EQ(t.captured_entries, 4 + 1 + t.crossings.size() + expected.size());
      for (auto& cross : t.crossings) {
        EXPECT_EQ(cross.fraction, .5);
        EXPECT_FALSE(cross.scaled);
        EXPECT_EQ(cross.point, points[cross.edge]);
      }
      if (rule == MarchingSaddleConnectivity::AboveDisconnected)
        EXPECT_EQ(t.analysis.segments, marching_squares(grid, .5));
    }
}
TEST(MarchingSquaresTrace, OriginalRadialAndExactAffineCrossings) {
  auto original = marching_radial_study();
  EXPECT_EQ(original.grid.size(), 21);
  EXPECT_EQ(original.contour.segments.size(), 44);
  EXPECT_NEAR(original.mean_midpoint_error, .0215291, 1e-7);
  EXPECT_NEAR(original.maximum_midpoint_error, .0279425, 1e-7);
  auto t = marching_squares_trace(original.grid, 6);
  EXPECT_EQ(t.analysis.segments, original.contour.segments);
  EXPECT_EQ(t.cells.size(), 400);
  std::size_t previous = 0;
  for (auto& cell : t.cells) {
    EXPECT_EQ(cell.segment_begin, previous);
    EXPECT_LE(cell.segment_end, cell.segment_begin + 2);
    previous = cell.segment_end;
  }
  EXPECT_EQ(previous, t.analysis.segments.size());
  std::vector<std::vector<double>> grid(5, std::vector<double>(7));
  for (std::size_t r = 0; r < 5; ++r)
    for (std::size_t c = 0; c < 7; ++c)
      grid[r][c] = 2 * c + 3 * r;
  auto affine = marching_squares_trace(grid, 7.5);
  for (auto& seg : affine.analysis.segments) {
    EXPECT_NEAR(2 * seg[0] + 3 * seg[1], 7.5, 1e-14);
    EXPECT_NEAR(2 * seg[2] + 3 * seg[3], 7.5, 1e-14);
  }
  EXPECT_TRUE(marching_radial_study(1, 0).midpoint_errors.empty());
}
TEST(MarchingSquaresTrace, TiesFiniteRangeCompleteGuardsQuotaAndOwnership) {
  EXPECT_TRUE(marching_squares({{1, 1}, {1, 1}}, 1).empty());
  auto tied = marching_squares_trace({{0, -1}, {-1, -1}}, 0);
  ASSERT_EQ(tied.analysis.segments.size(), 1);
  EXPECT_EQ(tied.analysis.segments.front(), (ContourSegment{0, 0, 0, 0}));
  EXPECT_EQ(tied.crossings.size(), 2);
  const double m       = std::numeric_limits<double>::max();
  auto         extreme = marching_squares_trace({{-m, m}, {-m, m}}, 0);
  ASSERT_EQ(extreme.crossings.size(), 2);
  for (auto& c : extreme.crossings) {
    EXPECT_TRUE(c.scaled);
    EXPECT_EQ(c.fraction, .5);
  }
  EXPECT_EQ(extreme.analysis.segments[0], (ContourSegment{.5, 0, .5, 1}));
  EXPECT_THROW(marching_squares({{1}, {1, 2}}, 0), std::invalid_argument);
  EXPECT_THROW(marching_squares({{NAN}}, 0), std::invalid_argument);
  EXPECT_THROW(marching_squares({}, NAN), std::invalid_argument);
  EXPECT_THROW(
      marching_squares_trace({}, 0, static_cast<MarchingSaddleConnectivity>(8)),
      std::invalid_argument);
  EXPECT_THROW(marching_radial_study(3, -1), std::invalid_argument);
  EXPECT_TRUE(marching_squares({}, 0).empty());
  EXPECT_TRUE(marching_squares({{}, {}}, 0).empty());
  EXPECT_TRUE(marching_squares({{1, 2, 3}}, 0).empty());
  auto t = marching_squares_trace({{1, 0}, {0, 0}}, .5);
  EXPECT_THROW(
      marching_squares_trace(
          {{1, 0}, {0, 0}},
          .5,
          MarchingSaddleConnectivity::AboveDisconnected,
          t.captured_entries - 1),
      std::length_error);
  EXPECT_EQ(
      marching_squares_trace(
          {{1, 0}, {0, 0}}, .5, MarchingSaddleConnectivity::AboveDisconnected, t.captured_entries)
          .captured_entries,
      t.captured_entries);
  std::vector<std::vector<double>> grid{{1, 0}, {0, 0}};
  auto own   = marching_squares_trace(grid, .5, MarchingSaddleConnectivity::AboveDisconnected, 0);
  grid[0][0] = 9;
  EXPECT_EQ(own.grid[0][0], 1);
}
