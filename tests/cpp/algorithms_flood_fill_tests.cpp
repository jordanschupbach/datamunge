#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/connected_components.hpp>
#include <datamunge/algorithms/flood_fill.hpp>
#include <deque>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
using namespace datamunge::algorithms;
using Grid = std::vector<std::vector<int>>;
namespace {
// Independent breadth-first membership oracle: separate visited mask, deferred paint.
std::pair<Grid, std::size_t> bfs(Grid g, std::size_t sr, std::size_t sc, int value, bool eight) {
  if (g.empty() || g[0].empty() || sr >= g.size() || sc >= g[0].size() || g[sr][sc] == value)
    return {g, 0};
  const int                       target = g[sr][sc];
  std::vector<std::vector<bool>>  seen(g.size(), std::vector<bool>(g[0].size()));
  std::deque<std::pair<int, int>> q{{int(sr), int(sc)}};
  seen[sr][sc]      = true;
  std::size_t count = 0;
  while (!q.empty()) {
    auto [r, c] = q.front();
    q.pop_front();
    ++count;
    for (int dr = -1; dr <= 1; ++dr)
      for (int dc = -1; dc <= 1; ++dc) {
        if ((!dr && !dc) || (!eight && std::abs(dr) + std::abs(dc) != 1))
          continue;
        const int a = r + dr, b = c + dc;
        if (a >= 0 && b >= 0 && a < int(g.size()) && b < int(g[0].size()) && !seen[a][b] &&
            g[a][b] == target) {
          seen[a][b] = true;
          q.emplace_back(a, b);
        }
      }
  }
  for (std::size_t r = 0; r < g.size(); ++r)
    for (std::size_t c = 0; c < g[r].size(); ++c)
      if (seen[r][c])
        g[r][c] = value;
  return {g, count};
}
void compare(const Grid& g, std::size_t r, std::size_t c, int value, bool eight) {
  auto expected = bfs(g, r, c, value, eight);
  auto actual   = g;
  auto w        = flood_fill_work(actual, r, c, value, eight);
  ASSERT_EQ(actual, expected.first);
  ASSERT_EQ(w.filled, expected.second);
  EXPECT_EQ(w.discovered, w.filled);
  EXPECT_EQ(w.neighbor_slots, (eight ? 8 : 4) * w.filled);
  EXPECT_EQ(w.accepted_neighbors, w.filled ? w.filled - 1 : 0);
}
void replay(const FloodFillTrace& t, bool eight) {
  auto                                             g = t.input;
  const auto&                                      w = t.work;
  std::vector<std::pair<std::size_t, std::size_t>> stack;
  std::set<std::pair<std::size_t, std::size_t>>    discovered;
  std::size_t                                      peak = 0, probes = 0, reads = 0, accepted = 0;
  if (w.stop == FloodFillStop::Filled) {
    stack.emplace_back(w.seed_row, w.seed_col);
    discovered.insert(stack[0]);
    g[w.seed_row][w.seed_col] = w.new_value;
    peak                      = 1;
  }
  for (const auto& v : t.visits) {
    ASSERT_FALSE(stack.empty());
    EXPECT_EQ(stack.back(), std::make_pair(v.row, v.col));
    stack.pop_back();
    EXPECT_EQ(v.stack_after_pop, stack.size());
    EXPECT_EQ(v.probe_begin, probes);
    EXPECT_EQ(v.probe_end - v.probe_begin, eight ? 8 : 4);
    for (auto k = v.probe_begin; k < v.probe_end; ++k) {
      const auto& p = t.probes[k];
      const auto& n = p.location;
      EXPECT_EQ(n.direction, k - v.probe_begin);
      ++probes;
      if (!n.in_bounds) {
        EXPECT_EQ(n.row, 0);
        EXPECT_EQ(n.col, 0);
        EXPECT_EQ(p.value_before, 0);
        EXPECT_FALSE(p.accepted);
        continue;
      }
      ++reads;
      EXPECT_EQ(p.value_before, g[n.row][n.col]);
      EXPECT_EQ(p.accepted, p.value_before == w.target);
      if (p.accepted) {
        ++accepted;
        ASSERT_TRUE(discovered.emplace(n.row, n.col).second);
        g[n.row][n.col] = w.new_value;
        stack.emplace_back(n.row, n.col);
        peak = std::max(peak, stack.size());
      }
    }
    EXPECT_EQ(v.stack_after_scan, stack.size());
  }
  EXPECT_TRUE(stack.empty());
  EXPECT_EQ(g, t.output);
  EXPECT_EQ(t.visits.size(), w.filled);
  EXPECT_EQ(probes, w.neighbor_slots);
  EXPECT_EQ(reads, w.in_bounds_neighbors);
  EXPECT_EQ(accepted, w.accepted_neighbors);
  EXPECT_EQ(peak, w.peak_stack);
  EXPECT_EQ(discovered.size(), w.discovered);
}
} // namespace
TEST(FloodFill, ExhaustiveBinaryFourByFourMatchesIndependentBfs) {
  for (unsigned mask = 0; mask < 65536; ++mask) {
    Grid g(4, std::vector<int>(4));
    for (int i = 0; i < 16; ++i)
      g[i / 4][i % 4] = (mask >> i) & 1;
    for (bool eight : {false, true}) {
      compare(g, 0, 0, 9, eight);
      if (mask % 257 == 0)
        for (int seed = 1; seed < 16; ++seed)
          compare(g, seed / 4, seed % 4, 9, eight);
    }
  }
}
TEST(FloodFill, SignedRandomRectanglesAndActualTraceReplay) {
  std::mt19937_64 rng(115);
  for (int fixture = 0; fixture < 600; ++fixture) {
    Grid g(1 + rng() % 13, std::vector<int>(1 + rng() % 15));
    for (auto& row : g)
      for (auto& v : row) {
        auto k = rng() % 12;
        v      = k == 0 ? INT_MIN : k == 1 ? INT_MAX : int(k % 4) - 2;
      }
    auto r = rng() % g.size(), c = rng() % g[0].size();
    for (bool eight : {false, true}) {
      compare(g, r, c, INT_MAX, eight);
      const auto t = flood_fill_trace(g, r, c, INT_MAX, eight);
      replay(t, eight);
      EXPECT_EQ(t.input, g);
    }
  }
}
TEST(FloodFill, TinyOrderAndMarkOnPushAreActual) {
  const auto t = flood_fill_trace({{1, 1}, {1, 1}}, 0, 0, 9);
  const std::vector<std::pair<std::size_t, std::size_t>> order{{0, 0}, {0, 1}, {1, 1}, {1, 0}};
  const std::vector<std::size_t> after_pop{0, 1, 1, 0}, after_scan{2, 2, 1, 0};
  ASSERT_EQ(t.visits.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(std::make_pair(t.visits[i].row, t.visits[i].col), order[i]);
    EXPECT_EQ(t.visits[i].stack_after_pop, after_pop[i]);
    EXPECT_EQ(t.visits[i].stack_after_scan, after_scan[i]);
  }
  EXPECT_EQ(t.work.in_bounds_neighbors, 8);
  EXPECT_EQ(t.work.accepted_neighbors, 3);
  EXPECT_EQ(t.work.peak_stack, 2);
  replay(t, false);
}
TEST(FloodFill, FullShapeValidationPrecedesEveryNoopAndQuota) {
  for (const Grid g : {Grid{{1, 1}, {1}}, Grid{{}, {1}}, Grid{{1}, {1, 1}}})
    for (auto seed : {std::size_t(0), SIZE_MAX})
      for (int value : {1, 9}) {
        auto actual = g;
        EXPECT_THROW(flood_fill(actual, seed, seed, value), std::invalid_argument);
        EXPECT_EQ(actual, g);
        EXPECT_THROW(flood_fill_trace(g, seed, seed, value, false, 1), std::invalid_argument);
      }
}
TEST(FloodFill, NoopStatesAndPreexistingReplacementBarrier) {
  for (const Grid g : {Grid{}, Grid{{}, {}}, Grid{{1}}})
    for (bool eight : {false, true}) {
      compare(g, SIZE_MAX, SIZE_MAX, 9, eight);
      replay(flood_fill_trace(g, SIZE_MAX, SIZE_MAX, 9, eight), eight);
    }
  EXPECT_EQ(flood_fill_trace({}, 0, 0, 9).work.stop, FloodFillStop::EmptyGrid);
  EXPECT_EQ(flood_fill_trace({{1}}, 1, 0, 9).work.stop, FloodFillStop::SeedOutside);
  auto same = flood_fill_trace({{1, 1}}, 0, 0, 1);
  EXPECT_EQ(same.work.stop, FloodFillStop::SameValue);
  EXPECT_TRUE(same.work.has_target);
  EXPECT_TRUE(same.visits.empty());
  Grid g{{1, 9, 1}};
  EXPECT_EQ(flood_fill(g, 0, 0, 9), 1);
  EXPECT_EQ(g, (Grid{{9, 9, 1}}));
  Grid signed_values{{INT_MIN, INT_MIN}, {0, INT_MIN}};
  EXPECT_EQ(flood_fill(signed_values, 0, 0, INT_MAX), 3);
  EXPECT_EQ(signed_values, (Grid{{INT_MAX, INT_MAX}, {0, INT_MAX}}));
}
TEST(FloodFill, WholeUpperEntryQuotaAndOwnedCopies) {
  const Grid g{{1, 1}, {1, 1}};
  EXPECT_THROW(flood_fill_trace(g, 0, 0, 9, false, 31), std::length_error);
  EXPECT_EQ(flood_fill_trace(g, 0, 0, 9, false, 32).work.filled, 4);
  EXPECT_EQ(flood_fill_trace(g, 0, 0, 9, true, 48).work.filled, 4);
  EXPECT_THROW(flood_fill_trace(g, 0, 0, 9, true, 47), std::length_error);
  EXPECT_NO_THROW(flood_fill_trace(g, 0, 0, 9, true, 0));
  EXPECT_THROW(flood_fill_trace(g, 0, 0, 1, false, 11), std::length_error);
  EXPECT_NO_THROW(flood_fill_trace(g, 0, 0, 1, false, 12));
  EXPECT_THROW(flood_fill_trace({{}, {}}, 0, 0, 9, false, 3), std::length_error);
  EXPECT_NO_THROW(flood_fill_trace({{}, {}}, 0, 0, 9, false, 4));
  Grid caller    = g;
  auto t         = flood_fill_trace(caller, 0, 0, 9);
  caller[0][0]   = 44;
  t.output[1][1] = 33;
  EXPECT_EQ(t.input, g);
  EXPECT_EQ(t.output[0][0], 9);
}
TEST(FloodFill, GuardedNeighborhoodMatchesSignedWideOracle) {
  const std::vector<std::size_t> dims{0, 1, 2, 3, 17, SIZE_MAX};
  const int dr4[]{-1, 1, 0, 0}, dc4[]{0, 0, -1, 1}, dr8[]{-1, -1, -1, 0, 0, 1, 1, 1},
      dc8[]{-1, 0, 1, -1, 1, -1, 0, 1};
  for (auto rows : dims)
    for (auto cols : dims)
      for (auto r : std::vector<std::size_t>{0, 1, rows / 2, rows ? rows - 1 : 0, SIZE_MAX})
        for (auto c : std::vector<std::size_t>{0, 1, cols / 2, cols ? cols - 1 : 0, SIZE_MAX})
          for (bool eight : {false, true}) {
            const auto n = flood_fill_neighborhood(rows, cols, r, c, eight);
            if (r >= rows || c >= cols) {
              EXPECT_EQ(n.count, 0);
              continue;
            }
            ASSERT_EQ(n.count, eight ? 8 : 4);
            for (std::size_t k = 0; k < n.count; ++k) {
              const __int128 a  = __int128(r) + (eight ? dr8[k] : dr4[k]),
                             b  = __int128(c) + (eight ? dc8[k] : dc4[k]);
              const bool  valid = a >= 0 && b >= 0 && a < __int128(rows) && b < __int128(cols);
              const auto& p     = n.candidates[k];
              EXPECT_EQ(p.direction, k);
              EXPECT_EQ(p.in_bounds, valid);
              EXPECT_EQ(p.row, valid ? std::size_t(a) : 0);
              EXPECT_EQ(p.col, valid ? std::size_t(b) : 0);
            }
          }
}
TEST(FloodFill, ConnectivityMatchesBinaryComponentsButNotMixedForeground) {
  const Grid checker{{1, 0, 1}, {0, 1, 0}, {1, 0, 1}};
  for (bool eight : {false, true}) {
    auto g = checker;
    EXPECT_EQ(flood_fill(g, 0, 0, 9, eight), eight ? 5 : 1);
  }
  Grid mixed{{1, 2}};
  EXPECT_EQ(connected_components(mixed).count, 1);
  EXPECT_EQ(flood_fill(mixed, 0, 0, 9), 1);
}
TEST(FloodFill, LargeUniformRegionsUseHeapFrontier) {
  Grid       g(256, std::vector<int>(256, 1));
  const auto w = flood_fill_work(g, 0, 0, 9, true);
  EXPECT_EQ(w.filled, 65536);
  EXPECT_EQ(w.neighbor_slots, 8 * 65536);
  EXPECT_GT(w.peak_stack, 1);
  Grid strip(1, std::vector<int>(100000, 1));
  EXPECT_EQ(flood_fill(strip, 0, 0, 9), 100000);
}
TEST(FloodFill, OriginalFixtureAndLegacySignature) {
  using Signature = std::size_t (*)(Grid&, std::size_t, std::size_t, int, bool);
  Signature fn    = &flood_fill;
  auto      g     = flood_fill_example();
  EXPECT_EQ(g.size(), 6);
  EXPECT_EQ(g[0].size(), 8);
  EXPECT_EQ(fn(g, 0, 0, 9, false), 8);
  EXPECT_EQ(g[3][5], 1);
  EXPECT_EQ(std::count(g[0].begin(), g[0].end(), 9), 3);
}
