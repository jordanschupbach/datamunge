#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/hirschberg.hpp>
#include <datamunge/algorithms/needleman_wunsch.hpp>
#include <functional>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NeedlemanWunschTrace, ExhaustiveAlignmentPathsAndEveryActualRecord) {
  std::vector<std::string> words{""};
  for (int n = 1; n <= 4; ++n)
    for (int mask = 0; mask < (1 << n); ++mask) {
      std::string s;
      for (int i = 0; i < n; ++i)
        s += ((mask >> i) & 1) ? 'A' : 'B';
      words.push_back(s);
    }
  for (auto scoring :
       std::vector<AlignmentScoring>{{2, -1, -2}, {1, -1, -1}, {0, 0, 0}, {1, -1, 2}})
    for (auto& a : words)
      for (auto& b : words) {
        std::function<int(std::size_t, std::size_t)> oracle = [&](std::size_t i, std::size_t j) {
          if (i == a.size() && j == b.size())
            return 0;
          int best = INT_MIN;
          if (i < a.size())
            best = std::max(best, scoring.gap + oracle(i + 1, j));
          if (j < b.size())
            best = std::max(best, scoring.gap + oracle(i, j + 1));
          if (i < a.size() && j < b.size())
            best = std::max(
                best, (a[i] == b[j] ? scoring.match : scoring.mismatch) + oracle(i + 1, j + 1));
          return best;
        };
        auto  t = needleman_wunsch_trace(a, b, scoring);
        auto& r = t.run;
        EXPECT_EQ(r.alignment.score, oracle(0, 0));
        EXPECT_EQ(needleman_wunsch(a, b, scoring).a_aligned, r.alignment.a_aligned);
        EXPECT_EQ(r.cells, (a.size() + 1) * (b.size() + 1));
        EXPECT_EQ(r.boundary_assignments, a.size() + b.size() + 2);
        EXPECT_EQ(r.interior_cells, a.size() * b.size());
        for (auto& c : t.cells) {
          auto& h = r.scores;
          EXPECT_EQ(
              c.diagonal,
              h[c.i - 1][c.j - 1] + (a[c.i - 1] == b[c.j - 1] ? scoring.match : scoring.mismatch));
          EXPECT_EQ(c.up, h[c.i - 1][c.j] + scoring.gap);
          EXPECT_EQ(c.left, h[c.i][c.j - 1] + scoring.gap);
          EXPECT_EQ(c.score, std::max({c.diagonal, c.up, c.left}));
          EXPECT_EQ(
              c.chosen,
              c.diagonal == c.score ? NeedlemanWunschMove::Diagonal
              : c.up == c.score     ? NeedlemanWunschMove::Up
                                    : NeedlemanWunschMove::Left);
        }
        std::string ar, br;
        std::size_t i = a.size(), j = b.size();
        int         sum = 0;
        for (auto& s : t.traceback) {
          EXPECT_EQ(s.i_before, i);
          EXPECT_EQ(s.j_before, j);
          EXPECT_EQ(s.i_after, i - (s.move != NeedlemanWunschMove::Left));
          EXPECT_EQ(s.j_after, j - (s.move != NeedlemanWunschMove::Up));
          i = s.i_after;
          j = s.j_after;
          ar += s.a;
          br += s.b;
          sum += s.move == NeedlemanWunschMove::Diagonal
                     ? (s.a == s.b ? scoring.match : scoring.mismatch)
                     : scoring.gap;
        }
        EXPECT_EQ(i + j, 0);
        EXPECT_EQ(sum, r.alignment.score);
        std::reverse(ar.begin(), ar.end());
        std::reverse(br.begin(), br.end());
        EXPECT_EQ(ar, r.alignment.a_aligned);
        EXPECT_EQ(br, r.alignment.b_aligned);
        EXPECT_EQ(
            t.captured_entries,
            3 + a.size() + b.size() + a.size() + 1 + r.cells + r.interior_cells +
                3 * r.traceback_steps);
      }
}
TEST(NeedlemanWunschTrace, OriginalTiesGuardsAndExactQuotas) {
  auto t = needleman_wunsch_trace("AAAC", "AGC", {1, -1, -1});
  EXPECT_EQ(t.run.alignment.score, 0);
  EXPECT_EQ(t.run.alignment.a_aligned, "AAAC");
  EXPECT_EQ(t.run.alignment.b_aligned, "-AGC");
  EXPECT_EQ(t.captured_entries, 59);
  EXPECT_NO_THROW(needleman_wunsch_trace("AAAC", "AGC", {1, -1, -1}, {20, 59}));
  EXPECT_THROW(needleman_wunsch_trace("AAAC", "AGC", {1, -1, -1}, {20, 58}), std::length_error);
  EXPECT_THROW(needleman_wunsch_run("AAAC", "AGC", {1, -1, -1}, {19, 0}), std::length_error);
  EXPECT_THROW(needleman_wunsch_trace("-", "", {}, {1, 1}), std::invalid_argument);
  EXPECT_THROW(needleman_wunsch("A", "", {INT_MIN, 0, 0}), std::overflow_error);
  EXPECT_THROW(needleman_wunsch("A", "A", {INT_MAX, 0, 0}), std::overflow_error);
  EXPECT_EQ(needleman_wunsch_trace("", "").captured_entries, 5);
  auto tie = needleman_wunsch_trace("AA", "A", {1, -1, -1});
  EXPECT_EQ(tie.cells.back().chosen, NeedlemanWunschMove::Diagonal);
  auto p = needleman_wunsch("A", "A", {1, -1, 2});
  EXPECT_EQ(p.score, 4);
  EXPECT_EQ(p.a_aligned, "-A");
  EXPECT_EQ(p.b_aligned, "A-");
  std::string binary("\0\xff", 2);
  EXPECT_EQ(needleman_wunsch(binary, binary).score, 4);
  for (auto& [a, b] : needleman_wunsch_demo_pairs())
    EXPECT_EQ(needleman_wunsch(a, b).score, hirschberg_run(a, b).alignment.score);
}
