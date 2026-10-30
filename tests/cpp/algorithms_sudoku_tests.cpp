// algorithms::sudoku: CSP encoding, AC-3 propagation (with its work counters), exact solving
// and the board picture. The puzzle is the CC0 example from Wikipedia's "Sudoku" article.
#include <datamunge/algorithms/sudoku.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>

namespace sudoku = datamunge::algorithms::sudoku;

namespace {
const std::string kPuzzle =
    "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
const std::string kSolution =
    "534678912672195348198342567859761423426853791713924856961537284287419635345286179";
}  // namespace

TEST(Sudoku, EncodesPeersArcsAndDomains) {
  const auto p = sudoku::peers(40);  // centre cell
  EXPECT_EQ(p.size(), 20u);
  EXPECT_EQ(std::set<int>(p.begin(), p.end()).count(40), 0u);
  EXPECT_EQ(sudoku::arcs().size(), 1620u);
  const auto d = sudoku::domains(kPuzzle);
  EXPECT_EQ(d[0], std::vector<int>{5});
  EXPECT_EQ(d[2].size(), 9u);
  EXPECT_EQ(sudoku::candidates(d), 30 + 51 * 9);
  EXPECT_THROW((void)sudoku::domains("123"), std::invalid_argument);
  EXPECT_THROW((void)sudoku::domains(std::string(81, 'x')), std::invalid_argument);
  EXPECT_THROW((void)sudoku::peers(81), std::out_of_range);
}

TEST(Sudoku, PropagationIsSoundAndCounted) {
  const auto r = sudoku::propagate(kPuzzle);
  ASSERT_TRUE(r.consistent);
  for (int c = 0; c < 81; ++c) {
    const auto& dom = r.domains[static_cast<std::size_t>(c)];
    EXPECT_NE(std::find(dom.begin(), dom.end(), kSolution[static_cast<std::size_t>(c)] - '0'), dom.end()) << c;
  }
  const auto board = sudoku::to_string(r.domains);
  for (int c = 0; c < 81; ++c)
    if (board[static_cast<std::size_t>(c)] != '.') EXPECT_EQ(board[static_cast<std::size_t>(c)], kSolution[static_cast<std::size_t>(c)]);
  EXPECT_EQ(r.candidates.front(), 30 + 51 * 9);
  EXPECT_EQ(r.candidates.back(), sudoku::candidates(r.domains));
  EXPECT_EQ(r.candidates.front() - r.candidates.back(), r.removals);
  EXPECT_GE(r.revisions, 1620);
  EXPECT_GT(r.checks, 0);
}

TEST(Sudoku, SolvesAndDetectsContradictions) {
  const auto s = sudoku::solve(kPuzzle);
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ(*s, kSolution);
  EXPECT_TRUE(sudoku::is_solution(kPuzzle, kSolution));
  std::string wrong = kSolution;
  std::swap(wrong[2], wrong[3]);
  EXPECT_FALSE(sudoku::is_solution(kPuzzle, wrong));

  // Two 5s in the first row: AC-3 empties a domain and no solution exists.
  std::string bad = kPuzzle;
  bad[1] = '5';
  EXPECT_FALSE(sudoku::propagate(bad).consistent);
  EXPECT_FALSE(sudoku::solve(bad).has_value());
}

TEST(Sudoku, BoardDrawsEveryDigit) {
  const auto r = sudoku::propagate(kPuzzle);
  const std::string svg = sudoku::board(r.domains, kPuzzle, sudoku::peers(0)).to_svg();
  EXPECT_NE(svg.find("<svg"), std::string::npos);
  EXPECT_THROW((void)sudoku::board({}, kPuzzle), std::invalid_argument);
}
