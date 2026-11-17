#include <gtest/gtest.h>

#include <datamunge/algorithms/string_metrics.hpp>
#include <map>
#include <queue>
using namespace datamunge::algorithms;
TEST(LevenshteinTrace, IndependentEditGraphShortestPaths) {
  std::vector<std::string> words;
  for (int n = 0; n <= 3; ++n)
    for (int mask = 0; mask < (1 << n); ++mask) {
      std::string s;
      for (int i = 0; i < n; ++i)
        s += mask >> i & 1 ? 'b' : 'a';
      words.push_back(s);
    }
  for (auto& a : words) {
    std::map<std::string, int> d;
    std::queue<std::string>    q;
    d[a] = 0;
    q.push(a);
    while (!q.empty()) {
      auto u = q.front();
      q.pop();
      std::vector<std::string> next;
      for (std::size_t i = 0; i < u.size(); ++i) {
        next.push_back(u.substr(0, i) + u.substr(i + 1));
        for (char c : {'a', 'b'}) {
          auto v = u;
          v[i]   = c;
          next.push_back(v);
        }
      }
      if (u.size() < 3)
        for (std::size_t i = 0; i <= u.size(); ++i)
          for (char c : {'a', 'b'})
            next.push_back(u.substr(0, i) + c + u.substr(i));
      for (auto& v : next)
        if (!d.contains(v)) {
          d[v] = d[u] + 1;
          q.push(v);
        }
    }
    for (auto& b : words) {
      auto t = levenshtein_trace(a, b);
      EXPECT_EQ(t.analysis.distance, d[b]);
      EXPECT_EQ(t.analysis.distance, levenshtein_distance(a, b));
      EXPECT_EQ(t.analysis.work.cells, a.size() * b.size());
      std::size_t cost = 0, i = t.row_bytes.size(), j = t.column_bytes.size();
      for (auto& e : t.path) {
        EXPECT_EQ(e.row, i);
        EXPECT_EQ(e.column, j);
        EXPECT_EQ(t.table[i][j], t.table[e.next_row][e.next_column] + e.cost);
        i = e.next_row;
        j = e.next_column;
        cost += e.cost;
      }
      EXPECT_EQ(i, 0);
      EXPECT_EQ(j, 0);
      EXPECT_EQ(cost, t.analysis.distance);
    }
  }
}
TEST(LevenshteinTrace, OriginalBytesOrientationQuotaOwnership) {
  EXPECT_EQ(levenshtein_distance("kitten", "sitting"), 3);
  EXPECT_EQ(levenshtein_distance("flaw", "lawn"), 2);
  EXPECT_EQ(levenshtein_distance("saturday", "sunday"), 3);
  EXPECT_EQ(levenshtein_distance("book", "back"), 2);
  EXPECT_EQ(levenshtein_distance("", "abc"), 3);
  EXPECT_EQ(levenshtein_distance("distance", "distance"), 0);
  EXPECT_EQ(levenshtein_distance(std::string("a\0b", 3), std::string("a\0c", 3)), 1);
  EXPECT_EQ(levenshtein_distance("\xc3\xa9", "e"), 2);
  auto t = levenshtein_trace("kitten", "sitting");
  EXPECT_EQ(t.row_bytes, "sitting");
  EXPECT_FALSE(t.rows_are_a);
  EXPECT_EQ(t.table.back().back(), 3);
  EXPECT_THROW(levenshtein_trace("a", "b", 7), std::length_error);
  EXPECT_EQ(levenshtein_trace("a", "b", 8).captured_entries, 7);
  std::string a = "a";
  auto        u = levenshtein_trace(a, "b", 0);
  a[0]          = 'c';
  EXPECT_EQ(u.column_bytes, "a");
  EXPECT_EQ(levenshtein_trace("", "").captured_entries, 1);
}
