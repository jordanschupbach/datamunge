#include <gtest/gtest.h>

#include <datamunge/algorithms/long_division.hpp>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
TEST(LongDivisionTrace, FullWidthAndEveryActualPrefix) {
  std::mt19937_64 rng(912);
  for (int i = 0; i < 10000; ++i) {
    const auto n = rng(), d = rng() | 1;
    auto       t = long_division_trace(n, d);
    ASSERT_EQ(t.steps.size(), 64);
    EXPECT_EQ(t.result.quotient, n / d);
    EXPECT_EQ(t.result.remainder, n % d);
    EXPECT_EQ(t.result.iterations, 64);
    std::uint64_t r = 0;
    for (auto& s : t.steps) {
      EXPECT_EQ(s.before, r);
      EXPECT_EQ(s.trial, (r << 1) | ((n >> s.bit) & 1));
      EXPECT_EQ(s.subtracted, s.trial >= d);
      r                 = s.remainder;
      const auto prefix = n >> s.bit;
      const auto q      = s.quotient >> s.bit;
      EXPECT_EQ(static_cast<unsigned __int128>(q) * d + r, prefix);
      EXPECT_LT(r, d);
    }
  }
  const auto m = std::numeric_limits<std::uint64_t>::max();
  for (auto d : {std::uint64_t{1}, m, m - 1, (std::uint64_t{1} << 63)})
    for (auto n : {std::uint64_t{0}, m, m - 1}) {
      auto r = long_division(n, d);
      EXPECT_EQ(r.quotient, n / d);
      EXPECT_EQ(r.remainder, n % d);
    }
}
TEST(LongDivisionTrace, GuardsAndOriginals) {
  EXPECT_THROW(long_division(1, 0), std::invalid_argument);
  EXPECT_THROW(long_division_trace(1, 0), std::invalid_argument);
  EXPECT_THROW(long_division_trace(1, 2, 65), std::length_error);
  EXPECT_EQ(long_division_trace(220, 13, 66).result.remainder, 12);
  EXPECT_EQ(long_division_trace(0, 1, 0).steps.size(), 64);
  for (auto p : std::vector<std::pair<std::uint64_t, std::uint64_t>>{
           {100, 7}, {220, 13}, {1000000, 999}, {42, 42}}) {
    auto t = long_division_trace(p.first, p.second);
    EXPECT_EQ(t.result.quotient, p.first / p.second);
    EXPECT_EQ(t.result.remainder, p.first % p.second);
  }
}
