#include <gtest/gtest.h>

#include <datamunge/random/linear_congruential_analysis.hpp>
#include <limits>
using namespace datamunge::random;
TEST(LCGAnalysis, OriginalAndActualWrapWords) {
  LinearCongruential64 g(1, 5, 3);
  auto                 t = lcg64_trace(g, 3);
  EXPECT_EQ(t.steps[0].after, 8);
  EXPECT_EQ(t.steps[1].after, 43);
  EXPECT_EQ(t.steps[2].after, 218);
  EXPECT_EQ(t.initial_state, 1);
  EXPECT_EQ(t.final_state, 218);
  EXPECT_EQ(g.state(), 218);
  const auto m = std::numeric_limits<std::uint64_t>::max();
  auto       s = lcg64_step(m, m, m);
  EXPECT_TRUE(s.multiplication_wrapped);
  EXPECT_EQ(s.product, 1);
  EXPECT_TRUE(s.addition_wrapped);
  EXPECT_EQ(s.after, 0);
  EXPECT_FALSE(lcg64_step(1, 0, m).multiplication_wrapped);
}
TEST(LCGAnalysis, ActualJumpCompositionAndRawLoop) {
  for (std::uint64_t a : {1ULL, 5ULL, 9ULL, 6364136223846793005ULL})
    for (std::uint64_t c : {1ULL, 3ULL, 1442695040888963407ULL}) {
      LinearCongruential64 g(123, a, c);
      std::uint64_t        expected = 123;
      for (std::uint64_t n = 0; n <= 1000; ++n) {
        auto j = lcg64_jump(123, a, c, n);
        EXPECT_EQ(j.after, expected);
        if (n < 1000)
          expected = g.next_u64();
      }
      auto p = lcg64_jump(123, a, c, 0xffffffffffffffffULL), q = lcg64_jump(p.after, a, c, 1);
      EXPECT_EQ(q.after, 123);
      for (auto& s : p.stages) {
        EXPECT_EQ(s.squared_multiplier, s.power_multiplier * s.power_multiplier);
        EXPECT_EQ(s.squared_increment, (s.power_multiplier + 1) * s.power_increment);
      }
    }
}
TEST(LCGAnalysis, ReducedOrbitsAndIncompleteBudget) {
  for (unsigned bits = 1; bits <= 12; ++bits) {
    auto o = lcg64_low_bits_orbit(1, 5, 3, bits, std::size_t{1} << bits);
    EXPECT_TRUE(o.complete);
    EXPECT_EQ(o.repeat_offset, 0);
    EXPECT_EQ(o.period, std::size_t{1} << bits);
    EXPECT_EQ(o.states.front(), o.states.back());
    auto short_run = lcg64_low_bits_orbit(1, 5, 3, bits, (std::size_t{1} << bits) - 1);
    EXPECT_FALSE(short_run.complete);
    EXPECT_EQ(short_run.period, 0);
  }
  EXPECT_EQ(lcg64_low_bits_orbit(0, 3, 1, 4, 16).period, 8);
  EXPECT_EQ(lcg64_low_bits_orbit(1, 5, 0, 4, 16).period, 4);
  EXPECT_EQ(lcg64_low_bits_orbit(0, 5, 0, 4, 16).period, 1);
  EXPECT_FALSE(lcg64_low_bits_orbit(1, 5, 3, 4, 0).complete);
  EXPECT_THROW(lcg64_low_bits_orbit(1, 5, 3, 0), std::invalid_argument);
  EXPECT_THROW(lcg64_low_bits_orbit(1, 5, 3, 21), std::invalid_argument);
  EXPECT_THROW(lcg64_low_bits_orbit(1, 5, 3, 4, 1048577), std::invalid_argument);
}
TEST(LCGAnalysis, QuotaAtomicityOwnershipAndNormalCache) {
  LinearCongruential64 g(1, 5, 3);
  EXPECT_THROW(lcg64_trace(g, 3, 3), std::length_error);
  EXPECT_EQ(g.state(), 1);
  auto t = lcg64_trace(g, 3, 4);
  g.next_u64();
  EXPECT_EQ(t.final_state, 218);
  auto z = lcg64_trace(g, 0, 0);
  EXPECT_EQ(z.initial_state, z.final_state);
  EXPECT_EQ(z.captured_entries, 1);
  EXPECT_THROW(LinearCongruential64(1, 3, 1), std::invalid_argument);
  EXPECT_THROW(LinearCongruential64(1, 5, 0), std::invalid_argument);
  LinearCongruential64 a(42), b(42);
  EXPECT_EQ(a.normal(), b.normal());
  lcg64_trace(a, 7);
  for (int i = 0; i < 7; ++i)
    b.next_u64();
  EXPECT_EQ(a.normal(), b.normal());
  EXPECT_EQ(a.next_u64(), b.next_u64());
}
