#include <datamunge/random/quality.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/ks_test.hpp>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <vector>

namespace rnd = datamunge::random;

TEST(Acorn64Discard, MatchesStepping) {
  for (std::size_t order : {1U, 2U, 5U, 12U}) {
    for (std::uint64_t steps : {0ULL, 1ULL, 2ULL, 7ULL, 64ULL, 1000ULL, 4097ULL}) {
      rnd::Acorn64 stepped(99, order), jumped(99, order);
      for (std::uint64_t t = 0; t < steps; ++t) (void)stepped.next_u64();
      jumped.discard(steps);
      EXPECT_EQ(stepped.state(), jumped.state()) << "order " << order << " steps " << steps;
      EXPECT_EQ(stepped.next_u64(), jumped.next_u64());
    }
  }
}

TEST(Acorn64Discard, HandChecksAndComposes) {
  rnd::Acorn64 small(std::vector<std::uint64_t>{1, 2, 3});
  small.discard(3);  // (1,3,6), (1,4,10), (1,5,15)
  EXPECT_EQ(small.state(), (std::vector<std::uint64_t>{1, 5, 15}));
  // Jumps compose even when the binomials wrap modulo 2^64.
  rnd::Acorn64 a(5, 12), b(5, 12);
  a.discard(1ULL << 40);
  a.discard((1ULL << 40) + 3);
  b.discard((1ULL << 41) + 3);
  EXPECT_EQ(a.state(), b.state());
}

TEST(BitPeriods, FollowTheBinomialPeriodLaw) {
  // Bit b of ACORN order k has period 2^(b + 1 + floor(log2 k)).
  for (std::size_t k : {1U, 3U, 4U, 12U}) {
    const auto periods = rnd::bit_periods(rnd::Acorn64(11, k), 6, 1U << 14);
    const int lg = std::bit_width(k) - 1;
    for (int b = 0; b < 6; ++b) EXPECT_EQ(periods[static_cast<std::size_t>(b)], 1ULL << (b + 1 + lg)) << k;
  }
  // A full-period LCG modulo 2^64: bit b has period 2^(b+1). A good generator has none.
  const auto lcg = rnd::bit_periods(rnd::LinearCongruential64(1), 4, 4096);
  EXPECT_EQ(lcg, (std::vector<std::uint64_t>{2, 4, 8, 16}));
  for (auto p : rnd::bit_periods(rnd::Xoshiro256StarStar(1), 8, 4096)) EXPECT_EQ(p, 0U);
  EXPECT_THROW((void)rnd::bit_periods_of({1, 2}, 0), std::invalid_argument);
}

TEST(QualityTests, GoodGeneratorPassesAndLowBitsAreFlagged) {
  rnd::Xoshiro256StarStar good(3);
  const auto table = rnd::quality_tests(good, 20000);
  ASSERT_EQ(table.nrows(), 7U);
  for (std::size_t r = 0; r < table.nrows(); ++r) EXPECT_GT(table.double_at("p_value", r), 1e-4) << r;
  EXPECT_EQ(table.double_at("df", 0), 99.0);
  EXPECT_EQ(table.double_at("df", 6), 255.0);

  const auto freq = rnd::bit_frequencies(good, 20000);
  ASSERT_EQ(freq.size(), 64U);
  for (double f : freq) EXPECT_NEAR(f, 0.5, 0.02);
  EXPECT_THROW((void)rnd::quality_tests(good, 10), std::invalid_argument);
}

TEST(KsUniform, AcceptsUniformRejectsSquares) {
  rnd::Pcg32 rng(8);
  auto u = rnd::uniform_sample(rng, 2000);
  EXPECT_GT(datamunge::stats::ks_test_one_sample_uniform(u).p_value, 0.01);
  for (auto& v : u) v *= v;
  EXPECT_LT(datamunge::stats::ks_test_one_sample_uniform(u).p_value, 1e-6);
  // D for {0.1, 0.6} against U(0,1): max(1/2 - 0.1, 1 - 0.6, 0.1, 0.6 - 1/2) = 0.4.
  EXPECT_NEAR(datamunge::stats::ks_test_one_sample_uniform({0.1, 0.6}).statistic, 0.4, 1e-12);
  EXPECT_THROW((void)datamunge::stats::ks_test_one_sample_uniform({0.1, 0.2}, 1.0, 1.0), std::invalid_argument);
}

TEST(BitPlot, DrawsOneRectanglePerRunOfOnes) {
  // bit 0: 1,1,0,1 -> 2 runs; bit 1: 0,1,1,0 -> 1 run.
  const auto plot = rnd::bit_plot({1, 3, 2, 1}, 2);
  EXPECT_EQ(plot.series().size(), 3U);
  EXPECT_THROW((void)rnd::bit_plot({}, 2), std::invalid_argument);
  EXPECT_THROW((void)rnd::bit_plot({1}, 65), std::invalid_argument);
}

TEST(UniformU64, UsesHighBitsNotThePeriodicLowBits) {
  // The lowest LCG bit alternates; a modulo-based uniform_u64(0, 1) would copy that pattern.
  rnd::LinearCongruential64 lcg(5);
  int changes = 0, previous = -1, ones = 0;
  for (int t = 0; t < 4000; ++t) {
    const int v = static_cast<int>(lcg.uniform_u64(0, 1));
    ones += v;
    if (previous >= 0) changes += v != previous;
    previous = v;
  }
  EXPECT_NEAR(changes, 2000, 200);  // about half the steps change, not all of them
  EXPECT_NEAR(ones, 2000, 200);
  // The result is floor(x * range / 2^64): the top bits pick the value.
  rnd::Acorn64 a(std::vector<std::uint64_t>{1, 0x8000000000000000ULL - 1});  // next output = 2^63
  EXPECT_EQ(a.uniform_u64(0, 3), 2U);
}
