#include <datamunge/algorithms/checksums.hpp>
#include <datamunge/random/random.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace alg = datamunge::algorithms;

namespace {

// Reference: reduce after every byte.
std::uint32_t naive_adler32(const std::string& s) {
  std::uint32_t a = 1, b = 0;
  for (unsigned char c : s) {
    a = (a + c) % 65521U;
    b = (b + a) % 65521U;
  }
  return (b << 16) | a;
}

std::string random_bytes(datamunge::random::SplitMix64& rng, std::size_t n) {
  std::string s(n, '\0');
  for (auto& c : s) c = static_cast<char>(rng.next_u64() & 0xFFU);
  return s;
}

} // namespace

TEST(Adler32, ReferenceValues) {
  EXPECT_EQ(alg::adler32(""), 1U);
  EXPECT_EQ(alg::adler32("Wikipedia"), 0x11E60398U);
  EXPECT_EQ(alg::adler32("abc"), 0x024D0127U);  // zlib.adler32(b"abc")
}

TEST(Adler32, DeferredReductionMatchesNaiveIncludingAllFFBytes) {
  datamunge::random::SplitMix64 rng(7);
  for (std::size_t n : {0UL, 1UL, 5551UL, 5552UL, 5553UL, 20000UL, 200000UL}) {
    const auto s = random_bytes(rng, n);
    EXPECT_EQ(alg::adler32(s), naive_adler32(s)) << n;
  }
  const std::string ff(100000, '\xFF');  // the worst case NMAX guards against
  EXPECT_EQ(alg::adler32(ff), naive_adler32(ff));
}

TEST(Adler32, UpdateAndCombineAgreeWithWholeMessage) {
  datamunge::random::SplitMix64 rng(11);
  for (int trial = 0; trial < 50; ++trial) {
    const auto x = random_bytes(rng, rng.next_u64() % 9000);
    const auto y = random_bytes(rng, rng.next_u64() % 9000);
    EXPECT_EQ(alg::adler32_update(alg::adler32(x), y), alg::adler32(x + y));
    EXPECT_EQ(alg::adler32_combine(alg::adler32(x), alg::adler32(y), y.size()), alg::adler32(x + y));
  }
  // A second message longer than the modulus.
  const auto x = random_bytes(rng, 300), y = random_bytes(rng, 140000);
  EXPECT_EQ(alg::adler32_combine(alg::adler32(x), alg::adler32(y), y.size()), alg::adler32(x + y));
}

TEST(Adler32, RollingWindowsMatchDirectSums) {
  datamunge::random::SplitMix64 rng(13);
  const auto s = random_bytes(rng, 3000);
  for (std::size_t w : {1UL, 2UL, 16UL, 257UL, 3000UL}) {
    const auto sums = alg::adler32_windows(s, w);
    ASSERT_EQ(sums.size(), s.size() - w + 1);
    for (std::size_t i = 0; i < sums.size(); i += 37) EXPECT_EQ(sums[i], alg::adler32(s.substr(i, w))) << w << ' ' << i;
    EXPECT_EQ(sums.back(), alg::adler32(s.substr(s.size() - w)));
  }
  EXPECT_TRUE(alg::adler32_windows("ab", 3).empty());
  EXPECT_THROW((void)alg::adler32_windows("ab", 0), std::invalid_argument);
}

TEST(Adler32, CollisionModelLimitsAndOrdering) {
  // Long messages approach 1 / 65521^2; short ones are far more likely to collide.
  const double floor = 1.0 / (65521.0 * 65521.0);
  EXPECT_NEAR(alg::adler32_collision_probability(100'000'000) / floor, 1.0, 1e-6);
  EXPECT_GT(alg::adler32_collision_probability(16), 500 * floor);
  EXPECT_GT(alg::adler32_collision_probability(16), alg::adler32_collision_probability(64));
  // n = 3: 1 / (4 pi sqrt(det Sigma)) with det Sigma = 6 v^2, v = 65535 / 12.
  const double v = 65535.0 / 12.0;
  EXPECT_NEAR(alg::adler32_collision_probability(3), 1.0 / (4 * 3.14159265358979 * std::sqrt(6.0) * v), 1e-9);
  EXPECT_THROW((void)alg::adler32_collision_probability(2), std::invalid_argument);
  EXPECT_THROW((void)alg::adler32_collision_probability(10, 0.0), std::invalid_argument);
}
