#include <datamunge/algorithms/addition_chain.hpp>
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <datamunge/random/random.hpp>

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <vector>

namespace alg = datamunge::algorithms;

TEST(AdditionChain, BinaryAndWindowChainsAreValidWithTheRightLength) {
  EXPECT_EQ(alg::binary_chain(13), (std::vector<std::uint64_t>{1, 2, 3, 6, 12, 13}));
  EXPECT_EQ(alg::binary_chain(1), (std::vector<std::uint64_t>{1}));
  datamunge::random::SplitMix64 rng(3);
  for (int trial = 0; trial < 500; ++trial) {
    const std::uint64_t e = (rng.next_u64() >> (trial % 60)) | 1U;
    const auto bin = alg::binary_chain(e);
    ASSERT_TRUE(alg::is_addition_chain(bin));
    EXPECT_EQ(bin.back(), e);
    EXPECT_EQ(static_cast<int>(bin.size()) - 1, (63 - std::countl_zero(e)) + std::popcount(e) - 1);
    for (int k = 2; k <= 6; ++k) {
      const auto w = alg::window_chain(e, k);
      ASSERT_TRUE(alg::is_addition_chain(w)) << e << " k=" << k;
      EXPECT_EQ(w.back(), e);
      EXPECT_GE(w.size(), static_cast<std::size_t>(alg::chain_lower_bound(e)) + 1);
    }
  }
  EXPECT_THROW((void)alg::window_chain(0, 2), std::invalid_argument);
  EXPECT_THROW((void)alg::window_chain(5, 0), std::invalid_argument);
}

TEST(AdditionChain, ValidatorStepsAndModPow) {
  EXPECT_TRUE(alg::is_addition_chain({1, 2, 3, 6, 12, 15}));
  EXPECT_FALSE(alg::is_addition_chain({1, 2, 5}));
  EXPECT_FALSE(alg::is_addition_chain({2, 4}));
  EXPECT_FALSE(alg::is_addition_chain({1, 2, 2}));
  const auto steps = alg::chain_steps({1, 2, 3, 6, 12, 15});
  ASSERT_EQ(steps.size(), 5U);
  EXPECT_EQ(steps.back(), (std::pair<std::size_t, std::size_t>{4, 2}));  // 15 = 12 + 3
  EXPECT_THROW((void)alg::chain_steps({1, 3}), std::invalid_argument);
  for (std::uint64_t e : {1ULL, 2ULL, 15ULL, 255ULL, 65537ULL, 1000003ULL})
    for (const auto& chain : {alg::addition_chain(e), alg::window_chain(e, 3)})
      EXPECT_EQ(alg::mod_pow_chain(7, chain, 1'000'000'007ULL), alg::mod_pow(7, e, 1'000'000'007ULL)) << e;
  EXPECT_EQ(alg::mod_pow_chain(7, {1, 2}, 1), 0U);
}

TEST(AdditionChain, LowerBoundHoldsAndOptimalMatchesKnownValues) {
  // Smallest n needing r steps (OEIS A003064), r = 0..11.
  const std::vector<std::uint64_t> first{1, 2, 3, 5, 7, 11, 19, 29, 47, 71, 127, 191};
  std::vector<std::uint64_t> seen(first.size(), 0);
  for (std::uint64_t e = 1; e <= 200; ++e) {
    const auto chain = alg::addition_chain(e);
    ASSERT_TRUE(alg::is_addition_chain(chain));
    const auto len = chain.size() - 1;
    EXPECT_GE(static_cast<int>(len), alg::chain_lower_bound(e)) << e;
    EXPECT_LE(len, alg::binary_chain(e).size() - 1);
    if (len < seen.size() && seen[len] == 0) seen[len] = e;
  }
  EXPECT_EQ(seen, first);
  EXPECT_EQ(alg::chain_lower_bound(1), 0);
  EXPECT_EQ(alg::chain_lower_bound(7), 4);
}
