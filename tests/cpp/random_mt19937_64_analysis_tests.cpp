#include <gtest/gtest.h>

#include <datamunge/random/mt19937_64_analysis.hpp>
#include <datamunge/random/random.hpp>
#include <limits>
#include <random>
namespace r = datamunge::random;
TEST(MT19937_64Analysis, FullReferenceStreamsAcrossRefillsAndSeedBits) {
  const std::vector<std::uint64_t> seeds{
      0, 1, 42, 5489, 0xffffffffULL, 0x100000000ULL, 0x8000000000000000ULL, ~std::uint64_t{0}};
  for (auto seed : seeds) {
    r::MersenneTwister64 actual(seed);
    std::mt19937_64      expected(seed);
    for (int i = 0; i < 20000; ++i)
      ASSERT_EQ(actual.next_u64(), expected()) << "seed=" << seed << ", i=" << i;
  }
}
TEST(MT19937_64Analysis, StandardTenThousandthWord) {
  r::MersenneTwister64 rng(5489);
  std::uint64_t        word = 0;
  for (int i = 0; i < 10000; ++i)
    word = rng.next_u64();
  EXPECT_EQ(word, 9981545732273789042ULL);
}
TEST(MT19937_64Analysis, SeedAndRefillPositionsAndLastWordUseOrderedUpdates) {
  r::MersenneTwister64 rng(0);
  EXPECT_EQ(rng.block_index(), 312);
  EXPECT_EQ(rng.state_words()[0], 0);
  EXPECT_EQ(rng.state_words()[1], 1);
  const auto initial = rng.state_words();
  const auto first   = rng.next_u64();
  EXPECT_EQ(rng.block_index(), 1);
  EXPECT_EQ(first, r::mt19937_64_temper_steps(rng.state_words()[0]).back());
  EXPECT_EQ(
      rng.state_words()[0], r::mt19937_64_twist_steps(initial[0], initial[1], initial[156]).back());
  EXPECT_EQ(
      rng.state_words()[156],
      r::mt19937_64_twist_steps(initial[156], initial[157], rng.state_words()[0]).back());
  EXPECT_EQ(
      rng.state_words()[311],
      r::mt19937_64_twist_steps(initial[311], rng.state_words()[0], rng.state_words()[155]).back());
  for (int i = 1; i < 312; ++i)
    (void)rng.next_u64();
  EXPECT_EQ(rng.block_index(), 312);
  (void)rng.next_u64();
  EXPECT_EQ(rng.block_index(), 1);
}
TEST(MT19937_64Analysis, TwistOddEvenAndIgnoredBits) {
  const auto even = r::mt19937_64_twist_steps(0x8000000000000000ULL, 0, 0);
  const auto odd  = r::mt19937_64_twist_steps(0x8000000000000000ULL, 1, 0);
  EXPECT_EQ(even[0], 0x8000000000000000ULL);
  EXPECT_EQ(even[1], 0x4000000000000000ULL);
  EXPECT_EQ(even[2], 0);
  EXPECT_EQ(even[3], 0x4000000000000000ULL);
  EXPECT_EQ(odd[0], 0x8000000000000001ULL);
  EXPECT_EQ(odd[2], 0xb5026f5aa96619e9ULL);
  EXPECT_EQ(odd[3], 0xf5026f5aa96619e9ULL);
  EXPECT_EQ(r::mt19937_64_twist_steps(0x800000007fffffffULL, 0xffffffff80000001ULL, 0), odd);
  EXPECT_EQ(r::mt19937_64_twist_steps(0x8000000000000000ULL, 1, 123)[3], odd[3] ^ 123);
}
TEST(MT19937_64Analysis, TemperExactWordAndEveryBasisVector) {
  EXPECT_EQ(
      (r::mt19937_64_temper_steps(1)),
      (std::array<std::uint64_t, 5>{1, 1, 0x20001, 0x0040002000020001, 0x0040002000020801}));
  for (unsigned i = 0; i < 64; ++i) {
    const auto x = std::uint64_t{1} << i;
    EXPECT_EQ(r::mt19937_64_untemper(r::mt19937_64_temper_steps(x).back()), x);
  }
  EXPECT_EQ(r::mt19937_64_untemper(0), 0);
}
TEST(MT19937_64Analysis, BothInverseDirectionsAndLinearity) {
  std::mt19937_64 rng(9823);
  for (int i = 0; i < 10000; ++i) {
    const auto a = rng(), b = rng();
    const auto t = r::mt19937_64_temper_steps(a).back();
    EXPECT_EQ(r::mt19937_64_untemper(t), a);
    EXPECT_EQ(r::mt19937_64_temper_steps(r::mt19937_64_untemper(a)).back(), a);
    EXPECT_EQ(r::mt19937_64_temper_steps(a ^ b).back(), t ^ r::mt19937_64_temper_steps(b).back());
  }
}
TEST(MT19937_64Analysis, AlignedOutputBlockPredictsManySubsequentRefills) {
  for (auto skip : {0, 312, 936}) {
    r::MersenneTwister64 source(5489);
    for (int i = 0; i < skip; ++i)
      (void)source.next_u64();
    std::array<std::uint64_t, 312> block{};
    for (auto& word : block)
      word = source.next_u64();
    auto clone = r::MersenneTwister64::from_output_block(block);
    EXPECT_EQ(clone.block_index(), 312);
    EXPECT_EQ(clone.state_words(), source.state_words());
    for (int i = 0; i < 10000; ++i)
      ASSERT_EQ(source.next_u64(), clone.next_u64());
  }
}
TEST(MT19937_64Analysis, ArbitraryZeroBlockIsExplicitlyDegenerate) {
  auto clone = r::MersenneTwister64::from_output_block({});
  for (int i = 0; i < 1000; ++i)
    EXPECT_EQ(clone.next_u64(), 0);
}
TEST(MT19937_64Analysis, CopyIncludesNormalCacheAndRawPosition) {
  r::MersenneTwister64 source(9);
  (void)source.normal();
  auto copy = source;
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(source.next_u64(), copy.next_u64());
    EXPECT_DOUBLE_EQ(source.normal(), copy.normal());
  }
  r::MersenneTwister64           raw(42);
  std::array<std::uint64_t, 312> block{};
  for (auto& word : block)
    word = raw.next_u64();
  auto clone = r::MersenneTwister64::from_output_block(block);
  EXPECT_DOUBLE_EQ(clone.normal(), raw.normal());
  EXPECT_DOUBLE_EQ(clone.normal(), raw.normal());
}
TEST(MT19937_64Analysis, UniformUsesExactlyTheUpper53Bits) {
  r::MersenneTwister64 a(5489), b(5489);
  for (int i = 0; i < 1000; ++i) {
    const auto raw = a.next_u64();
    EXPECT_DOUBLE_EQ(b.uniform01(), double(raw >> 11) * 0x1p-53);
  }
}
TEST(MT19937_64Analysis, ConformanceTablesRetainOrderAndExactHexWords) {
  const auto d = r::mt19937_64_conformance({5489, 0, 5489}, 10000);
  EXPECT_EQ(d.nrows(), 3);
  EXPECT_EQ(d.string_at("seed", 0), "0000000000001571");
  EXPECT_EQ(d.string_at("native_first", 0), "c96d191cf6f6aea6");
  EXPECT_EQ(d.string_at("native_last", 0), "8a8592f5817ed872");
  for (std::size_t i = 0; i < 3; ++i)
    EXPECT_EQ(d.double_at("mismatches", i), 0);
  EXPECT_EQ(r::mt19937_64_conformance({}).nrows(), 0);
  EXPECT_THROW(r::mt19937_64_conformance({}, 0), std::invalid_argument);
}
