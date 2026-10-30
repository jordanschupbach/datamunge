#include <gtest/gtest.h>

#include <array>
#include <datamunge/random/quality.hpp>
#include <datamunge/random/random.hpp>
#include <vector>

namespace rnd = datamunge::random;

TEST(Xoroshiro128Plus, MatchesAuthorsReferenceStatesAndOutputs) {
  const std::array<std::array<std::uint64_t, 2>, 3> states{
      {{1, 2},
       {7420758724034209717ULL, 9442990532527272306ULL},
       {4387707342976528954ULL, 3072119776036644419ULL}}};
  const std::array<std::array<std::uint64_t, 4>, 3> words{
      {{3, 412333834243ULL, 2360170716294286339ULL, 9295852285959843169ULL},
       {16863749256561482023ULL,
        15988492901402843592ULL,
        16860311396414380700ULL,
        3258968728841841858ULL},
       {7459827119013173373ULL,
        16629812729731364797ULL,
        17067482968129184606ULL,
        6083857043340806358ULL}}};
  for (int mode = 0; mode < 3; ++mode) {
    rnd::Xoroshiro128Plus rng(std::array<std::uint64_t, 2>{1, 2});
    if (mode == 1)
      rng.jump();
    if (mode == 2)
      rng.long_jump();
    EXPECT_EQ(rng.state(), states[mode]);
    for (const auto expected : words[mode])
      EXPECT_EQ(rng.next_u64(), expected);
  }
}
TEST(Xoroshiro128Plus, SnapshotsAndRejectsZeroState) {
  EXPECT_THROW((void)rnd::Xoroshiro128Plus(std::array<std::uint64_t, 2>{}), std::invalid_argument);
  for (std::uint64_t seed : {0ULL, 1ULL, 42ULL, ~0ULL}) {
    rnd::Xoroshiro128Plus original(seed);
    EXPECT_NE(original.state(), (std::array<std::uint64_t, 2>{}));
    for (int i = 0; i < 17; ++i)
      (void)original.next_u64();
    rnd::Xoroshiro128Plus restored(original.state());
    for (int i = 0; i < 50; ++i)
      EXPECT_EQ(original.next_u64(), restored.next_u64());
  }
}
TEST(Xoroshiro128Plus, JumpsCommuteAndClearNormalCache) {
  rnd::Xoroshiro128Plus a(42), b(42);
  for (int i = 0; i < 19; ++i)
    (void)a.next_u64();
  a.jump();
  b.jump();
  for (int i = 0; i < 19; ++i)
    (void)b.next_u64();
  EXPECT_EQ(a.state(), b.state());
  a.jump();
  a.long_jump();
  b.long_jump();
  b.jump();
  EXPECT_EQ(a.state(), b.state());
  for (bool large : {false, true}) {
    rnd::Xoroshiro128Plus rng(42);
    (void)rng.normal();
    if (large)
      rng.long_jump();
    else
      rng.jump();
    rnd::Xoroshiro128Plus fresh(rng.state());
    EXPECT_DOUBLE_EQ(rng.normal(), fresh.normal());
  }
}
TEST(BitLinearComplexities, RecoversKnownRecurrencesAndUsesHeldoutSuffix) {
  std::vector<std::uint64_t> words(80);
  for (std::size_t i = 0; i < words.size(); ++i)
    words[i] = (i % 2) | ((std::uint64_t(i % 2)) << 63);
  const auto out = rnd::bit_linear_complexities(words, 40, {0, 1, 63});
  EXPECT_EQ(out.double_at("linear_complexity", 0), 2);
  EXPECT_EQ(out.double_at("linear_complexity", 1), 0);
  EXPECT_EQ(out.double_at("heldout_errors", 0), 0);
  EXPECT_EQ(out.double_at("heldout_errors", 2), 0);
  for (std::size_t i = 40; i < 80; ++i)
    words[i] ^= 1;
  const auto changed = rnd::bit_linear_complexities(words, 40, {0});
  EXPECT_EQ(changed.double_at("linear_complexity", 0), 2);
  // The recurrence predicts from actual previous bits: only the two boundary errors remain.
  EXPECT_EQ(changed.double_at("heldout_errors", 0), 2);
  EXPECT_EQ(changed.double_at("error_rate", 0), 0.05);
  EXPECT_EQ(changed.double_at("one_frequency", 0), 0.5);
}
TEST(BitLinearComplexities, DetectsLinearLowestBitAcrossSeeds) {
  for (std::uint64_t seed : {0ULL, 1ULL, 42ULL, 2026ULL}) {
    rnd::Xoroshiro128Plus      rng(seed);
    std::vector<std::uint64_t> words(1024);
    for (auto& word : words)
      word = rng.next_u64();
    const auto out = rnd::bit_linear_complexities(words, 512, {0, 63});
    EXPECT_EQ(out.double_at("linear_complexity", 0), 128);
    EXPECT_EQ(out.double_at("heldout_errors", 0), 0);
    EXPECT_GT(out.double_at("linear_complexity", 1), 200);
    EXPECT_GT(out.double_at("error_rate", 1), 0.3);
    EXPECT_LT(out.double_at("error_rate", 1), 0.7);
  }
}
TEST(BitLinearComplexities, ValidatesSplitAndPositionsAndPreservesOrder) {
  std::vector<std::uint64_t> words(16);
  EXPECT_THROW((void)rnd::bit_linear_complexities({}, 2), std::invalid_argument);
  EXPECT_THROW((void)rnd::bit_linear_complexities(words, 1), std::invalid_argument);
  EXPECT_THROW((void)rnd::bit_linear_complexities(words, 16), std::invalid_argument);
  EXPECT_THROW((void)rnd::bit_linear_complexities(words, 8, {-1}), std::invalid_argument);
  EXPECT_THROW((void)rnd::bit_linear_complexities(words, 8, {64}), std::invalid_argument);
  const auto out = rnd::bit_linear_complexities(words, 8, {63, 0, 63});
  EXPECT_EQ(out.nrows(), 3);
  EXPECT_EQ(out.double_at("bit", 0), 63);
  EXPECT_EQ(out.double_at("bit", 1), 0);
  EXPECT_EQ(out.double_at("bit", 2), 63);
  const auto empty = rnd::bit_linear_complexities(words, 8, {});
  EXPECT_EQ(empty.nrows(), 0);
  EXPECT_EQ(empty.ncols(), 7);
}
