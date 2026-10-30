#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <datamunge/random/quality.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/bootstrap.hpp>
#include <numeric>
#include <random>

namespace rnd = datamunge::random;
namespace {
double xoshiro_test_mean(const std::vector<double>& x) {
  return std::accumulate(x.begin(), x.end(), 0.0) / x.size();
}
} // namespace
TEST(Xoshiro256StarStar, MatchesAuthorsReferenceStatesAndWords) {
  const std::array<std::uint64_t, 4>                initial{1, 2, 3, 4};
  const std::array<std::array<std::uint64_t, 4>, 3> states{
      {{1, 2, 3, 4},
       {10122426448480695249ULL,
        8079205330032121950ULL,
        7289065458748526725ULL,
        9477464255293849680ULL},
       {678511610814637056ULL,
        15850499779492529430ULL,
        6002989639035333134ULL,
        3559352929785830385ULL}}};
  const std::array<std::array<std::uint64_t, 4>, 3> words{
      {{11520, 0, 1509978240, 1215971899390074240ULL},
       {13534147089533256664ULL,
        7126240192422241655ULL,
        3805973808039778091ULL,
        11547880530658420384ULL},
       {5942309088398569549ULL,
        15625447729937358436ULL,
        6925613901769781251ULL,
        16198770605655666946ULL}}};
  for (std::size_t mode = 0; mode < 3; ++mode) {
    rnd::Xoshiro256StarStar rng(initial);
    if (mode == 1)
      rng.jump();
    if (mode == 2)
      rng.long_jump();
    EXPECT_EQ(rng.state(), states[mode]);
    for (auto word : words[mode])
      EXPECT_EQ(rng.next_u64(), word);
  }
}
TEST(Xoshiro256StarStar, RejectsZeroStateAndSnapshotsRawStreams) {
  EXPECT_THROW(
      (void)rnd::Xoshiro256StarStar(std::array<std::uint64_t, 4>{}), std::invalid_argument);
  for (std::uint64_t seed : {0ULL, 1ULL, 42ULL, ~0ULL}) {
    rnd::Xoshiro256StarStar rng(seed);
    EXPECT_NE(rng.state(), (std::array<std::uint64_t, 4>{}));
    for (int i = 0; i < 17; ++i)
      (void)rng.next_u64();
    rnd::Xoshiro256StarStar restored(rng.state());
    for (int i = 0; i < 50; ++i)
      EXPECT_EQ(rng.next_u64(), restored.next_u64());
  }
}
TEST(Xoshiro256StarStar, JumpsCommuteWithTransitionsAndEachOther) {
  rnd::Xoshiro256StarStar a(42), b(42);
  for (int i = 0; i < 19; ++i)
    (void)a.next_u64();
  a.jump();
  b.jump();
  for (int i = 0; i < 19; ++i)
    (void)b.next_u64();
  EXPECT_EQ(a.state(), b.state());
  a.long_jump();
  a.jump();
  b.jump();
  b.long_jump();
  EXPECT_EQ(a.state(), b.state());
}
TEST(Xoshiro256StarStar, JumpClearsCachedNormalDraw) {
  for (bool large : {false, true}) {
    rnd::Xoshiro256StarStar rng(42);
    (void)rng.normal();
    if (large)
      rng.long_jump();
    else
      rng.jump();
    rnd::Xoshiro256StarStar fresh(rng.state());
    EXPECT_DOUBLE_EQ(rng.normal(), fresh.normal());
  }
}
TEST(QualityAcrossSeeds, MatchesManualFirstLevelAggregation) {
  const auto summary = rnd::quality_tests_over_seeds(
      [](std::uint64_t seed) { return std::make_unique<rnd::Xoshiro256StarStar>(seed); },
      4,
      1000,
      7);
  ASSERT_EQ(summary.nrows(), 7);
  std::vector<double> sums(7), small(7), large(7);
  for (int seed = 7; seed < 11; ++seed) {
    rnd::Xoshiro256StarStar rng(seed);
    const auto              first = rnd::quality_tests(rng, 1000);
    for (std::size_t t = 0; t < 7; ++t) {
      const double p = first.double_at("p_value", t);
      sums[t] += p;
      small[t] += p < .05;
      large[t] += p > .95;
    }
  }
  for (std::size_t t = 0; t < 7; ++t) {
    EXPECT_DOUBLE_EQ(summary.double_at("mean_p", t), sums[t] / 4);
    EXPECT_EQ(summary.double_at("small_p_count", t), small[t]);
    EXPECT_EQ(summary.double_at("large_p_count", t), large[t]);
    EXPECT_GE(summary.double_at("ks_p_value", t), 0);
    EXPECT_LE(summary.double_at("ks_p_value", t), 1);
  }
}
TEST(QualityAcrossSeeds, RejectsInvalidFactoriesAndCounts) {
  const rnd::GeneratorFactory valid = [](std::uint64_t seed) {
    return std::make_unique<rnd::Xoshiro256StarStar>(seed);
  };
  EXPECT_THROW(rnd::quality_tests_over_seeds({}, 4, 1000), std::invalid_argument);
  EXPECT_THROW(rnd::quality_tests_over_seeds(valid, 1, 1000), std::invalid_argument);
  EXPECT_THROW(rnd::quality_tests_over_seeds(valid, 4, 999), std::invalid_argument);
  const rnd::GeneratorFactory empty = [](std::uint64_t) {
    return std::unique_ptr<rnd::RandomGenerator>{};
  };
  EXPECT_THROW(rnd::quality_tests_over_seeds(empty, 2, 1000), std::invalid_argument);
}
TEST(BootstrapWithEngine, MatchesManualUnbiasedResamplingAndIgnoresOptionSeed) {
  const std::vector<double> data{1, 2, 3, 4};
  rnd::Xoshiro256StarStar   rng(123), reference(123), other(123);
  const auto result = datamunge::stats::Bootstrap({16, .95, 1}).run(data, xoshiro_test_mean, rng);
  std::vector<double> means;
  for (int b = 0; b < 16; ++b) {
    double sum = 0;
    for (int i = 0; i < 4; ++i)
      sum += data[reference.uniform_u64(0, 3)];
    means.push_back(sum / 4);
  }
  EXPECT_EQ(result.replicates, means);
  EXPECT_EQ(rng.state(), reference.state());
  EXPECT_EQ(
      result.replicates,
      datamunge::stats::Bootstrap({16, .95, 999}).run(data, xoshiro_test_mean, other).replicates);
  EXPECT_THROW(
      (void)datamunge::stats::Bootstrap().run({}, xoshiro_test_mean, rng), std::invalid_argument);
}
TEST(BootstrapWithEngine, DefaultEngineSequenceIsPreserved) {
  const std::vector<double> data{1, 2, 3, 4};
  const auto      result = datamunge::stats::Bootstrap({16, .95, 17}).run(data, xoshiro_test_mean);
  std::mt19937_64 rng(17);
  std::uniform_int_distribution<std::size_t> pick(0, 3);
  std::vector<double>                        means;
  for (int b = 0; b < 16; ++b) {
    double sum = 0;
    for (int i = 0; i < 4; ++i)
      sum += data[pick(rng)];
    means.push_back(sum / 4);
  }
  EXPECT_EQ(result.replicates, means);
}
