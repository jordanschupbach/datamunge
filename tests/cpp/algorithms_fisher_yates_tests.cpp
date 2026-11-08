#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/fisher_yates.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <type_traits>
namespace {
using namespace datamunge::algorithms;
// Independently written unsigned SplitMix and full-product threshold range oracle.
std::uint64_t word(std::uint64_t& state) {
  state += 0x9e3779b97f4a7c15ULL;
  auto z = state;
  z ^= z >> 30;
  z *= 0xbf58476d1ce4e5b9ULL;
  z ^= z >> 27;
  z *= 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
std::vector<int> oracle(std::vector<int> a, std::uint64_t seed, std::uint64_t& words) {
  for (std::size_t end = a.size(); end > 1; --end) {
    const std::uint64_t bound = end;
    const auto          threshold =
        static_cast<std::uint64_t>((static_cast<unsigned __int128>(1) << 64) % bound);
    unsigned __int128 p;
    do {
      p = static_cast<unsigned __int128>(word(seed)) * bound;
      ++words;
    } while (static_cast<std::uint64_t>(p) < threshold);
    std::swap(a[end - 1], a[static_cast<std::size_t>(p >> 64)]);
  }
  return a;
}
TEST(FisherYatesNative, OriginalGoldenAndActualTrace) {
  const std::vector<int> a{1, 2, 3, 4, 5, 6};
  const auto             t = fisher_yates_shuffle_trace(a, 42);
  EXPECT_EQ(t.outcome.values, (std::vector<int>{3, 6, 4, 2, 1, 5}));
  EXPECT_EQ(t.input, a);
  EXPECT_EQ(t.steps.size(), 5);
  EXPECT_EQ(t.outcome.stages, 5);
  EXPECT_EQ(t.outcome.random_words, 5);
  auto          replay  = a;
  std::uint64_t raw     = 0;
  std::size_t   changed = 0;
  for (std::size_t k = 0; k < t.steps.size(); ++k) {
    const auto& s = t.steps[k];
    EXPECT_EQ(s.i, a.size() - 1 - k);
    EXPECT_LE(s.j, s.i);
    EXPECT_EQ(s.at_i, replay[s.i]);
    EXPECT_EQ(s.at_j, replay[s.j]);
    std::swap(replay[s.i], replay[s.j]);
    EXPECT_EQ(replay[s.i], t.outcome.values[s.i]);
    raw += s.random_words;
    if (s.i != s.j)
      ++changed;
  }
  EXPECT_EQ(replay, t.outcome.values);
  EXPECT_EQ(raw, t.outcome.random_words);
  EXPECT_EQ(changed, t.outcome.nontrivial_swaps);
}
TEST(FisherYatesNative, IndependentRawOracleAndTraceState) {
  std::mt19937_64 rng(112);
  for (int fixture = 0; fixture < 2400; ++fixture) {
    std::vector<int> a(rng() % 100);
    for (auto& v : a)
      v = int(rng() % 21) - 10;
    const auto    seed     = rng();
    std::uint64_t words    = 0;
    const auto    expected = oracle(a, seed, words);
    const auto    t        = fisher_yates_shuffle_trace(a, seed);
    EXPECT_EQ(t.outcome.values, expected);
    EXPECT_EQ(t.outcome.random_words, words);
    EXPECT_EQ(fisher_yates_shuffle(a, seed), expected);
    auto sorted = expected;
    std::sort(sorted.begin(), sorted.end());
    auto original = a;
    std::sort(original.begin(), original.end());
    EXPECT_EQ(sorted, original);
    EXPECT_EQ(t.input, a);
  }
}
TEST(FisherYatesNative, ActualRejectionRetry) {
  constexpr std::uint64_t seed  = 0x61c8864680b583ebULL;
  auto                    state = seed;
  EXPECT_EQ(word(state), 0);
  const auto t = fisher_yates_shuffle_trace({1, 2, 3}, seed);
  ASSERT_EQ(t.steps.size(), 2);
  EXPECT_EQ(t.steps[0].random_words, 2);
  EXPECT_EQ(t.steps[1].random_words, 1);
  EXPECT_EQ(t.outcome.random_words, 3);
  EXPECT_EQ(t.outcome.values, (std::vector<int>{2, 1, 3}));
  std::uint64_t words = 0;
  EXPECT_EQ(oracle({1, 2, 3}, seed, words), t.outcome.values);
  EXPECT_EQ(words, 3);
}
TEST(FisherYatesNative, AllChoiceSequencesBijectionThroughEight) {
  std::uint64_t fact = 1;
  for (int n = 0; n <= 8; ++n) {
    if (n)
      fact *= n;
    const auto c = shuffle_choice_census(n);
    EXPECT_EQ(c.outcomes, fact);
    EXPECT_EQ(c.permutation_counts.size(), fact);
    std::vector<int> a(n);
    std::iota(a.begin(), a.end(), 0);
    auto it = c.permutation_counts.begin();
    do {
      ASSERT_NE(it, c.permutation_counts.end());
      EXPECT_EQ(it->first, a);
      EXPECT_EQ(it->second, 1);
      ++it;
    } while (std::next_permutation(a.begin(), a.end()));
    EXPECT_EQ(it, c.permutation_counts.end());
    for (const auto& row : c.by_position)
      for (auto v : row)
        EXPECT_EQ(v, fact / n);
  }
}
TEST(FisherYatesNative, ExactBiasedAndFlatMarginalControls) {
  const auto bad = shuffle_choice_census(3, ShuffleCensusRule::AllPositions);
  EXPECT_EQ(bad.outcomes, 27);
  EXPECT_EQ(bad.permutation_counts.size(), 6);
  std::multiset<std::uint64_t> counts;
  for (const auto& [p, c] : bad.permutation_counts)
    counts.insert(c);
  EXPECT_EQ(counts, (std::multiset<std::uint64_t>{4, 4, 4, 5, 5, 5}));
  // Independent exhaustive triples use a differently ordered recursion in the test.
  std::map<std::vector<int>, std::uint64_t> m;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      for (int c = 0; c < 3; ++c) {
        std::vector<int> p{0, 1, 2};
        std::swap(p[0], p[a]);
        std::swap(p[1], p[b]);
        std::swap(p[2], p[c]);
        ++m[p];
      }
  EXPECT_EQ(m, bad.permutation_counts);
  const auto cyclic = shuffle_choice_census(3, ShuffleCensusRule::CyclicShift);
  EXPECT_EQ(cyclic.outcomes, 3);
  EXPECT_EQ(cyclic.permutation_counts.size(), 3);
  for (const auto& r : cyclic.by_position)
    EXPECT_EQ(r, (std::vector<std::uint64_t>{1, 1, 1}));
}
TEST(FisherYatesNative, SeedCensusOriginalTallyAndAllMarginals) {
  const auto c = fisher_yates_seed_census(6, 0, 60000);
  EXPECT_EQ(c.work_units, 420036);
  EXPECT_EQ(c.by_position[0], (std::vector<std::uint64_t>{10118, 9980, 9856, 10141, 9886, 10019}));
  EXPECT_EQ(c.permutation_counts.size(), 720);
  for (const auto& row : c.by_position)
    EXPECT_EQ(std::accumulate(row.begin(), row.end(), std::uint64_t{0}), 60000);
  for (int label = 0; label < 6; ++label) {
    std::uint64_t total = 0;
    for (int pos = 0; pos < 6; ++pos)
      total += c.by_position[pos][label];
    EXPECT_EQ(total, 60000);
  }
  std::vector<std::vector<std::uint64_t>> reconstructed(6, std::vector<std::uint64_t>(6));
  std::uint64_t                           sum = 0;
  for (const auto& [p, count] : c.permutation_counts) {
    sum += count;
    for (int i = 0; i < 6; ++i)
      reconstructed[i][p[i]] += count;
  }
  EXPECT_EQ(sum, 60000);
  EXPECT_EQ(reconstructed, c.by_position);
}
TEST(FisherYatesNative, ExplicitChoicesAndValidation) {
  EXPECT_EQ(fisher_yates_from_choices({1, 2, 3}, {0, 0}), (std::vector<int>{2, 3, 1}));
  EXPECT_EQ(fisher_yates_from_choices({1, 2, 3}, {2, 1}), (std::vector<int>{1, 2, 3}));
  EXPECT_THROW(fisher_yates_from_choices({1, 2, 3}, {0}), std::invalid_argument);
  EXPECT_THROW(fisher_yates_from_choices({1, 2, 3}, {3, 0}), std::invalid_argument);
  EXPECT_THROW(fisher_yates_from_choices({1, 2, 3}, {0, 2}), std::invalid_argument);
  EXPECT_THROW(fisher_yates_from_choices({}, {0}), std::invalid_argument);
  EXPECT_TRUE(fisher_yates_from_choices({}, {}).empty());
}
TEST(FisherYatesNative, CaptureCapsAndOwnership) {
  const std::vector<int> a{INT_MIN, 0, INT_MAX};
  const auto             t = fisher_yates_shuffle_trace(a, 42, 8);
  EXPECT_THROW(fisher_yates_shuffle_trace(a, 42, 7), std::length_error);
  EXPECT_EQ(fisher_yates_shuffle_trace(a, 42, 0).outcome.values, t.outcome.values);
  auto copy              = t;
  copy.input[0]          = 99;
  copy.steps[0].j        = 99;
  copy.outcome.values[0] = 99;
  EXPECT_NE(copy.input, t.input);
  EXPECT_NE(copy.steps[0].j, t.steps[0].j);
  EXPECT_NE(copy.outcome.values, t.outcome.values);
  EXPECT_EQ(a, (std::vector<int>{INT_MIN, 0, INT_MAX}));
  const auto singleton = fisher_yates_shuffle_trace({7}, 1, 2);
  EXPECT_EQ(singleton.outcome.random_words, 0);
  EXPECT_TRUE(singleton.steps.empty());
  EXPECT_THROW(fisher_yates_shuffle_trace({7}, 1, 1), std::length_error);
  const auto empty = fisher_yates_shuffle_trace({}, 1, 1);
  EXPECT_TRUE(empty.steps.empty());
  EXPECT_EQ(empty.outcome.random_words, 0);
}
TEST(FisherYatesNative, CensusWholeBudgetsAndDomain) {
  EXPECT_EQ(shuffle_choice_census(3, ShuffleCensusRule::FisherYates, 33).work_units, 33);
  EXPECT_THROW(shuffle_choice_census(3, ShuffleCensusRule::FisherYates, 32), std::length_error);
  EXPECT_EQ(shuffle_choice_census(3, ShuffleCensusRule::FisherYates, 0).outcomes, 6);
  EXPECT_EQ(fisher_yates_seed_census(2, 0, 10, 34).outcomes, 10);
  EXPECT_THROW(fisher_yates_seed_census(2, 0, 10, 33), std::length_error);
  const auto max = std::numeric_limits<std::uint64_t>::max();
  EXPECT_THROW(fisher_yates_seed_census(1, max, 2), std::invalid_argument);
  EXPECT_EQ(fisher_yates_seed_census(1, max, 1).outcomes, 1);
  EXPECT_THROW(shuffle_choice_census(21, ShuffleCensusRule::FisherYates, 0), std::length_error);
  EXPECT_THROW(shuffle_choice_census(16, ShuffleCensusRule::AllPositions, 0), std::length_error);
  EXPECT_THROW(shuffle_choice_census(static_cast<std::size_t>(INT_MAX) + 1), std::length_error);
  EXPECT_THROW(shuffle_choice_census(3, static_cast<ShuffleCensusRule>(99)), std::invalid_argument);
  EXPECT_THROW(fisher_yates_seed_census(2, 0, max, 0), std::length_error);
}
TEST(FisherYatesNative, EmptyCensusesAndNonzeroIntervals) {
  for (auto rule :
       {ShuffleCensusRule::FisherYates,
        ShuffleCensusRule::AllPositions,
        ShuffleCensusRule::CyclicShift}) {
    auto c = shuffle_choice_census(0, rule);
    EXPECT_EQ(c.outcomes, 1);
    EXPECT_EQ(c.permutation_counts.size(), 1);
    EXPECT_EQ(c.permutation_counts.begin()->first, (std::vector<int>{}));
    EXPECT_TRUE(c.by_position.empty());
  }
  const auto z = fisher_yates_seed_census(3, UINT64_MAX, 0, 9);
  EXPECT_EQ(z.outcomes, 0);
  EXPECT_TRUE(z.permutation_counts.empty());
  for (const auto& r : z.by_position)
    EXPECT_EQ(r, (std::vector<std::uint64_t>{0, 0, 0}));
  EXPECT_EQ(fisher_yates_seed_census(0, 0, 5).permutation_counts.begin()->second, 5);
  const auto                                c = fisher_yates_seed_census(4, 100, 300);
  std::map<std::vector<int>, std::uint64_t> expected;
  for (std::uint64_t s = 100; s < 400; ++s)
    ++expected[fisher_yates_shuffle({0, 1, 2, 3}, s)];
  EXPECT_EQ(c.permutation_counts, expected);
}
TEST(FisherYatesNative, LegacySignatureDuplicatesAndWork) {
  using Old = std::vector<int> (*)(const std::vector<int>&, std::uint64_t);
  static_assert(std::is_same_v<decltype(&fisher_yates_shuffle), Old>);
  const auto r = fisher_yates_shuffle_work({4, 4, 4, 4}, 42);
  EXPECT_EQ(r.values, (std::vector<int>{4, 4, 4, 4}));
  EXPECT_EQ(r.stages, 3);
  EXPECT_LE(r.nontrivial_swaps, 3);
  EXPECT_EQ(r.random_words, 3);
  std::vector<int> a(100001);
  std::iota(a.begin(), a.end(), 0);
  const auto out = fisher_yates_shuffle_work(a, 99);
  EXPECT_EQ(out.stages, 100000);
  auto sorted = out.values;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(sorted, a);
}
} // namespace
