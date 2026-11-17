#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <datamunge/algorithms/exchange_sorts.hpp>
#include <datamunge/algorithms/painters_algorithm.hpp>
#include <datamunge/algorithms/patience_sort.hpp>
#include <datamunge/algorithms/pbkdf2.hpp>
#include <datamunge/algorithms/pcnn.hpp>
#include <datamunge/algorithms/pearson_hashing.hpp>
#include <datamunge/algorithms/perceptron.hpp>
#include <datamunge/algorithms/petrick.hpp>
#include <datamunge/algorithms/quine_mccluskey.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/permutation_test.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <set>
namespace a = datamunge::algorithms;
namespace s = datamunge::stats;
TEST(PainterTrace, IndependentPerPixelVisibilityAndActualAssignments) {
  std::mt19937 rng(19);
  for (int repeat = 0; repeat < 300; ++repeat) {
    std::vector<a::DepthRect> rects;
    for (int i = 0; i < 10; ++i) {
      int x = rng() % 10 - 2, y = rng() % 10 - 2;
      rects.push_back(
          {x,
           y,
           x + int(rng() % 6) - 1,
           y + int(rng() % 6) - 1,
           double(rng() % 5),
           int(rng() % 4)});
    }
    auto t = a::painter_trace(rects, 7, 6);
    EXPECT_EQ(t.raster, a::painter_render(rects, 7, 6));
    for (int y = 0; y < 6; ++y)
      for (int x = 0; x < 7; ++x) {
        int winner = -1;
        for (std::size_t i = 0; i < rects.size(); ++i) {
          auto& r = rects[i];
          if (x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1 &&
              (winner < 0 || r.depth <= rects[winner].depth))
            winner = i;
        }
        EXPECT_EQ(t.raster[y][x], winner < 0 ? 0 : rects[winner].color);
      }
    std::vector<std::vector<int>> raster(6, std::vector<int>(7));
    for (auto& w : t.writes) {
      EXPECT_EQ(w.before, raster[w.y][w.x]);
      EXPECT_EQ(w.after, rects[w.object].color);
      raster[w.y][w.x] = w.after;
    }
    EXPECT_EQ(raster, t.raster);
  }
}
TEST(PainterTrace, ClippingEqualDepthEmptyAndFullGuards) {
  EXPECT_EQ(a::painter_order({1, 3, 3}), (std::vector<int>{1, 2, 0}));
  EXPECT_TRUE(a::painter_render({}, 0, 0).empty());
  EXPECT_EQ(a::painter_render({}, 0, 2).size(), 2);
  EXPECT_THROW(a::painter_render({}, -1, 0), std::invalid_argument);
  EXPECT_THROW(a::painter_order({0, std::nan("")}), std::invalid_argument);
  EXPECT_THROW(
      a::painter_render({{0, 0, 0, 0, std::numeric_limits<double>::infinity(), 1}}, 0, 0),
      std::invalid_argument);
  EXPECT_THROW(a::painter_trace({{0, 0, 1, 0, 1, 1}}, 2, 1, 1), std::length_error);
  EXPECT_EQ(a::painter_trace({{0, 0, 1, 0, 1, 1}}, 2, 1, 2).writes.size(), 2);
}
TEST(PancakeTrace, ExhaustivePermutationsActualFlipsSearchesAndBounds) {
  for (int n = 0; n <= 8; ++n) {
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    do {
      auto t        = a::pancake_sort_trace(v);
      auto expected = v;
      std::sort(expected.begin(), expected.end());
      EXPECT_EQ(t.result.sorted, expected);
      EXPECT_EQ(t.result.flips, a::pancake_sort(v).flips);
      auto state = v;
      ASSERT_EQ(t.states.size(), t.result.flips.size() + 1);
      for (std::size_t i = 0; i < t.result.flips.size(); ++i) {
        auto k = t.result.flips[i];
        ASSERT_GE(k, 2);
        ASSERT_LE(k, v.size());
        std::reverse(state.begin(), state.begin() + k);
        EXPECT_EQ(state, t.states[i + 1]);
      }
      EXPECT_EQ(state, expected);
      EXPECT_LE(t.result.flips.size(), n ? 2 * (n - 1) : 0);
      EXPECT_EQ(t.steps.size(), n ? n - 1 : 0);
      state = v;
      for (auto& r : t.steps) {
        auto maxi = std::max_element(state.begin(), state.begin() + r.prefix) - state.begin();
        EXPECT_EQ(r.maximum_index, maxi);
        EXPECT_EQ(r.comparisons, r.prefix - 1);
        for (auto f = r.flip_begin; f < r.flip_end; ++f)
          std::reverse(state.begin(), state.begin() + t.result.flips[f]);
        EXPECT_EQ(state, r.values);
      }
    } while (std::next_permutation(v.begin(), v.end()));
  }
}
TEST(PancakeTrace, DuplicatesExtremesQuotaAndOriginal) {
  auto t = a::pancake_sort_trace({5, 3, 8, 1, 9, 2, 7});
  EXPECT_EQ(t.result.sorted, (std::vector<int>{1, 2, 3, 5, 7, 8, 9}));
  EXPECT_EQ(
      a::pancake_sort({INT_MAX, INT_MIN, INT_MAX}).sorted,
      (std::vector<int>{INT_MIN, INT_MAX, INT_MAX}));
  EXPECT_THROW(a::pancake_sort_trace({2, 1}, 6), std::length_error);
  EXPECT_EQ(a::pancake_sort_trace({2, 1}, 7).states.size(), 2);
  EXPECT_NO_THROW(a::pancake_sort_trace({2, 1}, 0));
}
TEST(PatienceTrace, IndependentQuadraticLISAndHeapSources) {
  std::mt19937 rng(25);
  for (int repeat = 0; repeat < 1000; ++repeat) {
    std::vector<int> v(rng() % 60);
    for (int& x : v)
      x = int(rng() % 17) - 8;
    auto t        = a::patience_sort_trace(v);
    auto expected = v;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(t.sorted, expected);
    auto plain = v;
    a::patience_sort(plain);
    EXPECT_EQ(plain, expected);
    std::vector<std::size_t> dp(v.size(), 1);
    std::size_t              lis = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
      for (std::size_t j = 0; j < i; ++j)
        if (v[j] < v[i])
          dp[i] = std::max(dp[i], dp[j] + 1);
      lis = std::max(lis, dp[i]);
    }
    EXPECT_EQ(t.piles.size(), lis);
    auto piles = t.piles;
    for (std::size_t i = 0; i < t.sorted.size(); ++i) {
      auto p = t.extracted_piles[i];
      ASSERT_FALSE(piles[p].empty());
      EXPECT_EQ(piles[p].back(), t.sorted[i]);
      piles[p].pop_back();
    }
    EXPECT_EQ(t.deals.size(), v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      EXPECT_EQ(t.deals[i].value, v[i]);
      EXPECT_TRUE(std::is_sorted(t.deals[i].tops.begin(), t.deals[i].tops.end()));
    }
  }
}
TEST(PatienceTrace, StrictDuplicatesEmptyGenericAndQuota) {
  EXPECT_EQ(a::patience_sort_trace({2, 2, 2}).piles.size(), 1);
  EXPECT_TRUE(a::patience_sort_trace({}).sorted.empty());
  std::vector<std::string> v{"c", "a", "b"};
  a::patience_sort(v);
  EXPECT_EQ(v, (std::vector<std::string>{"a", "b", "c"}));
  EXPECT_THROW(a::patience_sort_trace({1}, 5), std::length_error);
  EXPECT_EQ(a::patience_sort_trace({1}, 6).sorted, (std::vector<int>{1}));
}
TEST(Pbkdf2Trace, PublishedFixturesExactChainXORAndBlockBoundaries) {
  EXPECT_EQ(
      a::pbkdf2_hmac_sha256_hex("password", "salt", 1, 32),
      "120fb6cffcf8b32c43e7225256c4f837a86548c92ccc35480805987cb70be17b");
  EXPECT_EQ(
      a::pbkdf2_hmac_sha256_hex("password", "salt", 2, 32),
      "ae4d0c95af6b46d32d0adff928f06dd02a303f8ef3c251dfd6e2d85a95474c43");
  EXPECT_EQ(
      a::pbkdf2_hmac_sha256_hex("password", "salt", 4096, 32),
      "c5e478d59288c841aa530db6845c4c8d962893a001ce4e11a4963873aa98134a");
  for (std::size_t n : {0, 1, 31, 32, 33, 64, 65}) {
    auto t = a::pbkdf2_hmac_sha256_trace("password", "salt", 3, n);
    EXPECT_EQ(t.key, a::pbkdf2_hmac_sha256("password", "salt", 3, n));
    EXPECT_EQ(t.hmac_calls, 3 * (n / 32 + (n % 32 != 0)));
    EXPECT_EQ(t.steps.size(), t.hmac_calls);
    std::array<std::uint8_t, 32> acc{};
    for (auto& r : t.steps) {
      if (r.round == 1)
        acc.fill(0);
      for (std::size_t k = 0; k < 32; ++k)
        acc[k] ^= r.u[k];
      EXPECT_EQ(acc, r.accumulator);
    }
  }
}
TEST(Pbkdf2Trace, ZeroIterationsWorkPreflightWideLengthAndBinary) {
  EXPECT_THROW(a::pbkdf2_hmac_sha256("", "", 0, 0), std::invalid_argument);
  EXPECT_THROW(a::pbkdf2_hmac_sha256_run("p", "s", 3, 33, {5, 0}), std::length_error);
  EXPECT_EQ(a::pbkdf2_hmac_sha256_run("p", "s", 3, 33, {6, 0}).hmac_calls, 6);
  EXPECT_THROW(a::pbkdf2_hmac_sha256_trace("p", "s", 3, 33, {0, 5}), std::length_error);
  EXPECT_THROW(
      a::pbkdf2_hmac_sha256_run("p", "s", 1, std::numeric_limits<std::size_t>::max(), {0, 0}),
      std::length_error);
  EXPECT_NE(
      a::pbkdf2_hmac_sha256(std::string("p\0x", 3), std::string("s\0y", 3), 2, 32),
      a::pbkdf2_hmac_sha256("p", "s", 2, 32));
}
TEST(PcgActual, ReferenceSeedWordsRotateOperandsAndPairedOutput) {
  datamunge::random::Pcg32   rng(42, 54);
  std::vector<std::uint32_t> expected{
      0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e};
  auto paired = rng;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    auto t = rng.next_u32_trace();
    EXPECT_EQ(t.output, expected[i]);
    EXPECT_EQ(t.after, t.before * 6364136223846793005ULL + rng.increment());
    EXPECT_EQ(t.output, std::rotr(t.xorshifted, static_cast<int>(t.rotation)));
    EXPECT_EQ(t.after, rng.state());
  }
  EXPECT_EQ(paired.next_u64(), (std::uint64_t(expected[0]) << 32) | expected[1]);
}
TEST(PcgActual, SelectorAliasCopiesAndOriginalSchedule) {
  datamunge::random::Pcg32 a(42, 54), b(42, 54ULL + (1ULL << 63)), c = a;
  for (int i = 0; i < 1000; ++i) {
    auto word = a.next_u64();
    EXPECT_EQ(word, b.next_u64());
    EXPECT_EQ(word, c.next_u64());
  }
  datamunge::random::Pcg32 original(0xC0FFEEULL);
  EXPECT_EQ(original.next_u64(), 3763436240344971625ULL);
  EXPECT_NEAR(original.exponential(4), .018741, 1e-6);
  EXPECT_FALSE(original.bernoulli(.3));
}
TEST(PCNNActual, IndependentScalarNoLinkThresholdScheduleAndCapturedCells) {
  a::PCNNParameters p;
  p.iterations           = 12;
  p.beta                 = 0;
  const double stimulus  = .3;
  auto         t         = a::pcnn_trace({{stimulus}}, p);
  double       threshold = p.initial_threshold;
  int          first     = 0;
  for (std::size_t n = 0; n < p.iterations; ++n) {
    auto before = threshold;
    threshold *= p.threshold_decay;
    auto decayed = threshold;
    int  pulse   = stimulus > threshold;
    if (pulse) {
      threshold += p.threshold_gain;
      if (!first)
        first = n + 1;
    }
    EXPECT_EQ(t.result.time_signature[n], pulse);
    auto& r = t.cells[n];
    EXPECT_EQ(r.before, before);
    EXPECT_EQ(r.decayed, decayed);
    EXPECT_EQ(r.linking, 0);
    EXPECT_EQ(r.activity, stimulus);
    EXPECT_EQ(r.pulse, pulse);
    EXPECT_EQ(r.after, threshold);
  }
  EXPECT_EQ(t.result.first_fire[0][0], first);
}
TEST(PCNNActual, SynchronousNeighborsOriginalDiskAndGuards) {
  a::PCNNParameters p;
  p.iterations = 2;
  auto t       = a::pcnn_trace({{.9, .3}}, p);
  EXPECT_EQ(t.cells[1].linking, 0);
  EXPECT_EQ(t.cells[3].linking, 1);
  EXPECT_EQ(t.cells[0].pulse, 1);
  p.iterations = 25;
  std::vector<std::vector<double>> img(16, std::vector<double>(16, .3));
  for (int y = 0; y < 16; ++y)
    for (int x = 0; x < 16; ++x)
      if (std::hypot(y - 7.5, x - 7.5) <= 4)
        img[y][x] = .9;
  auto r = a::pcnn_run(img, p);
  EXPECT_EQ(r.pulses.size(), 25);
  for (std::size_t n = 0; n < r.pulses.size(); ++n) {
    int sum = 0;
    for (auto& row : r.pulses[n])
      sum = std::accumulate(row.begin(), row.end(), sum);
    EXPECT_EQ(sum, r.time_signature[n]);
  }
  EXPECT_THROW(a::pcnn_run({{1}, {1, 2}}, p), std::invalid_argument);
  EXPECT_THROW(a::pcnn_run({{std::nan("")}}, p), std::invalid_argument);
  p.beta = -1;
  EXPECT_THROW(a::pcnn_run({}, p), std::invalid_argument);
  p.beta       = .2;
  p.iterations = 2;
  EXPECT_THROW(a::pcnn_trace({{1}}, p, {0, 1}), std::length_error);
  EXPECT_EQ(a::pcnn_trace({{1}}, p, {0, 2}).cells.size(), 2);
  EXPECT_THROW(a::pcnn_trace({{1}}, p, {4, 0}), std::length_error);
  p.initial_threshold = std::numeric_limits<double>::max();
  p.threshold_gain    = std::numeric_limits<double>::max();
  p.threshold_decay   = 1;
  EXPECT_NO_THROW(a::pcnn_run({{1}}, p));
  p.initial_threshold = 0;
  p.threshold_gain    = std::numeric_limits<double>::max();
  p.beta              = std::numeric_limits<double>::max();
  EXPECT_THROW(a::pcnn_run({{2, 2}}, p), std::overflow_error);
}
TEST(PearsonActual, CompletePermutationOneByteAndSingleSubstitutionBijection) {
  std::set<unsigned> outputs;
  for (int b = 0; b < 256; ++b)
    outputs.insert(a::pearson_hash(std::string(1, char(b))));
  EXPECT_EQ(outputs.size(), 256);
  for (std::size_t pos = 0; pos < 5; ++pos) {
    std::string value = "abcde";
    outputs.clear();
    for (int b = 0; b < 256; ++b) {
      value[pos] = char(b);
      outputs.insert(a::pearson_hash(value));
    }
    EXPECT_EQ(outputs.size(), 256);
  }
  auto         t = a::pearson_hash_trace("hash");
  std::uint8_t h = 0;
  for (auto& r : t.steps) {
    EXPECT_EQ(r.before, h);
    EXPECT_EQ(r.table_index, std::uint8_t(h ^ r.byte));
    h = a::pearson_default_table()[r.table_index];
    EXPECT_EQ(r.after, h);
  }
  EXPECT_EQ(t.hash, a::pearson_hash("hash"));
}
TEST(PearsonActual, CompleteCustomTableValidationWidthAndBinary) {
  std::array<std::uint8_t, 256> t{};
  EXPECT_THROW(a::pearson_hash("", t), std::invalid_argument);
  std::iota(t.begin(), t.end(), 0);
  EXPECT_EQ(a::pearson_hash(std::string("\0\x80\xff", 3), t), 0x7f);
  EXPECT_THROW(a::pearson_hash_wide("", 0), std::invalid_argument);
  EXPECT_THROW(a::pearson_hash_wide("", 9), std::invalid_argument);
  EXPECT_THROW(a::pearson_hash_trace("hash", a::pearson_default_table(), 3), std::length_error);
  EXPECT_EQ(a::pearson_hash_trace("hash", a::pearson_default_table(), 4).steps.size(), 4);
  EXPECT_EQ(a::pearson_hash_wide("hash", 1), a::pearson_hash("hash"));
}
TEST(PerceptronActual, IndependentIntegerUpdatesCertificationAndOriginalBoundary) {
  std::vector<std::vector<double>> x{
      {2, 2}, {3, 1}, {2, 3}, {1, 2}, {3, 3}, {-2, -1}, {-1, -2}, {-3, -1}, {-1, -3}, {-2, -2}};
  std::vector<int>       y{1, 1, 1, 1, 1, -1, -1, -1, -1, -1};
  auto                   t = a::perceptron_trace(x, y);
  std::vector<long long> w(2);
  long long              bias    = 0;
  std::size_t            updates = 0;
  for (auto& r : t.visits) {
    long long score = bias;
    for (int j = 0; j < 2; ++j)
      score += w[j] * static_cast<long long>(x[r.row][j]);
    EXPECT_EQ(r.score, score);
    bool change = y[r.row] * score <= 0;
    EXPECT_EQ(r.updated, change);
    if (change) {
      for (int j = 0; j < 2; ++j)
        w[j] += y[r.row] * static_cast<long long>(x[r.row][j]);
      bias += y[r.row];
      ++updates;
    }
    EXPECT_EQ(r.bias, bias);
    for (int j = 0; j < 2; ++j)
      EXPECT_EQ(r.weights[j], w[j]);
  }
  EXPECT_EQ(updates, t.model.total_mistakes);
  EXPECT_TRUE(t.model.converged);
  for (std::size_t i = 0; i < x.size(); ++i)
    EXPECT_EQ(a::perceptron_predict(t.model, x[i]), y[i]);
  EXPECT_EQ(t.model.weights, a::perceptron_train(x, y).weights);
}
TEST(PerceptronActual, ContradictionsZeroEpochFullGuardsAndHistoryAdmission) {
  auto m = a::perceptron_train({{1}, {1}}, {1, -1}, 5);
  EXPECT_FALSE(m.converged);
  EXPECT_EQ(m.epochs_run, 5);
  EXPECT_EQ(a::perceptron_train({{1}}, {1}, 0).weights, (std::vector<double>{0}));
  EXPECT_THROW(a::perceptron_train({{1}, {1, 2}}, {1, -1}), std::invalid_argument);
  EXPECT_THROW(a::perceptron_train({{1}}, {0}), std::invalid_argument);
  EXPECT_THROW(a::perceptron_train({}, {}, 1, 0), std::invalid_argument);
  EXPECT_THROW(a::perceptron_train({{std::nan("")}}, {1}, 0), std::invalid_argument);
  EXPECT_THROW(a::perceptron_predict(m, {}), std::invalid_argument);
  EXPECT_THROW(a::perceptron_trace({{1}}, {1}, 1, 1, 1), std::length_error);
  EXPECT_EQ(a::perceptron_trace({{1}}, {1}, 1, 1, 2).visits.size(), 1);
  EXPECT_THROW(
      a::perceptron_train({{std::numeric_limits<double>::max()}}, {1}, 2), std::overflow_error);
  EXPECT_FALSE(a::perceptron_train({}, {}).converged);
}
TEST(PermutationActual, IndependentAssignmentsTailCountsCallbackScheduleAndSingleton) {
  std::vector<double> pool{1, 2, 3, 4};
  for (auto alt : {s::Alternative::TwoSided, s::Alternative::Less, s::Alternative::Greater}) {
    std::size_t calls     = 0;
    auto        statistic = [&](const auto& x, const auto& y) {
      ++calls;
      return std::accumulate(x.begin(), x.end(), 0.0) / x.size() -
             std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    };
    auto t = s::permutation_test_trace({1, 2}, {3, 4}, statistic, {999, alt, 42});
    EXPECT_EQ(calls, 1000);
    std::size_t extreme = 0;
    long double mean    = 0;
    for (std::size_t i = 0; i < t.draws.size(); ++i) {
      auto&                 r = t.draws[i];
      std::set<std::size_t> ids(r.first_group.begin(), r.first_group.end());
      ASSERT_EQ(ids.size(), 2);
      double first = 0, second = 0;
      for (std::size_t j = 0; j < 4; ++j)
        (ids.contains(j) ? first : second) += pool[j] / 2;
      EXPECT_EQ(first - second, r.statistic);
      bool hit = alt == s::Alternative::Less      ? r.statistic <= -2
                 : alt == s::Alternative::Greater ? r.statistic >= -2
                                                  : std::abs(r.statistic) >= 2;
      EXPECT_EQ(hit, r.extreme);
      extreme += hit;
      mean += r.statistic;
    }
    EXPECT_DOUBLE_EQ(t.result.p_value, double(1 + extreme) / 1000);
    EXPECT_NEAR(t.result.null_mean, double(mean / 999), 1e-14);
  }
  EXPECT_EQ(
      s::permutation_test_two_sample({1}, {2}, {}, {1, s::Alternative::TwoSided, 42}).p_value, 1);
}
TEST(PermutationActual, CompletePreCallbackGuardsNonfiniteStatisticsAndWideMeans) {
  int  calls = 0;
  auto stat  = [&](const auto&, const auto&) {
    ++calls;
    return 0.;
  };
  EXPECT_THROW(
      s::permutation_test_trace({1}, {2}, stat, {2, s::Alternative::TwoSided, 42}, 3),
      std::length_error);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(s::permutation_test_two_sample({std::nan("")}, {2}, stat), std::invalid_argument);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(
      s::permutation_test_two_sample({1}, {2}, {}, {1, static_cast<s::Alternative>(99), 42}),
      std::invalid_argument);
  EXPECT_THROW(
      s::permutation_test_two_sample(
          {1}, {2}, [](const auto&, const auto&) { return std::nan(""); }),
      std::invalid_argument);
  double maximum = std::numeric_limits<double>::max();
  EXPECT_EQ(
      s::permutation_test_two_sample(
          {maximum, maximum}, {maximum, maximum}, {}, {2, s::Alternative::TwoSided, 42})
          .observed,
      0);
  EXPECT_THROW(
      s::permutation_test_two_sample(
          {1}, {2}, {}, {std::numeric_limits<std::size_t>::max(), s::Alternative::TwoSided, 42}),
      std::length_error);
}
TEST(PetrickActual, IndependentExhaustiveAllSmallChartsAndActualAbsorption) {
  std::mt19937 rng(31);
  for (int repeat = 0; repeat < 2000; ++repeat) {
    int                           n = 1 + rng() % 7, clauses = rng() % 7;
    std::vector<std::vector<int>> chart(clauses);
    for (auto& c : chart)
      for (int i = 0; i < n; ++i)
        if (rng() % 2)
          c.push_back(i);
    std::set<std::vector<int>> expected;
    int                        best = n + 1;
    for (int bits = 0; bits < (1 << n); ++bits) {
      bool valid = true;
      for (auto& c : chart) {
        bool covered = false;
        for (int id : c)
          covered |= bits & (1 << id);
        valid &= covered;
      }
      if (!valid)
        continue;
      int size = std::popcount(unsigned(bits));
      if (size < best) {
        best = size;
        expected.clear();
      }
      if (size == best) {
        std::vector<int> ids;
        for (int i = 0; i < n; ++i)
          if (bits & (1 << i))
            ids.push_back(i);
        expected.insert(ids);
      }
    }
    auto t = a::petrick_trace(chart);
    EXPECT_EQ(
        (std::set<std::vector<int>>(
            t.result.minimal_covers.begin(), t.result.minimal_covers.end())),
        expected);
    EXPECT_EQ(t.result.minimal_covers, a::petrick(chart).minimal_covers);
    std::size_t generated = 0;
    for (auto& r : t.stages) {
      generated += r.generated;
      EXPECT_EQ(r.generated - r.absorbed, r.terms.size());
      for (auto& term : r.terms)
        for (std::size_t ci = 0; ci <= r.clause; ++ci) {
          bool hit = false;
          for (int id : chart[ci])
            hit |= std::binary_search(term.begin(), term.end(), id);
          EXPECT_TRUE(hit);
        }
    }
    EXPECT_EQ(t.generated, generated);
  }
}
TEST(PetrickActual, LogicalIdentityFalseClauseOriginalChartAndLimits) {
  EXPECT_EQ(a::petrick({}).minimal_covers, (std::vector<std::vector<int>>{{}}));
  EXPECT_TRUE(a::petrick({{}}).minimal_covers.empty());
  EXPECT_THROW(a::petrick({{}, {-1}}), std::invalid_argument);
  EXPECT_THROW(a::petrick_trace({{0, 1}, {1, 2}}, {5, 0}), std::length_error);
  EXPECT_EQ(a::petrick_trace({{0, 1}, {1, 2}}, {6, 0}).generated, 6);
  EXPECT_THROW(a::petrick_trace({{0}}, {0, 2}), std::length_error);
  EXPECT_EQ(a::petrick_trace({{0}}, {0, 3}).stages.size(), 1);
  std::vector<std::uint32_t> minterms{0, 1, 2, 5, 6, 7};
  auto                       q = a::quine_mccluskey(3, minterms);
  EXPECT_TRUE(q.essential.empty());
  std::vector<std::vector<int>> chart(minterms.size());
  for (std::size_t i = 0; i < minterms.size(); ++i)
    for (std::size_t j = 0; j < q.prime_implicants.size(); ++j)
      if (q.prime_implicants[j].covers(minterms[i]))
        chart[i].push_back(j);
  auto r = a::petrick(chart);
  ASSERT_EQ(r.minimal_covers.size(), 2);
  EXPECT_EQ(r.minimal_covers.front().size(), 3);
}
