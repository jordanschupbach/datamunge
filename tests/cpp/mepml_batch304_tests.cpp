#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <datamunge/algorithms/qr_algorithm.hpp>
#include <datamunge/algorithms/quadratic_sieve.hpp>
#include <datamunge/algorithms/quickselect.hpp>
#include <datamunge/algorithms/quicksort.hpp>
#include <datamunge/algorithms/quine_mccluskey.hpp>
#include <datamunge/algorithms/rabin_karp.hpp>
#include <datamunge/algorithms/radix_sort.hpp>
#include <datamunge/algorithms/ramer_douglas_peucker.hpp>
#include <datamunge/geometry/convex_hull.hpp>
#include <datamunge/geometry/orientation.hpp>
#include <datamunge/geometry/quickhull.hpp>
#include <datamunge/random/sampling_analysis.hpp>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <string>
namespace a = datamunge::algorithms;
namespace h = datamunge::geometry;
namespace r = datamunge::random;
TEST(Batch304QR, SingularShiftsAndKnownDenseSpectrum) {
  const a::QrMatrix input{
      {2, -2.5, -3.5, 0}, {-2.5, 2, 0, -3.5}, {-3.5, 0, 2, -2.5}, {0, -3.5, -2.5, 2}};
  const auto                t = a::qr_eigenvalues_trace(input);
  const std::vector<double> expected{-4, 1, 3, 8};
  ASSERT_EQ(t.eigenvalues.size(), 4);
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(t.eigenvalues[i], expected[i], 1e-10);
  EXPECT_EQ(t.deflations, 4);
  EXPECT_FALSE(t.steps.empty());
  EXPECT_EQ(a::qr_eigenvalues_run({{2, 0}, {0, 2}}).iterations, 0);
  for (const auto& s : t.steps)
    for (std::size_t i = 0; i < s.active; ++i)
      for (std::size_t j = 0; j < s.active; ++j)
        EXPECT_DOUBLE_EQ(s.matrix[i][j], s.matrix[j][i]);
}
TEST(Batch304QR, CompleteRowDeflationAndAnalyticToeplitz) {
  const auto t = a::qr_eigenvalues_run({{2, 1, 1}, {1, 3, 0}, {1, 0, 4}});
  EXPECT_GT(t.iterations, 0);
  EXPECT_NEAR(std::accumulate(t.eigenvalues.begin(), t.eigenvalues.end(), 0.), 9, 1e-10);
  for (std::size_t n = 1; n <= 24; ++n) {
    const auto e = a::qr_eigenvalues(a::qr_tridiagonal(n));
    for (std::size_t j = 0; j < n; ++j)
      EXPECT_NEAR(e[j], 2 - 2 * std::cos((j + 1) * std::numbers::pi / (n + 1)), 1e-9);
  }
}
TEST(Batch304QR, SymmetryRoundingUsesWholeMatrixScale) {
  const auto t = a::qr_eigenvalues_run({{2, 1e-18}, {2e-18, 3}});
  EXPECT_NEAR(t.eigenvalues[0], 2, 1e-12);
  EXPECT_NEAR(t.eigenvalues[1], 3, 1e-12);
  EXPECT_THROW(a::qr_eigenvalues({{2, 1e-6}, {2e-6, 3}}), std::invalid_argument);
}
TEST(Batch304QR, ValidationBudgetAndExactHistoryAdmission) {
  EXPECT_THROW(a::qr_eigenvalues({{1, 2}, {0, 1}}), std::invalid_argument);
  EXPECT_THROW(a::qr_eigenvalues({{1}, {2}}), std::invalid_argument);
  EXPECT_THROW(a::qr_eigenvalues({{1}}, -1), std::invalid_argument);
  EXPECT_THROW(a::qr_eigenvalues({{2, 1}, {1, 2}}, 0), std::runtime_error);
  a::QrOptions o;
  o.max_trace_entries = 10;
  EXPECT_EQ(a::qr_eigenvalues_trace({{1, 0}, {0, 2}}, o).steps.size(), 2);
  o.max_trace_entries = 9;
  EXPECT_THROW(a::qr_eigenvalues_trace({{1, 0}, {0, 2}}, o), std::length_error);
  o           = {};
  o.max_order = 1;
  EXPECT_THROW(a::qr_eigenvalues_run({{1, 0}, {0, 1}}, o), std::length_error);
  EXPECT_TRUE(a::qr_eigenvalues({}).empty());
}
TEST(Batch304Sieve, ActualSingleRelationAndCongruence) {
  a::QuadraticSieveOptions o;
  o.factor_base_bound = 3;
  const auto t        = a::quadratic_sieve_trace(77, o);
  EXPECT_EQ(t.factor, 7);
  ASSERT_EQ(t.relations.size(), 1);
  EXPECT_EQ(t.factor_base, (std::vector<std::uint64_t>{2}));
  EXPECT_EQ(t.relations[0].x, 9);
  EXPECT_EQ(t.relations[0].exponents, (std::vector<unsigned>{2}));
  EXPECT_EQ(t.relations[0].pivot_column, -1);
  ASSERT_EQ(t.dependencies.size(), 1);
  EXPECT_EQ(t.dependencies[0].members, (std::vector<std::size_t>{0}));
  EXPECT_EQ(t.dependencies[0].X, 9);
  EXPECT_EQ(t.dependencies[0].Y, 2);
  EXPECT_EQ(t.dependencies[0].gcd, 7);
  EXPECT_EQ(t.smooth_relations, t.relations.size());
}
TEST(Batch304Sieve, ShortcutsBudgetsAndCompositeFactors) {
  EXPECT_THROW(a::quadratic_sieve(0), std::invalid_argument);
  EXPECT_THROW(a::quadratic_sieve(1), std::invalid_argument);
  EXPECT_EQ(a::quadratic_sieve(49), 7);
  a::QuadraticSieveOptions o;
  o.factor_base_bound = 3;
  o.max_windows       = 0;
  EXPECT_FALSE(a::quadratic_sieve_run(77, o).found);
  o.max_windows      = 1;
  o.max_sieve_values = 1;
  EXPECT_THROW(a::quadratic_sieve_run(77, o), std::length_error);
  o.max_sieve_values  = 0;
  o.max_trace_entries = 8;
  EXPECT_EQ(a::quadratic_sieve_trace(77, o).factor, 7);
  o.max_trace_entries = 7;
  EXPECT_THROW(a::quadratic_sieve_trace(77, o), std::length_error);
  for (std::uint64_t n : {15ULL, 77ULL, 10403ULL, 1000003ULL * 1000033ULL}) {
    auto q = a::quadratic_sieve_run(n);
    ASSERT_TRUE(q.found);
    EXPECT_EQ(n % q.factor, 0);
    EXPECT_GT(q.factor, 1);
    EXPECT_LT(q.factor, n);
  }
}
TEST(Batch304Hull, OriginalCloudAndExactPredicateControls) {
  const std::vector<h::Point2D> points{{0, 0},   {10, 0},  {18, 4},  {20, 12}, {15, 20},
                                       {6, 22},  {-2, 15}, {-4, 6},  {5, 5},   {10, 10},
                                       {8, 15},  {3, 10},  {12, 6},  {14, 14}, {2, 18},
                                       {16, 10}, {7, 3},   {11, 17}, {9, 8},   {13, 11}};
  auto                          t = h::quickhull_trace(points);
  EXPECT_EQ(t.hull.size(), 8);
  EXPECT_EQ(t.predicates, t.orientation_tests + t.area_comparisons);
  for (const auto& s : t.steps) {
    for (auto i : s.candidates) {
      EXPECT_GT(h::orientation_sign(t.points[s.a], t.points[s.b], t.points[i]), 0);
      EXPECT_LE(
          h::cross_difference_sign(t.points[s.a], t.points[s.b], t.points[i], t.points[s.farthest]),
          0);
    }
    for (auto i : s.left)
      EXPECT_GT(h::orientation_sign(t.points[s.a], t.points[s.farthest], t.points[i]), 0);
    for (auto i : s.right)
      EXPECT_GT(h::orientation_sign(t.points[s.farthest], t.points[s.b], t.points[i]), 0);
  }
  const double big  = std::numeric_limits<double>::max(),
               tiny = std::numeric_limits<double>::denorm_min();
  EXPECT_EQ(h::quickhull({{-big, -big}, {big, -big}, {big, big}, {-big, big}, {0, 0}}).size(), 4);
  EXPECT_EQ(h::quickhull({{0, 0}, {1, 0}, {0, tiny}}).size(), 3);
  EXPECT_GT(h::cross_difference_sign({-big, 0}, {big, 0}, {0, big}, {0, tiny}), 0);
}
TEST(Batch304Hull, IndependentMonotoneHullAndPolicies) {
  std::mt19937_64 rng(42);
  for (int trial = 0; trial < 100; ++trial) {
    std::vector<h::Point2D> p;
    for (int i = 0; i < 30; ++i)
      p.push_back({double(int(rng() % 41) - 20), double(int(rng() % 41) - 20)});
    auto x = h::quickhull(p), y = h::convex_hull(p);
    auto less = [](auto a, auto b) { return a.x < b.x || (a.x == b.x && a.y < b.y); };
    std::sort(x.begin(), x.end(), less);
    std::sort(y.begin(), y.end(), less);
    EXPECT_EQ(x, y);
  }
  EXPECT_THROW(h::quickhull({{0, std::numeric_limits<double>::infinity()}}), std::invalid_argument);
  h::QuickhullOptions o;
  o.max_predicates = 1;
  EXPECT_THROW(h::quickhull_run({{0, 0}, {1, 0}, {0, 1}}, o), std::length_error);
  o                   = {};
  o.max_trace_entries = 1;
  EXPECT_THROW(h::quickhull_trace({{0, 0}, {1, 0}, {0, 1}}, o), std::length_error);
}
TEST(Batch304Selection, EveryRankAndActualEqualBands) {
  std::mt19937_64 rng(42);
  for (std::size_t n = 1; n <= 50; ++n) {
    std::vector<std::int64_t> keys(n);
    for (auto& x : keys)
      x = int(rng() % 13) - 6;
    auto expected = keys;
    std::sort(expected.begin(), expected.end());
    for (std::size_t k = 0; k < n; ++k)
      for (bool mom : {false, true}) {
        a::QuickselectOptions o;
        o.median_of_medians = mom;
        const auto t        = a::quickselect_trace(keys, k, o);
        EXPECT_EQ(t.value, expected[k]);
        EXPECT_EQ(t.values[k], expected[k]);
        EXPECT_EQ(t.operations, t.comparisons + t.swaps + t.writes);
        for (const auto& s : t.steps) {
          EXPECT_LT(s.equal_begin, s.equal_end);
          for (std::size_t i = s.lo; i < s.hi; ++i) {
            auto v = s.values[i - s.lo];
            if (i < s.equal_begin)
              EXPECT_LT(v, s.pivot);
            else if (i < s.equal_end)
              EXPECT_EQ(v, s.pivot);
            else
              EXPECT_GT(v, s.pivot);
          }
        }
      }
  }
}
TEST(Batch304Selection, DuplicateLinearControlAndPolicies) {
  a::QuickselectOptions o;
  o.median_of_medians = true;
  for (std::size_t n : {32, 64, 128, 256, 512, 1024}) {
    auto r = a::quickselect_run(std::vector<std::int64_t>(n, 7), n / 2, o);
    EXPECT_EQ(r.value, 7);
    EXPECT_LT(r.comparisons, 8 * n);
  }
  EXPECT_THROW(a::quickselect({}, 0), std::invalid_argument);
  EXPECT_THROW(a::quickselect({1}, 1), std::invalid_argument);
  o.max_operations = 1;
  EXPECT_THROW(a::quickselect_run({3, 2, 1}, 1, o), std::length_error);
  o                   = {};
  o.max_trace_entries = 3;
  EXPECT_THROW(a::quickselect_trace({3, 2, 1}, 1, o), std::length_error);
  EXPECT_THROW(a::quickselect_scaling({10}, 0), std::invalid_argument);
  EXPECT_THROW(a::quickselect_scaling({10}, 2, 42, 19), std::length_error);
}
TEST(Batch304Quicksort, OriginalCutoffActualRangesAndCountContract) {
  const std::vector<std::int64_t> input{9, 3, 7, 1, 8, 5, 2};
  const auto                      t = a::quicksort_trace(input);
  ASSERT_EQ(t.steps.size(), 1);
  EXPECT_TRUE(t.steps[0].insertion);
  EXPECT_EQ(t.partitions, 0);
  EXPECT_EQ(t.values, (std::vector<std::int64_t>{1, 2, 3, 5, 7, 8, 9}));
  std::uint64_t count = 999;
  auto          copy  = input;
  a::quicksort(copy, count);
  EXPECT_EQ(count, t.comparisons);
  EXPECT_EQ(copy, t.values);
  auto u = a::quicksort_trace(std::vector<std::int64_t>(17, 7));
  EXPECT_EQ(u.partitions, 1);
  EXPECT_FALSE(u.steps[0].insertion);
  EXPECT_EQ(u.operations, u.comparisons + u.swaps + u.writes);
}
TEST(Batch304Quicksort, SortingQuadraticControlAndCaps) {
  std::mt19937_64 rng(42);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<std::int64_t> v(rng() % 250);
    for (auto& x : v)
      x = std::bit_cast<std::int64_t>(rng());
    auto expected = v;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(a::quicksort_run(v).values, expected);
  }
  for (std::size_t n : {32, 64, 128, 256}) {
    auto       t        = a::quicksort_run(std::vector<std::int64_t>(n, 7));
    const auto expected = (n * (n + 1) - 16 * 17) / 2 + 2 * (n - 16) + 15;
    EXPECT_EQ(t.comparisons, expected);
    EXPECT_LE(t.max_depth, 2);
  }
  a::QuicksortOptions o;
  o.max_operations = 1;
  EXPECT_THROW(a::quicksort_run({3, 2, 1}, o), std::length_error);
  o                   = {};
  o.max_trace_entries = 3;
  EXPECT_THROW(a::quicksort_trace({3, 2, 1}, o), std::length_error);
}
TEST(Batch304Boolean, ExhaustiveThreeVariableTruthAndMinimumTermCount) {
  for (unsigned truth = 0; truth < 256; ++truth) {
    std::vector<std::uint32_t> minterms;
    for (unsigned m = 0; m < 8; ++m)
      if (truth & (1U << m))
        minterms.push_back(m);
    auto t = a::quine_mccluskey_trace(3, minterms);
    for (unsigned m = 0; m < 8; ++m) {
      bool covered = false;
      for (auto p : t.result.minimal_cover)
        covered |= p.covers(m);
      EXPECT_EQ(covered, bool(truth & (1U << m)));
    }
    const auto& primes = t.result.prime_implicants;
    ASSERT_LT(primes.size(), 20);
    std::size_t best = 99, literals = 99;
    for (std::size_t subset = 0; subset < (std::size_t{1} << primes.size()); ++subset) {
      bool covers = true;
      for (auto m : minterms) {
        bool present = false;
        for (std::size_t i = 0; i < primes.size(); ++i)
          if (subset & (std::size_t{1} << i))
            present |= primes[i].covers(m);
        covers &= present;
      }
      if (covers) {
        const auto  terms = std::popcount(subset);
        std::size_t lits  = 0;
        for (std::size_t i = 0; i < primes.size(); ++i)
          if (subset & (std::size_t{1} << i))
            lits += 3 - std::popcount(primes[i].mask);
        if (terms < best || (terms == best && lits < literals)) {
          best     = terms;
          literals = lits;
        }
      }
    }
    std::size_t got = 0;
    for (auto p : t.result.minimal_cover)
      got += 3 - std::popcount(p.mask);
    EXPECT_EQ(t.result.minimal_cover.size(), best);
    EXPECT_EQ(got, literals);
  }
}
TEST(Batch304Boolean, DomainsDontcaresAndHistoryPolicies) {
  const auto t = a::quine_mccluskey_trace(4, {4, 8, 10, 11, 12, 15}, {9, 14});
  EXPECT_EQ(t.result.minimal_cover.size(), 3);
  EXPECT_EQ(a::quine_mccluskey(0, {0}).minimal_cover.size(), 1);
  EXPECT_TRUE(a::quine_mccluskey(0, {}).minimal_cover.empty());
  EXPECT_THROW(a::quine_mccluskey(-1, {}), std::invalid_argument);
  EXPECT_THROW(a::quine_mccluskey(2, {4}), std::invalid_argument);
  EXPECT_THROW(a::implicant_to_string({1, 0}, 0), std::invalid_argument);
  EXPECT_EQ(a::implicant_to_string({0xffffffffU, 0}, 32), std::string(32, '1'));
  a::QuineMcCluskeyOptions o;
  o.max_pair_tests = 1;
  EXPECT_THROW(a::quine_mccluskey_run(2, {0, 1, 2}, {}, o), std::length_error);
  o                   = {};
  o.max_trace_entries = 1;
  EXPECT_THROW(a::quine_mccluskey_trace(1, {0, 1}, {}, o), std::length_error);
}
TEST(Batch304Rabin, ExactBinaryMatchingAndCollisionRejection) {
  a::RabinKarpOptions o;
  o.modulus = 2;
  o.base    = 1;
  auto t    = a::rabin_karp_trace("ca", "a", o);
  ASSERT_EQ(t.windows.size(), 2);
  EXPECT_TRUE(t.windows[0].candidate);
  EXPECT_FALSE(t.windows[0].matched);
  EXPECT_EQ(t.matches, (std::vector<std::size_t>{1}));
  std::mt19937_64 rng(42);
  for (int trial = 0; trial < 300; ++trial) {
    std::string text(rng() % 70, '\0'), pattern(rng() % 12, '\0');
    for (auto& c : text)
      c = char(rng() % 256);
    for (auto& c : pattern)
      c = char(rng() % 256);
    if (trial % 3 == 0 && !text.empty())
      pattern = text.substr(0, std::min<std::size_t>(3, text.size()));
    std::vector<std::size_t> expected;
    for (std::size_t i = 0; i <= text.size(); ++i)
      if (text.compare(i, pattern.size(), pattern) == 0)
        expected.push_back(i);
    auto result = a::rabin_karp_trace(text, pattern);
    EXPECT_EQ(result.matches, expected);
    EXPECT_EQ(
        result.operations, result.visited_windows + result.hash_updates + result.compared_bytes);
  }
}
TEST(Batch304Rabin, NativeRollingIdentityAndPolicies) {
  auto t = a::rabin_karp_trace("abababbb", "aba");
  for (const auto& w : t.windows) {
    std::uint64_t hash = 0;
    for (std::size_t j = 0; j < 3; ++j)
      hash = (hash * 256 + static_cast<unsigned char>(std::string("abababbb")[w.start + j])) %
             1000000007;
    EXPECT_EQ(hash, w.hash);
  }
  a::RabinKarpOptions o;
  o.base = 0;
  EXPECT_THROW(a::rabin_karp_run("a", "a", o), std::invalid_argument);
  o                = {};
  o.max_operations = 1;
  EXPECT_THROW(a::rabin_karp_run("a", "a", o), std::length_error);
  o                   = {};
  o.max_trace_entries = 1;
  EXPECT_THROW(a::rabin_karp_trace("abc", "", o), std::length_error);
}
TEST(Batch304Radix, OriginalPassesStableIdentitiesAndFullSignedRange) {
  const std::vector<std::int64_t> keys{
      std::numeric_limits<std::int64_t>::max(),
      -1,
      std::numeric_limits<std::int64_t>::min(),
      0,
      -1,
      7};
  auto t = a::radix_sort_trace(keys);
  ASSERT_EQ(t.passes.size(), 8);
  std::vector<std::size_t> expected(keys.size());
  std::iota(expected.begin(), expected.end(), 0);
  for (unsigned pass = 0; pass < 8; ++pass) {
    std::stable_sort(expected.begin(), expected.end(), [&](auto i, auto j) {
      return ((static_cast<std::uint64_t>(keys[i]) ^ (1ULL << 63)) >> (pass * 8) & 255) <
             ((static_cast<std::uint64_t>(keys[j]) ^ (1ULL << 63)) >> (pass * 8) & 255);
    });
    EXPECT_EQ(t.passes[pass].origins, expected);
    EXPECT_EQ(t.passes[pass].offsets[256], keys.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
      EXPECT_EQ(t.passes[pass].values[i], keys[expected[i]]);
  }
  EXPECT_EQ(t.digit_reads, 16 * keys.size());
  EXPECT_EQ(t.scatters, 8 * keys.size());
  EXPECT_EQ(t.prefix_steps, 2048);
}
TEST(Batch304Radix, OriginalThousandSignedExtremeTrialsAndCaps) {
  std::mt19937_64 rng(20240927);
  for (int trial = 0; trial < 1000; ++trial) {
    std::vector<std::int64_t> v(1 + rng() % 200);
    for (auto& x : v) {
      const auto word = rng();
      switch (word % 6) {
        case 0:
          x = std::numeric_limits<std::int64_t>::min() + static_cast<std::int64_t>(rng() % 5);
          break;
        case 1:
          x = std::numeric_limits<std::int64_t>::max() - static_cast<std::int64_t>(rng() % 5);
          break;
        case 2:
          x = -static_cast<std::int64_t>(rng() % 100000);
          break;
        default:
          x = std::bit_cast<std::int64_t>(rng());
      }
    }
    auto expected = v;
    std::sort(expected.begin(), expected.end());
    a::radix_sort(v);
    EXPECT_EQ(v, expected);
  }
  EXPECT_TRUE(a::radix_sort_trace({7}).passes.empty());
  a::RadixSortOptions o;
  o.max_trace_entries = 8 * (2 * 2 + 514);
  EXPECT_EQ(a::radix_sort_trace({2, 1}, o).passes.size(), 8);
  --o.max_trace_entries;
  EXPECT_THROW(a::radix_sort_trace({2, 1}, o), std::length_error);
}
TEST(Batch304RDP, FiniteSegmentCorrectionDegeneracyAndActualDecisions) {
  EXPECT_DOUBLE_EQ(a::rdp_segment_distance({2, 0}, {0, 0}, {1, 0}), 1);
  EXPECT_DOUBLE_EQ(a::rdp_segment_distance({3, 4}, {0, 0}, {0, 0}), 5);
  EXPECT_EQ(a::ramer_douglas_peucker({{0, 0}, {2, 0}, {1, 0}}, .5).size(), 3);
  const auto t = a::ramer_douglas_peucker_trace({{0, 0}, {1, 1}, {2, 0}}, .5);
  ASSERT_EQ(t.steps.size(), 1);
  EXPECT_EQ(t.steps[0].farthest, 1);
  EXPECT_DOUBLE_EQ(t.steps[0].distance, 1);
  EXPECT_EQ(t.indices, (std::vector<std::size_t>{0, 1, 2}));
}
TEST(Batch304RDP, WaveformAssignedErrorAndPolicies) {
  const auto points = a::rdp_waveform();
  ASSERT_EQ(points.size(), 201);
  for (double epsilon : {0., .001, .01, .05, .2, .5}) {
    auto   t       = a::ramer_douglas_peucker_trace(points, epsilon);
    double maximum = 0;
    for (std::size_t j = 1; j < t.indices.size(); ++j)
      for (auto i = t.indices[j - 1] + 1; i < t.indices[j]; ++i)
        maximum = std::max(
            maximum,
            a::rdp_segment_distance(points[i], points[t.indices[j - 1]], points[t.indices[j]]));
    EXPECT_NEAR(maximum, t.max_segment_error, 1e-14);
    EXPECT_LE(maximum, epsilon + 1e-14);
    for (const auto& step : t.steps)
      EXPECT_EQ(step.split, step.distance > epsilon);
  }
  EXPECT_THROW(a::ramer_douglas_peucker({}, -1), std::invalid_argument);
  EXPECT_THROW(
      a::ramer_douglas_peucker({{0, std::numeric_limits<double>::quiet_NaN()}}, 1),
      std::invalid_argument);
  a::RDPOptions o;
  o.max_distance_tests = 1;
  EXPECT_THROW(a::ramer_douglas_peucker_run(points, .2, o), std::length_error);
}
TEST(Batch304Sampling, OriginalInterleavedStreamAndActualNormalCache) {
  auto          t = r::sample_distributions_trace(0xC0FFEEULL, 10000);
  r::SplitMix64 rng(0xC0FFEEULL);
  for (const auto& row : t.samples) {
    EXPECT_EQ(row.integer, rng.uniform_u64(3, 11));
    EXPECT_EQ(row.bernoulli, rng.bernoulli(.3));
    EXPECT_DOUBLE_EQ(row.normal, rng.normal(10, 2));
    EXPECT_DOUBLE_EQ(row.exponential, rng.exponential(4));
  }
  EXPECT_DOUBLE_EQ(t.final_uniform, rng.uniform(-2, 5));
  EXPECT_EQ(t.raw_draws, t.raw_words.size());
  EXPECT_EQ(t.samples[0].raw_offsets[3] - t.samples[0].raw_offsets[2], 2);
  EXPECT_EQ(t.samples[1].raw_offsets[3] - t.samples[1].raw_offsets[2], 0);
  EXPECT_EQ(t.minimum, 3);
  EXPECT_EQ(t.maximum, 11);
  EXPECT_NEAR(t.bernoulli_mean, .3007, 1e-12);
  EXPECT_NEAR(t.normal_mean, 9.9848, .00005);
  EXPECT_NEAR(t.exponential_mean, .2489, .00005);
}
TEST(Batch304Sampling, FiniteHalfOpenExtremesAndCheckedTransformOverflow) {
  r::SplitMix64 rng(42);
  const double  big = std::numeric_limits<double>::max();
  for (int i = 0; i < 100; ++i) {
    const double x = rng.uniform(-big, big);
    EXPECT_TRUE(std::isfinite(x));
    EXPECT_GE(x, -big);
    EXPECT_LT(x, big);
    EXPECT_DOUBLE_EQ(rng.uniform(1, std::nextafter(1., 2.)), 1);
  }
  class Words : public r::RandomGenerator {
   public:
    std::uint64_t next_u64() override { return std::numeric_limits<std::uint64_t>::max(); }
  } source;
  EXPECT_THROW(source.exponential(std::numeric_limits<double>::denorm_min()), std::overflow_error);
  EXPECT_THROW(
      r::sample_distributions_run(42, 0, r::SamplingParameters{.probability = 2}),
      std::invalid_argument);
  r::SamplingOptions o;
  o.max_raw_draws = 1;
  EXPECT_THROW(r::sample_distributions_run(42, 1, {}, o), std::length_error);
  o                   = {};
  o.max_trace_entries = 1;
  EXPECT_THROW(r::sample_distributions_trace(42, 1, {}, o), std::length_error);
  EXPECT_EQ(r::sample_distributions_run(42, 0).raw_draws, 1);
}
