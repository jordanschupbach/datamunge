#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <datamunge/algorithms/freivalds.hpp>
#include <random>
using namespace datamunge::algorithms;
namespace {
FreivaldsMatrix exact_product(const FreivaldsMatrix& a, const FreivaldsMatrix& b) {
  FreivaldsMatrix c(a.size(), std::vector<double>(b[0].size()));
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < b[0].size(); ++j) {
      long long x = 0;
      for (std::size_t k = 0; k < b.size(); ++k)
        x += static_cast<long long>(a[i][k]) * static_cast<long long>(b[k][j]);
      c[i][j] = static_cast<double>(x);
    }
  return c;
}
std::uint64_t word_at(std::uint64_t seed, std::uint64_t index) {
  auto x = seed + index * UINT64_C(0x9e3779b97f4a7c15);
  x      = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x      = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}
std::size_t terms(const FreivaldsCounts& w) {
  return w.br_terms + w.abr_terms + w.cr_terms;
}
} // namespace
TEST(FreivaldsChecked, ExhaustiveExactIntegerResidualOracle) {
  // Independent dense product first; compare D*r rather than projection reassociation.
  for (int ai = 0; ai < 81; ++ai)
    for (int bi = 0; bi < 81; ++bi) {
      FreivaldsMatrix a(2, std::vector<double>(2)), b = a;
      int             ac = ai, bc = bi;
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
          a[i][j] = ac % 3 - 1;
          b[i][j] = bc % 3 - 1;
          ac /= 3;
          bc /= 3;
        }
      const auto product = exact_product(a, b);
      for (int perturb = 0; perturb < 2; ++perturb) {
        auto c = product;
        c[0][0] += perturb;
        std::size_t accepted = 0;
        for (unsigned mask = 0; mask < 4; ++mask) {
          std::vector<std::uint8_t> r{
              static_cast<std::uint8_t>(mask & 1), static_cast<std::uint8_t>((mask >> 1) & 1)};
          bool expected = true;
          for (int i = 0; i < 2; ++i) {
            long long residual = 0;
            for (int j = 0; j < 2; ++j)
              residual += static_cast<long long>(product[i][j] - c[i][j]) * r[j];
            expected &= residual == 0;
          }
          const auto actual = freivalds_probe(a, b, c, r);
          ASSERT_EQ(actual.accepted, expected);
          ASSERT_EQ(actual.work.random_words, 0u);
          ASSERT_EQ(actual.work.provided_bits, 2u);
          ASSERT_EQ(terms(actual.work), 12u);
          accepted += expected;
        }
        const auto audit = freivalds_binary_audit(a, b, c);
        ASSERT_EQ(audit.accepted, accepted);
        ASSERT_EQ(audit.accepted, perturb ? 2u : 4u);
        ASSERT_EQ(audit.work.random_words, 0u);
      }
    }
}
TEST(FreivaldsChecked, RectangularExactCertificates) {
  std::mt19937 rng(122);
  for (int t = 0; t < 300; ++t) {
    const auto      m = 1 + rng() % 5, k = 1 + rng() % 5, n = 1 + rng() % 5;
    FreivaldsMatrix a(m, std::vector<double>(k)), b(k, std::vector<double>(n));
    for (auto& row : a)
      for (auto& x : row)
        x = int(rng() % 9) - 4;
    for (auto& row : b)
      for (auto& x : row)
        x = int(rng() % 9) - 4;
    const auto c    = exact_product(a, b);
    const auto good = freivalds_binary_audit(a, b, c);
    EXPECT_EQ(good.accepted, std::size_t(1) << n);
    auto wrong = c;
    wrong[rng() % m][rng() % n] += 1;
    const auto bad = freivalds_binary_audit(a, b, wrong);
    EXPECT_EQ(bad.accepted, std::size_t(1) << (n - 1));
  }
}
TEST(FreivaldsChecked, RNGFramesAndActualEarlyExitWork) {
  const FreivaldsMatrix a{{1, 2}, {3, 4}}, b{{1, 0}, {0, 1}};
  for (const std::uint64_t seed :
       std::array<std::uint64_t, 4>{0, 42, UINT64_MAX, UINT64_C(0xd1b54a32d192ed03)}) {
    const auto trace = freivalds_trace(a, b, a, 7, seed);
    EXPECT_EQ(trace.work.rounds_completed, 7u);
    EXPECT_EQ(trace.work.random_words, 14u);
    EXPECT_EQ(trace.work.compared_rows, 14u);
    EXPECT_EQ(terms(trace.work), 84u);
    for (std::size_t i = 0; i < 7; ++i) {
      const auto&  f = trace.frames[i];
      const double x = word_at(seed, 2 * i + 1) & 1, y = word_at(seed, 2 * i + 2) & 1;
      EXPECT_EQ(f.rv, (std::vector<double>{x, y}));
      EXPECT_EQ(f.br, f.rv);
      EXPECT_EQ(f.abr, (std::vector<double>{x + 2 * y, 3 * x + 4 * y}));
      EXPECT_EQ(f.cr, f.abr);
      EXPECT_EQ(f.difference, (std::vector<double>{0, 0}));
      EXPECT_EQ(f.state_before, seed + 2 * i * UINT64_C(0x9e3779b97f4a7c15));
      EXPECT_EQ(f.state_after, seed + (2 * i + 2) * UINT64_C(0x9e3779b97f4a7c15));
    }
  }
  const auto p = freivalds_probe(a, b, {{2, 2}, {3, 4}}, {1, 0});
  ASSERT_EQ(p.frames.size(), 1u);
  EXPECT_EQ(p.failed_round, 0u);
  EXPECT_EQ(p.failed_row, 0u);
  EXPECT_EQ(p.frames[0].difference, (std::vector<double>{1}));
  EXPECT_EQ(p.frames[0].abr, (std::vector<double>{1, 3}));
  EXPECT_EQ(p.work.compared_rows, 1u);
  EXPECT_EQ(terms(p.work), 12u);
}
TEST(FreivaldsChecked, ToleranceBoundaryAndCancellationAreNumeric) {
  const double e = 1e-6;
  EXPECT_TRUE(freivalds_probe({{0}}, {{1}}, {{e}}, {1}).accepted);
  EXPECT_FALSE(freivalds_probe({{0}}, {{1}}, {{std::nextafter(e, 1.)}}, {1}).accepted);
  EXPECT_TRUE(freivalds_probe({{0}}, {{1}}, {{std::nextafter(e, 0.)}}, {1}).accepted);
  EXPECT_TRUE(freivalds_verify({{1}}, {{1}}, {{1 + 1e-7}}, 100));
  EXPECT_TRUE(freivalds_probe({{-0.}}, {{1}}, {{0.}}, {1}).accepted);
  EXPECT_TRUE(
      freivalds_probe({{std::numeric_limits<double>::denorm_min()}}, {{1}}, {{0}}, {1}).accepted);
  const FreivaldsMatrix a{{1e16, 1, -1e16}}, b{{1, 1}, {1, 0}, {1, 1}}, c{{1, 0}};
  // Independently form the EXACT represented integer product using a wide accumulator.
  for (int j = 0; j < 2; ++j) {
    __int128 x = 0;
    for (int k = 0; k < 3; ++k)
      x += static_cast<__int128>(a[0][k]) * static_cast<__int128>(b[k][j]);
    EXPECT_EQ(x, static_cast<__int128>(c[0][j]));
  }
  const auto audit = freivalds_binary_audit(a, b, c);
  EXPECT_EQ(audit.accepted, 2u);
  EXPECT_EQ(audit.rejected, 2u);
  EXPECT_FALSE(freivalds_verify(a, b, c));
}
TEST(FreivaldsChecked, ValidateAllInputsBeforeCompatibilityZeroOrQuota) {
  const FreivaldsMatrix identity{{1, 0}, {0, 1}}, ragged{{1, 0}, {0}};
  for (int which = 0; which < 3; ++which) {
    auto a = identity, b = identity, c = identity;
    (which == 0 ? a : which == 1 ? b : c) = ragged;
    EXPECT_THROW(freivalds_trace(a, b, c, 0, 0, 1), std::invalid_argument);
    EXPECT_THROW(freivalds_verify(a, b, c, -3), std::invalid_argument);
  }
  for (const auto bad : std::array<double, 3>{NAN, INFINITY, -INFINITY})
    for (int which = 0; which < 3; ++which) {
      auto a = identity, b = identity, c = identity;
      (which == 0 ? a : which == 1 ? b : c)[1][1] = bad;
      EXPECT_THROW(freivalds_trace(a, b, c, 0, 0, 1), std::invalid_argument);
      a.push_back({1, 0});
      EXPECT_THROW(freivalds_verify(a, b, c), std::invalid_argument);
    }
  EXPECT_FALSE(freivalds_verify({{1}}, {{1}, {2}}, {{1}}, 0));
  EXPECT_FALSE(freivalds_verify({{1}}, {{1}}, {{1, 2}}, 0));
  EXPECT_THROW(freivalds_probe(identity, identity, identity, {1}), std::invalid_argument);
  EXPECT_THROW(freivalds_probe(identity, identity, identity, {0, 2}, 1), std::invalid_argument);
}
TEST(FreivaldsChecked, ActualOverflowAndUnusedValues) {
  const auto huge = std::numeric_limits<double>::max();
  EXPECT_THROW(freivalds_probe({{huge}}, {{2}}, {{0}}, {1}), std::overflow_error);
  EXPECT_TRUE(freivalds_probe({{huge}}, {{2}}, {{0}}, {0}).accepted);
  EXPECT_THROW(
      freivalds_probe({{1, 0}, {0, huge}}, {{1}, {2}}, {{2}, {0}}, {1}),
      std::overflow_error); // later row computed before row0 comparison
  EXPECT_THROW(freivalds_probe({{huge}}, {{1}}, {{-huge}}, {1}), std::overflow_error);
  EXPECT_THROW(freivalds_probe({{1}}, {{huge, huge}}, {{0, 0}}, {1, 1}), std::overflow_error);
}
TEST(FreivaldsChecked, AdmissionAndOwnedActualFrames) {
  FreivaldsMatrix a{{1, 2}, {3, 4}}, b{{1, 0}, {0, 1}}, c{{2, 2}, {3, 4}};
  EXPECT_EQ(freivalds_trace_entries(a, b, c, 1), 29u);
  EXPECT_EQ(freivalds_trace_entries(a, b, c, 2), 40u);
  EXPECT_THROW(freivalds_probe(a, b, c, {1, 0}, 28), std::length_error);
  auto t = freivalds_probe(a, b, c, {1, 0}, 29);
  EXPECT_EQ(t.frames[0].difference.size(), 1u);
  EXPECT_EQ(
      18 + 1 + t.frames[0].rv.size() + t.frames[0].br.size() + t.frames[0].abr.size() +
          t.frames[0].cr.size() + t.frames[0].difference.size(),
      28u);
  a[0][0] = 900;
  c.clear();
  EXPECT_EQ(t.a[0][0], 1);
  EXPECT_EQ(t.c[0][0], 2);
  t.frames[0].rv[0] = 9;
  EXPECT_EQ(t.frames[0].br[0], 1);
  EXPECT_NO_THROW(freivalds_trace(a, b, a, 2, 42, 0));
  EXPECT_THROW(freivalds_binary_audit(b, b, b, 3), std::length_error);
  EXPECT_EQ(freivalds_binary_audit(b, b, b, 4).vectors, 4u);
  EXPECT_THROW(freivalds_binary_vectors(SIZE_MAX), std::length_error);
  EXPECT_THROW(
      freivalds_binary_vectors(std::numeric_limits<std::size_t>::digits), std::length_error);
  EXPECT_EQ(freivalds_binary_vectors(0), 1u);
}
TEST(FreivaldsChecked, EmptyZeroWidthAndVacuousRounds) {
  const auto empty = freivalds_trace({}, {}, {});
  EXPECT_TRUE(empty.accepted);
  EXPECT_EQ(empty.frames.size(), 20u);
  EXPECT_EQ(empty.work.rounds_completed, 20u);
  EXPECT_EQ(empty.work.random_words, 0u);
  EXPECT_EQ(terms(empty.work), 0u);
  EXPECT_EQ(freivalds_trace_entries({}, {}, {}), 20u);
  EXPECT_EQ(freivalds_trace_entries({}, {}, {}, 0), 0u);
  EXPECT_EQ(freivalds_binary_audit({}, {}, {}).accepted, 1u);
  EXPECT_TRUE(freivalds_trace({{}, {}}, {}, {{}, {}}).accepted);
  const auto no_columns = freivalds_trace({{1, 2}}, {{}, {}}, {{}}, 3);
  EXPECT_TRUE(no_columns.accepted);
  EXPECT_EQ(no_columns.work.abr_terms, 6u);
  EXPECT_EQ(no_columns.work.br_terms, 0u);
  EXPECT_EQ(no_columns.work.random_words, 0u);
  for (int r : {INT_MIN, -3, 0}) {
    const auto v = freivalds_trace({{1}}, {{1}}, {{2}}, r, 42);
    EXPECT_TRUE(v.accepted);
    EXPECT_TRUE(v.frames.empty());
    EXPECT_EQ(v.work.final_state, 42u);
    EXPECT_EQ(v.work.kernel_calls, 1u);
  }
}
TEST(FreivaldsChecked, SeedSweepOrderingDuplicatesAndRange) {
  const auto s =
      freivalds_seed_sweep({{1}}, {{1}}, {{2}}, {3, -2, 0, 1, 3, 2}, 40, UINT64_MAX - 39);
  ASSERT_EQ(s.seeds.size(), 40u);
  for (std::size_t j = 0; j < s.rows.size(); ++j) {
    std::size_t accepted = 0;
    for (std::size_t i = 0; i < 40; ++i)
      accepted += freivalds_verify({{1}}, {{1}}, {{2}}, s.rows[j].rounds, UINT64_MAX - 39 + i);
    EXPECT_EQ(s.rows[j].accepted, accepted);
  }
  EXPECT_THROW(freivalds_seed_sweep({{1}}, {{1}}, {{1}}, {1}, 2, UINT64_MAX), std::length_error);
  const auto z = freivalds_seed_sweep({{1}}, {{1}}, {{1}}, {1}, 0);
  EXPECT_EQ(z.rows[0].accepted, 0u);
  EXPECT_TRUE(z.seeds.empty());
  EXPECT_TRUE(freivalds_seed_sweep({{1}}, {{1}}, {{1}}, {}, 20).seeds.empty());
  const auto bad = freivalds_seed_sweep({{1}}, {{1}, {1}}, {{1}}, {0, 2}, 3);
  EXPECT_FALSE(bad.compatible);
  EXPECT_EQ(bad.rows[0].accepted, 0u);
  EXPECT_EQ(bad.work.kernel_calls, 0u);
}
TEST(FreivaldsChecked, OriginalFixtureIndependentProductAndSweepCounts) {
  const auto                         f = freivalds_example_matrices();
  std::mt19937                       rng(3);
  std::uniform_int_distribution<int> uniform(-9, 9);
  for (const auto* m : {&f.a, &f.b})
    for (const auto& row : *m)
      for (const auto x : row)
        ASSERT_EQ(x, uniform(rng));
  EXPECT_EQ(f.product, exact_product(f.a, f.b));
  auto wrong = f.product;
  wrong[0][0] += 1;
  const auto s = freivalds_seed_sweep(f.a, f.b, wrong, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  const std::array<std::size_t, 12> counts{1982, 945, 465, 240, 123, 52, 19, 7, 4, 1, 0, 0};
  for (std::size_t i = 0; i < 12; ++i)
    EXPECT_EQ(s.rows[i].accepted, counts[i]);
  EXPECT_EQ(s.work.kernel_calls, 4000u);
  EXPECT_EQ(s.work.rounds_completed, 7838u);
  EXPECT_EQ(s.work.random_words, 313520u);
  EXPECT_EQ(terms(s.work), 37622400u);
  EXPECT_EQ(s.work.compared_rows, 157520u);
  for (const auto& trial : s.seeds) {
    EXPECT_FALSE(trial.accepted);
    EXPECT_EQ(trial.failed_row, 0u);
    EXPECT_EQ(trial.rounds_completed, trial.failed_round + 1);
  }
  const auto full = freivalds_verify_work(f.a, f.b, f.product);
  EXPECT_TRUE(full.accepted);
  EXPECT_EQ(terms(full.work), 96000u);
}
