#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/index_calculus.hpp>
#include <datamunge/algorithms/miller_rabin.hpp>
#include <datamunge/algorithms/pollard_rho.hpp>
#include <limits>
using namespace datamunge::algorithms;
namespace {
std::uint64_t power(std::uint64_t a, std::uint64_t e, std::uint64_t m) {
  std::uint64_t r = 1 % m;
  while (e) {
    if (e & 1)
      r = (unsigned __int128)r * a % m;
    a = (unsigned __int128)a * a % m;
    e >>= 1;
  }
  return r;
}
std::uint64_t root(std::uint64_t p) {
  auto f = factorize(p - 1);
  for (std::uint64_t g = 1; g < p; ++g)
    if (std::all_of(f.begin(), f.end(), [&](auto q) { return power(g, (p - 1) / q, p) != 1; }))
      return g;
  return 0;
}
} // namespace
TEST(IndexCalculusTrace, SmallPrimesEveryTargetAndActualRelations) {
  for (std::uint64_t p = 2; p <= 199; ++p) {
    if (!is_probable_prime(p))
      continue;
    auto                 g = root(p);
    IndexCalculusOptions options;
    options.relation_surplus = 1000;
    auto h                   = index_calculus_prepare_trace(g, p, 0x1D2C3B4A59687766ULL, options);
    ASSERT_EQ(h.table.status, IndexCalculusStatus::ready) << p;
    ASSERT_EQ(h.table.logarithms.size(), h.table.factor_base.size());
    for (auto q : h.table.factor_base)
      EXPECT_LT(q, p);
    for (const auto& r : h.relations) {
      EXPECT_GE(r.attempt, 1);
      EXPECT_LE(r.attempt, h.table.work.relation_attempts);
      EXPECT_EQ(power(g, r.exponent, p), r.residue);
      unsigned __int128 product = 1;
      for (std::size_t j = 0; j < r.powers.size(); ++j)
        for (std::uint64_t k = 0; k < r.powers[j]; ++k)
          product *= h.table.factor_base[j];
      EXPECT_EQ(product, r.residue);
    }
    for (const auto& stage : h.stages) {
      ASSERT_EQ(stage.rank, h.table.factor_base.size());
      ASSERT_EQ(stage.solution.size(), stage.rank);
      for (const auto& r : h.relations) {
        std::uint64_t sum = 0;
        for (std::size_t j = 0; j < stage.rank; ++j)
          sum = ((unsigned __int128)sum + (unsigned __int128)r.powers[j] * stage.solution[j]) %
                stage.modulus;
        EXPECT_EQ(sum, r.exponent % stage.modulus);
      }
    }
    for (std::uint64_t x = 0; x < p - 1; ++x) {
      auto target = power(g, x, p);
      auto q      = index_calculus_query(h.table, target);
      ASSERT_TRUE(q.result.found) << p << "/" << x;
      EXPECT_EQ(q.result.x, x);
      if (q.attempts) {
        EXPECT_EQ(q.residue, (unsigned __int128)target * power(g, q.shift, p) % p);
        std::uint64_t sum = 0;
        for (std::size_t j = 0; j < q.powers.size(); ++j)
          sum = ((unsigned __int128)sum + (unsigned __int128)q.powers[j] * h.table.logarithms[j]) %
                (p - 1);
        EXPECT_EQ((sum + (p - 1) - q.shift) % (p - 1), x);
      }
    }
  }
  EXPECT_TRUE(index_calculus(2, 4, 5).found);
  EXPECT_EQ(index_calculus(2, 4, 5).x, 2);
  EXPECT_EQ(index_calculus_prepare(2, 11).status, IndexCalculusStatus::rank_deficient);
}
TEST(IndexCalculusTrace, PrimePowerLiftsAndOriginalCases) {
  for (auto p : {17u, 97u, 257u, 65537u}) {
    IndexCalculusOptions options;
    options.relation_surplus = 1000;
    auto h = index_calculus_prepare_trace(root(p), p, 0x1D2C3B4A59687766ULL, options);
    ASSERT_EQ(h.table.status, IndexCalculusStatus::ready);
    EXPECT_FALSE(h.stages.empty());
    auto        f      = factorize(p - 1);
    std::size_t stages = 0;
    for (auto factor : f) {
      (void)factor;
      ++stages;
    }
    EXPECT_EQ(h.stages.size(), stages);
  }
  for (auto c : std::vector<std::vector<std::uint64_t>>{
           {10007, 5, 1234}, {100003, 2, 54321}, {1000003, 2, 424242}, {15485863, 6, 9999999}}) {
    auto r = index_calculus(c[1], power(c[1], c[2], c[0]), c[0]);
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.x, c[2]);
  }
}
TEST(IndexCalculusTrace, ExactPoliciesFailureStatusesAndOwnedHistory) {
  auto              h     = index_calculus_prepare_trace(2, 5);
  const std::size_t k     = h.table.factor_base.size(),
                    units = (h.relations.size() + h.stages.size()) * (k + 1);
  IndexCalculusOptions o;
  o.max_history_entries = units;
  EXPECT_NO_THROW(index_calculus_prepare_trace(2, 5, 0x1D2C3B4A59687766ULL, o));
  o.max_history_entries = units - 1;
  EXPECT_THROW(index_calculus_prepare_trace(2, 5, 0x1D2C3B4A59687766ULL, o), std::length_error);
  EXPECT_NO_THROW(index_calculus_prepare(2, 5, 0x1D2C3B4A59687766ULL, o));
  o                  = {};
  o.max_matrix_cells = 43;
  EXPECT_THROW(index_calculus_prepare(2, 5, 1, o), std::length_error);
  o.max_matrix_cells = 44;
  EXPECT_NO_THROW(index_calculus_prepare(2, 5, 1, o));
  o                   = {};
  o.max_sieve_entries = 4;
  EXPECT_THROW(index_calculus_prepare(2, 5, 1, o), std::length_error);
  o.max_sieve_entries = 5;
  EXPECT_NO_THROW(index_calculus_prepare(2, 5, 1, o));
  o                   = {};
  o.relation_attempts = 0;
  auto no             = index_calculus_prepare(2, 5, 1, o);
  EXPECT_EQ(no.status, IndexCalculusStatus::relation_limit);
  EXPECT_FALSE(index_calculus_query(no, 4).result.found);
  EXPECT_TRUE(index_calculus_query(no, 1, 0).result.found);
  EXPECT_TRUE(index_calculus_query(no, 2, 0).result.found);
  EXPECT_FALSE(index_calculus_query(h.table, 4, 0).result.found);
  EXPECT_TRUE(index_calculus_query(h.table, 2, 0).result.found);
  auto q     = index_calculus_query(h.table, 4);
  auto again = index_calculus_query(h.table, 4);
  EXPECT_EQ(q.shift, again.shift);
  EXPECT_EQ(q.attempts, again.attempts);
  auto saved                    = h.table.logarithms;
  h.relations.front().powers[0] = 99;
  EXPECT_EQ(h.table.logarithms, saved);
}
TEST(IndexCalculusTrace, RejectsDomainsAndCorruptTablesBeforeShortcuts) {
  for (auto p : {0u, 1u, 4u, 9u, 15u})
    EXPECT_THROW(index_calculus(0, 1, p), std::invalid_argument);
  EXPECT_THROW(index_calculus(0, 1, 5), std::invalid_argument);
  EXPECT_THROW(index_calculus(4, 1, 5), std::invalid_argument);
  EXPECT_THROW(index_calculus(2, 0, 5), std::invalid_argument);
  EXPECT_THROW(index_calculus(2, 5, 5), std::invalid_argument);
  auto t = index_calculus_prepare(2, 5), bad = t;
  bad.p = 0;
  EXPECT_THROW(index_calculus_query(bad, 1), std::invalid_argument);
  bad = t;
  bad.order_factors.pop_back();
  EXPECT_THROW(index_calculus_query(bad, 1), std::invalid_argument);
  bad               = t;
  bad.logarithms[0] = 0;
  EXPECT_THROW(index_calculus_query(bad, 1), std::invalid_argument);
  bad                = t;
  bad.factor_base[0] = 1;
  EXPECT_THROW(index_calculus_query(bad, 1), std::invalid_argument);
  bad = t;
  bad.factor_base.push_back(3);
  EXPECT_THROW(index_calculus_query(bad, 1), std::invalid_argument);
  EXPECT_EQ(index_calculus(7, 9, 5).x, 2);
  EXPECT_TRUE(index_calculus(1, 1, 2).found);
}

TEST(IndexCalculusTrace, FullWidthGroupAndForcedSmoothShiftArithmetic) {
  constexpr std::uint64_t p = 18446744073709551557ULL;
  IndexCalculusOptions    options;
  options.factor_base_bound = 2;
  options.relation_attempts = 0;
  auto t                    = index_calculus_prepare(2, p, 1, options);
  EXPECT_EQ(t.status, IndexCalculusStatus::relation_limit);
  EXPECT_EQ(t.order_factors, (std::vector<std::uint64_t>{2, 2, 11, 137, 547, 5594472617641ULL}));
  // Supply a mathematically verified one-prime table; this does not claim that
  // bounded random relation collection solved a cryptographic-sized instance.
  t.status             = IndexCalculusStatus::ready;
  t.logarithms         = {1};
  const auto undo_left = [](std::uint64_t y, unsigned shift) {
    auto x = y;
    for (unsigned i = 0; i < 64; ++i)
      x = y ^ (x << shift);
    return x;
  };
  const auto undo_right = [](std::uint64_t y, unsigned shift) {
    auto x = y;
    for (unsigned i = 0; i < 64; ++i)
      x = y ^ (x >> shift);
    return x;
  };
  t.query_state = undo_left(undo_right(undo_left(p - 3, 17), 7), 13);
  const auto q  = index_calculus_query(t, 4, 1);
  ASSERT_TRUE(q.result.found);
  EXPECT_EQ(q.result.x, 2);
  EXPECT_EQ(q.shift, p - 3);
  EXPECT_EQ(q.residue, 1);
  EXPECT_EQ(q.attempts, 1);
}

#include <datamunge/algorithms/index_calculus_analysis.hpp>
TEST(IndexCalculusTrace, NativeModelsAndPreparationDiagrams) {
  auto h = index_calculus_prepare_trace(2, 5);
  EXPECT_FALSE(index_calculus_preparation_diagram(h).to_svg().empty());
  EXPECT_THROW(index_calculus_preparation_diagram(h, 1), std::length_error);
  EXPECT_THROW(index_calculus_preparation_diagram(h, 2, 1), std::length_error);
  auto bad                 = h;
  bad.relations[0].attempt = 0;
  EXPECT_THROW(index_calculus_preparation_diagram(bad), std::invalid_argument);
  EXPECT_FALSE(
      index_calculus_preparation_diagram(index_calculus_prepare_trace(2, 11)).to_svg().empty());
  const auto models = index_calculus_cost_models({30, 60, 240, 1e300});
  ASSERT_EQ(models.size(), 4);
  EXPECT_NEAR(models[0].index_log10, 4.88, .01);
  EXPECT_NEAR(models[0].generic_log10, 4.52, .01);
  EXPECT_TRUE(std::isfinite(models.back().index_log10));
  EXPECT_THROW(index_calculus_cost_models({1}), std::invalid_argument);
  EXPECT_THROW(
      index_calculus_cost_models({std::numeric_limits<double>::infinity()}), std::invalid_argument);
  EXPECT_THROW(index_calculus_cost_models({}, 0), std::invalid_argument);
  EXPECT_THROW(
      index_calculus_cost_models({1e300}, std::numeric_limits<double>::max()), std::overflow_error);
}
