#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/integer_arithmetic.hpp>
#include <datamunge/algorithms/number_theory_extra.hpp>
#include <limits>
#include <random>
#include <stdexcept>
namespace {
using namespace datamunge::algorithms;
using U = unsigned __int128;
void same(const FermatOutcome& a, const FermatOutcome& b) {
  EXPECT_EQ(a.n, b.n);
  EXPECT_EQ(a.returned_factor, b.returned_factor);
  EXPECT_EQ(a.cofactor, b.cofactor);
  EXPECT_EQ(a.start_a, b.start_a);
  EXPECT_EQ(a.last_a, b.last_a);
  EXPECT_EQ(a.last_b, b.last_b);
  EXPECT_EQ(a.proper_split, b.proper_split);
  EXPECT_EQ(a.stop, b.stop);
  EXPECT_EQ(a.trials, b.trials);
  EXPECT_EQ(a.increments, b.increments);
}
void root_check(U x) {
  const auto r = integer_sqrt_u128(x);
  EXPECT_LE(U(r) * r, x);
  if (r != std::numeric_limits<std::uint64_t>::max())
    EXPECT_GT(U(r + 1) * (r + 1), x);
}
// Enumerate divisors, not square-test candidates: the largest small divisor
// gives the closest pair, independently of the production Fermat loop.
std::uint64_t closest(std::uint64_t n) {
  std::uint64_t p = 1;
  for (std::uint64_t d = 3; d <= n / d; d += 2)
    if (n % d == 0)
      p = d;
  return p;
}
void check(std::uint64_t n, std::size_t budget) {
  const auto t = fermat_factor_trace(n, budget, 0);
  same(t.outcome, fermat_factor_work(n, budget));
  const auto& o = t.outcome;
  EXPECT_EQ(t.rows.size(), o.trials);
  EXPECT_EQ(t.captured_entries, o.trials);
  if (n % 2 == 0) {
    EXPECT_TRUE(t.rows.empty());
    EXPECT_EQ(o.returned_factor, 2);
    EXPECT_EQ(o.cofactor, n / 2);
    EXPECT_EQ(o.proper_split, n > 2);
    EXPECT_EQ(o.stop, FermatStop::Even);
    EXPECT_EQ(o.start_a, 0);
    EXPECT_EQ(o.last_a, 0);
    EXPECT_EQ(o.last_b, 0);
    EXPECT_EQ(o.increments, 0);
    return;
  }
  const auto p = closest(n), q = n / p;
  const auto first = (p + q) / 2;
  const auto start = integer_sqrt_u64(n - 1) + 1;
  const auto need  = first - start + 1;
  EXPECT_EQ(o.start_a, start);
  EXPECT_EQ(o.trials, std::min<std::uint64_t>(need, budget));
  EXPECT_EQ(o.increments, o.trials - 1);
  EXPECT_EQ(o.last_a, start + o.increments);
  EXPECT_EQ(o.last_b, t.rows.back().b);
  for (std::size_t i = 0; i < t.rows.size(); ++i) {
    const auto& r = t.rows[i];
    EXPECT_EQ(r.a, start + i);
    EXPECT_EQ(r.difference, U(r.a) * r.a - n);
    root_check(r.difference);
    EXPECT_EQ(U(r.b) * r.b <= r.difference, true);
    EXPECT_EQ(r.square, U(r.b) * r.b == r.difference);
    if (i + 1 < t.rows.size())
      EXPECT_FALSE(r.square);
  }
  if (need <= budget) {
    EXPECT_TRUE(t.rows.back().square);
    EXPECT_EQ(o.returned_factor, p == 1 ? n : p);
    EXPECT_EQ(o.cofactor, p == 1 ? 1 : q);
    EXPECT_EQ(o.proper_split, p > 1);
    EXPECT_EQ(o.stop, p == 1 ? FermatStop::TrivialPair : FermatStop::Factor);
    EXPECT_EQ(U(o.returned_factor) * o.cofactor, n);
  } else {
    EXPECT_FALSE(t.rows.back().square);
    EXPECT_FALSE(o.proper_split);
    EXPECT_EQ(o.returned_factor, n);
    EXPECT_EQ(o.cofactor, 0);
    EXPECT_EQ(o.stop, FermatStop::TrialLimit);
  }
}
TEST(FermatTrace, RootFullWidth) {
  for (unsigned i = 0; i < 128; ++i) {
    const U x = U(1) << i;
    root_check(x);
    root_check(x - 1);
    root_check(x + 1);
  }
  EXPECT_EQ(integer_sqrt_u128(0), 0);
  EXPECT_EQ(integer_sqrt_u128(~U(0)), UINT64_MAX);
  for (std::uint64_t r :
       std::vector<std::uint64_t>{1, 2, 3, 4294967295ULL, 4294967296ULL, UINT64_MAX}) {
    const U s = U(r) * r;
    EXPECT_EQ(integer_sqrt_u128(s), r);
    EXPECT_EQ(integer_sqrt_u128(s - 1), r - 1);
    if (r > 1)
      EXPECT_EQ(integer_sqrt_u128(s + 1), r);
  }
}
TEST(FermatTrace, RootSeededBrackets) {
  std::mt19937_64 rng(106);
  for (int i = 0; i < 10000; ++i) {
    const U x = (U(rng()) << 64) | rng();
    root_check(x);
    const auto n = rng();
    EXPECT_EQ(integer_sqrt_u128(n), integer_sqrt_u64(n));
  }
}
TEST(FermatTrace, TinyActualRows) {
  const auto t = fermat_factor_trace(5959, 3, 3);
  ASSERT_EQ(t.rows.size(), 3);
  const std::uint64_t aa[] = {78, 79, 80}, bb[] = {11, 16, 21}, dd[] = {125, 282, 441};
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(t.rows[i].a, aa[i]);
    EXPECT_EQ(t.rows[i].b, bb[i]);
    EXPECT_EQ(t.rows[i].difference, dd[i]);
    EXPECT_EQ(t.rows[i].square, i == 2);
  }
  EXPECT_EQ(t.outcome.returned_factor, 59);
  EXPECT_EQ(t.outcome.cofactor, 101);
  EXPECT_EQ(t.outcome.trials, 3);
  EXPECT_EQ(t.outcome.increments, 2);
  EXPECT_EQ(fermat_factor(5959), 59);
}
TEST(FermatTrace, IndependentClosestDivisors) {
  for (std::uint64_t n = 2; n < 8192; ++n) {
    check(n, 1);
    check(n, 17);
  }
  for (std::uint64_t n = 3; n < 2048; n += 2)
    check(n, 2048);
}
TEST(FermatTrace, SeededClippedBudgets) {
  std::mt19937_64 rng(2024);
  for (int i = 0; i < 600; ++i) {
    const std::uint64_t p = 3 + 2 * (rng() % 50000), q = 3 + 2 * (rng() % 50000);
    check(p * q, 1 + rng() % 64);
  }
}
TEST(FermatTrace, FullWidthBoundaries) {
  const auto t = fermat_factor_trace(UINT64_MAX, 1, 1);
  EXPECT_EQ(t.outcome.start_a, 4294967296ULL);
  EXPECT_EQ(t.rows[0].difference, 1);
  EXPECT_EQ(t.outcome.returned_factor, 4294967295ULL);
  EXPECT_EQ(t.outcome.cofactor, 4294967297ULL);
  for (std::uint64_t p : std::vector<std::uint64_t>{3, 65537, 4294967291ULL, 4294967295ULL}) {
    const auto s = fermat_factor_trace(p * p, 1);
    EXPECT_EQ(s.outcome.returned_factor, p);
    EXPECT_EQ(s.outcome.cofactor, p);
    EXPECT_EQ(s.rows[0].difference, 0);
    EXPECT_EQ(s.rows[0].b, 0);
  }
  const auto wide = fermat_factor_trace(UINT64_MAX - 58, 3);
  EXPECT_EQ(wide.outcome.stop, FermatStop::TrialLimit);
  EXPECT_EQ(wide.rows.front().a, 4294967296ULL);
  EXPECT_EQ(wide.rows.front().difference, 59);
  EXPECT_GT(wide.rows.back().difference, UINT64_MAX >> 32);
  const auto near = fermat_factor_trace(18446744065119617027ULL, 1);
  EXPECT_EQ(near.outcome.start_a, 4294967296ULL);
  EXPECT_EQ(near.rows[0].difference, 8589934589ULL);
}
TEST(FermatTrace, StopsAndHardComposite) {
  check(2, 1);
  check(10, 1);
  check(13, 3);
  check(13, 4);
  check(945, 1);
  const auto f = fermat_factor_work(945);
  EXPECT_EQ(f.returned_factor, 27);
  EXPECT_EQ(f.cofactor, 35);
  const auto hard = fermat_factor_work(3000000021ULL);
  EXPECT_EQ(hard.stop, FermatStop::TrialLimit);
  EXPECT_EQ(hard.trials, 5000000);
  EXPECT_EQ(hard.increments, 4999999);
  EXPECT_EQ(fermat_factor(3000000021ULL), 3000000021ULL);
  EXPECT_EQ(3000000021ULL % 3, 0);
}
TEST(FermatTrace, QuotaAndOwnership) {
  EXPECT_THROW(fermat_factor_trace(5959, 3, 2), std::length_error);
  EXPECT_NO_THROW(fermat_factor_trace(5959, 2, 2));
  EXPECT_NO_THROW(fermat_factor_trace(5959, 3, 3));
  EXPECT_NO_THROW(fermat_factor_trace(5959, 3, 0));
  auto       a = fermat_factor_trace(5959);
  const auto b = a;
  a.rows[0].a  = 0;
  a.outcome.n  = 0;
  EXPECT_EQ(b.rows[0].a, 78);
  EXPECT_EQ(b.outcome.n, 5959);
  EXPECT_EQ(b.outcome.trials, 3);
  EXPECT_NO_THROW(fermat_factor_work(5959, 3));
  EXPECT_TRUE(fermat_factor_trace(10, 1, 1).rows.empty());
}
TEST(FermatTrace, DomainAndNames) {
  for (auto n : {0ULL, 1ULL}) {
    EXPECT_THROW(fermat_factor(n), std::invalid_argument);
    EXPECT_THROW(fermat_factor_work(n), std::invalid_argument);
    EXPECT_THROW(fermat_factor_trace(n), std::invalid_argument);
  }
  EXPECT_THROW(fermat_factor_work(10, 0), std::invalid_argument);
  EXPECT_THROW(fermat_factor_trace(10, 0, 0), std::invalid_argument);
  EXPECT_EQ(fermat_stop_name(FermatStop::Even), "even");
  EXPECT_EQ(fermat_stop_name(FermatStop::Factor), "factor");
  EXPECT_EQ(fermat_stop_name(FermatStop::TrivialPair), "trivial pair");
  EXPECT_EQ(fermat_stop_name(FermatStop::TrialLimit), "trial limit");
  EXPECT_EQ(fermat_stop_name(static_cast<FermatStop>(99)), "unknown");
  std::uint64_t (*legacy)(std::uint64_t) = &fermat_factor;
  EXPECT_EQ(legacy(49), 7);
}
TEST(FermatTrace, OriginalGapsAndExactBudget) {
  const std::uint64_t qs[]  = {10009, 10037, 10091, 10259, 10501, 11003, 12007};
  const std::size_t   inc[] = {0, 0, 0, 0, 2, 11, 45};
  for (int i = 0; i < 7; ++i) {
    auto t = fermat_factor_trace(10007 * qs[i], 100);
    EXPECT_EQ(t.outcome.returned_factor, 10007);
    EXPECT_EQ(t.outcome.cofactor, qs[i]);
    EXPECT_EQ(t.outcome.increments, inc[i]);
    EXPECT_EQ(t.outcome.trials, inc[i] + 1);
    check(10007 * qs[i], inc[i] + 1);
  }
  for (auto cap : {1U, 2U, 3U, 4U})
    check(5959, cap);
  EXPECT_EQ(fermat_factor(100000007ULL * 100000037ULL), 100000007);
  const auto unlimited = fermat_factor_work(13, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(unlimited.stop, FermatStop::TrivialPair);
  EXPECT_EQ(unlimited.trials, 4);
}
} // namespace
