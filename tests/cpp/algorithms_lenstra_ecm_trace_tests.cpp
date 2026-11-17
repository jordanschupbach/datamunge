#include <gtest/gtest.h>

#include <datamunge/algorithms/number_theory_advanced.hpp>
#include <limits>
#include <numeric>
using namespace datamunge::algorithms;
TEST(ECMTrace, CheckedActualFailedInverse) {
  auto e = ecm_add(91, 12, 1, {0, 1, false}, {7, 8, false});
  EXPECT_TRUE(e.evaluated);
  EXPECT_FALSE(e.inverted);
  EXPECT_EQ(e.denominator, 7);
  EXPECT_EQ(e.gcd, 7);
  EXPECT_EQ(e.factor, 7);
  EXPECT_TRUE(e.result.infinity);
  EXPECT_THROW(ecm_add(91, 12, 1, {0, 2, false}, {7, 8, false}), std::invalid_argument);
}
TEST(ECMTrace, OriginalInputsAndBoundedFailure) {
  for (std::uint64_t n : {1517ULL, 10403ULL, 1000003ULL * 101, 104729ULL * 1009, 999983ULL * 3}) {
    auto a = lenstra_ecm_analyze(n);
    EXPECT_TRUE(a.proper);
    EXPECT_GT(a.factor, 1);
    EXPECT_LT(a.factor, n);
    EXPECT_EQ(n % a.factor, 0);
    EXPECT_EQ(a.factor, lenstra_ecm(n));
  }
  auto f = lenstra_ecm_analyze(10403, 1, {20, 0});
  EXPECT_FALSE(f.proper);
  EXPECT_EQ(f.factor, 10403);
  EXPECT_EQ(f.work.curves, 0);
  for (std::uint64_t n : {0, 1, 2, 3, 47})
    EXPECT_EQ(lenstra_ecm(n), n);
  auto p = lenstra_ecm_analyze(101, 1, {20, 3});
  EXPECT_FALSE(p.proper);
  EXPECT_EQ(p.factor, 101);
}
TEST(ECMTrace, ActualCurvesDenominatorsAndQuota) {
  auto t = lenstra_ecm_trace(10403, 1, {20, 3});
  auto a = lenstra_ecm_analyze(10403, 1, {20, 3});
  EXPECT_EQ(t.analysis.factor, a.factor);
  EXPECT_EQ(t.analysis.work.additions, a.work.additions);
  EXPECT_EQ(t.curves.size(), t.analysis.work.curves);
  EXPECT_EQ(t.denominators.size(), t.analysis.work.denominator_gcds);
  EXPECT_EQ(t.captured_entries, t.prime_powers.size() + t.curves.size() + t.denominators.size());
  for (auto& e : t.denominators) {
    EXPECT_EQ(e.gcd, std::gcd(e.denominator, t.analysis.n));
    if (e.inverted)
      EXPECT_EQ(e.gcd, 1);
    if (e.factor)
      EXPECT_EQ(t.analysis.n % e.factor, 0);
  }
  ASSERT_GT(t.captured_entries, 1);
  EXPECT_THROW(lenstra_ecm_trace(10403, 1, {20, 3}, t.captured_entries - 1), std::length_error);
  EXPECT_EQ(lenstra_ecm_trace(10403, 1, {20, 3}, t.captured_entries).analysis.factor, a.factor);
  EXPECT_EQ(lenstra_ecm_trace(10403, 1, {20, 3}, 0).analysis.factor, a.factor);
}
TEST(ECMTrace, Unsigned64ResiduesAndInverse) {
  const std::uint64_t n = 18446744073709551557ULL;
  auto                e = ecm_add(n, 12, 1, {0, 1, false}, {0, 1, false});
  EXPECT_TRUE(e.inverted);
  EXPECT_EQ(e.slope, 6);
  EXPECT_EQ(e.result.x, 36);
  EXPECT_EQ(e.result.y, n - 217);
  EXPECT_THROW(ecm_add(1, 0, 0, {}, {}), std::invalid_argument);
  EXPECT_THROW(lenstra_ecm_analyze(91, 1, {1, 1}), std::invalid_argument);
  EXPECT_THROW(lenstra_ecm_analyze(91, 1, {20, 1000001}), std::invalid_argument);
}
TEST(ECMTrace, SmoothProxyCompleteInterval) {
  auto h = ecm_hasse_smoothness(101, 50);
  EXPECT_EQ(h.lower, 82);
  EXPECT_EQ(h.upper, 122);
  EXPECT_EQ(h.total, 41);
  EXPECT_GT(h.smooth, 0);
  EXPECT_LT(h.smooth, h.total);
  EXPECT_THROW(ecm_hasse_smoothness(100, 50), std::invalid_argument);
  EXPECT_THROW(ecm_hasse_smoothness(101, 1), std::invalid_argument);
}
