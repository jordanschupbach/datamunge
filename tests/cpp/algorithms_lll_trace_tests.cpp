#include <gtest/gtest.h>

#include <datamunge/algorithms/lll.hpp>
#include <limits>
using namespace datamunge::algorithms;
namespace {
using Basis = std::vector<std::vector<long long>>;
void relation(const Basis& original, const Basis& basis, const Basis& u) {
  ASSERT_EQ(basis.size(), original.size());
  for (std::size_t i = 0; i < basis.size(); ++i)
    for (std::size_t j = 0; j < basis[i].size(); ++j) {
      __int128 sum = 0;
      for (std::size_t k = 0; k < original.size(); ++k)
        sum += static_cast<__int128>(u[i][k]) * original[k][j];
      EXPECT_EQ(sum, static_cast<__int128>(basis[i][j]));
    }
}
__int128 det3(const Basis& a) {
  return static_cast<__int128>(a[0][0]) *
             (static_cast<__int128>(a[1][1]) * a[2][2] - static_cast<__int128>(a[1][2]) * a[2][1]) -
         static_cast<__int128>(a[0][1]) *
             (static_cast<__int128>(a[1][0]) * a[2][2] - static_cast<__int128>(a[1][2]) * a[2][0]) +
         static_cast<__int128>(a[0][2]) *
             (static_cast<__int128>(a[1][0]) * a[2][1] - static_cast<__int128>(a[1][1]) * a[2][0]);
}
} // namespace
TEST(LLLTrace, ActualTinyReductionAndOriginalExactTransforms) {
  auto tiny = lll_trace({{1, 0}, {3, 1}});
  EXPECT_EQ(tiny.events[0].kind, LLLEventKind::SizeReduce);
  EXPECT_EQ(tiny.events[0].multiplier, 3);
  EXPECT_EQ(tiny.analysis.basis, (Basis{{1, 0}, {0, 1}}));
  EXPECT_EQ(tiny.analysis.transform, (Basis{{1, 0}, {-3, 1}}));
  Basis original{{201, 37, 5}, {143, 25, 3}, {88, 16, 2}};
  auto  t = lll_trace(original);
  EXPECT_EQ(t.analysis.stop, LLLStop::Completed);
  // The original fixture finishes at an exact half projection. Floating GS
  // observes 0.50000000000000011, so completion is not a strict computed certificate.
  EXPECT_FALSE(t.analysis.observed_size_reduced);
  EXPECT_GT(t.analysis.mu[2][1], .5);
  EXPECT_NEAR(t.analysis.mu[2][1], .5, 1e-14);
  EXPECT_TRUE(t.analysis.observed_lovasz);
  EXPECT_EQ(t.analysis.basis, lll_reduce(original));
  relation(original, t.analysis.basis, t.analysis.transform);
  for (auto& s : t.states)
    relation(original, s.basis, s.transform);
  const auto det = det3(t.analysis.transform);
  EXPECT_TRUE(det == 1 || det == -1);
  EXPECT_EQ(det3(t.analysis.basis), det * det3(original));
  std::size_t reductions = 0, swaps = 0, checks = 0;
  for (auto& e : t.events) {
    ASSERT_LT(e.state, t.states.size());
    if (e.kind == LLLEventKind::SizeReduce)
      ++reductions;
    else if (e.kind == LLLEventKind::Swap)
      ++swaps;
    else {
      ++checks;
      EXPECT_EQ(e.passed, e.left >= e.right);
    }
  }
  EXPECT_EQ(reductions, t.analysis.work.size_reductions);
  EXPECT_EQ(swaps, t.analysis.work.swaps);
  EXPECT_EQ(checks, t.analysis.work.lovasz_tests);
  EXPECT_EQ(t.states.size(), 1 + reductions + swaps);
}
TEST(LLLTrace, ExactUnimodularWitnessesOnIntegerFamilies) {
  for (long long a = -6; a <= 6; ++a)
    for (long long b = -6; b <= 6; ++b) {
      Basis original{{1, a, b}, {0, 2, a}, {0, 0, 3}};
      auto  t = lll_trace(original);
      ASSERT_EQ(t.analysis.stop, LLLStop::Completed);
      relation(original, t.analysis.basis, t.analysis.transform);
      EXPECT_TRUE(t.analysis.observed_size_reduced);
      EXPECT_TRUE(t.analysis.observed_lovasz);
      const auto determinant = det3(t.analysis.transform);
      EXPECT_TRUE(determinant == 1 || determinant == -1);
    }
}
TEST(LLLTrace, ExplicitBudgetAndFloatingRankBreakdown) {
  Basis original{{201, 37, 5}, {143, 25, 3}, {88, 16, 2}};
  auto  a = lll_analyze(original, .75, 0);
  EXPECT_EQ(a.stop, LLLStop::StepLimit);
  EXPECT_EQ(a.basis, original);
  relation(original, a.basis, a.transform);
  const auto max = std::numeric_limits<long long>::max();
  Basis      full_rank{{max, 1}, {max - 1, 1}};
  EXPECT_THROW(lll_analyze(full_rank), std::domain_error);
  EXPECT_THROW(lll_analyze({{1, 2}, {2, 4}}), std::domain_error);
  EXPECT_THROW(lll_analyze({{1, 0}, {max, 1}}), std::overflow_error);
  EXPECT_THROW(lll_analyze({{max, 2}, {max, -max}}), std::overflow_error);
}
TEST(LLLTrace, CompleteGuardsQuotaEmptyAndOwnership) {
  EXPECT_THROW(lll_analyze({{1, 2}, {3}}), std::invalid_argument);
  EXPECT_THROW(lll_analyze({{}}), std::invalid_argument);
  EXPECT_THROW(lll_analyze({{1}, {2}}), std::invalid_argument);
  for (double d : {.25, 1., std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(lll_analyze({{1, 0}, {0, 1}}, d), std::invalid_argument);
  EXPECT_EQ(lll_analyze({}).stop, LLLStop::Completed);
  EXPECT_TRUE(lll_trace({}).states.empty());
  auto t = lll_trace({{1, 0}, {3, 1}});
  EXPECT_THROW(lll_trace({{1, 0}, {3, 1}}, .75, 100000, t.captured_entries - 1), std::length_error);
  EXPECT_EQ(
      lll_trace({{1, 0}, {3, 1}}, .75, 100000, t.captured_entries).captured_entries,
      t.captured_entries);
  Basis original{{1, 0}, {3, 1}};
  auto  own      = lll_trace(original, .75, 100000, 0);
  original[1][0] = 9;
  EXPECT_EQ(own.original[1][0], 3);
}
