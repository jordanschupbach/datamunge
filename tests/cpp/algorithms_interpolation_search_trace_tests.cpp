#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/interpolation_search.hpp>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
void check(const std::vector<int>& a, int key) {
  auto       t     = interpolation_search_trace(a, key);
  auto       r     = interpolation_search(a, key);
  auto       run   = interpolation_search_run(a, key);
  const bool found = std::binary_search(a.begin(), a.end(), key);
  EXPECT_EQ(t.run.result.found, found);
  EXPECT_EQ(r.found, found);
  EXPECT_EQ(r.index, t.run.result.index);
  EXPECT_EQ(run.result.probes, r.probes);
  EXPECT_EQ(t.probes.size(), r.probes);
  EXPECT_LE(r.probes, a.size());
  if (found) {
    ASSERT_GE(r.index, 0);
    ASSERT_LT(std::size_t(r.index), a.size());
    EXPECT_EQ(a[r.index], key);
  } else
    EXPECT_EQ(r.index, -1);
  std::ptrdiff_t lo = 0, hi = std::ptrdiff_t(a.size()) - 1;
  std::size_t    guards = 0, flat = 0, comps = 0, index = 0;
  while (lo <= hi) {
    ++guards;
    if (key < a[lo])
      break;
    ++guards;
    if (key > a[hi])
      break;
    ++flat;
    ASSERT_LT(index, t.probes.size());
    const auto& p = t.probes[index++];
    EXPECT_EQ(p.lo, lo);
    EXPECT_EQ(p.hi, hi);
    EXPECT_EQ(p.low_value, a[lo]);
    EXPECT_EQ(p.high_value, a[hi]);
    if (a[lo] == a[hi]) {
      EXPECT_TRUE(p.flat);
      EXPECT_EQ(p.position, lo);
      EXPECT_EQ(p.denominator, 0);
      break;
    }
    auto num = (__int128)key - a[lo], den = (__int128)a[hi] - a[lo];
    auto offset = std::ptrdiff_t(num * (hi - lo) / den);
    EXPECT_FALSE(p.flat);
    EXPECT_EQ(p.numerator, num);
    EXPECT_EQ(p.denominator, den);
    EXPECT_EQ(p.offset, offset);
    EXPECT_EQ(p.position, lo + offset);
    EXPECT_EQ(p.value, a[p.position]);
    ++comps;
    if (p.value == key)
      break;
    ++comps;
    if (p.value < key)
      lo = p.position + 1;
    else
      hi = p.position - 1;
  }
  EXPECT_EQ(t.run.work.endpoint_guards, guards);
  EXPECT_EQ(t.run.work.flat_tests, flat);
  EXPECT_EQ(t.run.work.key_comparisons, comps);
}
} // namespace
TEST(InterpolationTrace, ExhaustiveSortedMultisetsAndTargets) {
  for (unsigned n = 0; n <= 9; ++n) {
    std::size_t total = 1;
    for (unsigned i = 0; i < n; ++i)
      total *= 4;
    for (std::size_t code = 0; code < total; ++code) {
      auto             x = code;
      std::vector<int> a(n);
      for (auto& v : a) {
        v = int(x % 4) - 2;
        x /= 4;
      }
      if (!std::is_sorted(a.begin(), a.end()))
        continue;
      for (int key = -3; key <= 2; ++key)
        check(a, key);
    }
  }
}
TEST(InterpolationTrace, FullSignedDomainAndRandomOracles) {
  const auto min = std::numeric_limits<int>::min(), max = std::numeric_limits<int>::max();
  for (auto a : std::vector<std::vector<int>>{
           {min, max}, {min, min, 0, max}, {min, -1, 0, 1, max}, {max, max}, {min, min}})
    for (int key : {min, min + 1, -1, 0, 1, max - 1, max})
      check(a, key);
  std::mt19937 rng(1985);
  for (unsigned i = 0; i < 2000; ++i) {
    std::vector<int> a(rng() % 64);
    for (auto& v : a)
      v = int(rng() % 1001) - 500;
    std::sort(a.begin(), a.end());
    check(a, int(rng() % 1201) - 600);
  }
}
TEST(InterpolationTrace, OriginalAndFlatOutsideControls) {
  const std::vector<int> a{1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
  for (int key : {13, 1, 19, 8, 20})
    check(a, key);
  EXPECT_EQ(interpolation_search(a, 13).probes, 1);
  EXPECT_EQ(interpolation_search(a, 8).probes, 1);
  EXPECT_EQ(interpolation_search(a, 20).probes, 0);
  EXPECT_EQ(interpolation_search({1, 2, 3, 4, 5, 6, 7, 8, 9, 1000}, 9).probes, 9);
  auto flat = interpolation_search_trace({2, 2, 2}, 2);
  ASSERT_EQ(flat.probes.size(), 1);
  EXPECT_TRUE(flat.probes[0].flat);
  EXPECT_EQ(flat.run.work.key_comparisons, 0);
  EXPECT_EQ(interpolation_search({2, 2}, 1).probes, 0);
}
TEST(InterpolationTrace, CompleteCapsAndOwnedRecords) {
  const std::vector<int> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 1000};
  EXPECT_NO_THROW(interpolation_search_trace(a, 9, {10, 9}));
  EXPECT_THROW(interpolation_search_trace(a, 9, {10, 8}), std::length_error);
  EXPECT_THROW(interpolation_search_trace(a, 20, {9, 0}), std::length_error);
  EXPECT_NO_THROW(interpolation_search_run(a, 9, {10, 1}));
  EXPECT_NO_THROW(interpolation_search_trace(a, 9, {0, 0}));
  auto source = a;
  auto t      = interpolation_search_trace(source, 9);
  source[0]   = 99;
  EXPECT_EQ(t.probes[0].low_value, 1);
  t.probes[0].value = 77;
  EXPECT_EQ(t.probes[1].value, 2);
}

#include <datamunge/algorithms/interpolation_search_analysis.hpp>
TEST(InterpolationTrace, NativeFixturesAndActualGraphics) {
  auto a = interpolation_example_values();
  auto t = interpolation_search_trace(a, 13);
  EXPECT_FALSE(interpolation_probe_diagram(t).to_svg().empty());
  EXPECT_FALSE(interpolation_secant_plot(a, 13, t).to_svg().empty());
  EXPECT_THROW(interpolation_probe_diagram(t, 1), std::out_of_range);
  EXPECT_THROW(interpolation_secant_plot(a, 13, t, 0, 9), std::length_error);
  auto bad               = t;
  bad.probes[0].position = 99;
  EXPECT_THROW(interpolation_probe_diagram(bad), std::invalid_argument);
  for (std::size_t n = 2; n < 200; ++n) {
    auto a = interpolation_adversary(n);
    EXPECT_EQ(interpolation_search(a, 1).probes, n - 1);
  }
  EXPECT_THROW(interpolation_adversary(1), std::invalid_argument);
  EXPECT_THROW(interpolation_adversary(10, 9), std::length_error);
}
