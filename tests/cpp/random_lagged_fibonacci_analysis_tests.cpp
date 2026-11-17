#include <gtest/gtest.h>

#include <datamunge/random/lagged_fibonacci_analysis.hpp>
#include <limits>
#include <map>
using namespace datamunge::random;
namespace {
std::uint64_t seed_word(std::uint64_t& s) {
  s += 0x9e3779b97f4a7c15ULL;
  auto z = s;
  z      = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z      = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
} // namespace
TEST(LFGAnalysis, ExactOriginalSeedingOperandsWrapAndRawStream) {
  for (auto [j, k] :
       std::vector<std::pair<std::size_t, std::size_t>>{{1, 2}, {2, 5}, {3, 7}, {24, 55}})
    for (std::uint64_t seed : {0ULL, 1ULL, 2026ULL, 0xffffffffffffffffULL}) {
      LaggedFibonacci64          g(seed, j, k), raw(seed, j, k);
      std::vector<std::uint64_t> model(k);
      auto                       s = seed;
      for (auto& x : model)
        x = seed_word(s);
      EXPECT_EQ(g.state_words(), model);
      const auto t = lagged_fibonacci_trace(g, 1000);
      EXPECT_EQ(t.initial_state, model);
      std::size_t at = 0;
      for (const auto& e : t.steps) {
        const auto other = at >= j ? at - j : k - (j - at);
        const auto want =
            static_cast<std::uint64_t>(static_cast<__uint128_t>(model[at]) + model[other]);
        EXPECT_EQ(e.write_index, at);
        EXPECT_EQ(e.short_index, other);
        EXPECT_EQ(e.long_word, model[at]);
        EXPECT_EQ(e.short_word, model[other]);
        EXPECT_EQ(e.value, want);
        EXPECT_EQ(raw.next_u64(), want);
        model[at] = want;
        at        = at + 1 == k ? 0 : at + 1;
        EXPECT_EQ(e.next_index, at);
      }
      EXPECT_EQ(t.final_state, model);
      EXPECT_EQ(g.state_words(), model);
      EXPECT_EQ(t.final_index, at);
      EXPECT_EQ(g.ring_index(), at);
      EXPECT_EQ(t.captured_entries, 2 * k + 1000);
    }
}
TEST(LFGAnalysis, CompleteAdmissionBeforeConsumptionAndOwnership) {
  LaggedFibonacci64 g(2026, 2, 5), copy = g;
  EXPECT_THROW((void)lagged_fibonacci_trace(g, 10, 19), std::length_error);
  EXPECT_THROW(
      (void)lagged_fibonacci_trace(g, std::numeric_limits<std::size_t>::max(), 0),
      std::length_error);
  EXPECT_EQ(g.state_words(), copy.state_words());
  EXPECT_EQ(g.ring_index(), copy.ring_index());
  const auto t       = lagged_fibonacci_trace(g, 10, 20);
  auto       c       = t;
  c.initial_state[0] = 99;
  c.final_state[0]   = 99;
  c.steps[0].value   = 99;
  EXPECT_NE(t.initial_state[0], 99U);
  EXPECT_NE(t.steps[0].value, 99U);
  EXPECT_NO_THROW((void)lagged_fibonacci_trace(g, 0, 10));
  EXPECT_NO_THROW((void)lagged_fibonacci_trace(g, 10, 0));
}
TEST(LFGAnalysis, ParityOrbitAgainstIndependentShiftRegisterAndAllSmallStates) {
  std::map<std::uint64_t, std::size_t> periods;
  for (std::uint64_t seed = 0; seed < 4096 && periods.size() < 32; ++seed) {
    LaggedFibonacci64 g(seed, 2, 5);
    const auto        before = g.state_words();
    const auto        o      = lagged_fibonacci_parity_orbit(g, 32);
    EXPECT_EQ(g.state_words(), before);
    EXPECT_EQ(g.ring_index(), 0U);
    ASSERT_TRUE(o.complete);
    std::uint64_t mask = 0;
    for (std::size_t i = 0; i < 5; ++i)
      mask |= (before[i] & 1ULL) << i;
    ASSERT_EQ(o.masks.front(), mask);
    std::map<std::uint64_t, std::size_t> seen;
    std::vector<std::uint64_t>           states;
    while (!seen.contains(mask)) {
      seen[mask] = states.size();
      states.push_back(mask);
      const auto bit = ((mask >> 0) ^ (mask >> 3)) & 1ULL;
      mask           = (mask >> 1) | (bit << 4);
    }
    EXPECT_EQ(o.masks, states);
    EXPECT_EQ(o.repeat_offset, seen[mask]);
    EXPECT_EQ(o.period, states.size() - seen[mask]);
    periods[states.front()] = o.period;
  }
  ASSERT_EQ(periods.size(), 32U);
  for (auto [mask, period] : periods)
    EXPECT_EQ(period, mask == 0 ? 1U : 31U);
}
TEST(LFGAnalysis, OrbitLimitAndMaskGeometry) {
  LaggedFibonacci64 g(1, 2, 5);
  const auto        o = lagged_fibonacci_parity_orbit(g, 1);
  EXPECT_EQ(o.masks.size(), 1U);
  if (!o.complete)
    EXPECT_EQ(o.period, 0U);
  EXPECT_TRUE(lagged_fibonacci_parity_orbit(g, 0).complete);
  EXPECT_THROW(
      (void)lagged_fibonacci_parity_orbit(LaggedFibonacci64(1, 1, 64)), std::invalid_argument);
}
TEST(LFGAnalysis, DirectRawInspectionLeavesInheritedNormalCacheIntact) {
  LaggedFibonacci64 a(2026, 2, 5), b = a;
  EXPECT_EQ(a.normal(), b.normal());
  (void)lagged_fibonacci_trace(a, 10);
  for (int i = 0; i < 10; ++i)
    b.next_u64();
  EXPECT_EQ(a.normal(), b.normal());
  EXPECT_EQ(a.next_u64(), b.next_u64());
}
