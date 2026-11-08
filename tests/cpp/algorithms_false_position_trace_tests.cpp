#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <datamunge/algorithms/root_finding.hpp>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
using namespace datamunge::algorithms;
namespace {
void same(const RootResult& a, const RootResult& b) {
  EXPECT_EQ(a.converged, b.converged);
  EXPECT_EQ(a.iterations, b.iterations);
  EXPECT_EQ(a.root, b.root);
  EXPECT_EQ(a.residual, b.residual);
}
// Independently check each actual call/step, true signs, secant weights and transitions.
void check(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    FalsePositionOptions                 options) {
  std::vector<std::pair<double, double>> calls;
  auto                                   t = false_position_trace(
      [&](double x) {
        const double v = f(x);
        calls.emplace_back(x, v);
        return v;
      },
      a,
      b,
      options,
      0);
  same(t.outcome.result, false_position_work(f, a, b, options).result);
  ASSERT_EQ(t.evaluations.size(), calls.size());
  EXPECT_EQ(t.outcome.evaluations, calls.size());
  EXPECT_EQ(t.captured_entries, calls.size() + t.steps.size());
  EXPECT_EQ(t.steps.size(), static_cast<std::size_t>(t.outcome.result.iterations));
  ASSERT_GE(calls.size(), 2u);
  EXPECT_EQ(calls[0].first, a);
  EXPECT_EQ(calls[1].first, b);
  EXPECT_EQ(t.evaluations[0].role, -1);
  EXPECT_EQ(t.evaluations[1].role, 1);
  for (std::size_t i = 0; i < calls.size(); ++i) {
    EXPECT_EQ(t.evaluations[i].coordinate, calls[i].first);
    EXPECT_EQ(t.evaluations[i].value, calls[i].second);
  }
  std::size_t guards = 0, halvings = 0;
  int         previous = 0;
  for (std::size_t i = 0; i < t.steps.size(); ++i) {
    const auto& s = t.steps[i];
    EXPECT_EQ(s.f_a, f(s.a));
    EXPECT_EQ(s.f_b, f(s.b));
    EXPECT_NE(std::signbit(s.f_a), std::signbit(s.f_b));
    EXPECT_EQ(std::signbit(s.weight_a), std::signbit(s.f_a));
    EXPECT_EQ(std::signbit(s.weight_b), std::signbit(s.f_b));
    EXPECT_NE(s.weight_a, 0);
    EXPECT_NE(s.weight_b, 0);
    EXPECT_EQ(s.estimate, calls[i + 2].first);
    EXPECT_EQ(s.f_estimate, calls[i + 2].second);
    EXPECT_GE(s.estimate, std::min(s.a, s.b));
    EXPECT_LE(s.estimate, std::max(s.a, s.b));
    EXPECT_EQ(t.evaluations[i + 2].role, 0);
    guards += s.guarded;
    halvings += s.damped;
    if (s.replaced) {
      EXPECT_EQ(s.replaced, (std::signbit(s.f_estimate) == std::signbit(s.f_b) ? 1 : -1));
      EXPECT_EQ(
          s.damped, options.policy == FalsePositionPolicy::Illinois && previous == s.replaced);
      previous = s.replaced;
    } else
      EXPECT_FALSE(s.damped);
    if (i + 1 < t.steps.size()) {
      const auto& next = t.steps[i + 1];
      ASSERT_NE(s.replaced, 0);
      EXPECT_EQ(next.a, s.replaced < 0 ? s.estimate : s.a);
      EXPECT_EQ(next.b, s.replaced > 0 ? s.estimate : s.b);
      EXPECT_EQ(next.f_a, s.replaced < 0 ? s.f_estimate : s.f_a);
      EXPECT_EQ(next.f_b, s.replaced > 0 ? s.f_estimate : s.f_b);
      EXPECT_EQ(next.weight_a, s.replaced < 0 ? s.f_estimate : s.weight_a * (s.damped ? .5 : 1));
      EXPECT_EQ(next.weight_b, s.replaced > 0 ? s.f_estimate : s.weight_b * (s.damped ? .5 : 1));
    }
  }
  EXPECT_EQ(t.outcome.guarded_interpolations, guards);
  EXPECT_EQ(t.outcome.weight_halvings, halvings);
  if (!t.steps.empty()) {
    const auto& s = t.steps.back();
    EXPECT_EQ(t.outcome.result.root, s.estimate);
    EXPECT_EQ(t.outcome.result.residual, std::abs(s.f_estimate));
    EXPECT_EQ(t.outcome.a, s.a);
    EXPECT_EQ(t.outcome.b, s.b);
    EXPECT_EQ(t.outcome.f_a, s.f_a);
    EXPECT_EQ(t.outcome.f_b, s.f_b);
    EXPECT_EQ(t.outcome.weight_a, s.weight_a);
    EXPECT_EQ(t.outcome.weight_b, s.weight_b);
  }
}
} // namespace
TEST(FalsePositionTrace, OriginalExponentialAndMatchedTolerances) {
  auto f   = [](double x) { return std::exp(x) - 2; };
  auto old = false_position(f, 0, 1);
  EXPECT_EQ(old.iterations, 7);
  EXPECT_NEAR(old.root, std::log(2.), 1e-10);
  auto t = false_position_trace(f, 0, 1);
  same(old, t.outcome.result);
  EXPECT_EQ(t.outcome.evaluations, 9u);
  EXPECT_EQ(t.captured_entries, 16u);
  FalsePositionOptions o;
  o.residual_tolerance = 1e-13;
  o.max_iterations     = 40;
  auto ill             = false_position_trace(f, 0, 1, o);
  o.policy             = FalsePositionPolicy::Plain;
  auto plain           = false_position_trace(f, 0, 1, o);
  EXPECT_EQ(ill.outcome.result.iterations, 8);
  EXPECT_EQ(plain.outcome.result.iterations, 16);
  EXPECT_EQ(ill.outcome.result.residual, 0);
  check(f, 0, 1, o);
  o.policy = FalsePositionPolicy::Illinois;
  check(f, 0, 1, o);
}
TEST(FalsePositionTrace, SeededQuadraticsAndScalingAgainstAnalyticRoot) {
  std::mt19937 rng(2024);
  for (int i = 0; i < 2000; ++i) {
    double root  = .5 + double(rng() % 1025) / 1024.,
           scale = std::ldexp(1., static_cast<int>(rng() % 41) - 20);
    for (auto policy : {FalsePositionPolicy::Plain, FalsePositionPolicy::Illinois}) {
      FalsePositionOptions o{policy, scale * 1e-12, 0, 500};
      auto                 f = [=](double x) { return scale * (x * x - root * root); };
      const auto           r = false_position_work(f, 0, 2, o);
      EXPECT_TRUE(r.result.converged);
      EXPECT_NEAR(r.result.root, root, 1e-10);
      if (i < 100)
        check(f, i % 2 ? 2 : 0, i % 2 ? 0 : 2, o);
    }
  }
}
TEST(FalsePositionTrace, RawValuesAndActualDampingRemainDistinct) {
  FalsePositionOptions o;
  o.residual_tolerance = 1e-13;
  auto f               = [](double x) { return std::pow(x, 10) - 1; };
  auto t               = false_position_trace(f, 0, 2, o);
  ASSERT_TRUE(t.outcome.result.converged);
  bool distinct = false;
  for (const auto& s : t.steps)
    distinct |= s.weight_a != s.f_a || s.weight_b != s.f_b;
  EXPECT_TRUE(distinct);
  EXPECT_GT(t.outcome.weight_halvings, 0u);
  EXPECT_FALSE(t.steps.front().damped);
  check(f, 0, 2, o);
  check(f, 2, 0, o);
  o.policy         = FalsePositionPolicy::Plain;
  o.max_iterations = 50;
  auto p           = false_position_trace(f, 0, 2, o);
  EXPECT_FALSE(p.outcome.result.converged);
  EXPECT_EQ(p.outcome.stop, FalsePositionStop::IterationLimit);
  EXPECT_EQ(p.outcome.weight_halvings, 0u);
  for (const auto& s : p.steps) {
    EXPECT_EQ(s.f_a, s.weight_a);
    EXPECT_EQ(s.f_b, s.weight_b);
  }
  check(f, 0, 2, o);
}
TEST(FalsePositionTrace, AffineTrialsAgainstLongDoubleInterpolation) {
  std::mt19937 rng(73);
  for (int i = 0; i < 2000; ++i) {
    double               a = -1. - double(rng() % 1024) / 64., b = 1. + double(rng() % 1024) / 64.;
    double               root  = double(static_cast<int>(rng() % 1001) - 500) / 1024.;
    double               scale = std::ldexp(1., static_cast<int>(rng() % 1001) - 500);
    auto                 f     = [=](double x) { return scale * (x - root); };
    FalsePositionOptions o;
    o.residual_tolerance = 0;
    o.max_iterations     = 1;
    auto t               = false_position_trace(f, a, b, o);
    ASSERT_EQ(t.steps.size(), 1u);
    const auto& s = t.steps[0];
    long double expected =
        ((long double)s.f_b * a - (long double)s.f_a * b) / ((long double)s.f_b - s.f_a);
    EXPECT_NEAR(
        s.estimate,
        static_cast<double>(expected),
        32 * std::numeric_limits<double>::epsilon() * std::max({1., std::abs(a), std::abs(b)}));
    EXPECT_NEAR(s.estimate, root, 1e-12);
  }
}
TEST(FalsePositionTrace, InvalidControlsBeforeCallbacksOrQuota) {
  std::size_t calls = 0;
  auto        f     = [&](double x) {
    ++calls;
    return x - 1;
  };
  const double nan = std::numeric_limits<double>::quiet_NaN(),
               inf = std::numeric_limits<double>::infinity();
  for (double bad : {nan, inf, -inf}) {
    EXPECT_THROW(false_position_trace(f, bad, 2, {}, 1), std::invalid_argument);
    EXPECT_THROW(false_position_trace(f, 0, bad, {}, 1), std::invalid_argument);
  }
  for (double bad : {nan, inf, -inf, -1.}) {
    FalsePositionOptions o;
    o.residual_tolerance = bad;
    EXPECT_THROW(false_position_trace(f, 0, 2, o, 1), std::invalid_argument);
    o             = {};
    o.x_tolerance = bad;
    EXPECT_THROW(false_position_trace(f, 0, 2, o, 1), std::invalid_argument);
  }
  for (int n : {0, -1}) {
    FalsePositionOptions o;
    o.max_iterations = n;
    EXPECT_THROW(false_position_trace(f, 0, 2, o, 1), std::invalid_argument);
  }
  FalsePositionOptions o;
  o.policy = static_cast<FalsePositionPolicy>(99);
  EXPECT_THROW(false_position_trace(f, 0, 2, o, 1), std::invalid_argument);
  EXPECT_THROW(false_position_trace({}, 0, 2, {}, 1), std::invalid_argument);
  for (double bad : {0., -1., nan, inf})
    EXPECT_THROW(false_position(f, 0, 2, bad), std::invalid_argument);
  EXPECT_EQ(calls, 0u);
  EXPECT_EQ(false_position_policy_name(static_cast<FalsePositionPolicy>(99)), "unknown");
  EXPECT_EQ(false_position_stop_name(static_cast<FalsePositionStop>(99)), "unknown");
}
TEST(FalsePositionTrace, EndpointOrderNoBracketAndCallbackFailures) {
  for (int which = 0; which < 3; ++which) {
    std::vector<double> calls;
    auto                t = false_position_trace(
        [&](double x) {
          calls.push_back(x);
          return which == 2 ? 0. : x - (which ? 2. : 0.);
        },
        0,
        2,
        {},
        2);
    EXPECT_EQ(calls, (std::vector<double>{0, 2}));
    EXPECT_EQ(t.outcome.stop, FalsePositionStop::Endpoint);
    EXPECT_EQ(t.outcome.result.iterations, 0);
    EXPECT_EQ(t.outcome.result.root, which == 1 ? 2 : 0);
    EXPECT_EQ(t.captured_entries, 2u);
    EXPECT_TRUE(t.steps.empty());
  }
  auto t = false_position_trace([](double x) { return x * x + 1; }, -1, 1, {}, 2);
  EXPECT_EQ(t.outcome.stop, FalsePositionStop::NoBracket);
  EXPECT_FALSE(t.outcome.result.converged);
  EXPECT_EQ(t.outcome.result.residual, 0);
  EXPECT_EQ(t.outcome.evaluations, 2u);
  for (int fail : {1, 2, 3}) {
    int  calls = 0;
    auto f     = [&](double x) {
      if (++calls == fail)
        return std::numeric_limits<double>::quiet_NaN();
      return x - 1;
    };
    EXPECT_THROW(false_position_trace(f, 0, 2), std::domain_error);
    EXPECT_EQ(calls, fail);
  }
  int  calls  = 0;
  auto throws = [&](double x) {
    if (++calls == 3)
      throw std::runtime_error("callback");
    return x - 1;
  };
  EXPECT_THROW(false_position_trace(throws, 0, 2), std::runtime_error);
  EXPECT_EQ(calls, 3);
}
TEST(FalsePositionTrace, TerminalReasonsAndResidualVersusCoordinateUnits) {
  auto                 f = [](double x) { return x * x - 2; };
  FalsePositionOptions o;
  o.max_iterations     = 1;
  o.residual_tolerance = 0;
  EXPECT_EQ(false_position_work(f, 1, 2, o).stop, FalsePositionStop::IterationLimit);
  o.residual_tolerance = 1.;
  EXPECT_EQ(false_position_work(f, 1, 2, o).stop, FalsePositionStop::Residual);
  o.residual_tolerance = 0;
  o.x_tolerance        = 1.;
  EXPECT_EQ(false_position_work(f, 1, 2, o).stop, FalsePositionStop::Interval);
  const double a = 1., b = std::nextafter(a, 2.);
  auto         jump    = [=](double x) { return x == a ? -1. : 1.; };
  o                    = {};
  o.residual_tolerance = 0;
  auto s               = false_position_work(jump, a, b, o);
  EXPECT_EQ(s.stop, FalsePositionStop::Stagnation);
  EXPECT_FALSE(s.result.converged);
  EXPECT_EQ(s.result.residual, 1);
  o.x_tolerance = 1e-10;
  auto accepted = false_position_work(jump, a, b, o);
  EXPECT_EQ(accepted.stop, FalsePositionStop::Interval);
  EXPECT_TRUE(accepted.result.converged);
  EXPECT_EQ(accepted.result.residual, 1);
  auto tiny = [](double x) { return 1e-30 * (x * x - 2); };
  auto r    = false_position_work(tiny, 1, 2);
  EXPECT_EQ(r.stop, FalsePositionStop::Residual);
  EXPECT_GT(std::abs(r.result.root - std::sqrt(2.)), .01);
}
TEST(FalsePositionTrace, WholeRecordQuotasAndOwnedActualCalls) {
  std::size_t calls = 0;
  auto        f     = [&](double x) {
    ++calls;
    return x - 1;
  };
  EXPECT_THROW(false_position_trace(f, 0, 2, {}, 1), std::length_error);
  EXPECT_EQ(calls, 0u);
  EXPECT_THROW(false_position_trace(f, 0, 2, {}, 3), std::length_error);
  EXPECT_EQ(calls, 2u);
  calls  = 0;
  auto t = false_position_trace(f, 0, 2, {}, 4);
  EXPECT_EQ(calls, 3u);
  EXPECT_EQ(t.captured_entries, 4u);
  same(t.outcome.result, false_position_trace(f, 0, 2, {}, 0).outcome.result);
  auto copy              = t;
  t.evaluations[0].value = 99;
  t.steps[0].a           = 99;
  EXPECT_EQ(copy.evaluations.front().value, -1);
  EXPECT_EQ(copy.steps.front().a, 0);
  EXPECT_EQ(copy.outcome.result.root, 1);
  EXPECT_THROW(
      false_position_trace([](double x) { return x * x - 2; }, 1, 2, {}, 4), std::length_error);
}
TEST(FalsePositionTrace, FullFiniteInterpolationAvoidsOverflowOrInvalid) {
  volatile double maximum = std::numeric_limits<double>::max();
  const double    M       = maximum;
  for (int which = 0; which < 2; ++which) {
    auto f = [=](double x) { return which ? (M * .75) * (x - .25) : x / M - .25; };
    std::feclearexcept(FE_ALL_EXCEPT);
    auto      o   = false_position_work(f, which ? -1 : -M, which ? 1 : M);
    const int bad = std::fetestexcept(FE_OVERFLOW | FE_INVALID);
    EXPECT_EQ(bad, 0);
    EXPECT_TRUE(o.result.converged);
    EXPECT_EQ(o.result.iterations, 1);
    EXPECT_EQ(o.guarded_interpolations, 1u);
    const double expected = which ? .25 : M * .25;
    EXPECT_NEAR(
        o.result.root, expected, 2 * std::numeric_limits<double>::epsilon() * std::abs(expected));
    EXPECT_TRUE(std::isfinite(o.result.residual));
  }
  check([](double x) { return x; }, -M, M, {});
  for (int i = 0; i < 10000; ++i) {
    const double coordinate_scale = std::ldexp(1., -1020 + (i * 73) % 2041);
    const double value_scale      = std::ldexp(1., -1020 + (i * 191) % 2041);
    const double fraction         = i % 2 ? -.25 : .25;
    auto         f = [=](double x) { return (x / coordinate_scale - fraction) * value_scale; };
    FalsePositionOptions options;
    options.residual_tolerance = 0;
    options.max_iterations     = 1;
    std::feclearexcept(FE_ALL_EXCEPT);
    const auto r   = false_position_work(f, -coordinate_scale, coordinate_scale, options);
    const int  bad = std::fetestexcept(FE_OVERFLOW | FE_INVALID);
    EXPECT_EQ(bad, 0);
    EXPECT_NEAR(
        r.result.root / coordinate_scale, fraction, 8 * std::numeric_limits<double>::epsilon());
  }
  const double         tiny = std::ldexp(1., -1020);
  FalsePositionOptions options;
  options.residual_tolerance = 0;
  auto small =
      false_position_work([=](double x) { return (x / tiny - .25) * tiny; }, -tiny, tiny, options);
  EXPECT_TRUE(small.result.converged);
  EXPECT_EQ(small.result.root / tiny, .25);
  EXPECT_GT(small.guarded_interpolations, 0u);
}
TEST(FalsePositionTrace, WeightUnderflowIsExplicitAfterActualHalving) {
  const double M         = std::numeric_limits<double>::max(),
               small     = std::numeric_limits<double>::denorm_min();
  auto                 f = [=](double x) { return x == -M ? -small : small; };
  FalsePositionOptions o;
  o.residual_tolerance = 0;
  o.max_iterations     = 10;
  auto t               = false_position_trace(f, -M, M, o);
  EXPECT_EQ(t.outcome.stop, FalsePositionStop::WeightUnderflow);
  EXPECT_FALSE(t.outcome.result.converged);
  EXPECT_EQ(t.outcome.result.iterations, 2);
  EXPECT_EQ(t.outcome.weight_halvings, 1u);
  ASSERT_EQ(t.steps.size(), 2u);
  EXPECT_TRUE(t.steps.back().damped);
  EXPECT_EQ(t.steps.back().replaced, 1);
  EXPECT_EQ(t.captured_entries, 6u);
  check(f, -M, M, o);
}
