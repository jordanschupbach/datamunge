#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/itp.hpp>
#include <datamunge/algorithms/itp_analysis.hpp>
#include <datamunge/algorithms/root_finding.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
void check(const ItpTrace& t) {
  int trials = 0;
  for (std::size_t i = 0; i < t.steps.size(); ++i) {
    const auto& s = t.steps[i];
    EXPECT_LT(s.lower, s.upper);
    EXPECT_NE(std::signbit(s.f_lower), std::signbit(s.f_upper));
    EXPECT_GE(s.x, s.lower);
    EXPECT_LE(s.x, s.upper);
    EXPECT_TRUE(std::isfinite(s.f_x));
    EXPECT_EQ(s.radius, std::max((long double)s.x - s.lower, (long double)s.upper - s.x));
    if (s.projected) {
      ++trials;
      EXPECT_LT(s.lower, s.x);
      EXPECT_LT(s.x, s.upper);
      const auto a = (long double)s.lower, b = (long double)s.upper;
      const auto xf = (b * s.f_lower - a * s.f_upper) / ((long double)s.f_lower - s.f_upper);
      EXPECT_NEAR((double)s.interpolated, (double)xf, 3e-15 * (1 + std::abs((double)xf)));
      EXPECT_EQ(s.midpoint, a + (b - a) / 2);
      EXPECT_GE(s.delta, 0);
      EXPECT_GE(s.projection_radius, 0);
      EXPECT_LE(s.projection_radius, (b - a) / 2);
      EXPECT_LE(std::abs(s.truncated - s.midpoint), std::abs(s.interpolated - s.midpoint) + 1e-18L);
    } else
      EXPECT_EQ(i, t.steps.size() - 1);
    EXPECT_EQ(s.iteration, trials);
    if (i + 1 < t.steps.size()) {
      const auto& next = t.steps[i + 1];
      if (std::signbit(s.f_x) == std::signbit(s.f_lower)) {
        EXPECT_EQ(next.lower, s.x);
        EXPECT_EQ(next.upper, s.upper);
        EXPECT_EQ(next.f_lower, s.f_x);
      } else {
        EXPECT_EQ(next.lower, s.lower);
        EXPECT_EQ(next.upper, s.x);
        EXPECT_EQ(next.f_upper, s.f_x);
      }
    }
  }
  EXPECT_EQ(t.run.result.iterations, trials);
  EXPECT_LE(trials, t.run.projection_bound);
  if (!t.steps.empty()) {
    EXPECT_EQ(t.run.result.root, t.steps.back().x);
    EXPECT_EQ(t.run.result.residual, std::abs(t.steps.back().f_x));
    EXPECT_EQ(t.run.evaluations, t.steps.size() + 2);
  }
}
} // namespace
TEST(ItpTrace, ActualStagesCallbackCountsAndInterfaceParity) {
  std::vector<double> calls;
  const auto          f = [&](double x) {
    calls.push_back(x);
    return std::cos(x) - x;
  };
  const auto t = itp_trace(f, 0, 1, {1e-12, 0, 100, 1});
  check(t);
  EXPECT_TRUE(t.run.result.converged);
  EXPECT_NEAR(t.run.result.root, .7390851332151607, 1e-12);
  ASSERT_EQ(calls.size(), t.steps.size() + 2);
  EXPECT_EQ(calls[0], 0);
  EXPECT_EQ(calls[1], 1);
  for (std::size_t i = 0; i < t.steps.size(); ++i)
    EXPECT_EQ(calls[i + 2], t.steps[i].x);
  const auto r = itp_run([](double x) { return std::cos(x) - x; }, 0, 1, {1e-12, 0, 100, 1});
  EXPECT_EQ(r.result.root, t.run.result.root);
  EXPECT_EQ(r.evaluations, t.run.evaluations);
  const auto legacy = itp([](double x) { return std::cos(x) - x; }, 0, 1, 1e-12);
  const auto combined =
      itp_run([](double x) { return std::cos(x) - x; }, 0, 1, {1e-12, 1e-12, 100, 1});
  EXPECT_EQ(legacy.root, combined.result.root);
  EXPECT_EQ(legacy.iterations, combined.result.iterations);
}
TEST(ItpTrace, GeneratedSignBracketsRespectActualProjectionAllowances) {
  std::mt19937 rng(19);
  for (int n0 : {0, 1})
    for (int k = 3; k <= 40; ++k)
      for (int trial = 0; trial < 32; ++trial) {
        const double root = (double(rng()) + .5) / 4294967296.;
        const auto   t    = itp_trace(
            [root](double x) { return x < root ? -1. : 1.; },
            0,
            1,
            {std::ldexp(1., -k), 0, 100, n0});
        check(t);
        EXPECT_EQ(t.run.bisection_bound, k - 1);
        EXPECT_EQ(t.run.projection_bound, k - 1 + n0);
        EXPECT_LE(t.run.result.iterations, k - 1 + n0);
        EXPECT_LE(t.run.lower, root);
        EXPECT_GE(t.run.upper, root);
        EXPECT_TRUE(t.run.stop == ItpStop::Interval || t.run.stop == ItpStop::ProjectionLimit);
        if (t.run.stop == ItpStop::Interval)
          EXPECT_LE(t.run.radius, std::ldexp(1., -k));
        else
          EXPECT_FALSE(t.run.result.converged);
      }
}
TEST(ItpTrace, SeparateResidualAndCoordinateUnits) {
  auto       f        = [](double x) { return 1e-20 * (x - .731); };
  const auto combined = itp_run(f, 0, 2, {1e-10, 1e-10});
  EXPECT_EQ(combined.stop, ItpStop::Residual);
  EXPECT_GT(combined.radius, .1);
  const auto positional = itp_trace(f, 0, 2, {1e-10, 0});
  check(positional);
  EXPECT_TRUE(positional.run.result.converged);
  EXPECT_NEAR(positional.run.result.root, .731, 1e-10);
  const auto jump = itp_run([](double x) { return x < .731 ? -1. : 1.; }, 0, 2, {1e-10, 0});
  EXPECT_TRUE(jump.result.converged);
  EXPECT_EQ(jump.stop, ItpStop::Interval);
  EXPECT_EQ(jump.result.residual, 1);
}
TEST(ItpTrace, EndpointsNoBracketInitialIntervalAndZeroBudget) {
  int  calls    = 0;
  auto endpoint = [&](double x) {
    ++calls;
    return x;
  };
  const auto first = itp_trace(endpoint, 0, 2);
  EXPECT_EQ(first.run.stop, ItpStop::Endpoint);
  EXPECT_EQ(calls, 1);
  EXPECT_TRUE(first.steps.empty());
  calls             = 0;
  const auto second = itp_trace(endpoint, -2, 0);
  EXPECT_EQ(second.run.stop, ItpStop::Endpoint);
  EXPECT_EQ(calls, 2);
  const auto failed = itp_trace([](double x) { return x * x + 1; }, -1, 1);
  EXPECT_EQ(failed.run.stop, ItpStop::NoBracket);
  EXPECT_EQ(failed.run.evaluations, 2);
  EXPECT_FALSE(failed.run.result.converged);
  const auto tight = itp_trace([](double x) { return x - .1; }, 0, .2, {1, 0, 0, 0});
  check(tight);
  EXPECT_TRUE(tight.run.result.converged);
  EXPECT_EQ(tight.run.result.iterations, 0);
  const auto cap = itp_trace([](double x) { return x * x - 2; }, 1, 2, {1e-12, 0, 0});
  check(cap);
  EXPECT_EQ(cap.run.stop, ItpStop::IterationLimit);
  EXPECT_EQ(cap.run.evaluations, 3);
  EXPECT_EQ(cap.run.result.iterations, 0);
  const auto ordered = itp_trace([](double x) { return x * x - 2; }, 2, 1);
  check(ordered);
  EXPECT_TRUE(ordered.run.result.converged);
}
TEST(ItpTrace, RecordAdmissionBeforeCallbacksAndOwnership) {
  const auto f  = [](double x) { return std::cos(x) - x; };
  const auto t  = itp_trace(f, 0, 1);
  auto       o  = ItpOptions{};
  o.max_records = t.steps.size();
  EXPECT_EQ(itp_trace(f, 0, 1, o).steps.size(), t.steps.size());
  int calls     = 0;
  o.max_records = 1;
  EXPECT_THROW(
      itp_trace(
          [&](double x) {
            ++calls;
            return f(x);
          },
          0,
          1,
          o),
      std::length_error);
  EXPECT_EQ(calls, 3);
  EXPECT_NO_THROW(itp_run(f, 0, 1, o));
  o.max_records = 0;
  EXPECT_NO_THROW(itp_trace(f, 0, 1, o));
  auto owned       = t;
  owned.steps[0].x = 9;
  EXPECT_NE(t.steps[0].x, owned.steps[0].x);
}
TEST(ItpTrace, InputValidationAndCallbackErrors) {
  int  calls = 0;
  auto f     = [&](double x) {
    ++calls;
    return x;
  };
  for (auto o :
       {ItpOptions{0},
        ItpOptions{-1},
        ItpOptions{1, -1},
        ItpOptions{1, 0, -1},
        ItpOptions{1, 0, 1, -1}})
    EXPECT_THROW(itp_run(f, -1, 1, o), std::invalid_argument);
  const auto inf = std::numeric_limits<double>::infinity(),
             nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(itp_run(f, nan, 1), std::invalid_argument);
  EXPECT_THROW(itp_run(f, -1, inf), std::invalid_argument);
  EXPECT_THROW(itp_run({}, -1, 1), std::invalid_argument);
  EXPECT_THROW(
      itp_run(f, -1, 1, {1e-10, 0, 1, std::numeric_limits<int>::max()}), std::length_error);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(itp_run([&](double) { return nan; }, 0, 1), std::domain_error);
  EXPECT_THROW(
      itp_run([&](double x) { return x == 0   ? -1.
                                     : x == 1 ? 1.
                                              : inf; }, 0, 1), std::domain_error);
  EXPECT_THROW(
      itp_run([](double) -> double { throw std::runtime_error("caller"); }, 0, 1),
      std::runtime_error);
  EXPECT_EQ(itp_stop_name(static_cast<ItpStop>(999)), "unknown");
}
TEST(ItpTrace, FullScaleInputsAndFinitePrecisionStagnation) {
  const auto m    = std::numeric_limits<double>::max();
  const auto wide = itp_trace([](double x) { return x; }, -m, m, {1e-12, 0, 100});
  check(wide);
  EXPECT_EQ(wide.run.result.root, 0);
  const double a = 1, b = std::nextafter(a, 2.);
  const auto   stalled = itp_trace(
      [a](double x) { return x == a ? -1. : 1.; },
      a,
      b,
      {std::numeric_limits<double>::denorm_min(), 0});
  check(stalled);
  EXPECT_EQ(stalled.run.stop, ItpStop::Stagnation);
  EXPECT_FALSE(stalled.run.result.converged);
  EXPECT_EQ(stalled.run.result.iterations, 0);
  const auto tiny =
      itp_trace([](double x) { return x < 1e-300 ? -1. : 1.; }, 0, 2e-300, {1e-310, 0, 100});
  check(tiny);
  EXPECT_LE(tiny.run.result.iterations, tiny.run.projection_bound);
  const auto large_slack =
      itp_run([](double x) { return x - .731; }, 0, 2, {1e-12, 0, 20, 1000000});
  EXPECT_TRUE(std::isfinite(large_slack.result.root));
}
TEST(ItpTrace, NativeGraphicsAndDisplayAdmission) {
  const auto f = [](double x) { return x * x * x - 2; };
  const auto t = itp_trace(f, 0, 2, {.125, 0, 100, 0});
  ASSERT_FALSE(t.steps.empty());
  const auto& first = t.steps[0];
  EXPECT_NEAR((double)first.interpolated, .5, 1e-15);
  EXPECT_NEAR((double)first.truncated, .9, 1e-15);
  EXPECT_EQ(first.x, 1);
  EXPECT_EQ(first.projection_radius, 0);
  EXPECT_FALSE(itp_stage_diagram(t).to_svg().empty());
  EXPECT_THROW(itp_stage_diagram(t, t.steps.size()), std::out_of_range);
  const auto bi = bisection_trace(f, 0, 2, {.125, 0, 100});
  EXPECT_FALSE(itp_bisection_plot(t, bi).to_svg().empty());
  const auto points = t.steps.size() + bi.steps.size();
  EXPECT_NO_THROW(itp_bisection_plot(t, bi, true, 1e-18, points));
  EXPECT_THROW(itp_bisection_plot(t, bi, true, 1e-18, points - 1), std::length_error);
  EXPECT_THROW(itp_bisection_plot(t, bi, true, 0), std::invalid_argument);
  auto bad               = t;
  bad.steps[0].projected = false;
  EXPECT_THROW(itp_stage_diagram(bad), std::invalid_argument);
  bad                 = t;
  bad.run.evaluations = 0;
  EXPECT_THROW(itp_bisection_plot(bad, bi), std::invalid_argument);
}
TEST(ItpTrace, RoundedProjectionLimitIsAnExplicitFailure) {
  const double ulp      = std::nextafter(1., 2.) - 1.;
  const double location = 1 + 3 * ulp;
  const auto   t        = itp_trace(
      [location](double x) { return x < location ? -1. : 1.; },
      1,
      1 + 5 * ulp,
      {.75 * ulp, 0, 100, 0});
  check(t);
  EXPECT_EQ(t.run.projection_bound, 2);
  EXPECT_EQ(t.run.result.iterations, 2);
  EXPECT_EQ(t.run.stop, ItpStop::ProjectionLimit);
  EXPECT_FALSE(t.run.result.converged);
  EXPECT_GT(t.run.radius, .75 * ulp);
  EXPECT_TRUE(std::any_of(t.steps.begin(), t.steps.end(), [](const auto& s) {
    return s.rounding_adjusted;
  }));
  auto bad           = t;
  bad.steps[0].lower = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(itp_stage_diagram(bad), std::invalid_argument);
}
