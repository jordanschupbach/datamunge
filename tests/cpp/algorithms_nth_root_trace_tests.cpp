#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/nth_root.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NthRootTrace, FullExponentDegreesIndependentLogReferenceAndActualUpdates) {
  for (int exponent = -1073; exponent <= 1024; exponent += 17)
    for (int degree : {1, 2, 3, 4, 7, 10, 32, 63, 64, 65, 1000, std::numeric_limits<int>::max()}) {
      double a = std::ldexp(.75, exponent);
      if (a == 0)
        continue;
      auto        t         = nth_root_trace(a, degree);
      auto&       r         = t.run;
      long double reference = std::exp(std::log(static_cast<long double>(a)) / degree);
      EXPECT_TRUE(r.converged) << exponent << " " << degree;
      EXPECT_LT(std::abs((static_cast<long double>(r.value) - reference) / reference), 2e-14L)
          << exponent << " " << degree;
      EXPECT_EQ(r.updates, t.steps.size());
      EXPECT_EQ(t.captured_entries, 4 + r.updates);
      double      x          = r.initial;
      std::size_t multiplies = 0, wide = 0, logs = 0;
      for (auto& s : t.steps) {
        EXPECT_EQ(s.before, x);
        EXPECT_GT(s.ratio, 0);
        if (!s.log_ratio && !s.widened) {
          EXPECT_TRUE(s.power_available);
          double power = 1;
          for (int k = 0; k < degree - 1; ++k)
            power *= x;
          EXPECT_EQ(s.power, power);
          EXPECT_EQ(s.ratio, a / power);
          EXPECT_EQ(s.after, ((degree - 1) * x + a / power) / degree);
        } else {
          long double expected = (static_cast<long double>(degree - 1) * x + s.ratio) / degree;
          EXPECT_EQ(s.after, double(expected));
        }
        EXPECT_EQ(s.relative_change, std::abs(s.after - x) / std::abs(s.after));
        multiplies += s.power_multiplications;
        wide += s.widened;
        logs += s.log_ratio;
        x = s.after;
      }
      EXPECT_EQ(r.power_multiplications, multiplies);
      EXPECT_EQ(r.widened_updates, wide);
      EXPECT_EQ(r.log_ratio_updates, logs);
      EXPECT_EQ(nth_root(a, degree), r.value);
    }
}
TEST(NthRootTrace, OriginalFiftyDiagnosticShortcutsControlsAndQuota) {
  auto s = nth_root_demo_study();
  EXPECT_EQ(s.trace.run.initial, 4);
  EXPECT_EQ(s.trace.steps[0].power, 16);
  EXPECT_EQ(s.trace.steps[0].ratio, 3.125L);
  EXPECT_EQ(s.trace.steps[0].after, 11.125 / 3);
  EXPECT_NEAR(s.trace.run.value, std::cbrt(50.), 1e-14);
  auto fixed = nth_root_demo_study({8, 0, 1000});
  EXPECT_EQ(fixed.trace.run.updates, 8);
  EXPECT_EQ(fixed.values.size(), 9);
  EXPECT_FALSE(fixed.trace.run.converged);
  auto t = nth_root_trace(50, 3);
  EXPECT_NO_THROW(nth_root_trace(50, 3, {100, 1e-15, t.captured_entries}));
  EXPECT_THROW(nth_root_trace(50, 3, {100, 1e-15, t.captured_entries - 1}), std::length_error);
  EXPECT_FALSE(nth_root_run(50, 3, {0, 1e-15, 0}).converged);
  EXPECT_EQ(nth_root_run(50, 3, {0, 1e-15, 0}).value, 4);
  EXPECT_EQ(nth_root(0, 4), 0);
  EXPECT_TRUE(nth_root_run(0, 4).zero_shortcut);
  EXPECT_EQ(nth_root(5, 1), 5);
  EXPECT_TRUE(nth_root_run(5, 1).identity_shortcut);
  EXPECT_THROW(nth_root(-1, 3), std::invalid_argument);
  EXPECT_THROW(nth_root(1, 0), std::invalid_argument);
  EXPECT_THROW(nth_root(NAN, 1), std::invalid_argument);
  EXPECT_THROW(nth_root(INFINITY, 1), std::invalid_argument);
  EXPECT_THROW(nth_root_run(1, 2, {-1, 1e-15, 0}), std::invalid_argument);
  EXPECT_THROW(nth_root_run(1, 2, {1, -1, 0}), std::invalid_argument);
  EXPECT_THROW(nth_root_trace(50, 3, {100, 1e-15, 3}), std::length_error);
}
