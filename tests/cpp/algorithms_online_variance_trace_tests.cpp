#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/online_statistics.hpp>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(OnlineVarianceTrace, ActualDoubleOperandsIndependentCenteredReferenceAndChunkMerge) {
  std::mt19937_64 rng(264273);
  for (std::size_t n = 0; n < 100; ++n) {
    std::vector<double> values(n);
    for (auto& v : values)
      v = 1000. + double(int(rng() % 100) - 50) / 8;
    auto t = online_variance_trace(values);
    EXPECT_EQ(t.captured_records, n + 2);
    OnlineVariance a, left, right;
    double         mean = 0, m2 = 0;
    for (std::size_t i = 0; i < n; ++i) {
      auto& u = t.updates[i];
      EXPECT_EQ(u.before.count, i);
      EXPECT_EQ(u.before.mean, mean);
      EXPECT_EQ(u.before.m2, m2);
      double delta = values[i] - mean;
      mean += delta / (i + 1);
      double delta2 = values[i] - mean, increment = delta * delta2;
      m2 += increment;
      EXPECT_FALSE(u.widened);
      EXPECT_EQ(u.delta, delta);
      EXPECT_EQ(u.delta_after, delta2);
      EXPECT_EQ(u.increment, increment);
      EXPECT_EQ(u.after.mean, mean);
      EXPECT_EQ(u.after.m2, m2);
      a.add(values[i]);
      if (i < n / 2)
        left.add(values[i]);
      else
        right.add(values[i]);
    }
    EXPECT_EQ(a.mean(), t.final.mean);
    EXPECT_EQ(a.sum_of_squares(), t.final.m2);
    auto merge = left.merge_observed(right);
    EXPECT_EQ(merge.after.count, n);
    long double reference_mean = 0, reference_m2 = 0;
    for (double v : values)
      reference_mean += v;
    if (n)
      reference_mean /= n;
    for (double v : values)
      reference_m2 += (v - reference_mean) * (v - reference_mean);
    EXPECT_NEAR(a.mean(), double(reference_mean), 2e-12);
    EXPECT_NEAR(a.sum_of_squares(), double(reference_m2), 3e-9);
    EXPECT_NEAR(left.mean(), double(reference_mean), 2e-12);
    EXPECT_NEAR(left.sum_of_squares(), double(reference_m2), 3e-9);
    EXPECT_NO_THROW(online_variance_trace(values, n + 2));
    EXPECT_THROW(online_variance_trace(values, n + 1), std::length_error);
  }
}
TEST(OnlineVarianceTrace, AtomicFiniteOverflowCountGuardsAndSelfAlias) {
  OnlineVariance s;
  s.add(2);
  s.add(4);
  auto before = s.state();
  EXPECT_THROW(s.add(NAN), std::invalid_argument);
  EXPECT_EQ(s.count(), before.count);
  EXPECT_EQ(s.mean(), before.mean);
  auto m = s.merge_observed(s);
  EXPECT_EQ(m.left.count, 2);
  EXPECT_EQ(m.right.count, 2);
  EXPECT_EQ(s.count(), 4);
  EXPECT_EQ(s.mean(), 3);
  EXPECT_EQ(s.sum_of_squares(), 4);
  OnlineVariance huge;
  huge.add(std::numeric_limits<double>::max());
  auto old = huge.state();
  EXPECT_THROW(huge.add(-std::numeric_limits<double>::max()), std::overflow_error);
  EXPECT_EQ(huge.count(), old.count);
  EXPECT_EQ(huge.mean(), old.mean);
  EXPECT_EQ(huge.sum_of_squares(), old.m2);
  OnlineVariance other;
  other.add(-std::numeric_limits<double>::max());
  EXPECT_THROW(huge.merge(other), std::overflow_error);
  EXPECT_EQ(huge.mean(), old.mean);
  OnlineVariance many;
  many.add(1);
  while (many.count() <= SIZE_MAX / 2)
    many.merge(many);
  auto count = many.count();
  EXPECT_THROW(many.merge(many), std::length_error);
  EXPECT_EQ(many.count(), count);
  EXPECT_THROW(online_variance_trace({1, NAN}), std::invalid_argument);
}
TEST(OnlineVarianceTrace, CovarianceIndependentReferenceAtomicPairAndOriginalRngStudy) {
  OnlineCovariance    c;
  std::vector<double> x{1, 2, 4, 8}, y{2, 3, 8, 15};
  for (std::size_t i = 0; i < x.size(); ++i)
    c.add(x[i], y[i]);
  long double mx = 0, my = 0, cross = 0, sx = 0, sy = 0;
  for (double v : x)
    mx += v;
  for (double v : y)
    my += v;
  mx /= x.size();
  my /= y.size();
  for (std::size_t i = 0; i < x.size(); ++i) {
    cross += (x[i] - mx) * (y[i] - my);
    sx += (x[i] - mx) * (x[i] - mx);
    sy += (y[i] - my) * (y[i] - my);
  }
  EXPECT_NEAR(c.sample_covariance(), double(cross / 3), 1e-14);
  EXPECT_NEAR(c.correlation(), double(cross / std::sqrt(sx * sy)), 1e-15);
  auto count = c.count();
  auto mean  = c.mean_x();
  EXPECT_THROW(c.add(100, NAN), std::invalid_argument);
  EXPECT_EQ(c.count(), count);
  EXPECT_EQ(c.mean_x(), mean);
  OnlineCovariance large;
  large.add(1e150, 1e150);
  large.add(-1e150, -1e150);
  EXPECT_NEAR(large.correlation(), 1., 1e-15);
  auto d = online_variance_demo_data();
  ASSERT_EQ(d.base.size(), 10000);
  EXPECT_EQ(d.x.size(), 5000);
  auto rows = online_variance_offset_study(d.base, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16});
  EXPECT_EQ(rows.size(), 11);
  for (auto& r : rows) {
    EXPECT_EQ(r.count, 10000);
    EXPECT_TRUE(r.wide_reference_spread);
    EXPECT_GE(r.welford, 0);
    EXPECT_TRUE(std::isfinite(r.wide_reference));
  }
  EXPECT_LT(rows.back().distinct_values, rows.front().distinct_values);
  EXPECT_THROW(online_variance_offset_study({1}, {0}), std::invalid_argument);
  EXPECT_THROW(online_variance_offset_study({-1, 1}, {0, 1}, 3), std::length_error);
}
