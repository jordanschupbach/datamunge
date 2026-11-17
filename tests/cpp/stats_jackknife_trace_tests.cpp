#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/stats/jackknife.hpp>
#include <datamunge/stats/jackknife_analysis.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
using namespace datamunge::stats;
namespace {
double mean(const std::vector<double>& x) {
  long double s = 0;
  for (auto v : x)
    s += v;
  return double(s / x.size());
}
double variance(const std::vector<double>& x) {
  long double s = 0;
  for (auto v : x)
    s += v;
  const auto m = s / x.size();
  s            = 0;
  for (auto v : x)
    s += (v - m) * (v - m);
  return double(s / x.size());
}
void check(
    const std::vector<double>&   x,
    const DeleteDJackknifeTrace& t,
    const SampleStatistic&       statistic) {
  ASSERT_EQ(t.result.num_subsets, t.replicates.size());
  ASSERT_EQ(t.replicates.size(), t.deletion_sets.size());
  EXPECT_EQ(t.evaluations, t.replicates.size() + 1);
  EXPECT_EQ(t.result.estimate, statistic(x));
  std::set<std::vector<std::size_t>> distinct;
  long double                        sum = 0;
  for (std::size_t i = 0; i < t.replicates.size(); ++i) {
    const auto& deleted = t.deletion_sets[i];
    ASSERT_EQ(deleted.size(), t.result.d);
    ASSERT_TRUE(std::is_sorted(deleted.begin(), deleted.end()));
    ASSERT_EQ(std::adjacent_find(deleted.begin(), deleted.end()), deleted.end());
    ASSERT_LT(deleted.back(), x.size());
    std::vector<double> kept;
    for (std::size_t j = 0; j < x.size(); ++j)
      if (!std::binary_search(deleted.begin(), deleted.end(), j))
        kept.push_back(x[j]);
    EXPECT_EQ(t.replicates[i], statistic(kept));
    sum += t.replicates[i];
    distinct.insert(deleted);
  }
  const auto  m  = sum / t.replicates.size();
  long double ss = 0;
  for (auto v : t.replicates)
    ss += (v - m) * (v - m);
  EXPECT_NEAR(t.mean_replicate, double(m), 1e-14 * (1 + std::abs(double(m))));
  EXPECT_NEAR(
      double(t.replicate_population_variance),
      double(ss / t.replicates.size()),
      1e-13 * (1 + double(ss / t.replicates.size())));
  EXPECT_NEAR(
      t.result.standard_error,
      double(std::sqrt(
          (x.size() - t.result.d) / ((long double)t.result.d * t.replicates.size()) * ss)),
      1e-13 * (1 + t.result.standard_error));
  if (t.result.complete) {
    EXPECT_EQ(distinct.size(), t.replicates.size());
    EXPECT_TRUE(std::is_sorted(t.deletion_sets.begin(), t.deletion_sets.end()));
  }
}
} // namespace
TEST(JackknifeTrace, DeleteOneActualOrderPseudoValuesAndCallbacks) {
  const std::vector<double>        x{1, 2, 4};
  std::vector<std::vector<double>> calls;
  const auto                       r = jackknife(x, [&](const auto& kept) {
    calls.push_back(kept);
    return mean(kept);
  });
  ASSERT_EQ(calls.size(), 4);
  EXPECT_EQ(calls[0], x);
  EXPECT_EQ(calls[1], (std::vector<double>{2, 4}));
  EXPECT_EQ(calls[2], (std::vector<double>{1, 4}));
  EXPECT_EQ(calls[3], (std::vector<double>{1, 2}));
  EXPECT_NEAR(r.bias, 0, 1e-15);
  EXPECT_NEAR(r.standard_error, std::sqrt(7. / 9), 1e-15);
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_EQ(r.leave_one_out[i], mean(calls[i + 1]));
    EXPECT_NEAR(r.pseudo_values[i], x[i], 1e-15);
  }
}
TEST(JackknifeTrace, ExhaustiveTernarySamplesAllDeleteCountsMatchMeanIdentity) {
  for (std::size_t n = 2; n <= 6; ++n) {
    std::size_t count = 1;
    for (std::size_t i = 0; i < n; ++i)
      count *= 3;
    for (std::size_t code = 0; code < count; ++code) {
      std::vector<double> x(n);
      auto                state = code;
      for (auto& v : x) {
        v = double(int(state % 3) - 1);
        state /= 3;
      }
      const auto expected = std::sqrt(variance(x) / (n - 1));
      const auto one      = jackknife(x, mean);
      EXPECT_NEAR(one.standard_error, expected, 1e-14);
      EXPECT_NEAR(one.bias, 0, 1e-14);
      const auto var = jackknife(x, variance);
      EXPECT_NEAR(var.bias_corrected, variance(x) * n / (n - 1), 1e-14);
      for (std::size_t d = 1; d < n; ++d) {
        const auto t = jackknife_delete_d_trace(x, mean, {d, 1000, 7});
        check(x, t, mean);
        EXPECT_TRUE(t.result.complete);
        EXPECT_NEAR(t.result.standard_error, expected, 1e-14);
      }
    }
  }
}
TEST(JackknifeTrace, ExactModeBoundaryLargeBinomialAndSampledOwnership) {
  const std::vector<double> x{1, 2, 3, 4, 5, 6};
  const auto                complete = jackknife_delete_d_trace(x, mean, {3, 20, 7});
  check(x, complete, mean);
  EXPECT_TRUE(complete.result.complete);
  EXPECT_EQ(complete.result.num_subsets, 20);
  const auto sampled = jackknife_delete_d_trace(x, mean, {3, 19, 7});
  check(x, sampled, mean);
  EXPECT_FALSE(sampled.result.complete);
  EXPECT_EQ(sampled.result.num_subsets, 19);
  EXPECT_EQ(sampled.deletion_sets, jackknife_delete_d_trace(x, mean, {3, 19, 7}).deletion_sets);
  const auto summary = jackknife_delete_d(x, mean, {3, 19, 7});
  EXPECT_EQ(summary.standard_error, sampled.result.standard_error);
  std::vector<double> large(100);
  std::iota(large.begin(), large.end(), 0);
  const auto big = jackknife_delete_d_trace(large, mean, {50, 37, 7});
  check(large, big, mean);
  EXPECT_FALSE(big.result.complete);
  auto copy                = sampled;
  copy.replicates[0]       = 99;
  copy.deletion_sets[0][0] = 99;
  EXPECT_NE(copy.replicates, sampled.replicates);
  EXPECT_NE(copy.deletion_sets, sampled.deletion_sets);
}
TEST(JackknifeTrace, MonteCarloRepeatsAndOneDrawDispersionAreExplicit) {
  const auto one = jackknife_delete_d_trace({1, 2, 3}, mean, {1, 1, 7});
  EXPECT_FALSE(one.result.complete);
  EXPECT_EQ(one.result.standard_error, 0);
  EXPECT_EQ(one.evaluations, 2);
  const auto two = jackknife_delete_d_trace({1, 2, 3, 4}, mean, {2, 5, 7});
  EXPECT_FALSE(two.result.complete);
  EXPECT_EQ(two.deletion_sets.size(), 5);
  // Complete mode needs no RNG and ignores the seed.
  EXPECT_EQ(
      jackknife_delete_d_trace({1, 2, 3}, mean, {1, 3, 1}).deletion_sets,
      jackknife_delete_d_trace({1, 2, 3}, mean, {1, 3, 999}).deletion_sets);
}
TEST(JackknifeTrace, CompleteAdmissionPrecedesCallbacks) {
  int  calls = 0;
  auto f     = [&](const auto& x) {
    ++calls;
    return mean(x);
  };
  EXPECT_THROW(jackknife_checked({1, 2, 3}, f, {3, 5}), std::length_error);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(jackknife_checked({1, 2, 3}, f, {2, 0}), std::length_error);
  EXPECT_EQ(calls, 0);
  EXPECT_NO_THROW(jackknife_checked({1, 2, 3}, f, {3, 6}));
  EXPECT_EQ(calls, 4);
  calls = 0;
  EXPECT_THROW(jackknife_delete_d_trace({1, 2, 3}, f, {1, 3, 7}, {3, 5}), std::length_error);
  EXPECT_EQ(calls, 0);
  EXPECT_NO_THROW(jackknife_delete_d_trace({1, 2, 3}, f, {1, 3, 7}, {3, 6}));
  EXPECT_EQ(calls, 4);
  calls = 0;
  EXPECT_NO_THROW(jackknife_delete_d_trace({1, 2, 3}, f, {1, 3, 7}, {0, 0}));
  EXPECT_NO_THROW(jackknife_checked({1, 2, 3}, mean, {0, 0}));
  const auto          maximum = std::numeric_limits<std::size_t>::max();
  std::vector<double> x(100, 1);
  EXPECT_THROW(jackknife_delete_d_trace(x, mean, {50, maximum, 7}, {0, 0}), std::length_error);
  EXPECT_THROW(jackknife_delete_d_trace(x, mean, {50, maximum / 2, 7}, {0, 0}), std::length_error);
}
TEST(JackknifeTrace, FiniteInputsAndCallbackFailures) {
  const auto nan   = std::numeric_limits<double>::quiet_NaN();
  int        calls = 0;
  auto       f     = [&](const auto&) {
    ++calls;
    return 1.;
  };
  EXPECT_THROW(jackknife({1}, f), std::invalid_argument);
  EXPECT_THROW(jackknife({1, 2}, {}), std::invalid_argument);
  EXPECT_THROW(jackknife({1, nan}, f), std::invalid_argument);
  EXPECT_THROW(jackknife_delete_d({1, 2}, f, {0, 2, 7}), std::invalid_argument);
  EXPECT_THROW(jackknife_delete_d({1, 2}, f, {2, 2, 7}), std::invalid_argument);
  EXPECT_THROW(jackknife_delete_d({1, 2}, f, {1, 0, 7}), std::invalid_argument);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(jackknife({1, 2}, [nan](const auto&) { return nan; }), std::domain_error);
  EXPECT_THROW(jackknife_delete_d({1, 2}, [nan](const auto&) { return nan; }), std::domain_error);
  EXPECT_THROW(
      jackknife({1, 2}, [](const auto&) -> double { throw std::runtime_error("caller"); }),
      std::runtime_error);
}
TEST(JackknifeTrace, CentredConstantAndOutputRange) {
  const auto tiny = jackknife({1e-300, 2e-300, 4e-300}, mean);
  EXPECT_NEAR(tiny.standard_error / 1e-300, std::sqrt(7. / 9), 1e-14);
  const auto m        = std::numeric_limits<double>::max();
  const auto constant = [m](const auto&) { return m; };
  const auto one      = jackknife({1, 2, 3}, constant);
  EXPECT_EQ(one.estimate, m);
  EXPECT_EQ(one.bias, 0);
  EXPECT_EQ(one.bias_corrected, m);
  EXPECT_EQ(one.standard_error, 0);
  for (auto v : one.pseudo_values)
    EXPECT_EQ(v, m);
  const auto many = jackknife_delete_d_trace({1, 2, 3}, constant, {1, 3, 7});
  EXPECT_EQ(many.result.standard_error, 0);
  EXPECT_EQ(many.mean_replicate, m);
  EXPECT_THROW(
      jackknife({1, 2, 3}, [m](const auto& x) { return x.size() == 3 ? m : -m; }),
      std::overflow_error);
}
TEST(JackknifeTrace, NativeStatisticBridgesAndGraphicsAdmission) {
  const std::vector<double> x{1, 2, 4};
  const auto                mean = jackknife_statistic(JackknifeStatistic::Mean),
             median              = jackknife_statistic(JackknifeStatistic::Median);
  const auto pop                 = jackknife_statistic(JackknifeStatistic::PopulationVariance),
             sample              = jackknife_statistic(JackknifeStatistic::SampleVariance);
  EXPECT_EQ(mean(x), 7. / 3);
  EXPECT_EQ(median(x), 2);
  EXPECT_NEAR(pop(x), 14. / 9, 1e-15);
  EXPECT_NEAR(sample(x), 7. / 3, 1e-15);
  EXPECT_EQ(pop({1}), 0);
  EXPECT_TRUE(std::isnan(pop({})));
  EXPECT_TRUE(std::isnan(sample({1})));
  EXPECT_THROW(jackknife_statistic(static_cast<JackknifeStatistic>(999)), std::invalid_argument);
  const auto r = jackknife(x, mean);
  EXPECT_FALSE(jackknife_deletion_diagram(x, r, 2, 3).to_svg().empty());
  EXPECT_FALSE(jackknife_influence_plot(x, r, 3).to_svg().empty());
  EXPECT_THROW(jackknife_deletion_diagram(x, r, 3), std::out_of_range);
  EXPECT_THROW(jackknife_deletion_diagram(x, r, 2, 2), std::length_error);
  EXPECT_THROW(jackknife_influence_plot(x, r, 2), std::length_error);
  auto bad             = r;
  bad.leave_one_out[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(jackknife_influence_plot(x, bad), std::invalid_argument);
  EXPECT_THROW(jackknife_deletion_diagram(x, bad), std::invalid_argument);
}
