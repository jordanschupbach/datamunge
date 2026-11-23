#include <datamunge/stats/edf_tests.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace {

using namespace datamunge::stats;
using datamunge::random::SplitMix64;

TEST(EdfTests, AndersonDarlingMatchesSpecifiedAndFittedReferences) {
  const std::vector<double> uniforms{.05, .11, .12, .19, .22, .44, .46, .61, .86};
  const std::vector<double> normal{-3.2, -1.1, -.6, -.15, 0, .1, .4, 1.4, 4.7};
  SplitMix64 specified_rng(73), fitted_rng(73), affine_rng(73);
  const auto specified = anderson_darling_test(uniforms, [](double x) { return x; }, 4095, specified_rng);
  const auto fitted = anderson_darling_normal_test(normal, 4095, fitted_rng);
  EXPECT_NEAR(specified.statistic, 1.5935273689511682, 1e-14);
  EXPECT_NEAR(specified.p_value, .15668, .026);
  EXPECT_NEAR(fitted.statistic, .52034842684478022, 1e-14);
  EXPECT_NEAR(fitted.p_value, .13926, .026);
  EXPECT_NEAR(fitted.estimate1, .17222222222222222, 1e-14);
  EXPECT_NEAR(fitted.estimate2, 2.1149927764520724, 1e-14);
  EXPECT_DOUBLE_EQ(fitted.p_value, (1 + fitted.parameter2) / 4096);
  auto affine = normal;
  for (auto& value : affine)
    value = 11 - 3 * value;
  const auto transformed = anderson_darling_normal_test(affine, 4095, affine_rng);
  EXPECT_NEAR(transformed.statistic, fitted.statistic, 1e-14);
  EXPECT_DOUBLE_EQ(transformed.parameter2, fitted.parameter2);
  EXPECT_NEAR(transformed.estimate1, 11 - 3 * fitted.estimate1, 1e-14);
  EXPECT_NEAR(transformed.estimate2, 3 * fitted.estimate2, 1e-14);
}

TEST(EdfTests, AndersonDarlingValidatesTheCdfAndKeepsExtremeStatistics) {
  SplitMix64 rng(7), fresh(7);
  EXPECT_THROW(anderson_darling_test({.2, .8}, [](double x) { return 1 - x; }, 99, rng),
               std::invalid_argument);
  EXPECT_THROW(anderson_darling_test({1}, {}, 99, rng), std::invalid_argument);
  EXPECT_THROW(anderson_darling_normal_test({2, 2, 2}, 99, rng), std::invalid_argument);
  EXPECT_THROW(anderson_darling_normal_test({1, 2}, 99, rng), std::invalid_argument);
  EXPECT_EQ(rng.next_u64(), fresh.next_u64());
  const auto edge = anderson_darling_test({0, .2, .8}, [](double x) { return x; }, 99, rng);
  EXPECT_TRUE(std::isinf(edge.statistic));
  EXPECT_DOUBLE_EQ(edge.p_value, .01);
  std::vector<double> separated(30000, 0);
  separated.back() = 1;
  EXPECT_TRUE(std::isfinite(anderson_darling_normal_test(separated, 1, rng).statistic));
}

TEST(EdfTests, CramerVonMisesMatchesSpecifiedAndExactPooledReferences) {
  const std::vector<double> uniforms{.05, .11, .12, .19, .22, .44, .46, .61, .86};
  const std::vector<double> first{0, 1, 1}, second{1, 2, 2, 4};
  SplitMix64 rng(27), copy(27);
  const auto specified = cramer_von_mises_test(uniforms, [](double x) { return x; }, 999, rng);
  const auto repeated = cramer_von_mises_test(uniforms, [](double x) { return x; }, 999, copy);
  EXPECT_NEAR(specified.statistic, .31040000000000001, 1e-14);
  EXPECT_DOUBLE_EQ(specified.parameter2, repeated.parameter2);
  EXPECT_DOUBLE_EQ(specified.p_value, (1 + specified.parameter2) / 1000);
  const auto exact = cramer_von_mises_test_two_sample_exact(first, second);
  EXPECT_NEAR(exact.statistic, .47108843537414968, 1e-14);
  EXPECT_DOUBLE_EQ(exact.parameter1, 35);
  EXPECT_DOUBLE_EQ(exact.parameter2, 4);
  EXPECT_DOUBLE_EQ(exact.p_value, 4. / 35);
  EXPECT_DOUBLE_EQ(cramer_von_mises_test_two_sample_exact(second, first).p_value, exact.p_value);
  const auto simulated = cramer_von_mises_test_two_sample(first, second, 4095, rng);
  EXPECT_DOUBLE_EQ(simulated.statistic, exact.statistic);
  EXPECT_NEAR(simulated.p_value, exact.p_value, .025);
  EXPECT_DOUBLE_EQ(cramer_von_mises_test_two_sample_exact({7, 7}, {7, 7, 7}).p_value, 1);
  EXPECT_THROW(cramer_von_mises_test_two_sample_exact(first, second, 34), std::length_error);
  EXPECT_THROW(cramer_von_mises_test_two_sample_exact(first, {}), std::invalid_argument);
}

TEST(EdfTests, MidrankAndersonDarlingMatchesDirectMultigroupEnumeration) {
  const std::vector<std::vector<double>> groups{{0, 1, 1}, {1, 2}, {2, 3}};
  const auto exact = anderson_darling_k_sample_test_exact(groups, 210);
  EXPECT_NEAR(exact.statistic, 3.4967948717948714, 1e-14);
  EXPECT_DOUBLE_EQ(exact.parameter1, 210);
  EXPECT_DOUBLE_EQ(exact.parameter2, 30);
  EXPECT_DOUBLE_EQ(exact.p_value, 1. / 7);
  const auto reordered = anderson_darling_k_sample_test_exact({{3, 2}, {2, 1}, {1, 0, 1}});
  EXPECT_DOUBLE_EQ(reordered.statistic, exact.statistic);
  EXPECT_DOUBLE_EQ(reordered.p_value, exact.p_value);
  EXPECT_THROW(anderson_darling_k_sample_test_exact(groups, 209), std::length_error);

  SplitMix64 rng(7), copy(7);
  const auto sampled = anderson_darling_k_sample_test(groups, 4095, rng);
  const auto repeated = anderson_darling_k_sample_test(groups, 4095, copy);
  EXPECT_DOUBLE_EQ(sampled.statistic, exact.statistic);
  EXPECT_DOUBLE_EQ(sampled.p_value, (1 + sampled.parameter2) / 4096);
  EXPECT_DOUBLE_EQ(sampled.p_value, repeated.p_value);
  EXPECT_NEAR(sampled.p_value, exact.p_value, .025);

  const auto constant = anderson_darling_k_sample_test_exact({{1, 1}, {1}, {1, 1}});
  EXPECT_DOUBLE_EQ(constant.statistic, 0);
  EXPECT_DOUBLE_EQ(constant.parameter1, 30);
  EXPECT_DOUBLE_EQ(constant.p_value, 1);
  EXPECT_THROW(anderson_darling_k_sample_test_exact({{1}}), std::invalid_argument);
  EXPECT_THROW(anderson_darling_k_sample_test({{1}, {}, {2}}, 99, rng), std::invalid_argument);
}

} // namespace
