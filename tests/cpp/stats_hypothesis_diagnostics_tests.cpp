#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/stats/stats.hpp>
#include <limits>
#include <numeric>
using namespace datamunge::stats;
namespace {
const std::vector<double> x{1, 2, 3, 4}, y{2, 3, 4, 5};
}
TEST(HypothesisDiagnostics, FourPointStudentClosedFormAndIntervalDuality) {
  const auto r = t_test_one_sample(x);
  EXPECT_NEAR(r.statistic, std::sqrt(15.0), 1e-13);
  EXPECT_DOUBLE_EQ(r.parameter1, 3);
  EXPECT_NEAR(r.p_value, 0.030466291662170977, 1e-13);
  EXPECT_NEAR(r.conf_int_lower, .445739743239121, 1e-10);
  EXPECT_GT(r.conf_int_lower, 0);
  const auto paired     = t_test_paired({2, 4, 7}, {1, 2, 4});
  const auto difference = t_test_one_sample({1, 2, 3});
  EXPECT_DOUBLE_EQ(paired.statistic, difference.statistic);
  EXPECT_DOUBLE_EQ(paired.p_value, difference.p_value);
}
TEST(HypothesisDiagnostics, StudentSmallTailRemainsRepresentableAndSymmetric) {
  const std::vector<double> high{9999998, 9999999, 10000000, 10000001};
  auto                      low = high;
  for (auto& v : low)
    v = -v;
  const auto a = t_test_one_sample(high), b = t_test_one_sample(low);
  EXPECT_GT(a.p_value, 0);
  EXPECT_LT(a.p_value, 1e-19);
  EXPECT_DOUBLE_EQ(a.p_value, b.p_value);
  EXPECT_NEAR(
      t_test_one_sample(high, 0, Alternative::Greater).p_value, a.p_value / 2, a.p_value * 1e-12);
}
TEST(HypothesisDiagnostics, RankTiesAndContinuityCorrectionHaveClosedForm) {
  const std::vector<double> a{1, 2, 2}, b{3, 4, 5};
  const auto                ranks = rank_sum_table(a, b);
  const auto                w     = wilcoxon_rank_sum_test(a, b);
  EXPECT_DOUBLE_EQ(ranks.double_at("rank", 1), 2.5);
  EXPECT_DOUBLE_EQ(ranks.double_at("rank", 2), 2.5);
  EXPECT_DOUBLE_EQ(ranks.double_at("rank", 5), 6);
  EXPECT_DOUBLE_EQ(w.statistic, 0);
  EXPECT_NEAR(w.p_value, std::erfc((4 / std::sqrt(5.1)) / std::sqrt(2.0)), 1e-15);
}
TEST(HypothesisDiagnostics, RankSumTinyTailDoesNotDependOnSampleOrder) {
  std::vector<double> a(50), b(50);
  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 100);
  const auto first = wilcoxon_rank_sum_test(a, b), second = wilcoxon_rank_sum_test(b, a);
  EXPECT_GT(first.p_value, 0);
  EXPECT_LT(first.p_value, 1e-16);
  EXPECT_DOUBLE_EQ(first.p_value, second.p_value);
}
TEST(HypothesisDiagnostics, ThreeGroupsHaveExactSumsOfSquaresAndRankStatistic) {
  const std::vector<double> v{1, 2, 3, 4, 5, 6, 7, 8, 9};
  const auto                a = one_way_anova(v, {3, 3, 3}), k = kruskal_wallis_test(v, {3, 3, 3});
  EXPECT_DOUBLE_EQ(a.statistic, 27);
  EXPECT_DOUBLE_EQ(a.parameter1, 2);
  EXPECT_DOUBLE_EQ(a.parameter2, 6);
  EXPECT_NEAR(a.p_value, .001, 1e-15); // F(2,6) survival at 27 = (6/(6+54))^3
  EXPECT_NEAR(k.statistic, 7.2, 1e-13);
  EXPECT_NEAR(k.p_value, std::exp(-3.6), 1e-14);
}
TEST(HypothesisDiagnostics, LargeChiSquareAndAnovaTailsDoNotCancel) {
  const auto chi = chi_squared_goodness_of_fit({1000, 0});
  EXPECT_GT(chi.p_value, 0);
  EXPECT_LT(chi.p_value, 1e-200);
  const auto a = one_way_anova({0, 1, 2, 100, 101, 102, 200, 201, 202}, {3, 3, 3});
  EXPECT_NEAR(a.p_value, std::pow(6.0 / (6 + 2 * a.statistic), 3), 1e-23);
  const std::vector<double> lo{1, 2, 3, 4}, hi{1, 200, 300, 400};
  const auto                less = f_test_variance(lo, hi), greater = f_test_variance(hi, lo);
  EXPECT_GT(less.p_value, 0);
  EXPECT_NEAR(less.p_value, greater.p_value, less.p_value * 1e-10);
}
TEST(HypothesisDiagnostics, BinomialTenTrialsHasRationalTailAndConservativeSize) {
  const auto b = binomial_test(9, 10);
  EXPECT_NEAR(b.p_value, 22.0 / 1024, 1e-15);
  const auto d    = binomial_rejection_table(10);
  double     mass = 0, size = 0;
  for (std::size_t i = 0; i < 11; ++i) {
    mass += d.double_at("null_probability", i);
    if (d.double_at("reject_exact", i))
      size += d.double_at("null_probability", i);
  }
  EXPECT_NEAR(mass, 1, 1e-14);
  EXPECT_NEAR(size, 22.0 / 1024, 1e-15);
  const auto operating = binomial_operating_characteristics({10}, .5, .7);
  EXPECT_NEAR(operating.double_at("exact_size", 0), 22.0 / 1024, 1e-15);
  EXPECT_NEAR(operating.double_at("exact_power", 0), .1494520318, 1e-12);
}
TEST(HypothesisDiagnostics, ProbabilityOrderedTestIsNotDualToCentralInterval) {
  const auto b = binomial_test(0, 10, .3);
  EXPECT_NEAR(b.p_value, .0388396033, 1e-12);
  EXPECT_LT(b.p_value, .05);
  EXPECT_GT(b.conf_int_upper, .3);
  EXPECT_DOUBLE_EQ(b.conf_int_lower, 0);
}
TEST(HypothesisDiagnostics, BinomialDirectUpperTailSurvivesTinyProbabilities) {
  const auto b = binomial_test(100, 100, .5, Alternative::Greater);
  EXPECT_NEAR(b.p_value, std::exp2(-100), std::exp2(-100) * 1e-12);
  const auto c = binomial_test(0, 100, .5, Alternative::Less);
  EXPECT_DOUBLE_EQ(b.p_value, c.p_value);
  EXPECT_NEAR(binomial_test(100, 100).p_value, 2 * std::exp2(-100), std::exp2(-100) * 1e-12);
}
TEST(HypothesisDiagnostics, BinomialNullEndpointsHaveDefinedInclusiveTails) {
  EXPECT_DOUBLE_EQ(binomial_test(0, 10, 0).p_value, 1);
  EXPECT_DOUBLE_EQ(binomial_test(1, 10, 0).p_value, 0);
  EXPECT_DOUBLE_EQ(binomial_test(10, 10, 1).p_value, 1);
  EXPECT_DOUBLE_EQ(binomial_test(9, 10, 1).p_value, 0);
  EXPECT_DOUBLE_EQ(binomial_test(0, 10, 1, Alternative::Greater).p_value, 1);
  EXPECT_DOUBLE_EQ(binomial_test(10, 10, 0, Alternative::Less).p_value, 1);
}
TEST(HypothesisDiagnostics, SparseScoreRuleCanExceedNominalSize) {
  const auto d = binomial_operating_characteristics(
      {1, 5, 10, 20, 40, 80}, .1, .3, .05, Alternative::TwoSided, false);
  EXPECT_NEAR(d.double_at("score_size", 0), .1, 1e-14);
  for (std::size_t i = 0; i < d.nrows(); ++i)
    EXPECT_LE(d.double_at("exact_size", i), .05 + 1e-14);
}
TEST(HypothesisDiagnostics, FisherFourTablesHaveRationalConditionalProbabilities) {
  const auto                d = fisher_exact_support(0, 3, 3, 0);
  const std::vector<double> mass{.05, .45, .45, .05};
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(d.double_at("probability", i), mass[i], 1e-15);
    EXPECT_DOUBLE_EQ(d.double_at("selected", i), i == 0 || i == 3);
  }
  const auto f = fisher_exact_test_2x2(0, 3, 3, 0);
  EXPECT_NEAR(f.p_value, .1, 1e-15);
  EXPECT_DOUBLE_EQ(f.estimate1, 0);
  EXPECT_NEAR(chi_squared_test_independence({0, 3, 3, 0}, 2, 2, false).statistic, 6, 1e-14);
  EXPECT_NEAR(chi_squared_test_independence({0, 3, 3, 0}, 2, 2, true).statistic, 8.0 / 3, 1e-14);
}
TEST(HypothesisDiagnostics, FisherTailsSymmetryZerosAndCountOverflow) {
  const auto a = fisher_exact_test_2x2(50, 0, 0, 50, Alternative::Greater);
  const auto b = fisher_exact_test_2x2(0, 50, 50, 0, Alternative::Less);
  EXPECT_GT(a.p_value, 0);
  EXPECT_LT(a.p_value, 1e-25);
  EXPECT_DOUBLE_EQ(a.p_value, b.p_value);
  const auto degenerate = fisher_exact_test_2x2(0, 0, 0, 0);
  EXPECT_DOUBLE_EQ(degenerate.p_value, 1);
  EXPECT_TRUE(std::isnan(degenerate.estimate1));
  EXPECT_THROW(
      fisher_exact_test_2x2(std::numeric_limits<std::size_t>::max(), 1, 0, 0),
      std::invalid_argument);
}
TEST(HypothesisDiagnostics, CorrelationThreePairsHasNoFisherIntervalAndPerfectLimits) {
  EXPECT_FALSE(pearson_correlation_test({1, 2, 3}, {2, 4, 6}).has_conf_int);
  const auto r = pearson_correlation_test(x, {2, 4, 6, 8});
  EXPECT_DOUBLE_EQ(r.estimate1, 1);
  EXPECT_DOUBLE_EQ(r.p_value, 0);
  EXPECT_TRUE(std::isinf(r.statistic));
  EXPECT_DOUBLE_EQ(r.conf_int_lower, 1);
  EXPECT_DOUBLE_EQ(r.conf_int_upper, 1);
  EXPECT_THROW(pearson_correlation_test(x, {1, 1, 1, 1}), std::invalid_argument);
}
TEST(HypothesisDiagnostics, ResultTablesOwnLabelsAndPreserveAbsentIntervals) {
  const auto t = t_test_one_sample(x), w = wilcoxon_rank_sum_test(x, y);
  const auto d = hypothesis_test_table({t, w}, {"mean", "ranks"});
  EXPECT_EQ(d.string_at("label", 0), "mean");
  EXPECT_DOUBLE_EQ(d.double_at("ci_lower", 0), t.conf_int_lower);
  EXPECT_TRUE(d.is_null("ci_lower", 1));
  EXPECT_TRUE(d.is_null("ci_upper", 1));
  EXPECT_EQ(hypothesis_test_table({}).nrows(), 0u);
  EXPECT_THROW((void)hypothesis_test_table({t}, {"one", "two"}), std::invalid_argument);
}
TEST(HypothesisDiagnostics, InvalidFiniteControlsAndDegenerateDomainsAreRejected) {
  const double nan     = std::numeric_limits<double>::quiet_NaN();
  const auto   invalid = static_cast<Alternative>(99);
  EXPECT_THROW(t_test_one_sample({1, nan}), std::invalid_argument);
  EXPECT_THROW(t_test_one_sample({1, 1}), std::invalid_argument);
  EXPECT_THROW(t_test_two_sample(x, y, false, invalid), std::invalid_argument);
  EXPECT_THROW(t_test_one_sample(x, 0, Alternative::TwoSided, 1), std::invalid_argument);
  EXPECT_THROW(wilcoxon_rank_sum_test({1, 1}, {1, 1}), std::invalid_argument);
  EXPECT_THROW(wilcoxon_signed_rank_test({0, 0}), std::invalid_argument);
  EXPECT_THROW(one_way_anova(x, {0, 4}), std::invalid_argument);
  EXPECT_THROW(one_way_anova({1, 2}, {1, 1}), std::invalid_argument);
  EXPECT_THROW(kruskal_wallis_test({1, 1, 1, 1}, {2, 2}), std::invalid_argument);
  EXPECT_THROW(chi_squared_goodness_of_fit({1, -1}), std::invalid_argument);
  EXPECT_THROW(chi_squared_goodness_of_fit({1, 2}, {.4, .4}), std::invalid_argument);
  EXPECT_THROW(chi_squared_test_independence({0, 0, 1, 2}, 2, 2), std::invalid_argument);
  EXPECT_THROW(f_test_variance(x, {1, 1, 1}), std::invalid_argument);
  EXPECT_THROW(shapiro_francia_test({1, 1, 1, 1, 1}), std::invalid_argument);
  EXPECT_THROW(shapiro_francia_test(std::vector<double>(5001, 1)), std::invalid_argument);
  EXPECT_THROW(ks_test_one_sample_normal(x, nan, 1), std::invalid_argument);
  EXPECT_THROW(ks_test_two_sample(x, {1, nan}), std::invalid_argument);
  EXPECT_THROW(binomial_test(1, 10, nan), std::invalid_argument);
  EXPECT_THROW(proportion_test_one_sample(0, 10, 0), std::invalid_argument);
  EXPECT_THROW((void)binomial_rejection_table(0), std::invalid_argument);
  EXPECT_THROW((void)binomial_operating_characteristics({}, .5, .7), std::invalid_argument);
}
TEST(HypothesisDiagnostics, ReferencePlotUsesStudentDensityAndSelectedTail) {
  const auto t    = t_test_one_sample(x);
  const auto plot = t_test_reference_plot(t);
  EXPECT_GE(plot.series().size(), 3u);
  auto invalid       = t;
  invalid.parameter1 = 0;
  EXPECT_THROW((void)t_test_reference_plot(invalid), std::invalid_argument);
}

TEST(HypothesisDiagnostics, HighConfidenceVarianceIntervalsBracketQuantilesBelowOneMicro) {
  const auto f = f_test_variance({1, 2}, {2, 4}, Alternative::TwoSided, .999);
  // With df1=df2=1, F quantile(p)=tan(pi*p/2)^2.
  const double q = std::pow(std::tan(std::acos(-1.0) * .0005 / 2), 2);
  EXPECT_NEAR(f.conf_int_lower, .25 * q, 1e-18);
  EXPECT_NEAR(f.conf_int_upper, .25 / q, 1e-5);
}
