#include <gtest/gtest.h>

#include <datamunge/stats/anova_test.hpp>
#include <datamunge/stats/chi_squared_test.hpp>
#include <datamunge/stats/correlation_test.hpp>
#include <datamunge/stats/fisher_exact_test.hpp>
#include <datamunge/stats/ks_test.hpp>
#include <datamunge/stats/normality_test.hpp>
#include <datamunge/stats/p_adjust.hpp>
#include <datamunge/stats/proportion_test.hpp>
#include <datamunge/stats/t_test.hpp>
#include <datamunge/stats/variance_test.hpp>
#include <datamunge/stats/wilcoxon_test.hpp>

#include <stdexcept>
#include <vector>

using namespace datamunge::stats;

// Reference values throughout were computed with base R (t.test/wilcox.test/ks.test/
// chisq.test/oneway.test/kruskal.test/cor.test/var.test/prop.test/binom.test/fisher.test)
// on a small fixed dataset -- see scratchpad/htest_oracle/check.cpp for the exact R session
// this was validated against interactively. Tolerances are tight where the method is
// identical to R's (t-tests, chi-squared tests, ANOVA, F-test, Pearson correlation,
// binomial test) and looser where it's documented to differ (asymptotic vs. exact KS,
// uncorrected vs. corrected Wilson proportion CI, sample vs. conditional-MLE odds ratio).

namespace {
const std::vector<double> kX{3.7471, 5.3673, 3.3287, 8.1906, 5.659,  3.3591, 5.9749, 6.4766, 6.1516, 4.3892,
                              8.0236, 5.7797, 3.7575, 0.5706, 7.2499, 4.9101, 4.9676, 6.8877, 6.6424, 6.1878};
const std::vector<double> kY{8.7569, 8.3464, 6.2237, 0.0319, 7.8595, 5.8316, 5.5326, 1.5877,
                              4.5655, 7.2538, 10.076, 5.6916, 7.163,  5.8386, 1.8688, 4.755, 4.8171, 5.8221};
} // namespace

TEST(TTest, OneSampleMatchesR) {
    const auto r = t_test_one_sample(kX, 4.0);
    EXPECT_NEAR(r.statistic, 3.3814, 1e-3);
    EXPECT_NEAR(r.parameter1, 19.0, 1e-9);
    EXPECT_NEAR(r.p_value, 0.003133, 1e-5);
    EXPECT_NEAR(r.conf_int_lower, 4.526214, 1e-3);
    EXPECT_NEAR(r.conf_int_upper, 6.235886, 1e-3);
}

TEST(TTest, WelchTwoSampleMatchesR) {
    const auto r = t_test_two_sample(kX, kY, /*equal_variance=*/false);
    EXPECT_NEAR(r.statistic, -0.39327, 1e-4);
    EXPECT_NEAR(r.parameter1, 30.404, 1e-2);
    EXPECT_NEAR(r.p_value, 0.6969, 1e-3);
}

TEST(TTest, PooledTwoSampleMatchesR) {
    const auto r = t_test_two_sample(kX, kY, /*equal_variance=*/true);
    EXPECT_NEAR(r.statistic, -0.40029, 1e-4);
    EXPECT_NEAR(r.parameter1, 36.0, 1e-9);
    EXPECT_NEAR(r.p_value, 0.6913, 1e-3);
}

TEST(TTest, PairedMatchesR) {
    const std::vector<double> x18(kX.begin(), kX.begin() + 18);
    const auto r = t_test_paired(x18, kY);
    EXPECT_NEAR(r.statistic, -0.46797, 1e-4);
    EXPECT_NEAR(r.parameter1, 17.0, 1e-9);
    EXPECT_NEAR(r.p_value, 0.6457, 1e-3);
}

TEST(TTest, PairedRejectsMismatchedLength) {
    EXPECT_THROW(t_test_paired(kX, kY), std::invalid_argument);
}

TEST(TTest, IdenticalSamplesGiveZeroStatisticAndPValueOne) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0};
    const auto r = t_test_two_sample(a, a);
    EXPECT_NEAR(r.statistic, 0.0, 1e-9);
    EXPECT_NEAR(r.p_value, 1.0, 1e-9);
}

TEST(Wilcoxon, SignedRankMatchesR) {
    const auto r = wilcoxon_signed_rank_test(kX, 4.0);
    EXPECT_NEAR(r.statistic, 180.0, 1e-9);
    EXPECT_NEAR(r.p_value, 0.005414, 5e-4);
}

TEST(Wilcoxon, RankSumMatchesR) {
    const auto r = wilcoxon_rank_sum_test(kX, kY);
    EXPECT_NEAR(r.statistic, 161.0, 1e-9);
    EXPECT_NEAR(r.p_value, 0.5886, 5e-3);
}

TEST(Wilcoxon, IdenticalSamplesGiveLargePValue) {
    const std::vector<double> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const auto r = wilcoxon_rank_sum_test(a, a);
    EXPECT_GT(r.p_value, 0.5);
}

TEST(KSTest, OneSampleStatisticMatchesRExactly) {
    // The D statistic computation is identical to R's regardless of exact vs. asymptotic
    // p-value method, so this should match to high precision even though the p-value uses
    // a different (documented) approximation.
    const auto r = ks_test_one_sample_normal(kX, 5.0, 2.0);
    EXPECT_NEAR(r.statistic, 0.18207, 1e-4);
    EXPECT_GT(r.p_value, 0.3); // both R's exact (0.4669) and our asymptotic (0.4797) agree: not significant
}

TEST(KSTest, TwoSampleStatisticMatchesRExactly) {
    const auto r = ks_test_two_sample(kX, kY);
    EXPECT_NEAR(r.statistic, 0.18333, 1e-4);
    EXPECT_GT(r.p_value, 0.5);
}

TEST(KSTest, SampleAgainstItsOwnDistributionIsNotSignificant) {
    const auto r = ks_test_two_sample(kX, kX);
    EXPECT_NEAR(r.statistic, 0.0, 1e-9);
    EXPECT_NEAR(r.p_value, 1.0, 1e-9);
}

TEST(ChiSquaredTest, GoodnessOfFitMatchesR) {
    const std::vector<double> observed{18, 22, 20, 15, 25};
    const auto r = chi_squared_goodness_of_fit(observed);
    EXPECT_NEAR(r.statistic, 2.9, 1e-9);
    EXPECT_NEAR(r.parameter1, 4.0, 1e-9);
    EXPECT_NEAR(r.p_value, 0.5747, 1e-4);
}

TEST(ChiSquaredTest, IndependenceThreeByTwoMatchesR) {
    const std::vector<double> table{10, 20, 15, 25, 12, 18};
    const auto r = chi_squared_test_independence(table, 3, 2);
    EXPECT_NEAR(r.statistic, 0.29315, 1e-4);
    EXPECT_NEAR(r.parameter1, 2.0, 1e-9);
    EXPECT_NEAR(r.p_value, 0.8637, 1e-4);
}

TEST(ChiSquaredTest, IndependenceTwoByTwoWithYatesMatchesR) {
    const std::vector<double> table{8, 12, 15, 5};
    const auto r = chi_squared_test_independence(table, 2, 2, /*correct=*/true);
    EXPECT_NEAR(r.statistic, 3.6829, 1e-3);
    EXPECT_NEAR(r.p_value, 0.05497, 1e-4);
}

TEST(ChiSquaredTest, RejectsTableSizeMismatch) {
    const std::vector<double> table{1, 2, 3};
    EXPECT_THROW(chi_squared_test_independence(table, 2, 2), std::invalid_argument);
}

TEST(AnovaTest, OneWayMatchesR) {
    const std::vector<double> v{6.1,    5.7632, 4.8355, 4.7466, 5.697,  5.5567, 4.3112, 4.2925, 5.3646, 5.7685,
                                 5.8877, 6.8811, 6.3981, 5.388,  6.3411, 4.8706, 7.433,  7.9804, 5.6328, 4.9559,
                                 6.0697, 5.3649, 7.9016, 5.4608, 6.1897, 5.528,  4.7567, 5.6888, 3.695,  6.9656};
    const std::vector<std::size_t> sizes{10, 10, 10};
    const auto r = one_way_anova(v, sizes);
    EXPECT_NEAR(r.statistic, 2.3351, 1e-3);
    EXPECT_NEAR(r.parameter1, 2.0, 1e-9);
    EXPECT_NEAR(r.parameter2, 27.0, 1e-9);
    EXPECT_NEAR(r.p_value, 0.116, 1e-3);
}

TEST(AnovaTest, KruskalWallisMatchesR) {
    const std::vector<double> v{6.1,    5.7632, 4.8355, 4.7466, 5.697,  5.5567, 4.3112, 4.2925, 5.3646, 5.7685,
                                 5.8877, 6.8811, 6.3981, 5.388,  6.3411, 4.8706, 7.433,  7.9804, 5.6328, 4.9559,
                                 6.0697, 5.3649, 7.9016, 5.4608, 6.1897, 5.528,  4.7567, 5.6888, 3.695,  6.9656};
    const std::vector<std::size_t> sizes{10, 10, 10};
    const auto r = kruskal_wallis_test(v, sizes);
    EXPECT_NEAR(r.statistic, 4.1368, 1e-3);
    EXPECT_NEAR(r.parameter1, 2.0, 1e-9);
    EXPECT_NEAR(r.p_value, 0.1264, 1e-3);
}

TEST(AnovaTest, RejectsGroupSizeMismatch) {
    const std::vector<double> v{1, 2, 3, 4};
    const std::vector<std::size_t> sizes{2, 3}; // sums to 5, not 4
    EXPECT_THROW(one_way_anova(v, sizes), std::invalid_argument);
}

TEST(CorrelationTest, PearsonMatchesR) {
    const std::vector<double> x18(kX.begin(), kX.begin() + 18);
    const auto r = pearson_correlation_test(x18, kY);
    EXPECT_NEAR(r.estimate1, -0.3203987, 1e-4);
    EXPECT_NEAR(r.statistic, -1.3529, 1e-3);
    EXPECT_NEAR(r.parameter1, 16.0, 1e-9);
    EXPECT_NEAR(r.p_value, 0.1949, 1e-3);
    EXPECT_NEAR(r.conf_int_lower, -0.684829, 1e-3);
    EXPECT_NEAR(r.conf_int_upper, 0.1722351, 1e-3);
}

TEST(CorrelationTest, SpearmanMatchesR) {
    const std::vector<double> x18(kX.begin(), kX.begin() + 18);
    const auto r = spearman_correlation_test(x18, kY);
    EXPECT_NEAR(r.estimate1, -0.4262126, 1e-4);
    EXPECT_NEAR(r.p_value, 0.07922, 5e-3);
}

TEST(CorrelationTest, PerfectCorrelationGivesRNearOne) {
    const std::vector<double> a{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<double> b{2, 4, 6, 8, 10, 12, 14, 16};
    const auto r = pearson_correlation_test(a, b);
    EXPECT_NEAR(r.estimate1, 1.0, 1e-9);
}

TEST(VarianceTest, FTestMatchesR) {
    const auto r = f_test_variance(kX, kY);
    EXPECT_NEAR(r.statistic, 0.50762, 1e-4);
    EXPECT_NEAR(r.parameter1, 19.0, 1e-9);
    EXPECT_NEAR(r.parameter2, 17.0, 1e-9);
    EXPECT_NEAR(r.p_value, 0.1554, 1e-3);
    EXPECT_NEAR(r.conf_int_lower, 0.1927821, 1e-3);
    EXPECT_NEAR(r.conf_int_upper, 1.3030578, 1e-3);
}

TEST(ProportionTest, OneSampleMatchesR) {
    const auto r = proportion_test_one_sample(45, 100, 0.5);
    EXPECT_NEAR(r.statistic, 0.81, 1e-6);
    EXPECT_NEAR(r.p_value, 0.3681, 1e-3);
    // CI uses uncorrected Wilson vs. R's corrected Wilson (documented); check it's close.
    EXPECT_NEAR(r.conf_int_lower, 0.3514281, 5e-3);
    EXPECT_NEAR(r.conf_int_upper, 0.5524574, 5e-3);
}

TEST(ProportionTest, TwoSampleMatchesR) {
    const auto r = proportion_test_two_sample(45, 100, 55, 90);
    EXPECT_NEAR(r.statistic, 4.3067, 1e-3);
    EXPECT_NEAR(r.p_value, 0.03796, 1e-4);
    EXPECT_NEAR(r.conf_int_lower, -0.31185, 1e-4);
    EXPECT_NEAR(r.conf_int_upper, -0.01037, 1e-4);
}

TEST(ProportionTest, BinomialTestTwoSidedMatchesR) {
    const auto r = binomial_test(45, 100, 0.5);
    EXPECT_NEAR(r.p_value, 0.3682, 1e-4);
    EXPECT_NEAR(r.conf_int_lower, 0.3503202, 1e-4);
    EXPECT_NEAR(r.conf_int_upper, 0.5527198, 1e-4);
}

TEST(ProportionTest, BinomialTestGreaterMatchesR) {
    const auto r = binomial_test(60, 100, 0.5, Alternative::Greater);
    EXPECT_NEAR(r.p_value, 0.02844, 1e-4);
    EXPECT_NEAR(r.conf_int_lower, 0.5129758, 1e-4);
    EXPECT_NEAR(r.conf_int_upper, 1.0, 1e-9);
}

TEST(ProportionTest, RejectsSuccessesGreaterThanTrials) {
    EXPECT_THROW(binomial_test(11, 10), std::invalid_argument);
}

TEST(FisherExactTest, TwoByTwoMatchesR) {
    const auto r = fisher_exact_test_2x2(8, 12, 15, 5);
    EXPECT_NEAR(r.p_value, 0.05355, 1e-4);
    // estimate1 is the naive sample odds ratio, not R's conditional MLE (0.2315317);
    // check against the documented naive value instead.
    EXPECT_NEAR(r.estimate1, (8.0 * 5.0) / (12.0 * 15.0), 1e-9);
}

TEST(FisherExactTest, IndependentTableGivesLargePValue) {
    // Rows and columns proportional -- perfectly consistent with independence.
    const auto r = fisher_exact_test_2x2(10, 10, 10, 10);
    EXPECT_GT(r.p_value, 0.5);
}

TEST(NormalityTest, RoughlyNormalSampleIsNotSignificant) {
    const std::vector<double> n1{4.3735, 5.1836, 4.1644, 6.5953, 5.3295, 4.1795, 5.4874, 5.7383, 5.5758, 4.6946,
                                  6.5118, 5.3898, 4.3788, 2.7853, 6.1249, 4.9551, 4.9838, 5.9438, 5.8212, 5.5939,
                                  5.919,  5.7821, 5.0746, 3.0106, 5.6198, 4.9439, 4.8442, 3.5292, 4.5218, 5.4179};
    const auto r = shapiro_francia_test(n1);
    EXPECT_NEAR(r.statistic, 0.950956, 1e-4); // W', cross-checked against R shapiro.test's W = 0.95011
    EXPECT_GT(r.p_value, 0.05);
}

TEST(NormalityTest, UniformSampleHasLowerWThanNormalSample) {
    const std::vector<double> u1{9.1288, 2.936,  4.5907, 3.3239, 6.5087, 2.5802, 4.7855, 7.6631, 0.8425, 8.7532,
                                  3.3907, 8.3944, 3.4668, 3.3377, 4.7635, 8.922,  8.6434, 3.8999, 7.7732, 9.6062,
                                  4.3466, 7.1251, 3.9999, 3.2535, 7.5709, 2.0269, 7.1112, 1.2169, 2.4549, 1.433};
    const auto r = shapiro_francia_test(u1);
    // Cross-checked against R shapiro.test's W = 0.92532 (a different but closely related
    // statistic) -- both correctly rate this uniform sample as less normal-looking.
    EXPECT_LT(r.statistic, 0.96);
}

TEST(NormalityTest, RejectsFewerThanFiveObservations) {
    const std::vector<double> tiny{1.0, 2.0, 3.0};
    EXPECT_THROW(shapiro_francia_test(tiny), std::invalid_argument);
}

// Reference values below computed with base R's p.adjust() -- see the two Rscript sessions
// this was validated against interactively (25-value and 6-value fixed p-vectors).
namespace {
const std::vector<double> kP25{0.001, 0.008, 0.039, 0.041, 0.042, 0.06,  0.074, 0.205, 0.212, 0.216, 0.222,
                                0.251, 0.269, 0.275, 0.34,  0.341, 0.384, 0.569, 0.594, 0.696, 0.762, 0.94,
                                0.942, 0.975, 0.986};

void expect_near_vec(const std::vector<double>& actual, const std::vector<double>& expected, double tol) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], tol) << "at index " << i;
}
} // namespace

TEST(PAdjust, BonferroniMatchesR) {
    const auto r = p_adjust(kP25, PAdjustMethod::Bonferroni);
    expect_near_vec(r, {0.025, 0.2, 0.975, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 1e-9);
}

TEST(PAdjust, HolmMatchesR) {
    const auto r = p_adjust(kP25, PAdjustMethod::Holm);
    expect_near_vec(
        r, {0.025, 0.192, 0.897, 0.902, 0.902, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 1e-9);
}

TEST(PAdjust, HochbergMatchesR) {
    const auto r = p_adjust(kP25, PAdjustMethod::Hochberg);
    expect_near_vec(r,
                     {0.025, 0.192, 0.882, 0.882, 0.882, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986,
                      0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986},
                     1e-9);
}

TEST(PAdjust, HommelMatchesR) {
    const auto r = p_adjust(kP25, PAdjustMethod::Hommel);
    expect_near_vec(r,
                     {0.025, 0.192, 0.682, 0.697, 0.714, 0.84,  0.962, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986,
                      0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.986},
                     1e-9);
}

TEST(PAdjust, BHMatchesR) {
    const auto r = p_adjust(kP25, PAdjustMethod::BH);
    expect_near_vec(r,
                     {0.025, 0.1, 0.21, 0.21, 0.21, 0.25, 0.2642857143, 0.4910714286, 0.4910714286, 0.4910714286,
                      0.4910714286, 0.4910714286, 0.4910714286, 0.4910714286, 0.5328125, 0.5328125, 0.5647058824,
                      0.7815789474, 0.7815789474, 0.87, 0.9071428571, 0.986, 0.986, 0.986, 0.986},
                     1e-9);
}

TEST(PAdjust, BYMatchesR) {
    const auto r = p_adjust(kP25, PAdjustMethod::BY);
    expect_near_vec(r,
                     {0.0953989544, 0.3815958178, 0.8013512173, 0.8013512173, 0.8013512173, 0.9539895444, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                     1e-9);
}

TEST(PAdjust, NoneReturnsInputUnchanged) {
    const auto r = p_adjust(kP25, PAdjustMethod::None);
    expect_near_vec(r, kP25, 1e-12);
}

TEST(PAdjust, UnsortedInputMatchesR) {
    const std::vector<double> p6{0.6, 0.01, 0.03, 0.5, 0.02, 0.9};
    expect_near_vec(p_adjust(p6, PAdjustMethod::Bonferroni), {1, 0.06, 0.18, 1, 0.12, 1}, 1e-9);
    expect_near_vec(p_adjust(p6, PAdjustMethod::Holm), {1, 0.06, 0.12, 1, 0.10, 1}, 1e-9);
    expect_near_vec(p_adjust(p6, PAdjustMethod::Hochberg), {0.9, 0.06, 0.12, 0.9, 0.10, 0.9}, 1e-9);
    expect_near_vec(p_adjust(p6, PAdjustMethod::Hommel), {0.9, 0.06, 0.12, 0.9, 0.08, 0.9}, 1e-9);
    expect_near_vec(p_adjust(p6, PAdjustMethod::BH), {0.72, 0.06, 0.06, 0.72, 0.06, 0.9}, 1e-9);
    expect_near_vec(p_adjust(p6, PAdjustMethod::BY), {1, 0.147, 0.147, 1, 0.147, 1}, 1e-9);
}

TEST(PAdjust, HommelDegeneratesToHochbergWhenNIsTwo) {
    const std::vector<double> p2{0.03, 0.5};
    const auto hommel = p_adjust(p2, PAdjustMethod::Hommel);
    const auto hochberg = p_adjust(p2, PAdjustMethod::Hochberg);
    expect_near_vec(hommel, hochberg, 1e-12);
    expect_near_vec(hommel, {0.06, 0.5}, 1e-9);
}

TEST(PAdjust, SingleValueIsAlwaysUnchanged) {
    const std::vector<double> p1{0.03};
    for (const auto method : {PAdjustMethod::Bonferroni, PAdjustMethod::Holm, PAdjustMethod::Hochberg,
                               PAdjustMethod::Hommel, PAdjustMethod::BH, PAdjustMethod::BY, PAdjustMethod::None}) {
        expect_near_vec(p_adjust(p1, method), {0.03}, 1e-12);
    }
}

TEST(PAdjust, EmptyInputReturnsEmpty) {
    EXPECT_TRUE(p_adjust({}, PAdjustMethod::BH).empty());
}

// Reference values below are hand-computed (not from R -- WY/RW aren't in base R's p.adjust).
// sorted_p = [0.01, 0.04, 0.20] (already ascending); resampled_p's 4 rows, columns aligned to
// sorted_p's order:
//   row0 = [0.02,  0.03,  0.50]
//   row1 = [0.10,  0.01,  0.60]
//   row2 = [0.005, 0.20,  0.30]
//   row3 = [0.15,  0.25,  0.05]
//
// Westfall-Young (single-step): per-row global min = [0.02, 0.01, 0.005, 0.05]; adjusted p[k] =
// fraction of those <= sorted_p[k]: k=0 (0.01) -> {0.01,0.005} -> 2/4=0.5; k=1 (0.04) ->
// {0.02,0.01,0.005} -> 3/4=0.75; k=2 (0.20) -> all 4 -> 1.0.
//
// Romano-Wolf (step-down), walking k=2 down to 0 with a running per-row min over columns k..2:
//   k=2: qmin = [0.50,0.60,0.30,0.05] vs 0.20 -> only row3 -> 1/4=0.25
//   k=1: qmin = [0.03,0.01,0.20,0.05] vs 0.04 -> row0,row1 -> 2/4=0.5
//   k=0: qmin = [0.02,0.01,0.005,0.05] vs 0.01 -> row1,row2 -> 2/4=0.5
// raw = [0.5, 0.5, 0.25] -> monotonic cummax -> [0.5, 0.5, 0.5].
namespace {
const std::vector<double> kRwWySortedP{0.01, 0.04, 0.20};
const std::vector<std::vector<double>> kRwWyResampled{
    {0.02, 0.03, 0.50},
    {0.10, 0.01, 0.60},
    {0.005, 0.20, 0.30},
    {0.15, 0.25, 0.05},
};
} // namespace

TEST(WestfallYoung, MatchesHandComputedExample) {
    const auto r = westfall_young_adjust(kRwWyResampled, kRwWySortedP);
    expect_near_vec(r, {0.5, 0.75, 1.0}, 1e-12);
}

TEST(RomanoWolf, MatchesHandComputedExample) {
    const auto r = romano_wolf_adjust(kRwWyResampled, kRwWySortedP);
    expect_near_vec(r, {0.5, 0.5, 0.5}, 1e-12);
}

TEST(RomanoWolf, NeverLessPowerfulThanWestfallYoung) {
    // Romano-Wolf narrows its comparison set at every step, so it can only be at least as
    // powerful (adjusted p <= Westfall-Young's) at every rank -- a general property, not
    // specific to this example.
    const auto wy = westfall_young_adjust(kRwWyResampled, kRwWySortedP);
    const auto rw = romano_wolf_adjust(kRwWyResampled, kRwWySortedP);
    ASSERT_EQ(wy.size(), rw.size());
    for (std::size_t i = 0; i < wy.size(); ++i) EXPECT_LE(rw[i], wy[i] + 1e-12);
}

TEST(WestfallYoungRomanoWolf, AgreeWhenOnlyOneHypothesis) {
    const std::vector<double> p1{0.03};
    const std::vector<std::vector<double>> resamples{{0.01}, {0.05}, {0.02}, {0.10}};
    const auto wy = westfall_young_adjust(resamples, p1);
    const auto rw = romano_wolf_adjust(resamples, p1);
    expect_near_vec(wy, {0.5}, 1e-12); // 2 of 4 resamples (0.01, 0.02) <= 0.03
    expect_near_vec(rw, wy, 1e-12);
}

TEST(WestfallYoungRomanoWolf, EmptyInputReturnsEmpty) {
    EXPECT_TRUE(westfall_young_adjust({}, {}).empty());
    EXPECT_TRUE(romano_wolf_adjust({}, {}).empty());
}

TEST(WestfallYoungRomanoWolf, RejectsNoResamples) {
    const std::vector<double> p1{0.03};
    EXPECT_THROW(westfall_young_adjust({}, p1), std::invalid_argument);
    EXPECT_THROW(romano_wolf_adjust({}, p1), std::invalid_argument);
}

TEST(WestfallYoungRomanoWolf, RejectsMismatchedRowLength) {
    const std::vector<std::vector<double>> badRows{{0.1, 0.2}, {0.1, 0.2, 0.3}};
    EXPECT_THROW(westfall_young_adjust(badRows, kRwWySortedP), std::invalid_argument);
    EXPECT_THROW(romano_wolf_adjust(badRows, kRwWySortedP), std::invalid_argument);
}
