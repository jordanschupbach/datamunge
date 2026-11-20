#include <gtest/gtest.h>

#include <datamunge/stats/additional_tests.hpp>
#include <datamunge/stats/bootstrap.hpp>
#include <datamunge/stats/classical_tests.hpp>
#include <datamunge/stats/survival.hpp>
#include <datamunge/stats/wilcoxon_test.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sp = datamunge::stats;

TEST(ClassicalTests, ScaleRanksAndMannWhitney) {
    const std::vector<double> wide{0, 1, 9, 10}, narrow{3, 4, 6, 7};
    const auto u = sp::mann_whitney_test(wide, narrow);
    EXPECT_DOUBLE_EQ(u.statistic, sp::wilcoxon_rank_sum_test(wide, narrow).statistic);
    EXPECT_DOUBLE_EQ(u.estimate1, 0.5);
    const auto ansari = sp::ansari_bradley_test(wide, narrow, sp::Alternative::Greater);
    EXPECT_DOUBLE_EQ(ansari.statistic, 6);
    EXPECT_NEAR(ansari.p_value, 0.008980238763039, 1e-13); // R ansari.test(exact=FALSE)
    EXPECT_NEAR(sp::ansari_bradley_test(narrow, wide, sp::Alternative::Less).p_value, ansari.p_value, 1e-14);
    const auto siegel = sp::siegel_tukey_test({16.55, 15.36, 15.94, 16.43, 16.01},
                                           {16.05, 15.98, 16.10, 15.88, 15.91});
    EXPECT_DOUBLE_EQ(siegel.statistic, 24); // NIST worksheet ranks, regardless of reference choice
    EXPECT_DOUBLE_EQ(sp::siegel_tukey_test({1, 2}, {1, 3}).statistic, 5.5); // Tied extremes.
    EXPECT_THROW(sp::ansari_bradley_test({1}, {2}), std::invalid_argument);
}

TEST(ClassicalTests, OrderedAndBinaryBlocksAndTieCorrections) {
    const std::vector<std::vector<double>> blocks{{1, 2, 3}, {2, 3, 4}};
    EXPECT_DOUBLE_EQ(sp::kendall_w_test(blocks).estimate1, 1);
    const auto page = sp::page_trend_test(blocks);
    EXPECT_DOUBLE_EQ(page.statistic, 28);
    EXPECT_NEAR(page.p_value, 0.5 * std::erfc(std::sqrt(2.)), 1e-14);
    EXPECT_DOUBLE_EQ(sp::page_trend_test({{1, 1, 3}, {2, 3, 4}}).statistic, 27.5);
    const auto q = sp::cochran_q_test({{0, 1, 1}, {0, 0, 1}, {1, 1, 1}});
    EXPECT_NEAR(q.statistic, 3, 1e-14);
    EXPECT_NEAR(q.p_value, std::exp(-1.5), 1e-14);
    EXPECT_THROW(sp::page_trend_test({{0, 0}, {1, 1}}), std::invalid_argument);
    EXPECT_THROW(sp::cochran_q_test({{0, 2}, {1, 1}}), std::invalid_argument);
}

TEST(ClassicalTests, CategoricalReferences) {
    const auto mh = sp::mantel_haenszel_test({{8, 2, 4, 6}, {5, 3, 2, 6}}, false);
    EXPECT_NEAR(mh.statistic, 5.2957906712173, 1e-13); // R mantelhaen.test(correct=FALSE)
    EXPECT_NEAR(mh.p_value, 0.02137702415284, 1e-14);
    EXPECT_NEAR(mh.estimate1, 5.516129032258065, 1e-14);
    EXPECT_LT(sp::mantel_haenszel_test({{8, 2, 4, 6}, {5, 3, 2, 6}}).statistic, mh.statistic);
    auto paired = sp::mcnemar_test({1, 1, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 0});
    EXPECT_DOUBLE_EQ(paired.statistic, 4);
    EXPECT_DOUBLE_EQ(paired.parameter1, 5);
    EXPECT_NEAR(paired.p_value, 0.375, 1e-14);
    EXPECT_DOUBLE_EQ(sp::mcnemar_test(0, 0).p_value, 1);
    EXPECT_THROW(sp::mantel_haenszel_test({{0, 0, 1, 2}}), std::invalid_argument);
}

TEST(ClassicalTests, KaplanMeierTiedRiskSetsAndPrenticeScores) {
    const std::vector<sp::SurvivalObservation> records{{3, true}, {1, true}, {1, false}, {4, false}};
    const auto km = sp::kaplan_meier(records);
    ASSERT_EQ(km.points.size(), 3U);
    EXPECT_EQ(km.points[0].at_risk, 4U);
    EXPECT_EQ(km.points[0].events, 1U);
    EXPECT_EQ(km.points[0].censored, 1U);
    EXPECT_DOUBLE_EQ(km.points[0].survival, 0.75);
    EXPECT_NEAR(km.points[0].standard_error, 0.75 * std::sqrt(1. / 12), 1e-14);
    EXPECT_EQ(km.points[1].at_risk, 2U);
    EXPECT_DOUBLE_EQ(km.points.back().survival, 0.375);
    EXPECT_EQ(km.median_time.value(), 3);
    EXPECT_FALSE(sp::kaplan_meier({{1, false}, {2, false}}).median_time.has_value());
    const auto scores = sp::prentice_wilcoxon_scores(records);
    EXPECT_NEAR(scores[0], -1. / 15, 1e-14);
    EXPECT_NEAR(scores[1], -0.6, 1e-14);
    EXPECT_NEAR(scores[2], 0.2, 1e-14);
    EXPECT_NEAR(scores[3], 7. / 15, 1e-14);
    EXPECT_NEAR(std::accumulate(scores.begin(), scores.end(), 0.), 0., 1e-14);
    const auto comparison = sp::prentice_wilcoxon_test({{1, true}, {2, true}}, {{3, true}, {4, false}},
                                                       sp::Alternative::Less);
    EXPECT_LT(comparison.p_value, 0.1);
    EXPECT_THROW(sp::kaplan_meier({{-1, false}}), std::invalid_argument);
    EXPECT_THROW(sp::prentice_wilcoxon_test({{1, false}}, {{2, false}}), std::invalid_argument);
    EXPECT_FALSE(sp::kaplan_meier_plot({km}).to_svg().empty());
}

TEST(ClassicalTests, BootstrapTInvertsTheSameSeededResamples) {
    const sp::Bootstrap bootstrap({100, 0.9, 42});
    const std::vector<double> sample{1, 3, 4, 5};
    const auto mean = [](const auto& x) { return std::accumulate(x.begin(), x.end(), 0.) / x.size(); };
    const auto ordinary = bootstrap.run(sample, mean);
    const auto student = bootstrap.run_studentized(sample, mean, [](const auto&) { return 2.; });
    ASSERT_EQ(student.pivots.size(), ordinary.replicates.size());
    EXPECT_EQ(student.discarded, 0U);
    for (std::size_t i = 0; i < student.pivots.size(); ++i)
        EXPECT_DOUBLE_EQ(student.pivots[i], (ordinary.replicates[i] - ordinary.estimate) / 2);
    EXPECT_NEAR(student.interval.first, ordinary.basic_interval.first, 1e-14);
    EXPECT_NEAR(student.interval.second, ordinary.basic_interval.second, 1e-14);
    const auto skipped = sp::Bootstrap({100, 0.95, 42}).run_studentized({0, 1}, mean,
        [](const auto& x) { return std::abs(x[0] - x[1]); });
    EXPECT_GT(skipped.discarded, 0U);
    EXPECT_EQ(skipped.discarded + skipped.pivots.size(), 100U);
    EXPECT_THROW((void)bootstrap.run_studentized(sample, mean, [](const auto&) { return 0.; }), std::domain_error);
}

TEST(ClassicalTests, KnownVarianceMeansAndChowUseTheirReferenceLaws) {
    const auto first = sp::z_test_one_sample({1, 3, 5, 7}, 2, 3, sp::Alternative::Greater);
    EXPECT_DOUBLE_EQ(first.statistic, 1);
    EXPECT_NEAR(first.p_value, 0.158655253931457, 1e-15);
    const auto second = sp::z_test_two_sample({4}, {2}, 3, 4);
    EXPECT_DOUBLE_EQ(second.statistic, .4);
    EXPECT_NEAR(second.p_value, 0.6891565167793516, 1e-15);
    EXPECT_THROW(sp::z_test_one_sample({1}, 0), std::invalid_argument);
    const datamunge::linalg::DenseMatrix<double> design(4, 2, {1, 0, 1, 1, 1, 2, 1, 3});
    const auto chow = sp::chow_test(design, {0, 1, 3, 3}, design, {1, 3, 6, 7});
    EXPECT_NEAR(chow.statistic, 150.0 / 7, 1e-12);
    EXPECT_NEAR(chow.p_value, std::pow(2 / (2 + 150.0 / 7), 2), 1e-13);
    EXPECT_DOUBLE_EQ(chow.parameter1, 2);
    EXPECT_DOUBLE_EQ(chow.parameter2, 4);
    const datamunge::linalg::DenseMatrix<double> deficient(4, 2, 1.0);
    EXPECT_THROW(sp::chow_test(deficient, {0, 1, 3, 3}, design, {1, 3, 6, 7}), std::invalid_argument);
}

TEST(ClassicalTests, EnumeratedTestsRespectTiesStructuralZerosAndWorkLimits) {
    EXPECT_NEAR(sp::multinomial_test({4, 2, 0}, {.5, .3, .2}).p_value, .6625, 1e-14);
    EXPECT_DOUBLE_EQ(sp::multinomial_test({0, 2}, {1., 0.}).p_value, 0);
    EXPECT_THROW(sp::multinomial_test({4, 2, 0}, {.5, .3, .2}, 27), std::length_error);
    EXPECT_NEAR(sp::multinomial_test({4, 2, 0}, {.5, .3, .2}, 28).p_value, .6625, 1e-14);
    const auto perm = sp::permutation_test_exact({{1, 2}, {3, 4}});
    EXPECT_DOUBLE_EQ(perm.parameter1, 6);
    EXPECT_DOUBLE_EQ(perm.statistic, 4);
    EXPECT_DOUBLE_EQ(perm.p_value, 1.0 / 3);
    const auto ranks = sp::kruskal_wallis_test_exact({{0, 0}, {1, 1}});
    EXPECT_DOUBLE_EQ(ranks.p_value, 1.0 / 3);
    EXPECT_DOUBLE_EQ(ranks.statistic, 3);
    EXPECT_THROW(sp::permutation_test_exact({{1, 2}, {3, 4}}, 5), std::length_error);
}

TEST(ClassicalTests, ConditionalAndUnconditionalExactTestsMatchIndependentEnumeration) {
    EXPECT_NEAR(sp::fisher_exact_test({{1, 3, 2}, {4, 0, 2}}).p_value, 0.1341991341991342, 1e-14);
    EXPECT_THROW(sp::fisher_exact_test({{1, 3, 2}, {4, 0, 2}}, 1), std::length_error);
    EXPECT_NEAR(sp::barnard_exact_test(0, 3, 3, 0).p_value, .03125, 1e-13);
    EXPECT_NEAR(sp::boschloo_exact_test(0, 3, 3, 0, sp::Alternative::Less).p_value, .015625, 1e-13);
    // Independent exact-rational rejection ordering and binomial-polynomial maximization.
    EXPECT_NEAR(sp::barnard_exact_test(7, 12, 8, 3).p_value, .06821830932319466, 1e-10);
    EXPECT_NEAR(sp::boschloo_exact_test(7, 12, 8, 3).p_value, .06821830932319489, 1e-10);
    EXPECT_NEAR(sp::barnard_exact_test(7, 12, 8, 3, sp::Alternative::Less).p_value,
                sp::barnard_exact_test(12, 7, 3, 8, sp::Alternative::Greater).p_value, 1e-13);
}

TEST(ClassicalTests, CompositeNormalityMatchesReferenceAndRefitsSimulatedSamples) {
    std::vector<double> sample{-1.5, -.8, -.3, -.1, .2, .4, .7, 1.7, 2.3};
    auto wilk = sp::shapiro_wilk_test(sample);
    EXPECT_NEAR(wilk.statistic, 0.9763522898334478, 1e-8); // R shapiro.test()
    EXPECT_NEAR(wilk.p_value, 0.9429425794296288, 1e-7);
    const auto lillie = sp::lilliefors_test(sample, 999, 42);
    EXPECT_NEAR(lillie.statistic, 0.1413696, 1e-7);
    EXPECT_GE(lillie.p_value, .001);
    for (double& value : sample) value = 10 + 3 * value;
    const auto shifted = sp::lilliefors_test(sample, 999, 42);
    EXPECT_NEAR(shifted.statistic, lillie.statistic, 1e-15);
    EXPECT_DOUBLE_EQ(shifted.p_value, lillie.p_value);
    sample = {-1.5, -.8, -.3, -.1, .2, .4, .7, 1.7, 2.3, 3, -1.1, 1.01};
    wilk = sp::shapiro_wilk_test(sample);
    EXPECT_NEAR(wilk.statistic, 0.9712558379558854, 1e-8);
    EXPECT_NEAR(wilk.p_value, 0.9235046005312417, 1e-7);
    EXPECT_THROW(sp::shapiro_wilk_test({0, 0, 0}), std::invalid_argument);
    EXPECT_THROW(sp::lilliefors_test({0, 0, 0, 0}, 19, 42), std::invalid_argument);
}
