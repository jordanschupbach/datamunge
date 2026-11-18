#include <gtest/gtest.h>

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
