#include <gtest/gtest.h>

#include <datamunge/stats/anova_test.hpp>
#include <datamunge/stats/ks_test.hpp>
#include <datamunge/stats/nonparametric.hpp>
#include <datamunge/stats/wilcoxon_test.hpp>

#include <limits>
#include <stdexcept>
#include <vector>

namespace sp = datamunge::stats;

TEST(Nonparametric, KendallExactPermutationTailsAndTies) {
    const std::vector<double> x{1, 2, 3, 4}, y{1, 2, 4, 3};
    const auto exact = sp::kendall_correlation_test(x, y);
    EXPECT_DOUBLE_EQ(exact.statistic, 4);
    EXPECT_NEAR(exact.estimate1, 2. / 3, 1e-14);
    EXPECT_NEAR(exact.p_value, 8. / 24, 1e-14);
    EXPECT_NEAR(sp::kendall_correlation_test(x, y, sp::Alternative::Greater).p_value, 4. / 24, 1e-14);
    EXPECT_NEAR(sp::kendall_correlation_test(x, y, sp::Alternative::Less).p_value, 23. / 24, 1e-14);

    // R cor.test(method = "kendall", exact = FALSE, continuity = FALSE).
    const auto tied = sp::kendall_correlation_test({1, 1, 2, 3, 4, 4}, {1, 3, 2, 2, 4, 6});
    EXPECT_NEAR(tied.estimate1, 0.5929994533288809, 1e-14);
    EXPECT_NEAR(tied.p_value, 0.1129044267625, 1e-13);
    EXPECT_THROW(sp::kendall_correlation_test({1, 1}, {1, 2}), std::invalid_argument);
}

TEST(Nonparametric, QuadrantUsesFisherMarginsAndOmitsMedianPairs) {
    const auto result = sp::quadrant_test({1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 7, 6});
    EXPECT_DOUBLE_EQ(result.parameter1, 6);
    EXPECT_DOUBLE_EQ(result.statistic, 3);
    EXPECT_DOUBLE_EQ(result.estimate1, 1);
    EXPECT_NEAR(result.p_value, 0.1, 1e-14);
    EXPECT_THROW(sp::quadrant_test({1, 1, 1}, {1, 2, 3}), std::invalid_argument);
}

TEST(Nonparametric, SignAndPairedWilcoxonUseDifferenceDirection) {
    const auto sign = sp::paired_sign_test({4, 4, 5, 7, 4}, {3, 4, 4, 3, 3});
    EXPECT_DOUBLE_EQ(sign.parameter1, 4);
    EXPECT_DOUBLE_EQ(sign.statistic, 4);
    EXPECT_NEAR(sign.p_value, 0.125, 1e-14);
    const auto paired = sp::wilcoxon_paired_test({4, 3, 5, 7}, {3, 4, 4, 3}, sp::Alternative::Greater);
    const auto differences = sp::wilcoxon_signed_rank_exact_test({1, -1, 1, 4}, 0, sp::Alternative::Greater);
    EXPECT_DOUBLE_EQ(paired.statistic, differences.statistic);
    EXPECT_DOUBLE_EQ(paired.p_value, differences.p_value);
    const auto tied = sp::wilcoxon_signed_rank_exact_test({0, 1, -1, 2});
    EXPECT_DOUBLE_EQ(tied.statistic, 4.5);
    EXPECT_DOUBLE_EQ(tied.parameter1, 3);
    EXPECT_DOUBLE_EQ(tied.p_value, 0.75);
    EXPECT_DOUBLE_EQ(sp::wilcoxon_signed_rank_exact_test({1, -1, 2}, 0, sp::Alternative::Greater).p_value, 0.375);
    const std::vector<double> many(101, 1.);
    EXPECT_DOUBLE_EQ(sp::wilcoxon_paired_test(many, std::vector<double>(101)).p_value,
                     sp::wilcoxon_signed_rank_test(many).p_value);
    EXPECT_THROW(sp::wilcoxon_signed_rank_exact_test(many), std::invalid_argument);
    EXPECT_THROW(sp::paired_sign_test({1, 2}, {1}), std::invalid_argument);
    EXPECT_THROW(sp::sign_test({1, 1}, 1), std::invalid_argument);
}

TEST(Nonparametric, GroupAndBlockedTestsMatchIndependentReference) {
    const std::vector<std::vector<double>> groups{{1, 1, 2, 5}, {2, 3, 5}, {4, 4, 6, 7}};
    const auto mood = sp::median_test(groups);
    EXPECT_DOUBLE_EQ(mood.estimate1, 4);
    EXPECT_NEAR(mood.statistic, 3.2625, 1e-13);
    EXPECT_NEAR(mood.p_value, 0.1956848151655, 1e-13);
    const auto vdw = sp::van_der_waerden_test(groups);
    EXPECT_NEAR(vdw.statistic, 4.3991065559589586, 1e-7);
    EXPECT_NEAR(vdw.p_value, 0.1108526676307344, 1e-8);
    EXPECT_DOUBLE_EQ(sp::kruskal_wallis_test(groups).statistic,
                     sp::kruskal_wallis_test({1, 1, 2, 5, 2, 3, 5, 4, 4, 6, 7}, {4, 3, 4}).statistic);
    const auto friedman = sp::friedman_test({{1, 1, 3}, {2, 3, 4}, {3, 3, 2}, {2, 4, 5}});
    EXPECT_NEAR(friedman.statistic, 16. / 7, 1e-14);
    EXPECT_NEAR(friedman.p_value, 0.318906557324, 1e-12);
    EXPECT_THROW(sp::friedman_test({{1, 1}, {3, 3}}), std::invalid_argument);
    EXPECT_THROW(sp::median_test({{1}, {1}}), std::invalid_argument);
    EXPECT_THROW(sp::van_der_waerden_test({{1}, {1}}), std::invalid_argument);
}

TEST(Nonparametric, GenericKsValidatesCdfAndPlotsActualSteps) {
    const std::vector<double> sample{0.2, 0.4, 0.9};
    for (auto direction : {sp::Alternative::TwoSided, sp::Alternative::Less, sp::Alternative::Greater}) {
        const auto expected = sp::ks_test_one_sample_uniform(sample, 0, 1, direction);
        const auto actual = sp::ks_test_one_sample(sample, [](double x) { return x; }, direction);
        EXPECT_DOUBLE_EQ(actual.statistic, expected.statistic);
        EXPECT_DOUBLE_EQ(actual.p_value, expected.p_value);
    }
    EXPECT_THROW(sp::ks_test_one_sample(sample, [](double x) { return 1 - x; }), std::invalid_argument);
    EXPECT_THROW(sp::ks_test_one_sample(sample, {}), std::invalid_argument);
    EXPECT_THROW(sp::ks_test_one_sample(sample, [](double) { return std::numeric_limits<double>::quiet_NaN(); }),
                 std::invalid_argument);
    const auto plot = sp::empirical_cdf_plot({1, 1, 2}, {0, 3});
    ASSERT_EQ(plot.series().size(), 2U);
    EXPECT_EQ(plot.series()[0].x, (std::vector<double>{0, 1, 1, 2, 2, 3}));
    EXPECT_EQ(plot.series()[0].y, (std::vector<double>{0, 0, 2. / 3, 2. / 3, 1, 1}));
}
