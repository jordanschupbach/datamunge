#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/anova_test.hpp>
#include <datamunge/stats/p_adjust.hpp>
#include <datamunge/stats/pairwise_test.hpp>
#include <datamunge/stats/t_test.hpp>
#include <datamunge/stats/tukey_hsd.hpp>

#include <stdexcept>
#include <string>
#include <vector>

using namespace datamunge::stats;

// Reference values from R 4.5: ptukey/qtukey, TukeyHSD(aov(...)), pairwise.t.test and p.adjust.

namespace {
const std::vector<double>      kX{4.2, 5.1, 3.9, 4.8, 6.3, 5.9, 7.1, 5.2, 4.9, 6.0, 5.5, 5.8};
const std::vector<std::string> kG{"a", "a", "a", "a", "b", "b", "b", "c", "c", "c", "c", "c"};
} // namespace

TEST(WelchAnova, MatchesRAndWelchT) {
    const auto fit = welch_anova({1, 2, 4, 3, 6, 9, 12, 15, 3, 4, 5, 5}, {3, 5, 4});
    EXPECT_NEAR(fit.statistic, 4.123846206567713, 1e-12);
    EXPECT_EQ(fit.parameter1, 2);
    EXPECT_NEAR(fit.parameter2, 4.775362600667653, 1e-12);
    EXPECT_NEAR(fit.p_value, 0.09113216473697176, 1e-12);

    const auto two = welch_anova({1, 2, 3, 7}, {2, 2});
    const auto t = t_test_two_sample({1, 2}, {3, 7});
    EXPECT_NEAR(two.statistic, t.statistic * t.statistic, 1e-12);
    EXPECT_NEAR(two.parameter2, t.parameter1, 1e-12);
    EXPECT_NEAR(two.p_value, t.p_value, 1e-12);
    EXPECT_THROW(welch_anova({1, 2, 3}, {1, 2}), std::invalid_argument);
}

TEST(GamesHowell, MatchesRStudentizedRangeAndHandlesLowDegreesOfFreedom) {
    const auto rows = games_howell({1, 2, 4, 3, 6, 9, 12, 15, 3, 4, 5, 5}, {3, 5, 4});
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_EQ(rows[0].group1, 1u);
    EXPECT_EQ(rows[0].group2, 0u);
    EXPECT_NEAR(rows[0].difference, 6.666666666666666, 1e-12);
    EXPECT_NEAR(rows[0].degrees_of_freedom, 5.192003451747447, 1e-12);
    EXPECT_NEAR(rows[0].lower, -0.7108761659750273, 1e-5);
    EXPECT_NEAR(rows[0].upper, 14.04420949930836, 1e-5);
    EXPECT_NEAR(rows[0].p_adjusted, 0.07077266853099473, 1e-10);
    EXPECT_NEAR(rows[1].p_adjusted, 0.2742276174560709, 1e-10);
    EXPECT_NEAR(rows[2].p_adjusted, 0.1785500652859811, 1e-10);

    // SciPy tukey_hsd(equal_var=False), which supports pairwise df below two.
    const auto small = games_howell({1, 2, 3, 7, 4, 5}, {2, 2, 2});
    ASSERT_EQ(small.size(), 3u);
    EXPECT_LT(small[0].degrees_of_freedom, 2);
    EXPECT_NEAR(small[0].p_adjusted, 0.47108070, 1e-7);
    EXPECT_NEAR(small[0].lower, -26.39546349, 1e-6);
    EXPECT_NEAR(small[0].upper, 33.39546349, 1e-6);
    EXPECT_THROW(games_howell({1, 1, 3, 4}, {2, 2}), std::invalid_argument);
}

TEST(GroupScaleTests, MatchesRAndScipy) {
    const std::vector<double> values{1, 2, 4, 3, 6, 9, 12, 15, 3, 4, 5, 5};
    const std::vector<std::size_t> sizes{3, 5, 4};
    const auto means = levene_test(values, sizes, false);
    const auto medians = levene_test(values, sizes);
    EXPECT_NEAR(means.statistic, 3.613626892623975, 1e-12);
    EXPECT_NEAR(means.p_value, 0.07046746767741865, 1e-12);
    EXPECT_NEAR(medians.statistic, 3.550089445438283, 1e-12);
    EXPECT_NEAR(medians.p_value, 0.073005089125065, 1e-12);
    EXPECT_EQ(medians.parameter1, 2);
    EXPECT_EQ(medians.parameter2, 9);
    const auto bartlett = bartlett_test(values, sizes);
    EXPECT_NEAR(bartlett.statistic, 6.4957755578345, 1e-12);
    EXPECT_NEAR(bartlett.p_value, 0.03885619408704, 1e-12);
    const auto fligner = fligner_killeen_test(values, sizes);
    EXPECT_NEAR(fligner.statistic, 4.1283268053792, 1e-7);
    EXPECT_NEAR(fligner.p_value, 0.1269244308009, 1e-8);
    EXPECT_THROW(bartlett_test({1, 1, 2, 4}, {2, 2}), std::invalid_argument);
    EXPECT_THROW(fligner_killeen_test({1, 3, 2, 4}, {2, 2}), std::invalid_argument);
    EXPECT_THROW(levene_test({1, 2, 3}, {1, 2}), std::invalid_argument);
}

TEST(DunnTest, UsesTiedPooledRanksAndAdjustsTheWholeFamily) {
    const std::vector<double> values{0, 2, 2, 1, 2, 4, 5, 3, 3};
    const std::vector<std::size_t> sizes{3, 4, 2};
    const auto rows = dunn_test(values, sizes);
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_EQ(rows[0].group1, 1u);
    EXPECT_EQ(rows[0].group2, 0u);
    EXPECT_NEAR(rows[0].statistic, 1.3430289304681915, 1e-14);
    EXPECT_NEAR(rows[1].statistic, 1.4301109716746978, 1e-14);
    EXPECT_NEAR(rows[2].statistic, .3230291412348993, 1e-14);
    EXPECT_NEAR(rows[0].p_value, .17926260621678772, 1e-14);
    EXPECT_NEAR(rows[0].p_adjusted, .45805551515979082, 1e-14);
    EXPECT_NEAR(rows[2].p_adjusted, .7466731726289577, 1e-14);
    const auto raw = dunn_test(values, sizes, PAdjustMethod::None);
    EXPECT_DOUBLE_EQ(raw[0].p_value, raw[0].p_adjusted);
    EXPECT_THROW(dunn_test({1, 1, 1}, {1, 2}), std::invalid_argument);
}

TEST(StudentizedRange, MatchesR) {
    EXPECT_NEAR(studentized_range_cdf(3.5, 3, 20), 0.94410815, 1e-7);
    EXPECT_NEAR(studentized_range_quantile(0.95, 3, 20), 3.5779346, 1e-6);
    EXPECT_NEAR(studentized_range_cdf(3.5, 5, 30000), 0.90366154, 1e-7);  // infinite-df limit
    EXPECT_NEAR(studentized_range_cdf(10, 3, 147, false), 1.732432e-10, 1e-12);
    EXPECT_EQ(studentized_range_cdf(0, 3, 20), 0.0);
    EXPECT_THROW(studentized_range_cdf(1, 1, 20), std::invalid_argument);
    EXPECT_THROW(studentized_range_quantile(1.0, 3, 20), std::invalid_argument);
}

TEST(TukeyHSD, UnbalancedMatchesR) {
    const TukeyHSD hsd(kX, kG);
    ASSERT_EQ(hsd.comparisons().size(), 3u);
    const double diff[] = {1.9333333333, 0.98, -0.9533333333};
    const double lwr[]  = {0.824227003167, 0.005860746335, -2.013841826620};
    const double upr[]  = {3.042439663, 1.954139254, 0.107175160};
    const double padj[] = {0.002289965874, 0.048706978422, 0.077538501614};
    const char*  name[] = {"b-a", "c-a", "c-b"};
    for (std::size_t k = 0; k < 3; ++k) {
        const auto& c = hsd.comparisons()[k];
        EXPECT_EQ(c.label(), name[k]);
        EXPECT_NEAR(c.diff, diff[k], 1e-9);
        EXPECT_NEAR(c.lower, lwr[k], 1e-6);
        EXPECT_NEAR(c.upper, upr[k], 1e-6);
        EXPECT_NEAR(c.p_adjusted, padj[k], 1e-7);
    }
    EXPECT_EQ(hsd.df(), 9.0);
}

TEST(TukeyHSD, DataFrameAndPlot) {
    const TukeyHSD hsd(datamunge::datasets::iris(), "Sepal.Length", "Species");
    const auto     frame = hsd.frame();
    ASSERT_EQ(frame.nrows(), 3u);
    EXPECT_EQ(frame.get_column<std::string>("comparison").values()[2], "virginica-versicolor");
    EXPECT_NEAR(frame.get_column<double>("lwr").values()[0], 0.68622729, 1e-6);
    EXPECT_NEAR(frame.get_column<double>("p_adj").values()[2], 8.287557529e-09, 1e-11);
    EXPECT_EQ(hsd.plot().y_tick_label_list().size(), 3u);
    EXPECT_THROW(TukeyHSD(datamunge::datasets::iris(), "Species", "Species"), std::invalid_argument);
    EXPECT_THROW(TukeyHSD({1, 2, 3}, {"a", "a", "a"}), std::invalid_argument);
}

TEST(TukeyHSD, FormulaMatchesColumns) {
    const auto     iris = datamunge::datasets::iris();
    const TukeyHSD by_formula(iris, "Sepal.Length ~ Species");
    const TukeyHSD by_columns(iris, "Sepal.Length", "Species");
    ASSERT_EQ(by_formula.comparisons().size(), by_columns.comparisons().size());
    for (std::size_t i = 0; i < by_formula.comparisons().size(); ++i) {
        EXPECT_EQ(by_formula.comparisons()[i].label(), by_columns.comparisons()[i].label());
        EXPECT_DOUBLE_EQ(by_formula.comparisons()[i].diff, by_columns.comparisons()[i].diff);
        EXPECT_DOUBLE_EQ(by_formula.comparisons()[i].p_adjusted, by_columns.comparisons()[i].p_adjusted);
    }
    EXPECT_DOUBLE_EQ(TukeyHSD(iris, "Sepal.Length~Species", 0.9).conf_level(), 0.9);
    EXPECT_THROW(TukeyHSD(iris, "Species ~ Sepal.Length"), std::invalid_argument);
    EXPECT_THROW(TukeyHSD(iris, "Sepal.Length ~ Species + Petal.Width"), std::invalid_argument);
    EXPECT_THROW(TukeyHSD(iris, "Sepal.Length ~ Nope"), std::out_of_range);
}

TEST(PairwiseTTest, MatchesR) {
    const auto iris   = datamunge::datasets::iris();
    const auto pooled = pairwise_t_test(iris, "Sepal.Length", "Species");
    const auto welch  = pairwise_t_test(iris, "Sepal.Length", "Species", PAdjustMethod::Holm, false);
    const auto bonf   = pairwise_t_test(iris, "Sepal.Length", "Species", p_adjust_method("bonferroni"));
    const double pooled_r[] = {1.754038848e-15, 6.644464047e-32, 2.765638236e-09};
    const double welch_r[]  = {7.493485228e-17, 1.190060181e-24, 1.866144387e-07};
    const double bonf_r[]   = {2.631058272e-15, 6.644464047e-32, 8.296914707e-09};
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(pooled.get_column<double>("p_adjusted").values()[k] / pooled_r[k], 1.0, 1e-6);
        EXPECT_NEAR(welch.get_column<double>("p_adjusted").values()[k] / welch_r[k], 1.0, 1e-6);
        EXPECT_NEAR(bonf.get_column<double>("p_adjusted").values()[k] / bonf_r[k], 1.0, 1e-6);
    }
    EXPECT_EQ(pooled.get_column<std::string>("group1").values()[0], "versicolor");
    EXPECT_NEAR(pooled.get_column<double>("estimate").values()[0], 0.93, 1e-12);
    EXPECT_THROW(pairwise_t_test({1.0, 2.0}, {"a"}), std::invalid_argument);
}

TEST(PAdjust, ByName) {
    const auto by = p_adjust({0.01, 0.02, 0.04, 0.3}, "BY");
    EXPECT_NEAR(by[0], 0.08333333333, 1e-9);
    EXPECT_NEAR(by[2], 0.1111111111, 1e-9);
    EXPECT_NEAR(by[3], 0.625, 1e-12);
    EXPECT_EQ(p_adjust_method("FDR"), PAdjustMethod::BH);
    EXPECT_EQ(p_adjust_method("Holm"), PAdjustMethod::Holm);
    EXPECT_THROW(p_adjust_method("tukey"), std::invalid_argument);
}

TEST(DataFrameSplit, GroupsByLabel) {
    const auto iris = datamunge::datasets::iris();
    const auto by   = iris.split("Sepal.Length", "Species");
    ASSERT_EQ(by.size(), 3u);
    EXPECT_EQ(by.begin()->first, "setosa");
    EXPECT_EQ(by.at("virginica").size(), 50u);
    EXPECT_EQ(by.at("setosa").front(), 5.1);
    EXPECT_THROW((void)iris.split("Species", "Species"), std::invalid_argument);
    EXPECT_THROW((void)iris.split("Sepal.Length", "Sepal.Width"), std::invalid_argument);
    const auto plt = datamunge::graphics::RPlot::boxplot(iris, "Sepal.Length", "Species");
    EXPECT_EQ(plt.series().size(), 3u);
    EXPECT_EQ(plt.x_tick_label_list().at(2), "virginica");
}
