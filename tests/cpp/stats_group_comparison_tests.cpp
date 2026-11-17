#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/p_adjust.hpp>
#include <datamunge/stats/pairwise_test.hpp>
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
