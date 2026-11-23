#include <datamunge/stats/power.hpp>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace {

using namespace datamunge::stats;

TEST(StudyPower, TTestPowerMatchesBothTailsAndPooledUnequalAllocation) {
  EXPECT_NEAR(one_sample_t_test_power(20, .5), .56450441843908394, 1e-10);
  EXPECT_NEAR(paired_t_test_power(20, .5, .05, Alternative::Greater), .69514933824465752, 1e-10);
  EXPECT_NEAR(pooled_t_test_power(20, 20, .5), .33793902892509031, 1e-10);
  EXPECT_NEAR(pooled_t_test_power(18, 27, .6), .48717719322026232, 1e-10);
  EXPECT_DOUBLE_EQ(one_sample_t_test_power(8, 0, .1), .1);
  EXPECT_DOUBLE_EQ(one_sample_t_test_power(25, -.7), one_sample_t_test_power(25, .7));
  const auto wrong_direction = one_sample_t_test_power(20, .5, .05, Alternative::Less);
  EXPECT_LT(wrong_direction, .05);
  EXPECT_DOUBLE_EQ(wrong_direction, one_sample_t_test_power(20, -.5, .05, Alternative::Greater));
  EXPECT_THROW(pooled_t_test_power(2, 1, .5), std::invalid_argument);
  EXPECT_THROW(one_sample_t_test_power(20, .5, 1), std::invalid_argument);
}

TEST(StudyPower, TTestSampleSizeFindsTheFirstAdmissibleInteger) {
  const auto one = one_sample_t_test_sample_size(.5, .8);
  EXPECT_EQ(one, 34u);
  EXPECT_GE(one_sample_t_test_power(one, .5), .8);
  EXPECT_LT(one_sample_t_test_power(one - 1, .5), .8);
  EXPECT_EQ(paired_t_test_sample_size(-.5, .8, .05, Alternative::Less), 27u);
  const auto balanced = pooled_t_test_sample_size(.5, .8);
  EXPECT_EQ(balanced.first, 64u);
  EXPECT_EQ(balanced.second, 64u);
  const auto [first, second] = pooled_t_test_sample_size(.6, .8, {4, 6});
  EXPECT_EQ(first, 38u);
  EXPECT_EQ(second, 57u);
  EXPECT_GE(pooled_t_test_power(first, second, .6), .8);
  EXPECT_LT(pooled_t_test_power(first - 2, second - 3, .6), .8);
  EXPECT_EQ(one_sample_t_test_sample_size(0, .05), 2u);
  EXPECT_THROW(one_sample_t_test_sample_size(0, .8), std::domain_error);
  EXPECT_THROW(paired_t_test_sample_size(-.5, .8, .05, Alternative::Greater), std::domain_error);
  EXPECT_THROW(pooled_t_test_sample_size(.5, .8, {0, 2}), std::invalid_argument);
  EXPECT_THROW(pooled_t_test_sample_size(.5, .8, {1, std::numeric_limits<std::size_t>::max()}),
               std::overflow_error);
}

TEST(StudyPower, FixedEffectsAnovaMatchesAllocationWeightedNoncentralF) {
  EXPECT_NEAR(one_way_anova_power({10, 11, 14}, {10, 18, 27}, 16), .79468034988169578, 1e-9);
  EXPECT_NEAR(one_way_anova_power({0, 1, 2, 3}, {5, 5, 5, 5}, 5), .35355942381530248, 1e-9);
  EXPECT_NEAR(one_way_anova_power({.6, 0}, {18, 27}, 1), pooled_t_test_power(18, 27, .6), 1e-12);
  const auto translated = one_way_anova_power({1e14 + 10, 1e14 + 11, 1e14 + 14}, {10, 18, 27}, 16);
  EXPECT_NEAR(translated, one_way_anova_power({10, 11, 14}, {10, 18, 27}, 16), 1e-14);
  EXPECT_DOUBLE_EQ(one_way_anova_power({2, 2}, {1, 4}, 2, .1), .1);
  EXPECT_DOUBLE_EQ(one_way_anova_power({-1e300, 1e300}, {10, 10}, 1), 1);
  EXPECT_GT(one_way_anova_power({1, 5}, {1, 4}, 2), .05);
  EXPECT_THROW(one_way_anova_power({1, 2}, {1, 1}, 1), std::invalid_argument);
  EXPECT_THROW(one_way_anova_power({1, 2}, {0, 5}, 1), std::invalid_argument);
  EXPECT_THROW(one_way_anova_power({1, 2}, {5, 5}, 0), std::invalid_argument);
}

TEST(StudyPower, TwoProportionsUseSpecifiedNullAndAlternativeVariances) {
  const std::pair null{.35, .25}, changed{.50, .25};
  EXPECT_NEAR(two_proportion_power(80, 120, null, changed), .61435608185312396, 1e-13);
  EXPECT_NEAR(two_proportion_power(80, 120, null, changed, .05, Alternative::Greater),
              .72446097509562379, 1e-13);
  EXPECT_NEAR(two_proportion_power(100, 100, {.4, .4}, {.6, .4}), .82298215348488801, 1e-13);
  EXPECT_NEAR(two_proportion_power(25, 40, {.1, .1}, {.35, .35}), .21766520132510678, 1e-13);
  EXPECT_DOUBLE_EQ(two_proportion_power(120, 80, {.25, .35}, {.25, .5}, .05, Alternative::Less),
                   two_proportion_power(80, 120, null, changed, .05, Alternative::Greater));
  EXPECT_DOUBLE_EQ(two_proportion_power(25, 40, null, null), .05);
  EXPECT_DOUBLE_EQ(two_proportion_power(1, 1, {.5, .5}, {1, 0}), 0);
  EXPECT_DOUBLE_EQ(two_proportion_power(2, 2, {.5, .5}, {1, 0}), 1);
  EXPECT_GT(two_proportion_power(10, 20, {0, .4}, {0, .7}, .05, Alternative::Less), .05);
}

TEST(StudyPower, TwoProportionSampleSizeFindsTheFirstAllocation) {
  const std::pair null{.35, .25}, changed{.50, .25};
  const auto [first, second] = two_proportion_sample_size(null, changed, .8, {4, 6});
  EXPECT_EQ(first, 126u);
  EXPECT_EQ(second, 189u);
  EXPECT_GE(two_proportion_power(first, second, null, changed), .8);
  EXPECT_LT(two_proportion_power(first - 2, second - 3, null, changed), .8);
  EXPECT_EQ(two_proportion_sample_size(null, null, .05, {2, 3}), (std::pair<std::size_t, std::size_t>{2, 3}));
  EXPECT_THROW(two_proportion_sample_size(null, changed, .8, {2, 3}, .05, Alternative::Less), std::domain_error);
  EXPECT_THROW(two_proportion_sample_size({.5, .5}, {.5000000000000001, .5}, .8), std::overflow_error);
  EXPECT_THROW(two_proportion_sample_size(null, changed, .8, {0, 3}), std::invalid_argument);
  EXPECT_THROW(two_proportion_power(0, 3, null, changed), std::invalid_argument);
  EXPECT_THROW(two_proportion_power(3, 3, {0, 1}, changed), std::invalid_argument);
  EXPECT_THROW(two_proportion_power(3, 3, null, {-.1, .25}), std::invalid_argument);
}

TEST(StudyPower, SchoenfeldCountsMatchTheProportionalHazardsInformationFormula) {
  EXPECT_EQ(schoenfeld_event_count(.75, .5, .8), 380u);
  EXPECT_EQ(schoenfeld_event_count(.75, .4, .8), 396u);
  EXPECT_EQ(schoenfeld_event_count(1 / .75, .6, .8), 396u);
  EXPECT_EQ(schoenfeld_event_count(.75, .4, .9, .025, Alternative::Less), 530u);
  EXPECT_EQ(schoenfeld_event_count(1.25, .6, .82, .05, Alternative::Greater), 549u);
  EXPECT_EQ(schoenfeld_event_count(1, .5, .05), 0u);
  EXPECT_THROW(schoenfeld_event_count(1, .5, .8), std::domain_error);
  EXPECT_THROW(schoenfeld_event_count(.75, .5, .8, .05, Alternative::Greater), std::domain_error);
  EXPECT_THROW(schoenfeld_event_count(.75, 0, .8), std::invalid_argument);
  EXPECT_THROW(schoenfeld_event_count(1.0000000000000002, .5, .8), std::overflow_error);
}

TEST(StudyPower, ClusterPowerMatchesBlockCovarianceForUnequalSizes) {
  const std::vector<std::size_t> first{5, 10, 15, 25, 30, 35}, second{18, 20, 22, 18, 20, 22};
  const auto means = cluster_randomized_mean_power(first, second, .05, .5);
  EXPECT_NEAR(means.first.design_effect, 2.24166666666666670, 1e-14);
  EXPECT_NEAR(means.first.effective_sample_size, 53.53159851301115424, 1e-13);
  EXPECT_NEAR(means.second.design_effect, 1.95666666666666678, 1e-14);
  EXPECT_NEAR(means.second.effective_sample_size, 61.32879045996592282, 1e-13);
  EXPECT_NEAR(means.power, .76213426205944612, 1e-13);
  const auto direct = cluster_randomized_proportion_power(first, second, .05, {.35, .25}, {.5, .25});
  EXPECT_NEAR(direct.power, .42107062077756235, 1e-13);
  EXPECT_DOUBLE_EQ(direct.first.design_effect, means.first.design_effect);
  EXPECT_DOUBLE_EQ(direct.second.effective_sample_size, means.second.effective_sample_size);
  const auto greater = cluster_randomized_mean_power(first, second, .05, .5, .05, Alternative::Greater);
  const auto less = cluster_randomized_mean_power(second, first, .05, -.5, .05, Alternative::Less);
  EXPECT_DOUBLE_EQ(greater.power, less.power);
}

TEST(StudyPower, ClusterDesignEffectsHandleCorrelationLimitsAndIndependence) {
  const auto equal = cluster_design_effect({20, 20, 20}, .05);
  EXPECT_DOUBLE_EQ(equal.design_effect, 1.95);
  EXPECT_GT(cluster_design_effect({10, 20, 30}, .05).design_effect, equal.design_effect);
  EXPECT_DOUBLE_EQ(cluster_design_effect({10, 20, 30}, 0).effective_sample_size, 60);
  EXPECT_DOUBLE_EQ(cluster_design_effect({12, 12, 12}, 1).effective_sample_size, 3);
  EXPECT_DOUBLE_EQ(cluster_design_effect({1, 1, 1}, 1).design_effect, 1);
  const auto largest = std::numeric_limits<std::size_t>::max();
  EXPECT_DOUBLE_EQ(cluster_design_effect({largest, largest}, 1).effective_sample_size, 2);
  const auto independent = cluster_randomized_proportion_power({10, 20}, {8, 16}, 0, {.3, .3}, {.4, .3});
  EXPECT_DOUBLE_EQ(independent.power, two_proportion_power(30, 24, {.3, .3}, {.4, .3}));
  EXPECT_DOUBLE_EQ(cluster_randomized_mean_power({4}, {24, 16}, .3, 0).power, .05);
  EXPECT_THROW(cluster_design_effect({}, .05), std::invalid_argument);
  EXPECT_THROW(cluster_design_effect({0, 4}, .05), std::invalid_argument);
  EXPECT_THROW(cluster_randomized_mean_power({10}, {10}, 1.1, .1), std::invalid_argument);
}

} // namespace
