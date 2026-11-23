#include <datamunge/random/count_distributions.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/random/poisson_binomial.hpp>
#include <datamunge/stats/chi_squared_test.hpp>
#include <datamunge/stats/proportion_test.hpp>
#include <datamunge/stats/robust_summary.hpp>
#include <datamunge/stats/sequential.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <stdexcept>

using namespace datamunge::stats;

TEST(RobustSummary, AllNineQuantileTypesMatchR) {
    const double references[][7]{
        {0, 0, 10, 10, 20, 20, 40}, {0, 5, 10, 15, 20, 30, 40},
        {0, 0, 10, 10, 10, 20, 40}, {0, 0, 5, 10, 15, 20, 32},
        {0, 5, 10, 15, 20, 30, 40}, {0, 2.5, 8.75, 15, 22.5, 35, 40},
        {3.75, 7.5, 11.25, 15, 18.75, 25, 34},
        {0, 25.0 / 6, 115.0 / 12, 15, 125.0 / 6, 95.0 / 3, 40},
        {0, 4.375, 9.6875, 15, 20.625, 31.25, 40}};
    for (int type = 1; type <= 9; ++type) {
        const auto q = sample_quantiles({40, 0, 20, 10}, {.125, .25, .375, .5, .625, .75, .9}, type);
        for (std::size_t i = 0; i < q.size(); ++i)
            EXPECT_NEAR(q[i], references[type - 1][i], 1e-12) << "type " << type << ", index " << i;
        EXPECT_EQ(sample_quantiles({40, 0, 20, 10}, {0, 1}, type), (std::vector<double>{0, 40}));
        EXPECT_DOUBLE_EQ(sample_quantiles({2}, {.5}, type).front(), 2);
    }
    EXPECT_THROW(sample_quantiles({1}, {.5}, 10), std::invalid_argument);
    EXPECT_THROW(sample_quantiles({1}, {1.01}), std::invalid_argument);
}

TEST(RobustSummary, WeightedEmpiricalQuantileInvertsCumulativeMass) {
    const auto q = weighted_empirical_quantiles({100, 30, 10, 20, 10, -100}, {0, 4, 1, 3, 2, 0},
                                                 {1, 0, .3, .30000000000000004, .6, .75});
    EXPECT_EQ(q, (std::vector<double>{30, 10, 10, 20, 20, 30}));
    EXPECT_THROW(weighted_empirical_quantiles({1, 2}, {0, 0}, {.5}), std::invalid_argument);
    EXPECT_THROW(weighted_empirical_quantiles({1, 2}, {1, -1}, {.5}), std::invalid_argument);
    EXPECT_THROW(weighted_empirical_quantiles({1, 2}, {1}, {.5}), std::invalid_argument);
}

TEST(RobustSummary, HarrellDavisQuantilesMatchRBetaWeights) {
    const auto q = harrell_davis_quantiles({63, 9, 2, 15, 4}, {0, .01, .25, .5, .9, .99, 1});
    ASSERT_EQ(q.size(), 7u);
    EXPECT_DOUBLE_EQ(q[0], 2);
    EXPECT_NEAR(q[1], 2.0256459392301274, 1e-11);
    EXPECT_NEAR(q[2], 4.308808437609553, 1e-11);
    EXPECT_NEAR(q[3], 11.98176, 1e-11);
    EXPECT_NEAR(q[4], 55.071768773312314, 1e-11);
    EXPECT_NEAR(q[5], 62.523008185092493, 1e-11);
    EXPECT_DOUBLE_EQ(q[6], 63);
    EXPECT_DOUBLE_EQ(harrell_davis_quantiles({2}, {.1}).front(), 2);
    EXPECT_THROW(harrell_davis_quantiles({2}, {-1}), std::invalid_argument);
}

TEST(RobustSummary, ExactQuantileIntervalHasConservativeBinomialCoverage) {
    const auto median = exact_quantile_interval({5, 1, 4, 2, 3, 0}, .5);
    EXPECT_EQ(median.lower_rank, 1u);
    EXPECT_EQ(median.upper_rank, 6u);
    EXPECT_DOUBLE_EQ(median.lower, 0);
    EXPECT_DOUBLE_EQ(median.upper, 5);
    EXPECT_DOUBLE_EQ(median.coverage, 1 - 2.0 / 64);
    std::vector<double> values(30);
    std::iota(values.begin(), values.end(), 1);
    const auto upper = exact_quantile_interval(values, .9);
    EXPECT_EQ(upper.lower_rank, 23u);
    EXPECT_EQ(upper.upper_rank, 31u);
    EXPECT_EQ(upper.upper, std::numeric_limits<double>::infinity());
    EXPECT_NEAR(upper.coverage, .9922163807620787, 1e-12);
    const auto tiny = exact_quantile_interval({12}, .5);
    EXPECT_EQ(tiny.lower, -std::numeric_limits<double>::infinity());
    EXPECT_EQ(tiny.upper, std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(tiny.coverage, 1);
    EXPECT_THROW(exact_quantile_interval({1, 2}, 1), std::invalid_argument);
}
TEST(RobustSummary, MedianAbsoluteDeviationWithEvenAndOddSampleSizes) {
    EXPECT_DOUBLE_EQ(median_absolute_deviation({100, 3, 1, 2}), 1.4826);
    EXPECT_DOUBLE_EQ(median_absolute_deviation({100, 3, 1, 2}, 0, 1), 2.5);
    EXPECT_DOUBLE_EQ(median_absolute_deviation({9, 1, 3}, {}, 1), 2);
    EXPECT_DOUBLE_EQ(median_absolute_deviation({9}), 0);
    EXPECT_THROW(median_absolute_deviation({}), std::invalid_argument);
    EXPECT_THROW(median_absolute_deviation({1, 2}, 0, -1), std::invalid_argument);
}

TEST(RobustSummary, EmpiricalCdfBandHandlesTiesAndRightContinuity) {
    const EmpiricalCdfBand band({7, 1, 1, 3, 4, 3, 4, 5}, .8);
    ASSERT_EQ(band.knots().size(), 5u);
    EXPECT_DOUBLE_EQ(band.evaluate(0).cdf, 0);
    EXPECT_DOUBLE_EQ(band.evaluate(1).cdf, .25);
    EXPECT_DOUBLE_EQ(band.evaluate(2).cdf, .25);
    const auto point = band.evaluate(3);
    const auto radius = std::sqrt(std::log(10.) / 16);
    EXPECT_DOUBLE_EQ(point.cdf, .5);
    EXPECT_NEAR(point.lower, .5 - radius, 1e-15);
    EXPECT_NEAR(point.upper, .5 + radius, 1e-15);
    EXPECT_EQ(band.evaluate(std::numeric_limits<double>::infinity()).upper, 1);
    EXPECT_THROW(static_cast<void>(band.evaluate(std::numeric_limits<double>::quiet_NaN())),
                 std::invalid_argument);
    EXPECT_THROW(EmpiricalCdfBand({}), std::invalid_argument);
}

TEST(CountQuantiles, MatchRAndHandleSupportEndpoints) {
    using datamunge::random::binomial_quantile;
    using datamunge::random::hypergeometric_quantile;
    EXPECT_EQ(binomial_quantile(1e-35, 1000, .4), 215u);
    EXPECT_EQ(binomial_quantile(1e-35, 1000, .7), 512u);
    EXPECT_EQ(binomial_quantile(.5, 1000, .7), 700u);
    EXPECT_EQ(binomial_quantile(.999999999999, 1000, .7), 798u);
    EXPECT_EQ(binomial_quantile(.25, 2, .5), 0u);
    EXPECT_EQ(binomial_quantile(0, 5, 1), 5u);
    EXPECT_EQ(binomial_quantile(1, 5, 0), 0u);
    EXPECT_EQ(binomial_quantile(.5, 0, .5), 0u);
    if constexpr (std::numeric_limits<std::size_t>::digits > 60) {
        const auto n = static_cast<std::size_t>(1ULL << 60);
        const auto tiny_p = std::ldexp(1.0, -60);
        EXPECT_EQ(binomial_quantile(.5, n, tiny_p), 1u);
        EXPECT_EQ(binomial_quantile(.95, n, tiny_p), 3u);
        EXPECT_EQ(binomial_quantile(.999999999999, n, tiny_p), 14u);
    }
    EXPECT_THROW(binomial_quantile(-.1, 5, .5), std::invalid_argument);

    EXPECT_EQ(hypergeometric_quantile(0, 120, 50, 80), 10u);
    EXPECT_EQ(hypergeometric_quantile(1e-12, 120, 50, 80), 16u);
    EXPECT_EQ(hypergeometric_quantile(.5, 120, 50, 80), 33u);
    EXPECT_EQ(hypergeometric_quantile(.99, 120, 50, 80), 39u);
    EXPECT_EQ(hypergeometric_quantile(1, 120, 50, 80), 50u);
    EXPECT_EQ(hypergeometric_quantile(1e-30, 10000, 4000, 2000), 579u);
    EXPECT_EQ(hypergeometric_quantile(.5, 10000, 4000, 2000), 800u);
    EXPECT_EQ(hypergeometric_quantile(.5, 20, 7, 5), 2u);
    EXPECT_EQ(hypergeometric_quantile(.5, 0, 0, 0), 0u);
    EXPECT_THROW(hypergeometric_quantile(.5, 8, 9, 2), std::invalid_argument);
}

TEST(ContinuousDistributions, GammaMatchesRAcrossShapesAndTails) {
    using namespace datamunge::random;
    EXPECT_NEAR(gamma_pdf(3.25, .4, 1.1), .01114749642710071, 1e-15);
    EXPECT_NEAR(gamma_cdf(.5, 1e-100, 1, false) / 5.5977359477616092e-101, 1, 1e-13);
    EXPECT_NEAR(gamma_cdf(1e6, 1e6), .50013298076087254, 1e-12);
    EXPECT_NEAR(gamma_cdf(1005000, 1e6, 1, false) / 2.9874901401146387e-7, 1, 1e-10);
    EXPECT_NEAR(gamma_quantile(1e-30, .8, 3) / 8.6802470827539162e-38, 1, 1e-13);
    EXPECT_NEAR(gamma_quantile(1e-100, 3, 2, false), 481.07888968819827, 1e-10);
    EXPECT_NEAR(gamma_quantile(.995, 5e5), 501823.26462924038, 2e-6);
    EXPECT_DOUBLE_EQ(gamma_cdf(0, 1), 0);
    EXPECT_DOUBLE_EQ(gamma_cdf(std::numeric_limits<double>::infinity(), 1), 1);
    EXPECT_DOUBLE_EQ(gamma_quantile(1, 1, 1, false), 0);
    EXPECT_EQ(gamma_quantile(0, 1, 1, false), std::numeric_limits<double>::infinity());
    EXPECT_THROW(gamma_cdf(1, 0), std::invalid_argument);
}

TEST(CountDistributions, PoissonBinomialMatchesEnumeratedBernoulliTrials) {
    const datamunge::random::PoissonBinomial count({0, 1, .1, .4, .9});
    EXPECT_DOUBLE_EQ(count.pmf(0), 0);
    EXPECT_NEAR(count.pmf(1), .054, 1e-15);
    EXPECT_NEAR(count.pmf(2), .528, 1e-15);
    EXPECT_NEAR(count.pmf(3), .382, 1e-15);
    EXPECT_NEAR(count.pmf(4), .036, 1e-15);
    EXPECT_DOUBLE_EQ(count.cdf(0), 0);
    EXPECT_NEAR(count.cdf(2), .582, 1e-15);
    EXPECT_NEAR(count.cdf(2, false), .418, 1e-15);
    EXPECT_DOUBLE_EQ(count.cdf(4, false), 0);
    EXPECT_EQ(count.quantile(0), 1u);
    EXPECT_EQ(count.quantile(.5), 2u);
    EXPECT_EQ(count.quantile(.95), 3u);
    EXPECT_EQ(count.quantile(1), 4u);
    const datamunge::random::PoissonBinomial rare({1e-40, 1e-40});
    EXPECT_NEAR(rare.cdf(1, false) / 1e-80, 1, 1e-14);
    EXPECT_DOUBLE_EQ(datamunge::random::PoissonBinomial({}).pmf(0), 1);
    EXPECT_THROW(datamunge::random::PoissonBinomial({1.1}), std::invalid_argument);
}

TEST(CountDistributions, NegativeBinomialMatchesRMeanPrecisionParameterization) {
    using namespace datamunge::random;
    EXPECT_NEAR(negative_binomial_pmf(8, 7.3, 2.5), .065877785492543903, 1e-14);
    EXPECT_NEAR(negative_binomial_cdf(8, 7.3, 2.5), .66545589923774506, 1e-14);
    EXPECT_NEAR(negative_binomial_cdf(1024, 7.3, .7, false) / 3.4532539356970574e-42, 1, 1e-12);
    EXPECT_EQ(negative_binomial_quantile(.95, 7.3, 2.5), 18u);
    EXPECT_EQ(negative_binomial_quantile(.75, 1, 1), 1u);
    EXPECT_DOUBLE_EQ(negative_binomial_pmf(0, 0, 3), 1);
    EXPECT_DOUBLE_EQ(negative_binomial_cdf(0, 0, 3, false), 0);
    EXPECT_EQ(negative_binomial_quantile(1, 0, 3), 0u);
    EXPECT_EQ(negative_binomial_quantile(1, 1, 3), std::numeric_limits<std::size_t>::max());
    EXPECT_NEAR(negative_binomial_cdf(8, 7.3, 1e30), gamma_cdf(7.3, 9, 1, false), 1e-15);
    EXPECT_NEAR(negative_binomial_cdf(0, 7, 1e-100, false) / 2.3220441944845988e-98, 1, 1e-14);
    EXPECT_NEAR(negative_binomial_cdf(8, 7, 1e-100, false) / 2.2948656230560276e-98, 1, 1e-14);
    EXPECT_NEAR(negative_binomial_cdf(300000000, 1e-8, 1e-16, false) / 1.3048381260153831e-18, 1, 1e-13);
    EXPECT_NEAR(negative_binomial_pmf(5, 1e-20, 1) / 1e-100, 1, 1e-14);
    EXPECT_NEAR(negative_binomial_cdf(100000000, 1e8, 1e8), .50002820947914206, 1e-10);
    if constexpr (std::numeric_limits<std::size_t>::digits >= 48) {
        constexpr auto center = static_cast<std::size_t>(1e14);
        EXPECT_NEAR(negative_binomial_cdf(center, 1e14, 1e14), .5000000282094792, 1e-14);
        EXPECT_NEAR(negative_binomial_pmf(center, 1e14, 1e14) / 2.8209479177387786e-8, 1, 1e-14);
        EXPECT_EQ(negative_binomial_quantile(.5, 1e14, 1e14), center - 1);
    }
    EXPECT_THROW(negative_binomial_cdf(1, -1, 2), std::invalid_argument);
}

TEST(CountDistributions, BetaBinomialAgreesWithRBetaMixtureMasses) {
    using namespace datamunge::random;
    EXPECT_NEAR(beta_binomial_pmf(6, 27, .3, 2.5), .030028049798311359, 1e-14);
    EXPECT_NEAR(beta_binomial_cdf(6, 27, .3, 2.5), .838176837837557698, 1e-14);
    EXPECT_NEAR(beta_binomial_cdf(6, 27, .3, 2.5, false), .161823162162442247, 1e-14);
    EXPECT_EQ(beta_binomial_quantile(.9, 27, .3, 2.5), 9u);
    EXPECT_EQ(beta_binomial_quantile(.5, 3, 1, 1), 1u);
    EXPECT_EQ(beta_binomial_quantile(.75, 3, 1, 1), 2u);
    EXPECT_EQ(beta_binomial_quantile(1, 27, .3, 2.5), 27u);
    EXPECT_DOUBLE_EQ(beta_binomial_pmf(0, 0, .3, 2.5), 1);
    EXPECT_DOUBLE_EQ(beta_binomial_pmf(4, 3, 1, 1), 0);
    EXPECT_NEAR(beta_binomial_pmf(1000, 2000, 1e100, 1e100), binomial_pdf(1000, 2000, .5), 1e-12);
    EXPECT_NEAR(beta_binomial_cdf(1000, 2000, 1e100, 1e100),
                .5 + .5 * binomial_pdf(1000, 2000, .5), 1e-12);
    EXPECT_THROW(beta_binomial_cdf(1, 3, 0, 1), std::invalid_argument);
}

TEST(CountIntervals, NewcombeWilsonCombinationAndPoissonIntervalsMatchR) {
    const auto difference = newcombe_difference_interval(2, 11, 8, 17);
    EXPECT_NEAR(difference.difference, -.288770053475935817, 1e-15);
    EXPECT_NEAR(difference.lower, -.544348459736240908, 1e-9);
    EXPECT_NEAR(difference.upper, .072858999591777163, 1e-9);
    const auto reversed = newcombe_difference_interval(8, 17, 2, 11);
    EXPECT_DOUBLE_EQ(reversed.lower, -difference.upper);
    EXPECT_DOUBLE_EQ(reversed.upper, -difference.lower);
    const auto zeros = newcombe_difference_interval(0, 10, 0, 10);
    EXPECT_LT(zeros.lower, 0);
    EXPECT_GT(zeros.upper, 0);
    EXPECT_THROW(newcombe_difference_interval(11, 10, 0, 10), std::invalid_argument);

    const auto rate = poisson_rate_interval(10, 22);
    ASSERT_TRUE(rate.estimate);
    EXPECT_DOUBLE_EQ(*rate.estimate, 10. / 22);
    EXPECT_NEAR(rate.lower, .2179722134605652, 1e-12);
    EXPECT_NEAR(rate.upper, .8359252746371717, 1e-12);
    const auto ratio = poisson_rate_ratio_interval(10, 22, 5, 35);
    ASSERT_TRUE(ratio.estimate);
    EXPECT_NEAR(*ratio.estimate, 35. / 11, 1e-14);
    EXPECT_NEAR(ratio.lower, .99091292737397729, 1e-12);
    EXPECT_NEAR(ratio.upper, 11.86387985818238, 1e-12);
    const auto no_first = poisson_rate_ratio_interval(0, 22, 5, 35);
    EXPECT_NEAR(no_first.upper, 1.73612584915405, 1e-12);
    const auto no_second = poisson_rate_ratio_interval(5, 22, 0, 35);
    EXPECT_NEAR(no_second.lower, 1.45783886391096, 1e-12);
    EXPECT_EQ(no_second.upper, std::numeric_limits<double>::infinity());
    EXPECT_FALSE(poisson_rate_ratio_interval(0, 22, 0, 35).estimate);
    EXPECT_THROW(poisson_rate_interval(3, 0), std::invalid_argument);
}

TEST(CountIntervals, RiskRatioScoreInvertsConstrainedBinomialFits) {
    const auto ratio = risk_ratio_score_interval(2, 11, 8, 17);
    ASSERT_TRUE(ratio.estimate);
    EXPECT_DOUBLE_EQ(*ratio.estimate, 17. / 44);
    EXPECT_NEAR(ratio.lower, .1019464147728496, 1e-9);
    EXPECT_NEAR(ratio.upper, 1.2292134867073266, 1e-8);
    const auto reversed = risk_ratio_score_interval(8, 17, 2, 11);
    EXPECT_DOUBLE_EQ(reversed.lower, 1 / ratio.upper);
    EXPECT_DOUBLE_EQ(reversed.upper, 1 / ratio.lower);
    const auto all = risk_ratio_score_interval(20, 20, 15, 15);
    EXPECT_NEAR(all.lower, .8388748419471922, 1e-9);
    EXPECT_NEAR(all.upper, 1.2560972547129416, 1e-8);
    const auto different_confidence = risk_ratio_score_interval(5, 20, 2, 37, .90);
    EXPECT_NEAR(different_confidence.lower, 1.373987064823162, 1e-8);
    EXPECT_NEAR(different_confidence.upper, 15.794674804132546, 1e-7);
    const auto high = risk_ratio_score_interval(9, 10, 1, 100);
    EXPECT_NEAR(high.lower, 16.24326073641212, 1e-7);
    EXPECT_NEAR(high.upper, 512.87500149759887, 3e-6);
}

TEST(CountIntervals, RiskRatioScoreHandlesZeroEvents) {
    constexpr auto inf = std::numeric_limits<double>::infinity();
    const auto no_first = risk_ratio_score_interval(0, 20, 4, 20);
    ASSERT_TRUE(no_first.estimate);
    EXPECT_DOUBLE_EQ(*no_first.estimate, 0);
    EXPECT_DOUBLE_EQ(no_first.lower, 0);
    EXPECT_NEAR(no_first.upper, .8703403479684769, 2e-9);
    const auto no_second = risk_ratio_score_interval(4, 20, 0, 20);
    ASSERT_TRUE(no_second.estimate);
    EXPECT_EQ(*no_second.estimate, inf);
    EXPECT_DOUBLE_EQ(no_second.lower, 1 / no_first.upper);
    EXPECT_EQ(no_second.upper, inf);
    const auto none = risk_ratio_score_interval(0, 5, 0, 20);
    EXPECT_FALSE(none.estimate);
    EXPECT_DOUBLE_EQ(none.lower, 0);
    EXPECT_EQ(none.upper, inf);
    EXPECT_THROW(risk_ratio_score_interval(0, 0, 1, 2), std::invalid_argument);
    EXPECT_THROW(risk_ratio_score_interval(2, 1, 0, 3), std::invalid_argument);
    EXPECT_THROW(risk_ratio_score_interval(1, 4, 1, 4, 1), std::invalid_argument);
}

TEST(CategoricalInference, PowerDivergenceHasCorrectLimitsAndZeroHandling) {
    const std::vector<double> counts{25, 15}, probabilities{.5, .5};
    EXPECT_NEAR(power_divergence_test(counts, probabilities, 0).statistic,
                2.526715392157062, 1e-14);
    const auto pearson = chi_squared_goodness_of_fit(counts, probabilities);
    const auto power_one = power_divergence_test(counts, probabilities, 1);
    EXPECT_DOUBLE_EQ(power_one.statistic, pearson.statistic);
    EXPECT_DOUBLE_EQ(power_one.p_value, pearson.p_value);
    const auto reverse = power_divergence_test(counts, probabilities, -1).statistic;
    EXPECT_NEAR(reverse, 40 * (std::log(20. / 25) + std::log(20. / 15)), 1e-14);
    EXPECT_NEAR(power_divergence_test(counts, probabilities, -1 - 1e-9).statistic, reverse, 1e-8);
    EXPECT_NEAR(power_divergence_test({0, 20}, {}, -.5).statistic, 8 * (20 - std::sqrt(200)), 1e-13);
    EXPECT_NEAR(power_divergence_test({0, 20}).statistic, 40 * std::log(2.), 1e-13);
    EXPECT_NEAR(power_divergence_test({1e15 + 1, 1e15 - 1}).statistic, 2e-15, 1e-25);
    EXPECT_EQ(power_divergence_test({0, 20}, {}, -1).statistic, std::numeric_limits<double>::infinity());
    EXPECT_EQ(power_divergence_test({10, 20, 30}, {}, 0, 1).parameter1, 1);
    EXPECT_THROW(power_divergence_test(counts, {}, 0, 1), std::invalid_argument);
}

TEST(CategoricalInference, CochranArmitageMatchesRAndTracksDirection) {
    const auto simple = cochran_armitage_test({2, 5, 8}, {10, 10, 10});
    EXPECT_NEAR(simple.statistic, 2.683281572999748, 1e-13);
    EXPECT_NEAR(simple.p_value, .007290358091535639, 1e-13);
    const auto fit = cochran_armitage_test({2, 6, 9}, {10, 11, 15}, {-2, 1, 7});
    EXPECT_NEAR(fit.statistic, 1.711173562204016, 1e-13);
    EXPECT_NEAR(fit.p_value, .08704907761347239, 1e-13);
    const auto negative = cochran_armitage_test({2, 6, 9}, {10, 11, 15},
                                                {2, -1, -7}, Alternative::Less);
    EXPECT_DOUBLE_EQ(negative.statistic, -fit.statistic);
    EXPECT_DOUBLE_EQ(negative.p_value, .5 * fit.p_value);
    EXPECT_THROW(cochran_armitage_test({0, 0}, {2, 3}), std::invalid_argument);
    EXPECT_THROW(cochran_armitage_test({1, 1}, {2, 3}, {1, 1}), std::invalid_argument);
}

TEST(RobustSummary, TrimAndWinsorizeWithOriginalVarianceDenominator) {
    const auto result = trimmed_summary({100, 2, 3, 0, 4, 1}, .2);
    EXPECT_DOUBLE_EQ(result.trimmed_mean, 2.5);
    EXPECT_DOUBLE_EQ(result.winsorized_mean, 2.5);
    EXPECT_DOUBLE_EQ(result.winsorized_variance, 1.9);
    const auto untrimmed = trimmed_summary({4, 1, 1}, 0);
    EXPECT_DOUBLE_EQ(untrimmed.trimmed_mean, 2);
    EXPECT_DOUBLE_EQ(untrimmed.winsorized_mean, 2);
    EXPECT_DOUBLE_EQ(untrimmed.winsorized_variance, 3);
    EXPECT_THROW(trimmed_summary({1, 2}, .5), std::invalid_argument);
    EXPECT_THROW(trimmed_summary({1, std::numeric_limits<double>::infinity()}, 0), std::invalid_argument);
}

TEST(RobustSummary, HuberLocationAndScaleMatchTheCalibratedFixedPoint) {
    const std::vector<double> x{-2, -1, 0, 1, 2, 100};
    const auto fit = huber_location_scale(x, 1.5, 1e-12, 500);
    EXPECT_TRUE(fit.converged);
    EXPECT_NEAR(fit.location, .86880784309926862, 1e-10);
    EXPECT_NEAR(fit.scale, 2.89602614366422895, 1e-10);
    EXPECT_GT(fit.iterations, 1u);
    EXPECT_FALSE(huber_location_scale(x, 1.5, 1e-12, 1).converged);
    const auto degenerate = huber_location_scale({5, 5, 5, 500});
    EXPECT_DOUBLE_EQ(degenerate.location, 5);
    EXPECT_DOUBLE_EQ(degenerate.scale, 0);
    EXPECT_FALSE(degenerate.converged);
    EXPECT_THROW(huber_location_scale(x, 0), std::invalid_argument);
}

TEST(RobustSummary, QnAndSnMatchEnumeratedPairwiseDistances) {
    for (const auto& sample : {std::vector<double>{0, 11}, {0, 0, 1, 8},
                              {3, 1, 19, -1, 8}, {-7, 1, -2, -1, -3, 2},
                              {5, 4, 8, -2, 5, 4, 5, 1, 0}}) {
        std::vector<double> distinct_pairs, inner;
        for (std::size_t i = 0; i < sample.size(); ++i) {
            std::vector<double> distances;
            for (std::size_t j = 0; j < sample.size(); ++j) {
                const double distance = std::abs(sample[i] - sample[j]);
                distances.push_back(distance);
                if (i < j)
                    distinct_pairs.push_back(distance);
            }
            std::sort(distances.begin(), distances.end());
            inner.push_back(distances[sample.size() / 2]);
        }
        std::sort(distinct_pairs.begin(), distinct_pairs.end());
        std::sort(inner.begin(), inner.end());
        const auto h = sample.size() / 2 + 1;
        EXPECT_NEAR(qn_scale(sample, false), 2.21914 * distinct_pairs[h * (h - 1) / 2 - 1], 1e-14);
        EXPECT_NEAR(sn_scale(sample, false), 1.1926 * inner[(sample.size() - 1) / 2], 1e-14);
    }
    EXPECT_NEAR(qn_scale({1, 2, 3, 100}), 2.21914 * 2 * .51321, 1e-14);
    EXPECT_NEAR(sn_scale({1, 2, 3, 100}), 1.1926 * 2 * .954, 1e-14);
    EXPECT_DOUBLE_EQ(qn_scale({5, 5, 5, 500}), 0);
    EXPECT_DOUBLE_EQ(sn_scale({5, 5, 5, 500}), 0);
    EXPECT_DOUBLE_EQ(qn_scale({1}), 0);
    EXPECT_DOUBLE_EQ(sn_scale({1}), 0);
}

TEST(RobustSummary, EmpiricalLikelihoodMatchesClosedFormAndConstrainsWeights) {
    const auto fit = empirical_likelihood_mean_test({0, 2}, .5);
    ASSERT_EQ(fit.weights.size(), 2u);
    EXPECT_NEAR(fit.weights[0], .75, 1e-15);
    EXPECT_NEAR(fit.weights[1], .25, 1e-15);
    EXPECT_NEAR(fit.statistic, -2 * std::log(.75), 1e-14);

    const std::vector<double> data{1, 1, 4, 5, 9};
    const auto constrained = empirical_likelihood_mean_test(data, 6);
    ASSERT_EQ(constrained.weights.size(), data.size());
    EXPECT_NEAR(std::accumulate(constrained.weights.begin(), constrained.weights.end(), 0.),
                1, 1e-14);
    EXPECT_NEAR(std::inner_product(data.begin(), data.end(), constrained.weights.begin(), 0.),
                6, 1e-14);
    const auto boundary = empirical_likelihood_mean_test(data, 9);
    EXPECT_EQ(boundary.statistic, std::numeric_limits<double>::infinity());
    EXPECT_EQ(boundary.p_value, 0);
    EXPECT_TRUE(boundary.weights.empty());

    const auto interval = empirical_likelihood_mean_interval({0, 2});
    EXPECT_DOUBLE_EQ(interval.estimate, 1);
    EXPECT_NEAR(interval.cutoff, 3.841458820694124, 1e-12);
    const auto radius = std::sqrt(1 - std::exp(-.5 * interval.cutoff));
    EXPECT_NEAR(interval.lower, 1 - radius, 1e-14);
    EXPECT_NEAR(interval.upper, 1 + radius, 1e-14);
    EXPECT_THROW(empirical_likelihood_mean_test({2, 2}, 2), std::invalid_argument);
}

TEST(SequentialInference, WaldBoundariesAreAbsorbing) {
    SequentialProbabilityRatioTest sprt(.05, .2);
    EXPECT_NEAR(sprt.lower_boundary(), std::log(.2 / .95), 1e-15);
    EXPECT_NEAR(sprt.upper_boundary(), std::log(.8 / .05), 1e-15);
    EXPECT_EQ(sprt.observe(1), SequentialDecision::Continue);
    EXPECT_EQ(sprt.observe(1), SequentialDecision::Continue);
    EXPECT_EQ(sprt.observe(1), SequentialDecision::AcceptAlternative);
    EXPECT_EQ(sprt.observe(-100), SequentialDecision::AcceptAlternative);
    EXPECT_DOUBLE_EQ(sprt.log_likelihood_ratio(), 3);
    EXPECT_EQ(sprt.decision(), SequentialDecision::AcceptAlternative);
    SequentialProbabilityRatioTest lower(.05, .2);
    EXPECT_EQ(lower.observe(-std::numeric_limits<double>::infinity()), SequentialDecision::AcceptNull);
    EXPECT_THROW(lower.observe(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(SequentialProbabilityRatioTest(.9, .2), std::invalid_argument);
}

TEST(SequentialInference, BoundedMeanConfidenceSequenceMatchesMixtureBoundary) {
    BoundedMeanConfidenceSequence sequence(2, 4, .95, 3);
    EXPECT_FALSE(sequence.current().estimate);
    EXPECT_EQ(sequence.current().lower, 2);
    EXPECT_EQ(sequence.current().upper, 4);
    for (int i = 0; i < 100; ++i)
        sequence.observe(i % 2 == 0 ? 2 : 4);
    const auto bounds = sequence.current();
    ASSERT_TRUE(bounds.estimate);
    EXPECT_DOUBLE_EQ(*bounds.estimate, 3);
    EXPECT_EQ(bounds.size, 100u);
    const auto radius = 2 * std::sqrt(28 * (std::log(28. / 3) + 2 * std::log(20.))) / 100;
    EXPECT_NEAR(bounds.lower, 3 - radius, 1e-14);
    EXPECT_NEAR(bounds.upper, 3 + radius, 1e-14);
    EXPECT_THROW(sequence.observe(5), std::invalid_argument);
    EXPECT_EQ(sequence.current().size, 100u);
    EXPECT_THROW(BoundedMeanConfidenceSequence(4, 4), std::invalid_argument);
}
