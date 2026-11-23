#include <datamunge/stats/sample_moments.hpp>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

using namespace datamunge::stats;

TEST(SampleMoments, AnalyticCovarianceMatchesRAndFrequencyMatchesExpandedRows) {
    const datamunge::linalg::DenseMatrix<double> x(3, 2, {1, 2, 3, 6, 99, 99});
    const auto analytic = weighted_sample_moments(x, {2, 1, 0});
    EXPECT_NEAR(analytic.mean[0], 5.0 / 3, 1e-14);
    EXPECT_NEAR(analytic.mean[1], 10.0 / 3, 1e-14);
    EXPECT_DOUBLE_EQ(analytic.covariance(0, 0), 2);
    EXPECT_DOUBLE_EQ(analytic.covariance(0, 1), 4);
    EXPECT_DOUBLE_EQ(analytic.covariance(1, 0), 4);
    EXPECT_DOUBLE_EQ(analytic.covariance(1, 1), 8);
    const auto rescaled = weighted_sample_moments(x, {2e200, 1e200, 0});
    EXPECT_DOUBLE_EQ(rescaled.covariance(0, 1), 4);

    const auto counts = weighted_sample_moments(x, {2, 1, 0}, SampleWeightType::Frequency);
    const auto expanded = weighted_sample_moments({3, 2, {1, 2, 1, 2, 3, 6}}, {1, 1, 1});
    EXPECT_DOUBLE_EQ(counts.covariance(0, 1), expanded.covariance(0, 1));
    EXPECT_NEAR(counts.covariance(0, 1), 8.0 / 3, 1e-14);
    EXPECT_DOUBLE_EQ(weighted_sample_moments(x, {1, 1e-90, 0}).covariance(0, 1), 4);
    EXPECT_THROW(weighted_sample_moments(x, {1, 0, 0}), std::invalid_argument);
    EXPECT_THROW(weighted_sample_moments(x, {1.5, 1, 0}, SampleWeightType::Frequency),
                 std::invalid_argument);
}

TEST(SampleMoments, CorrectedAndUncorrectedShapeMatchSciPy) {
    const auto raw = sample_shape({1, 2, 3, 10});
    EXPECT_NEAR(raw.skewness, 1.0182337649086284, 1e-14);
    EXPECT_NEAR(raw.excess_kurtosis, -0.7696, 1e-14);
    const auto corrected = sample_shape({1, 2, 3, 10}, true);
    EXPECT_NEAR(corrected.skewness, 1.763632614803888, 1e-14);
    EXPECT_NEAR(corrected.excess_kurtosis, 3.228, 1e-14);
    const double big = std::numeric_limits<double>::max();
    EXPECT_DOUBLE_EQ(sample_shape({-big, big}).excess_kurtosis, -2);
    EXPECT_THROW(sample_shape({1, 2, 3}, true), std::invalid_argument);
    EXPECT_THROW(sample_shape({1, 1}), std::invalid_argument);
}

TEST(SampleMoments, LMomentsMatchEnumeratedOrderStatisticContrasts) {
    const auto fit = sample_l_moments({63, 4, 15, 2, 9});
    EXPECT_NEAR(fit.location, 18.6, 1e-12);
    EXPECT_NEAR(fit.scale, 13.3, 1e-12);
    EXPECT_NEAR(fit.third, 9.3, 1e-12);
    EXPECT_NEAR(fit.fourth, 7.8, 1e-12);
    ASSERT_TRUE(fit.skewness && fit.kurtosis);
    EXPECT_NEAR(*fit.skewness, 93.0 / 133, 1e-12);
    EXPECT_NEAR(*fit.kurtosis, 78.0 / 133, 1e-12);
    const auto constant = sample_l_moments({7, 7, 7, 7});
    EXPECT_DOUBLE_EQ(constant.location, 7);
    EXPECT_DOUBLE_EQ(constant.scale, 0);
    EXPECT_FALSE(constant.skewness || constant.kurtosis);
    EXPECT_THROW(sample_l_moments({1, 2, 3}), std::invalid_argument);
}

TEST(SampleMoments, WeightedLorenzCurveMatchesReplicatedDistribution) {
    const auto fit = lorenz_curve({3, 80, 1}, {1, 0, 3});
    EXPECT_EQ(fit.population_share, (std::vector<double>{0, .75, 1}));
    EXPECT_EQ(fit.value_share, (std::vector<double>{0, .5, 1}));
    EXPECT_DOUBLE_EQ(fit.gini, .25);
    EXPECT_DOUBLE_EQ(lorenz_curve({1, 1, 1, 3}).gini, fit.gini);
    EXPECT_DOUBLE_EQ(lorenz_curve({5, 5}, {1e200, 1e200}).gini, 0);
    EXPECT_DOUBLE_EQ(lorenz_curve({0, 1}, {1e200, 1e-100}).gini, 1);
    EXPECT_THROW(lorenz_curve({0, 0}), std::invalid_argument);
    EXPECT_THROW(lorenz_curve({1, 0}, {0, 1}), std::invalid_argument);
    EXPECT_THROW(lorenz_curve({1, -1}), std::invalid_argument);
}

TEST(SampleMoments, FiellerConfidenceSetCanBeBoundedOrDisconnected) {
    const auto bounded = fieller_confidence_set(1, 2, .01, 0, 0, 1);
    ASSERT_EQ(bounded.size(), 1u);
    EXPECT_NEAR(bounded[0].first, .45, 1e-15);
    EXPECT_NEAR(bounded[0].second, .55, 1e-15);
    const auto correlated = fieller_confidence_set(1, 2, .04, .09, .01, 1.96);
    ASSERT_EQ(correlated.size(), 1u);
    for (const auto endpoint : {correlated[0].first, correlated[0].second}) {
        const auto residual = 1 - 2 * endpoint;
        const auto variance = .04 - .02 * endpoint + .09 * endpoint * endpoint;
        EXPECT_NEAR(residual * residual, 1.96 * 1.96 * variance, 1e-15);
    }
    constexpr auto inf = std::numeric_limits<double>::infinity();
    using Set = std::vector<std::pair<double, double>>;
    EXPECT_EQ(fieller_confidence_set(2, 0, 0, 1, 0, 1), (Set{{-inf, -2}, {2, inf}}));
    EXPECT_EQ(fieller_confidence_set(2, 1, 0, 1, 0, 1), (Set{{1, inf}}));
    EXPECT_EQ(fieller_confidence_set(0, 0, 1, 1, 0, 1), (Set{{-inf, inf}}));
    EXPECT_EQ(fieller_confidence_set(1, 3, 0, 0, 0, 1), (Set{{1. / 3, 1. / 3}}));
    EXPECT_TRUE(fieller_confidence_set(1, 0, 0, 0, 0, 1).empty());
    EXPECT_THROW(fieller_confidence_set(1, 2, 1, 1, 2), std::invalid_argument);
}

TEST(SampleMoments, DeltaMethodPropagatesJointAndSingularCovariance) {
    const auto cov = delta_method_covariance({2, 2, {1, 1, 1, -1}}, {2, 2, {1, .5, .5, 2}});
    EXPECT_DOUBLE_EQ(cov(0, 0), 4);
    EXPECT_DOUBLE_EQ(cov(0, 1), -1);
    EXPECT_DOUBLE_EQ(cov(1, 0), -1);
    EXPECT_DOUBLE_EQ(cov(1, 1), 2);
    const auto singular = delta_method_covariance({2, 2, {1, 1, 1, -1}}, {2, 2, {1, 1, 1, 1}});
    EXPECT_DOUBLE_EQ(singular(0, 0), 4);
    EXPECT_DOUBLE_EQ(singular(1, 1), 0);
    EXPECT_THROW(delta_method_covariance({1, 2, {1, 1}}, {2, 2, {1, 0, .5, 1}}),
                 std::invalid_argument);
}
