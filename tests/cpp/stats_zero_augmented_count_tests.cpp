#include <datamunge/stats/zero_augmented_count.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using namespace datamunge::stats;

TEST(ZeroAugmentedCount, MatchesIndependentWeightedLikelihoodsAndObservedInformation) {
    const std::vector<std::size_t> counts{
        0,0,0,0,1,2,3,4,5,8,12,15, 0,0,0,0,0,0,0,1,2,4,10,17,
        0,0,0,1,2,3,4,7,10,13,20,30, 0,0,0,0,0,0,0,2,3,8,24,35};
    DenseMatrix<double> x(counts.size(), 2, 1), z(counts.size(), 2, 1);
    const double exposure[]{1, .7, 1.5, 1.2};
    const double weights[]{1, .8, 1.2, 1};
    ZeroAugmentedCountOptions options;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        x(i, 1) = i / 24;
        z(i, 1) = (i / 12) % 2;
        options.offset.push_back(std::log(exposure[i % 4]));
        options.weights.push_back(weights[i % 4]);
    }
    using Family = ZeroAugmentedCountFamily;
    struct Reference {
        decltype(&zero_inflated_count_regression) fit;
        Family family;
        std::array<double, 5> coefficients;
        double log_likelihood;
        std::array<double, 3> covariance;
    };
    // R optim() and optimHess() on the weighted dpois() or dnbinom() log likelihood.
    const Reference references[]{
        {zero_inflated_count_regression, Family::Poisson,
         {1.753024897,.586091493,-.896395510,1.228368239,0}, -164.8383062181106,
         {.01180031565,.00046382574,.37632319232}},
        {zero_inflated_count_regression, Family::NegativeBinomial,
         {1.580255681,.656749718,-1.434401470,1.535794205,.975118565}, -114.0898935204382,
         {.09564487796,.07612983002,.75137130022}},
        {hurdle_count_regression, Family::Poisson,
         {1.753634493,.585503907,-.887303196,1.223775431,0}, -164.8518458312314,
         {.01177307008,0,.37310923410}},
        {hurdle_count_regression, Family::NegativeBinomial,
         {1.581881758,.655050559,-.887303242,1.223775489,.973356650}, -114.2026095827522,
         {.10073772770,0,.37310924856}},
    };
    for (const auto& reference : references) {
        options.family = reference.family;
        const auto fit = reference.fit(x, z, counts, options);
        const bool hurdle = fit.model == ZeroAugmentedCountModel::Hurdle;
        SCOPED_TRACE((hurdle ? 10 : 0) + static_cast<int>(reference.family));
        ASSERT_EQ(fit.status, ZeroAugmentedCountStatus::Converged);
        ASSERT_EQ(fit.count_coefficients.size(), 2u);
        ASSERT_EQ(fit.zero_coefficients.size(), 2u);
        for (std::size_t j = 0; j < 2; ++j) {
            EXPECT_NEAR(fit.count_coefficients[j], reference.coefficients[j], 5e-7);
            EXPECT_NEAR(fit.zero_coefficients[j], reference.coefficients[j + 2], 5e-7);
        }
        EXPECT_NEAR(fit.dispersion, reference.coefficients[4], 5e-7);
        EXPECT_NEAR(fit.log_likelihood, reference.log_likelihood, 1e-10);
        ASSERT_EQ(fit.covariance.rows(), reference.family == Family::NegativeBinomial ? 5u : 4u);
        EXPECT_NEAR(fit.covariance(0, 0), reference.covariance[0], 5e-7);
        EXPECT_NEAR(fit.covariance(0, 2), reference.covariance[1], 2e-7);
        EXPECT_NEAR(fit.covariance(3, 3), reference.covariance[2], 5e-7);

        const DenseMatrix<double> new_x(2, 2, {1, 0, 1, 0}), new_z(2, 2, {1, 1, 1, 1});
        const auto pred = fit.predict(new_x, new_z, {0, std::log(2.)});
        const auto pi = 1 / (1 + std::exp(-fit.zero_coefficients[0] - fit.zero_coefficients[1]));
        EXPECT_NEAR(pred.count_mean[1], 2 * pred.count_mean[0], 1e-13);
        for (std::size_t i = 0; i < 2; ++i) {
            const auto mu = pred.count_mean[i];
            const auto f0 = fit.dispersion == 0 ? std::exp(-mu)
                : std::pow(1 + fit.dispersion * mu, -1 / fit.dispersion);
            EXPECT_NEAR(pred.zero_component_probability[i], pi, 1e-13);
            EXPECT_NEAR(pred.zero_probability[i], hurdle ? pi : pi + (1 - pi) * f0, 1e-13);
            EXPECT_NEAR(pred.mean[i], (1 - pi) * mu / (hurdle ? 1 - f0 : 1), 1e-13);
        }
    }
}

TEST(ZeroAugmentedCount, SelectsPoissonAndZeroProbabilityBoundaries) {
    const DenseMatrix<double> intercept(10, 1, 1);
    const std::vector<std::size_t> pulsed{0,0,0,0,0,4,5,5,5,6};
    for (const auto fit_model : {zero_inflated_count_regression, hurdle_count_regression}) {
        const auto fit = fit_model(intercept, intercept, pulsed,
            {.family = ZeroAugmentedCountFamily::NegativeBinomial});
        EXPECT_EQ(fit.status, ZeroAugmentedCountStatus::PoissonBoundary);
        EXPECT_DOUBLE_EQ(fit.dispersion, 0);
        EXPECT_EQ(fit.covariance.rows(), 2u);
    }
    const DenseMatrix<double> positive_intercept(6, 1, 1);
    const std::vector<std::size_t> positive{2,3,2,2,3,2};
    for (const auto fit_model : {zero_inflated_count_regression, hurdle_count_regression}) {
        const auto fit = fit_model(positive_intercept, positive_intercept, positive, {});
        EXPECT_EQ(fit.status, ZeroAugmentedCountStatus::NoZeroComponent);
        EXPECT_TRUE(fit.zero_coefficients.empty());
        EXPECT_EQ(fit.covariance.rows(), 1u);
        EXPECT_DOUBLE_EQ(fit.fitted.zero_component_probability[0], 0);
        EXPECT_NEAR(fit.fitted.mean[0], 14. / 6, 2e-8);
    }
}

TEST(ZeroAugmentedCount, ValidatesPredictorsOnRowsUsedByEachComponent) {
    const DenseMatrix<double> intercept(4, 1, 1), x(4, 2, {1,1, 1,1, 1,0, 1,0});
    const std::vector<std::size_t> counts{0,0,2,3};
    EXPECT_THROW((void)hurdle_count_regression(x, intercept, counts), std::invalid_argument);
    EXPECT_THROW((void)zero_inflated_count_regression(intercept, intercept, counts, {.weights = {1,1,0,1}}),
                 std::invalid_argument);
    EXPECT_THROW((void)hurdle_count_regression(intercept, intercept, {0,0,0,0}), std::invalid_argument);
}
}
