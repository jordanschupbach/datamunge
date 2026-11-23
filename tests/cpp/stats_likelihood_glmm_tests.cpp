#include <datamunge/stats/likelihood_glmm.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fit_likelihood_glmm;
using datamunge::stats::GLMMFamily;
using datamunge::stats::LikelihoodGLMMOptions;
using datamunge::stats::LikelihoodGLMMStatus;

TEST(LikelihoodGLMM, AdaptiveQuadratureAndLaplaceMatchIndependentGroupIntegrals) {
    const std::vector<std::size_t> binomial{
        0,1,2,3, 1,2,4,5, 0,0,1,1, 2,4,4,6, 0,1,1,2, 3,4,5,6, 0,2,2,4, 1,3,3,5};
    const std::vector<std::size_t> poisson{
        0,1,1,1, 1,2,2,3, 2,3,3,4, 3,4,7,6, 0,0,1,2, 5,8,13,9, 1,1,2,2, 0,2,4,3};
    const std::array<double,8> ref[][2]{
        {{-.280058172663,.978265668735,1.466904940848,-39.225937814932,-.849898545384,
          .311797544014,.031810779325,.208334874141},
         {-.280066013625,.977501605814,1.476474720854,-39.145915442127,-.850893076426,
          .315392857777,.031798699121,.211911595887}},
        {{.687030685016,.110369049836,.732813319595,-56.992289079921,-.736858970330,
          .085731772668,0,.050802925133},
         {.686453903546,.110369046977,.736472929075,-56.969002435238,-.738643628142,
          .086426096295,0,.051672694898}},
    };
    const double x_values[]{-1,-.4,.35,1}, exposure[]{.8,1,1.2,1.5};
    const std::size_t trial_counts[]{4,5,5,6};
    DenseMatrix<double> design(binomial.size(), 2, 1);
    std::vector<std::size_t> groups, trials;
    std::vector<double> offset;
    for (std::size_t i = 0; i < binomial.size(); ++i) {
        design(i, 1) = x_values[i % 4];
        groups.push_back(10 * (i / 4) + 7);
        trials.push_back(trial_counts[i % 4]);
        offset.push_back(std::log(exposure[i % 4]));
    }
    // Independent R uniroot/curvature Laplace and numerical integration
    // of dpois/dbinom times the standard Gaussian density.
    for (const bool is_poisson : {false,true})
        for (const auto points : {1u,31u}) {
            const bool adaptive = points > 1;
            SCOPED_TRACE((is_poisson ? 10 : 0) + adaptive);
            const auto fit = fit_likelihood_glmm(design, is_poisson ? poisson : binomial, groups,
                {.family = is_poisson ? GLMMFamily::Poisson : GLMMFamily::Binomial,
                 .quadrature_points = points, .trials = is_poisson ? std::vector<std::size_t>{} : trials,
                 .offset = offset});
            ASSERT_EQ(fit.status, LikelihoodGLMMStatus::Converged);
            const auto& expected = ref[is_poisson][adaptive];
            EXPECT_NEAR(fit.coefficients[0], expected[0], 1e-7);
            EXPECT_NEAR(fit.coefficients[1], expected[1], 1e-7);
            EXPECT_NEAR(fit.random_intercept_sd, expected[2], 1e-7);
            EXPECT_NEAR(fit.log_likelihood, expected[3], 1e-9);
            EXPECT_NEAR(fit.group_modes.front(), expected[4], 1e-7);
            ASSERT_EQ(fit.covariance.rows(), 3u);
            EXPECT_NEAR(fit.covariance(0,0), expected[5], 2e-7);
            EXPECT_NEAR(fit.covariance(1,2), expected[6], 2e-7);
            EXPECT_NEAR(fit.covariance(2,2), expected[7], 2e-7);
            EXPECT_EQ(fit.group_levels[0], 7u);
            const DenseMatrix<double> query(2, 2, {1,1, 1,1});
            const auto prediction = fit.predict(query, {7,999}, {0,0});
            const auto baseline = fit.coefficients[0] + fit.coefficients[1];
            const auto transform = [is_poisson](double eta) {
                return is_poisson ? std::exp(eta) : 1 / (1 + std::exp(-eta));
            };
            EXPECT_NEAR(prediction[0], transform(baseline + fit.group_modes.front()), 1e-12);
            EXPECT_NEAR(prediction[1], transform(baseline), 1e-12);
        }
}

TEST(LikelihoodGLMM, EstimatesExactZeroVarianceAndFixedEffectInformation) {
    const DenseMatrix<double> original(6, 1, 2);
    const std::vector<std::size_t> y{0,1,0,1,0,1}, groups{2,2,5,5,9,9};
    const auto fit = fit_likelihood_glmm(original, y, groups, {.quadrature_points = 32});
    ASSERT_EQ(fit.status, LikelihoodGLMMStatus::ZeroVariance);
    EXPECT_DOUBLE_EQ(fit.random_intercept_sd, 0);
    EXPECT_NEAR(fit.log_likelihood, -6 * std::log(2.), 1e-12);
    ASSERT_EQ(fit.covariance.rows(), 1u);
    EXPECT_NEAR(fit.covariance(0,0), 1./6, 1e-12);
    EXPECT_EQ(fit.group_modes, (std::vector<double>{0,0,0}));
}

TEST(LikelihoodGLMM, RejectsInvalidTrialsGroupsAndQuadratureOrder) {
    const DenseMatrix<double> x(4, 1, 1), aliased(4, 2, 1);
    EXPECT_THROW((void)fit_likelihood_glmm(x, {0,2,0,1}, {0,0,1,1}), std::invalid_argument);
    EXPECT_THROW((void)fit_likelihood_glmm(x, {0,1,0,1}, {0,0,0,0}), std::invalid_argument);
    EXPECT_THROW((void)fit_likelihood_glmm(x, {0,1,0,1}, {0,0,1,1}, {.quadrature_points = 49}),
        std::invalid_argument);
    EXPECT_THROW((void)fit_likelihood_glmm(aliased, {0,1,0,1}, {0,0,1,1}), std::invalid_argument);
}
}
