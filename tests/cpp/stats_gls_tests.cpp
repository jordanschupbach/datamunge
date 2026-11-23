#include <datamunge/stats/gls.hpp>

#include <array>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {

using datamunge::linalg::DenseMatrix;
using namespace datamunge::stats;

TEST(GeneralizedLeastSquares, SuppliedCovarianceMatchesDenseReference) {
    const DenseMatrix<double> design(4, 2, {1, 0, 1, 1, 1, 3, 1, 4});
    const DenseMatrix<double> covariance(4, 4, {
        3, 1, 0, 0, 1, 2, .5, 0, 0, .5, 1, .25, 0, 0, .25, 2,
    });
    const auto fit = generalized_least_squares(design, {1, 3, 5, 9}, covariance);

    // Base R solves the dense covariance and coefficient information systems.
    ASSERT_EQ(fit.coefficients.size(), 2u);
    EXPECT_NEAR(fit.coefficients[0], .795868772782505, 1e-13);
    EXPECT_NEAR(fit.coefficients[1], 1.688942891859051, 1e-13);
    EXPECT_NEAR(fit.covariance(0, 0), 2.309842041312271, 1e-13);
    EXPECT_NEAR(fit.covariance(1, 0), -.670716889428918, 1e-13);
    EXPECT_DOUBLE_EQ(fit.covariance(0, 1), fit.covariance(1, 0));
    EXPECT_NEAR(fit.covariance(1, 1), .263669501822600, 1e-13);
    ASSERT_EQ(fit.fitted.size(), 4u);
    ASSERT_EQ(fit.residuals.size(), 4u);
    EXPECT_NEAR(fit.fitted[2], 5.862697448359659, 1e-13);
    EXPECT_NEAR(fit.residuals[2], -.862697448359659, 1e-13);
    EXPECT_NEAR(fit.log_likelihood, -6.165697558451260, 1e-13);
}

TEST(GeneralizedLeastSquares, StructuredModelsMatchDenseProfileWithInterleavedBlocks) {
    const DenseMatrix<double> design(12, 2, {
        1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 3, 1, 3, 1, 3,
    });
    const std::vector<double> response{4, 4.1, 1.4, 5.2, 6.8, 2.2, 7.4, 6.9, 4.6, 7.7, 8.6, 6.2};
    const StructuredGlsOptions options{.groups = {10, 20, -1, 10, 20, -1, 10, 20, -1, 10, 20, -1}};
    struct Reference {
        GlsCorrelation structure;
        double rho, intercept, slope, variance, log_likelihood, slope_variance;
    };
    // Base R builds each dense correlation matrix and optimizes the Gaussian profile.
    const std::array references{
        Reference{GlsCorrelation::AR1, .780994950641697, 3.198292671378758, 1.445318745597646,
                  1.641246394796772, -15.763300641326275, .063650520838341},
        Reference{GlsCorrelation::CompoundSymmetry, .851213041966795, 3.24, 1.456666666666667,
                  1.926194350860277, -14.288918842799653, .019106173203016},
    };
    for (const auto& ref : references) {
        const auto result = structured_gls(design, response, ref.structure, options);
        EXPECT_TRUE(result.converged);
        EXPECT_FALSE(result.at_boundary);
        EXPECT_NEAR(result.correlation, ref.rho, 1e-7);
        EXPECT_NEAR(result.fit.coefficients[0], ref.intercept, 1e-7);
        EXPECT_NEAR(result.fit.coefficients[1], ref.slope, 1e-7);
        EXPECT_NEAR(result.residual_variance, ref.variance, 1e-6);
        EXPECT_NEAR(result.fit.log_likelihood, ref.log_likelihood, 1e-12);
        EXPECT_NEAR(result.fit.covariance(1, 1), ref.slope_variance, 1e-7);
        EXPECT_NEAR(result.fit.fitted[8], ref.intercept + 2 * ref.slope, 1e-7);
    }
}

TEST(GeneralizedLeastSquares, ReportsSearchBoundaryAndRefinementLimit) {
    const DenseMatrix<double> intercept(4, 1, 1);
    const auto fit = structured_gls(intercept, {0, 1, 0, 1}, GlsCorrelation::CompoundSymmetry);
    EXPECT_TRUE(fit.converged);
    EXPECT_TRUE(fit.at_boundary);
    EXPECT_GT(fit.correlation, -1.0 / 3);
    EXPECT_NEAR(fit.correlation, -1.0 / 3, 2e-8);
    EXPECT_NEAR(fit.fit.coefficients[0], .5, 1e-12);

    const StructuredGlsOptions limit{.max_iterations = 1};
    EXPECT_FALSE(structured_gls(intercept, {0, 1, 0, 1}, GlsCorrelation::AR1, limit).converged);
}

TEST(GeneralizedLeastSquares, RejectsInvalidCovarianceRankAndVarianceModels) {
    const DenseMatrix<double> design(3, 2, {1, 0, 1, 1, 1, 2});
    const std::vector<double> response{0, 1, 3};
    const auto identity = DenseMatrix<double>::identity(3);
    EXPECT_THROW(generalized_least_squares(design, response, DenseMatrix<double>(2, 2)),
                 std::invalid_argument);
    auto covariance = identity;
    covariance(0, 1) = .1;
    EXPECT_THROW(generalized_least_squares(design, response, covariance), std::invalid_argument);
    covariance(1, 0) = covariance(0, 1) = 2;
    EXPECT_THROW(generalized_least_squares(design, response, covariance), std::invalid_argument);
    EXPECT_THROW(generalized_least_squares(DenseMatrix<double>(3, 2, 1), response, identity),
                 std::invalid_argument);
    EXPECT_THROW(structured_gls(design, {1, 2, 3}, GlsCorrelation::AR1), std::domain_error);
    EXPECT_THROW(structured_gls(design, response, GlsCorrelation::AR1, {.groups = {0, 1, 2}}),
                 std::domain_error);
}

} // namespace
