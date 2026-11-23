#include <datamunge/stats/confirmatory_factor_analysis.hpp>

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::CFASpecification;
using datamunge::stats::CFAStatus;
using datamunge::stats::fit_confirmatory_factor_analysis;
using Mask = DenseMatrix<std::uint8_t>;

TEST(ConfirmatoryFactorAnalysis, OneFactorLikelihoodMatchesIndependentRFit) {
    const DenseMatrix<double> l(5,1,{.75,.62,.83,.51,.67});
    auto sample = l*l.transpose();
    const double psi[]{.5,.42,.3,.66,.55};
    for (std::size_t j=0;j<5;++j) sample(j,j) += psi[j];
    sample(0,1) = sample(1,0) += .13;
    sample(2,3) = sample(3,2) -= .08;
    sample(3,4) = sample(4,3) += .07;
    const CFASpecification spec{{DenseMatrix<double>(5,1,.55),Mask(5,1,1)},
        {DenseMatrix<double>(1,1,1),{}},{.6*DenseMatrix<double>::identity(5),Mask::identity(5)}};
    const auto fit = fit_confirmatory_factor_analysis(sample,spec);
    ASSERT_EQ(fit.status,CFAStatus::Converged);
    EXPECT_EQ(fit.degrees_of_freedom,5u);
    EXPECT_NEAR(fit.discrepancy,.0607315143094223,1e-10);
    EXPECT_NEAR(std::abs(fit.loadings(0,0)),.819798079772940,1e-6);
    EXPECT_NEAR(std::abs(fit.loadings(4,0)),.661346708304648,1e-6);
    EXPECT_NEAR(fit.residual_covariance(0,0),.390431113641338,1e-6);
    EXPECT_NEAR(fit.residual_covariance(4,4),.561520531559461,1e-6);
    EXPECT_DOUBLE_EQ(fit.factor_covariance(0,0),1);
    EXPECT_DOUBLE_EQ(fit.residual_covariance(1,0),0);
    EXPECT_NEAR(fit.covariance_residuals(0,1),sample(0,1)-fit.implied_covariance(0,1),1e-14);
}

TEST(ConfirmatoryFactorAnalysis, MarkerLoadingsIdentifyCorrelatedFactorsWithFreeResidualCovariance) {
    const DenseMatrix<double> l(6,2,{1,0,.8,0,1.1,0,0,1,0,.72,0,.9});
    const DenseMatrix<double> phi(2,2,{1.4,.32,.32,.85});
    auto e = .5*DenseMatrix<double>::identity(6);
    e(0,5) = e(5,0) = .09;
    const auto sample = (l*phi)*l.transpose()+e;
    auto loading = l;
    loading(1,0) = loading(2,0) = loading(4,1) = loading(5,1) = .6;
    auto errors = Mask::identity(6);
    errors(0,5) = errors(5,0) = 1;
    const CFASpecification spec{{loading,Mask(6,2,{0,0,1,0,1,0,0,0,0,1,0,1})},
        {DenseMatrix<double>::identity(2),Mask(2,2,1)},
        {.7*DenseMatrix<double>::identity(6),errors}};
    const auto fit = fit_confirmatory_factor_analysis(sample,spec);
    ASSERT_EQ(fit.status,CFAStatus::Converged);
    EXPECT_EQ(fit.degrees_of_freedom,7u);
    EXPECT_NEAR(fit.discrepancy,0,1e-14);
    EXPECT_NEAR(fit.factor_covariance(0,0),1.4,1e-7);
    EXPECT_NEAR(fit.factor_covariance(0,1),.32,1e-7);
    EXPECT_NEAR(fit.residual_covariance(0,5),.09,1e-7);
    EXPECT_NEAR(fit.loadings(2,0),1.1,1e-7);
    EXPECT_NEAR(fit.loadings(5,1),.9,1e-7);
}

TEST(ConfirmatoryFactorAnalysis, DetectsUnidentifiedScalesAndInvalidCovarianceMasks) {
    const auto sample = DenseMatrix<double>::identity(4);
    CFASpecification spec{{DenseMatrix<double>(4,1,.6),Mask(4,1,1)},
        {DenseMatrix<double>(1,1,1),Mask(1,1,1)},
        {DenseMatrix<double>::identity(4),Mask::identity(4)}};
    EXPECT_THROW((void)fit_confirmatory_factor_analysis(sample,spec),std::domain_error);
    spec.factor_covariance.free_entries = {};
    spec.residual_covariance.free_entries(0,1) = 1;
    EXPECT_THROW((void)fit_confirmatory_factor_analysis(sample,spec),std::invalid_argument);
}
}
