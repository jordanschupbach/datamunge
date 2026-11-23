#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/stats/covariance_shrinkage.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::ledoit_wolf_covariance;
using datamunge::stats::oracle_approximating_covariance;

TEST(CovarianceShrinkage, BothMethodsMatchIndependentOuterProductCalculationsWhenWide) {
    DenseMatrix<double> x(6,12);
    for (std::size_t row = 0; row < x.rows(); ++row)
        for (std::size_t col = 0; col < x.cols(); ++col) {
            const double i = row+1, j = col+1;
            x(row,col) = 1.6*std::sin(.82*i)*(1+.05*j)+.33*std::cos(.47*i*j)+.9*j;
        }
    // Base R cross-products, explicit outer-product residuals, and Chen et al. equation (23).
    const auto lw = ledoit_wolf_covariance(x);
    const auto oas = oracle_approximating_covariance(x);
    EXPECT_NEAR(lw.shrinkage,.0722555774530225,1e-13);
    EXPECT_NEAR(oas.shrinkage,.298344413974846,1e-13);
    EXPECT_NEAR(lw.target_variance,2.7298947078425151,1e-13);
    EXPECT_DOUBLE_EQ(lw.target_variance,oas.target_variance);
    EXPECT_NEAR(lw.covariance(0,0),2.2865193195267088,1e-13);
    EXPECT_NEAR(lw.covariance(1,0),1.7824715739525796,1e-13);
    EXPECT_NEAR(oas.covariance(0,11),2.258286038503828,1e-13);
    EXPECT_NEAR(oas.covariance(1,1),1.983556341285428,1e-13);
    EXPECT_EQ(lw.location,oas.location);
    EXPECT_TRUE(datamunge::linalg::cholesky(lw.covariance).ok);
    EXPECT_TRUE(datamunge::linalg::cholesky(oas.covariance).ok);
    const auto lw_values = datamunge::linalg::symmetric_eigen(lw.covariance).eigenvalues;
    const auto oas_values = datamunge::linalg::symmetric_eigen(oas.covariance).eigenvalues;
    EXPECT_NEAR(lw_values.back(),.1972501185011068,1e-13);
    EXPECT_NEAR(oas_values.back(),.814448836824307,1e-13);
    double trace = 0;
    for (std::size_t j = 0; j < x.cols(); ++j)
        trace += oas.covariance(j,j);
    EXPECT_NEAR(trace,x.cols()*oas.target_variance,1e-13);
}

TEST(CovarianceShrinkage, KnownZeroUsesOriginalFiniteDimensionTerms) {
    const DenseMatrix<double> x(6,2,{2,1,4,0,-2,0,3,1,4,1,-6,0});
    const auto lw = ledoit_wolf_covariance(x,true);
    const auto oas = oracle_approximating_covariance(x,true);
    EXPECT_EQ(oas.location,(std::vector<double>{0,0}));
    EXPECT_DOUBLE_EQ(lw.location[0],0);
    EXPECT_NEAR(lw.shrinkage,.212637154748392,1e-13);
    EXPECT_NEAR(oas.shrinkage,.366250472947408,1e-13);
    EXPECT_NEAR(oas.target_variance,22.0/3,1e-13);
    EXPECT_NEAR(oas.covariance(0,1),.950624290578888,1e-13);

    auto shifted = x;
    for (std::size_t i = 0; i < x.rows(); ++i) {
        shifted(i,0) += 1e12;
        shifted(i,1) -= 2e12;
    }
    const auto centered = oracle_approximating_covariance(shifted);
    const auto ordinary = oracle_approximating_covariance(x);
    EXPECT_DOUBLE_EQ(centered.shrinkage,ordinary.shrinkage);
    EXPECT_DOUBLE_EQ(centered.covariance(1,0),ordinary.covariance(1,0));
    EXPECT_DOUBLE_EQ(centered.location[0],1e12+5.0/6);
    EXPECT_DOUBLE_EQ(centered.location[1],-2e12+.5);
    for (const double magnitude : {1e-140,1e140}) {
        const auto scaled = oracle_approximating_covariance(magnitude*x,true);
        EXPECT_NEAR(scaled.shrinkage,oas.shrinkage,1e-13);
        EXPECT_NEAR((scaled.covariance(0,0)/magnitude)/magnitude,oas.covariance(0,0),1e-13);
    }
}

TEST(CovarianceShrinkage, DegenerateLimitsAreFiniteAndDefined) {
    const DenseMatrix<double> isotropic(4,2,{1,0,-1,0,0,1,0,-1});
    for (const auto method : {ledoit_wolf_covariance,oracle_approximating_covariance}) {
        const auto exact = method(isotropic,false);
        EXPECT_DOUBLE_EQ(exact.shrinkage,1);
        EXPECT_DOUBLE_EQ(exact.covariance(0,0),.5);
        EXPECT_DOUBLE_EQ(exact.covariance(0,1),0);
        const auto flat = method(DenseMatrix<double>(3,4,12),false);
        EXPECT_DOUBLE_EQ(flat.shrinkage,1);
        EXPECT_DOUBLE_EQ(flat.target_variance,0);
        EXPECT_DOUBLE_EQ(flat.covariance(2,2),0);
        EXPECT_DOUBLE_EQ(flat.location[0],12);
    }

    const DenseMatrix<double> same_outer(4,2,{1,2,-1,-2,1,2,-1,-2});
    EXPECT_DOUBLE_EQ(ledoit_wolf_covariance(same_outer).shrinkage,0);
    EXPECT_DOUBLE_EQ(ledoit_wolf_covariance(same_outer).covariance(0,1),2);
    EXPECT_NEAR(oracle_approximating_covariance(same_outer).shrinkage,.5,1e-15);
    const DenseMatrix<double> single(1,2,{1,2});
    EXPECT_DOUBLE_EQ(ledoit_wolf_covariance(single,true).shrinkage,0);
    EXPECT_DOUBLE_EQ(oracle_approximating_covariance(single,true).shrinkage,1);
}

TEST(CovarianceShrinkage, ValidatesShapeFiniteValuesAndCovarianceRange) {
    EXPECT_THROW((void)ledoit_wolf_covariance(DenseMatrix<double>(1,2,{1,2})),std::invalid_argument);
    EXPECT_THROW((void)oracle_approximating_covariance(DenseMatrix<double>(2,0)),std::invalid_argument);
    const auto inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)ledoit_wolf_covariance(DenseMatrix<double>(2,1,{0,inf})),std::invalid_argument);
    EXPECT_THROW((void)oracle_approximating_covariance(DenseMatrix<double>(2,1,{1e200,-1e200})),std::overflow_error);
    EXPECT_THROW((void)ledoit_wolf_covariance(DenseMatrix<double>(2,1,{1e-200,-1e-200})),std::overflow_error);
}
} // namespace
