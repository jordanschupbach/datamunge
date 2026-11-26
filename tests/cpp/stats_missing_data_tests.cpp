#include <datamunge/stats/missing_data.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

const double na = std::numeric_limits<double>::quiet_NaN();
#include "stats_missing_data_data.inc"

Matrix as_matrix(const std::vector<double>& values, const std::vector<double>& dim) {
    const auto rows = static_cast<std::size_t>(dim[0]), cols = static_cast<std::size_t>(dim[1]);
    Matrix x(rows,cols);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            x(i,j) = values[i*cols+j];
    return x;
}

void check(const Matrix& x, const std::vector<double>& mu, const std::vector<double>& sigma, double ll,
           const std::vector<double>& little, const std::vector<double>& naniar) {
    const auto t = little_mcar_test(x,{.tolerance = 1e-13});
    const auto& r = t.fit;
    EXPECT_TRUE(r.converged);
    const std::size_t p = x.cols();
    for (std::size_t j = 0; j < p; ++j) {
        EXPECT_NEAR(r.mean[j],mu[j],1e-8*(1+std::abs(mu[j]))) << j;
        for (std::size_t k = 0; k < p; ++k)
            EXPECT_NEAR(r.covariance(j,k),sigma[j+k*p],1e-8*(1+std::abs(sigma[j+k*p]))) << j << "," << k;
    }
    EXPECT_NEAR(r.log_likelihood,ll,1e-8*std::abs(ll));
    EXPECT_NEAR(t.statistic,little[0],1e-7*little[0]);
    EXPECT_DOUBLE_EQ(t.df,little[1]);
    EXPECT_NEAR(t.p_value,little[2],1e-9);
    EXPECT_EQ(t.patterns,static_cast<std::size_t>(little[3]));
    // naniar runs em.norm with its default criterion 1e-4, so agreement is only approximate.
    EXPECT_NEAR(t.statistic,naniar[0],1e-3*naniar[0]);
    EXPECT_DOUBLE_EQ(t.df,naniar[1]);
}

// References: norm em.norm with a tight criterion, exact log-likelihood, Little's d2, naniar mcar_test.
TEST(MissingData, AirqualityMatchesNorm) {
    check(as_matrix(aq_data,aq_dim),aq_mu,aq_sigma,aq_ll[0],aq_little,aq_naniar);
}

TEST(MissingData, NhanesMatchesNorm) {
    check(as_matrix(nh_data,nh_dim),nh_mu,nh_sigma,nh_ll[0],nh_little,nh_naniar);
}

TEST(MissingData, CompleteDataGiveSampleMoments) {
    const Matrix x(4,2,{1,2,2,1,3,5,4,4});
    const auto r = em_multivariate_normal(x);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.mean[0],2.5,1e-12);
    EXPECT_NEAR(r.mean[1],3,1e-12);
    EXPECT_NEAR(r.covariance(0,0),1.25,1e-12);
    EXPECT_NEAR(r.covariance(0,1),1.25,1e-12);
    EXPECT_THROW((void)little_mcar_test(x),std::invalid_argument); // one pattern
}

// Complete data with the ML covariance reproduce least squares: y = 2x + 1 + (0, 1, -1, 0) residuals.
TEST(MissingData, RegressionFromMomentsMatchesLeastSquares) {
    const Matrix x(4,2,{1,3,2,6,3,6,4,9});
    const auto fit = em_multivariate_normal(x);
    const auto r = mvn_regression(fit.mean,fit.covariance,1,{0});
    EXPECT_NEAR(r.slopes[0],1.8,1e-12);
    EXPECT_NEAR(r.intercept,1.5,1e-12);
    EXPECT_NEAR(r.residual_variance,0.45,1e-12); // RSS / n = 1.8 / 4
    EXPECT_THROW((void)mvn_regression(fit.mean,fit.covariance,1,{1}),std::invalid_argument);
}

TEST(MissingData, PatternsAndValidation) {
    const Matrix x(4,2,{na,1,2,na,3,4,na,na});
    const auto p = missing_patterns(x);
    ASSERT_EQ(p.size(),4u);
    EXPECT_EQ(p[0].rows,(std::vector<std::size_t>{2}));
    EXPECT_EQ(p[3].rows,(std::vector<std::size_t>{3}));
    EXPECT_THROW((void)em_multivariate_normal(Matrix(3,2,{1,na,2,na,3,na})),std::invalid_argument);
    EXPECT_THROW((void)missing_patterns(Matrix(1,1,std::numeric_limits<double>::infinity())),std::invalid_argument);
}

} // namespace
