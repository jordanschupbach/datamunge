#include <datamunge/stats/minimum_covariance_determinant.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::minimum_covariance_determinant;

const DenseMatrix<double> sample(9,2,{-.2,.1,.1,-.4,.4,.3,-.45,-.2,.2,.08,.05,.25,.55,-.2,6,-5,-8,4});

TEST(MinimumCovarianceDeterminant, MatchesExhaustiveRAndGaussianReweighting) {
    const auto fit = minimum_covariance_determinant(sample);
    EXPECT_TRUE(fit.exact_search);
    EXPECT_TRUE(fit.converged);
    EXPECT_EQ(fit.support,(std::vector<std::size_t>{0,1,2,4,5,6}));
    EXPECT_EQ(fit.reweighted_rows,(std::vector<std::size_t>{0,1,2,3,4,5,6}));
    EXPECT_NEAR(fit.raw_location[0],.1833333333333333,1e-14);
    EXPECT_NEAR(fit.raw_location[1],.0216666666666667,1e-14);
    EXPECT_NEAR(fit.raw_covariance(0,0),.130662728476461,1e-13);
    EXPECT_NEAR(fit.raw_covariance(1,0),-.016764274596980,1e-13);
    EXPECT_NEAR(fit.raw_covariance(1,1),.135377680706862,1e-13);
    EXPECT_NEAR(fit.raw_consistency_factor,2.21880104960029,1e-12);
    EXPECT_NEAR(fit.reweight_consistency_factor,1.10446792390291,1e-12);
    EXPECT_NEAR(fit.reweight_cutoff,7.37775890822787,1e-12);
    EXPECT_NEAR(fit.location[0],.0928571428571429,1e-14);
    EXPECT_NEAR(fit.location[1],-.01,1e-14);
    EXPECT_NEAR(fit.covariance(0,0),.1099959891560445,1e-13);
    EXPECT_NEAR(fit.covariance(1,0),.0118335848989597,1e-13);
    EXPECT_NEAR(fit.covariance(1,1),.0644062580767380,1e-13);
    EXPECT_NEAR(fit.raw_squared_distances[3],3.7585953061049415,1e-12);
    EXPECT_NEAR(fit.squared_distances[3],2.95344399160291,1e-13);
    EXPECT_NEAR(fit.squared_distances[7],818.49497427826213,1e-10);
    EXPECT_NEAR(fit.squared_distances[8],972.73352240171312,1e-10);
}

TEST(MinimumCovarianceDeterminant, SearchAndDistancesAreAffineEquivariant) {
    const auto fit = minimum_covariance_determinant(sample);
    auto transformed = sample;
    for (std::size_t i = 0; i < sample.rows(); ++i) {
        transformed(i,0) = 9+2*sample(i,0)-.5*sample(i,1);
        transformed(i,1) = -6+sample(i,0)+3*sample(i,1);
    }
    const auto equivalent = minimum_covariance_determinant(transformed);
    EXPECT_EQ(equivalent.support,fit.support);
    EXPECT_EQ(equivalent.reweighted_rows,fit.reweighted_rows);
    for (std::size_t i = 0; i < sample.rows(); ++i)
        EXPECT_NEAR(equivalent.squared_distances[i],fit.squared_distances[i],1e-10);

    const auto at_limit = minimum_covariance_determinant(sample,{.exact_subset_limit=84});
    const auto below_limit = minimum_covariance_determinant(sample,{.exact_subset_limit=83,.random_starts=32});
    EXPECT_TRUE(at_limit.exact_search);
    EXPECT_FALSE(below_limit.exact_search);
    EXPECT_EQ(below_limit.support,fit.support);
}

TEST(MinimumCovarianceDeterminant, FarOutliersDoNotHideFullRankInlierScatter) {
    auto extreme = sample;
    for (std::size_t i = 7; i < sample.rows(); ++i)
        for (std::size_t j = 0; j < sample.cols(); ++j)
            extreme(i,j) *= 1e10;
    const auto ordinary = minimum_covariance_determinant(sample);
    const auto robust = minimum_covariance_determinant(extreme);
    EXPECT_EQ(robust.support,ordinary.support);
    EXPECT_EQ(robust.reweighted_rows,ordinary.reweighted_rows);
    EXPECT_NEAR(robust.covariance(1,0),ordinary.covariance(1,0),1e-14);
    EXPECT_GT(robust.squared_distances[7],1e22);
}

TEST(MinimumCovarianceDeterminant, ConcentrationSearchIsReproducibleOnLargerContaminatedSamples) {
    DenseMatrix<double> x(120,3);
    for (std::size_t i = 0; i < x.rows(); ++i) {
        const double t = static_cast<double>(i+1);
        x(i,0) = std::sin(.37*t)+.35*std::sin(1.4*t);
        x(i,1) = .25*x(i,0)+std::cos(.46*t)+.2*std::cos(1.5*t);
        x(i,2) = std::sin(.7*t)+.4*std::sin(.32*t);
        if (i >= 92) {
            x(i,0) += 15;
            x(i,1) -= 18;
            x(i,2) += 12;
        }
    }
    const datamunge::stats::MCDOptions controls{.exact_subset_limit=0,.random_starts=40,.seed=129};
    const auto fit = minimum_covariance_determinant(x,controls);
    const auto repeated = minimum_covariance_determinant(x,controls);
    EXPECT_FALSE(fit.exact_search);
    EXPECT_TRUE(fit.converged);
    EXPECT_EQ(fit.support,repeated.support);
    EXPECT_EQ(fit.reweighted_rows,repeated.reweighted_rows);
    ASSERT_EQ(fit.support.size(),62u);
    EXPECT_LT(fit.support.back(),92u);
    ASSERT_GE(fit.reweighted_rows.size(),70u);
    EXPECT_LT(fit.reweighted_rows.back(),92u);
    EXPECT_GT(fit.squared_distances.back(),100);
}

TEST(MinimumCovarianceDeterminant, ValidatesDegenerateSubsetsAndReweighting) {
    const DenseMatrix<double> singular_subset(6,2,{0,0,1,1,2,2,3,3,0,1,2,4});
    EXPECT_THROW((void)minimum_covariance_determinant(singular_subset),std::domain_error);
    EXPECT_THROW((void)minimum_covariance_determinant(sample,{.support_size=2}),std::invalid_argument);
    EXPECT_THROW((void)minimum_covariance_determinant(sample,{.reweight_probability=1e-5}),std::domain_error);
    auto nonfinite = sample;
    nonfinite(0,0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)minimum_covariance_determinant(nonfinite),std::invalid_argument);
    const auto full = minimum_covariance_determinant(DenseMatrix<double>(3,1,{1,2,3}),{.support_size=3});
    EXPECT_TRUE(full.exact_search);
    EXPECT_DOUBLE_EQ(full.raw_consistency_factor,1);
    EXPECT_NEAR(full.raw_covariance(0,0),2.0/3,1e-15);
}
} // namespace
