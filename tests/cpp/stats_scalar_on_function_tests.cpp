#include <datamunge/fda/scalar_on_function.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::fda::BSplineBasis;
using datamunge::fda::fit_scalar_on_function;
using datamunge::linalg::DenseMatrix;

TEST(ScalarOnFunctionRegression, IrregularGridMatchesIndependentRDesignAndPenalty) {
    const std::vector<double> grid{0,.2,.6,1.1,1.8,2.7,3};
    DenseMatrix<double> x(9,grid.size());
    for (std::size_t i = 1; i <= x.rows(); ++i)
        for (std::size_t j = 1; j <= x.cols(); ++j)
            x(i-1,j-1) = std::sin(.63*i+.47*j) + .1*i*std::cos(.15*i*j)
                + .14*(static_cast<double>((i+2*j)%4)-1.5);
    const std::vector<double> y{-.3,.2,.9,.8,.15,-.2,-.1,.8,1.2};
    const auto basis = BSplineBasis::open_uniform(3,5,0,3);
    const auto fit = fit_scalar_on_function(grid,x,y,basis,{.035,2});
    EXPECT_NEAR(fit.intercept,.30037488597379,1e-13);
    EXPECT_NEAR(fit.roughness,7.013200201992,3e-12);
    const std::array coefficients{-2.17481735242063,-.13720258620502,.96624999525465,
        .11918869067672,-2.25628250491163};
    ASSERT_EQ(fit.coefficients.size(),coefficients.size());
    for (std::size_t j = 0; j < coefficients.size(); ++j)
        EXPECT_NEAR(fit.coefficients[j],coefficients[j],1e-13);
    EXPECT_NEAR(fit.fitted[0],.0094904596770178,1e-13);
    EXPECT_NEAR(fit.fitted[8],1.1587403465943127,1e-13);
    EXPECT_DOUBLE_EQ(fit.residuals[8],y[8]-fit.fitted[8]);
    EXPECT_EQ(fit.predict(x),fit.fitted);
    const auto function = fit.coefficient_function({-.5,1.1,2.7,4});
    EXPECT_DOUBLE_EQ(function.front(),0);
    EXPECT_NEAR(function[1],.30438589363075,1e-13);
    EXPECT_NEAR(function[2],-1.05351815618023,1e-13);
    EXPECT_DOUBLE_EQ(function.back(),0);

    const auto rough = fit_scalar_on_function(grid,x,y,basis,{0,2});
    EXPECT_GT(rough.roughness,fit.roughness);
}

TEST(ScalarOnFunctionRegression, PenaltyIdentifiesUnderdeterminedFitWhenNullSpaceIsObserved) {
    const std::vector<double> grid{0,.25,.5,.75,1};
    const DenseMatrix<double> x(3,5,{0,0,0,0,0,1,1,1,1,1,0,0,0,1,2});
    const auto basis = BSplineBasis::open_uniform(3,5);
    const auto fit = fit_scalar_on_function(grid,x,{3,4,5},basis,{2,2});
    EXPECT_NEAR(fit.intercept,3,1e-13);
    EXPECT_NEAR(fit.roughness,0,1e-28);
    EXPECT_NEAR(fit.fitted[1],4,1e-13);
    EXPECT_NEAR(fit.fitted[2],5,1e-13);
    EXPECT_THROW((void)fit_scalar_on_function(grid,x,{3,4,5},basis,{0,2}),std::domain_error);

    const DenseMatrix<double> identical(8,5,1);
    EXPECT_THROW((void)fit_scalar_on_function(grid,identical,std::vector<double>(8),basis,{1,2}),
        std::domain_error);
}

TEST(ScalarOnFunctionRegression, ValidatesGridAndPredictionInputs) {
    const std::vector<double> grid{0,1};
    DenseMatrix<double> x(3,2,{0,0,1,0,0,1});
    const auto basis = BSplineBasis::open_uniform(1,2);
    const auto fit = fit_scalar_on_function(grid,x,{2,3,4},basis,{0,1});
    EXPECT_NEAR(fit.predict(DenseMatrix<double>(1,2,{1,1}))[0],5,1e-14);
    EXPECT_THROW((void)fit.predict(DenseMatrix<double>(1,3,0)),std::invalid_argument);
    EXPECT_THROW((void)fit_scalar_on_function({0,2},x,{2,3,4},basis,{0,1}),std::invalid_argument);
    EXPECT_THROW((void)fit_scalar_on_function(grid,x,{2,3,4},basis,{-1,1}),std::invalid_argument);
    x(0,0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)fit.predict(x),std::invalid_argument);
}
}
