#include <gtest/gtest.h>

#include <datamunge/fda/pspline.hpp>

#include <cmath>
#include <limits>
#include <vector>

using datamunge::fda::BSplineBasis;
using datamunge::fda::PSplineOptions;
using datamunge::fda::fit_pspline;
using datamunge::fda::pspline_penalty_matrix;

TEST(PSpline, SecondOrderPenaltyMatchesKnownStructure) {
    const auto penalty = pspline_penalty_matrix(5, 2);

    const double expected[5][5] = {
        {1, -2, 1, 0, 0},
        {-2, 5, -4, 1, 0},
        {1, -4, 6, -4, 1},
        {0, 1, -4, 5, -2},
        {0, 0, 1, -2, 1},
    };

    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            EXPECT_DOUBLE_EQ(penalty(i, j), expected[i][j]);
}

TEST(PSpline, ZeroPenaltyInterpolatesNoiselessDataWell) {
    std::vector<double> x(20);
    std::vector<double> y(20);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<double>(i) / static_cast<double>(x.size() - 1);
        y[i] = std::sin(2.0 * 3.14159265358979323846 * x[i]);
    }

    PSplineOptions opts;
    opts.lambda = 0.0;
    opts.penalty_order = 2;
    const auto fit = fit_pspline(x, y, 3, 8, opts);

    EXPECT_LT(fit.residual_sum_of_squares, 0.02);
    EXPECT_GT(fit.r_squared, 0.99);
    EXPECT_EQ(fit.fitted.size(), x.size());
}

TEST(PSpline, LargerPenaltyShrinksRoughness) {
    std::vector<double> x(40);
    std::vector<double> y(40);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 2.0 * 3.14159265358979323846 * static_cast<double>(i)
             / static_cast<double>(x.size() - 1);
        y[i] = std::sin(x[i]) + 0.2 * std::cos(4.0 * x[i]);
    }

    const auto basis = BSplineBasis::open_uniform(3, 14, x.front(), x.back());

    PSplineOptions light;
    light.lambda = 1e-4;
    light.penalty_order = 2;
    const auto fit_light = fit_pspline(x, y, basis, light);

    PSplineOptions heavy;
    heavy.lambda = 10.0;
    heavy.penalty_order = 2;
    const auto fit_heavy = fit_pspline(x, y, basis, heavy);

    EXPECT_LT(fit_heavy.penalty, fit_light.penalty);
    EXPECT_LT(fit_heavy.effective_degrees_of_freedom, fit_light.effective_degrees_of_freedom);
    EXPECT_EQ(fit_heavy.predict({x.front(), x.back()}).size(), 2U);
}

TEST(PSpline, RejectsInvalidPenaltyOrder) {
    std::vector<double> x = {0.0, 0.5, 1.0};
    std::vector<double> y = {0.0, 1.0, 0.0};

    PSplineOptions opts;
    opts.penalty_order = 0;
    EXPECT_THROW(fit_pspline(x, y, 1, 3, opts), std::invalid_argument);

    EXPECT_THROW(pspline_penalty_matrix(3, 3), std::invalid_argument);
}

TEST(PSpline, ThreePointClosedFormAndFixedLambdaDeletion) {
    const auto fit=fit_pspline({0,1,2},{0,1,0},1,3,{1.0,2});
    const double beta[]={2.0/7,3.0/7,2.0/7};
    const double leverage[]={6.0/7,3.0/7,6.0/7};
    const double loo[]={-2,1,-2};
    for (std::size_t i=0;i<3;++i) {
        EXPECT_NEAR(fit.coefficients[i],beta[i],1e-14);
        EXPECT_NEAR(fit.leverage[i],leverage[i],1e-14);
        EXPECT_NEAR(fit.loo_residuals()[i],loo[i],1e-13);
    }
    EXPECT_NEAR(fit.penalty,4.0/49,1e-14);
    EXPECT_NEAR(fit.residual_sum_of_squares,24.0/49,1e-14);
    EXPECT_NEAR(fit.objective,4.0/7,1e-14);
    EXPECT_NEAR(fit.effective_degrees_of_freedom,15.0/7,1e-14);
    EXPECT_NEAR(fit.gcv(),2.0,1e-13);
    EXPECT_NEAR(fit.loo_cv(),3.0,1e-12);
}

TEST(PSpline, DifferenceGramAndPolynomialNullspaceAtSeveralOrders) {
    for (std::size_t d=1;d<=4;++d) {
        const auto D=datamunge::fda::pspline_difference_matrix(8,d);
        const auto P=pspline_penalty_matrix(8,d);
        const auto gram=D.transpose()*D;
        EXPECT_EQ(D.rows(),8-d);
        for (std::size_t i=0;i<8;++i)
            for (std::size_t j=0;j<8;++j) EXPECT_DOUBLE_EQ(P(i,j),gram(i,j));
        for (std::size_t power=0;power<d;++power) {
            std::vector<double> beta(8);
            for (std::size_t i=0;i<8;++i) beta[i]=std::pow(static_cast<double>(i),power);
            for (double delta : D*beta) EXPECT_NEAR(delta,0.0,1e-10);
        }
    }
}

TEST(PSpline, LeverageAndLooAgreeWithIndependentFixedBasisRefits) {
    const std::vector<double> x{0,.1,.3,.4,.65,.7,.85,1};
    const std::vector<double> y{0,1,.4,-.2,.7,.8,-.3,0};
    const auto basis=BSplineBasis::open_uniform(3,6,0,1);
    for (double lambda : {0.01,1.0,100.0}) {
        const auto fit=fit_pspline(x,y,basis,{lambda,2});
        const auto B=basis.evaluate_dense(x);
        const auto A=B.transpose()*B+lambda*fit.penalty_matrix;
        // Unit-response fits independently reconstruct the full smoother column by column.
        double trace=0;
        for (std::size_t i=0;i<x.size();++i) {
            std::vector<double> unit(x.size(),0); unit[i]=1;
            const auto column=fit_pspline(x,unit,basis,{lambda,2});
            EXPECT_NEAR(column.fitted[i],fit.leverage[i],1e-11);
            trace+=column.fitted[i];
            auto xminus=x,yminus=y;
            xminus.erase(xminus.begin()+i); yminus.erase(yminus.begin()+i);
            const auto deleted=fit_pspline(xminus,yminus,basis,{lambda,2});
            EXPECT_NEAR(y[i]-deleted.predict({x[i]})[0],fit.loo_residuals()[i],1e-10);
        }
        EXPECT_NEAR(trace,fit.effective_degrees_of_freedom,1e-11);
        const auto rhs=B.transpose()*y;
        const auto lhs=A*fit.coefficients;
        for (std::size_t j=0;j<rhs.size();++j) EXPECT_NEAR(lhs[j],rhs[j],1e-10);
    }
}

TEST(PSpline, InterpolatingAndUnidentifiedDeletionScoresAreExplicit) {
    const auto fit=fit_pspline({0,1,2},{0,1,0},1,3,{0.0,2});
    EXPECT_TRUE(std::isinf(fit.gcv()));
    EXPECT_TRUE(std::isinf(fit.loo_cv()));
    for (double value : fit.loo_residuals()) EXPECT_TRUE(std::isnan(value));
    EXPECT_THROW(fit_pspline({0},{1},BSplineBasis::open_uniform(3,6,0,1),{1.0,2}),std::runtime_error);
}

TEST(PSpline, RejectsNonFiniteValuesAndOutOfDomainTraining) {
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const double inf=std::numeric_limits<double>::infinity();
    const auto basis=BSplineBasis::open_uniform(1,3,0,2);
    EXPECT_THROW(fit_pspline({0,1,2},{0,nan,0},basis),std::invalid_argument);
    EXPECT_THROW(fit_pspline({0,inf,2},{0,1,0},basis),std::invalid_argument);
    EXPECT_THROW(fit_pspline({-1,1,2},{0,1,0},basis),std::invalid_argument);
    EXPECT_THROW(fit_pspline({0,1,2},{0,1,0},basis,{inf,2}),std::invalid_argument);
    EXPECT_THROW(fit_pspline({0,1,2},{0,1,0},basis,{nan,2}),std::invalid_argument);
    EXPECT_THROW(fit_pspline({0,nan,2},{0,1,0},1,3),std::invalid_argument);
    const auto fit=fit_pspline({0,1,2},{0,1,0},basis);
    EXPECT_THROW(fit.predict({inf}),std::invalid_argument);
    EXPECT_DOUBLE_EQ(fit.predict({-1})[0],0.0); // Low-level basis semantics are preserved.
    EXPECT_THROW(datamunge::fda::pspline_difference_matrix(3,0),std::invalid_argument);
}
