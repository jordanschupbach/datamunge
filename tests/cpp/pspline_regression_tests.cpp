#include <gtest/gtest.h>

#include <datamunge/datamunge.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace dm = datamunge;

TEST(PSplineRegression, SecondOrderPenaltyLeavesLinesUnpenalized) {
    // A second-order difference penalty vanishes on lines, so any lambda reproduces one exactly.
    dm::Vector x = dm::seql(0.0, 2.0, 60);
    dm::Vector y = 1.5 - 0.75 * x;
    dm::PSR fit(x, y, 20, 1e4);
    EXPECT_EQ(fit.basis_size(), 20u);
    EXPECT_EQ(fit.degree(), 3u);
    EXPECT_EQ(fit.penalty_order(), 2u);
    EXPECT_DOUBLE_EQ(fit.lambda(), 1e4);
    EXPECT_NEAR(fit.r_squared(), 1.0, 1e-10);
    EXPECT_NEAR(fit.predict({1.25})[0], 1.5 - 0.75 * 1.25, 1e-8);
}

TEST(PSplineRegression, LargerLambdaMeansFewerEffectiveDf) {
    dm::set_seed(11);
    dm::Vector x = dm::unif(0.0, 1.0, 300);
    dm::Vector y = dm::sin(6.0 * x) + dm::rnorm(0.0, 0.2, 300);
    dm::PSR rough(x, y, 40, 1e-3);
    dm::PSR smooth(x, y, 40, 1e3);
    EXPECT_GT(rough.effective_df(), smooth.effective_df());
    EXPECT_LT(smooth.effective_df(), 10.0);
    EXPECT_GT(smooth.effective_df(), 2.0);  // never below the unpenalized line
}

TEST(PSplineRegression, GcvTracksNoisySignalWithRichBasis) {
    // More basis functions than BSplineRegression could support on this data: the penalty keeps it
    // well posed and GCV picks a lambda that is no worse than any of a few fixed choices.
    dm::set_seed(7);
    dm::Vector x = dm::unif(0.0, 1.0, 400);
    dm::Vector y = dm::sin(6.0 * x) + dm::rnorm(0.0, 0.1, 400);
    dm::PSR fit(x, y, 80);
    EXPECT_GT(fit.lambda(), 0.0);
    for (double lambda : {1e-4, 1e-1, 1e2, 1e5}) {
        EXPECT_LE(fit.gcv(), dm::PSR(x, y, 80, lambda).gcv() * (1.0 + 1e-9));
    }
    dm::Vector grid = dm::seql(0.05, 0.95, 50);
    dm::Vector err = dm::abs(fit.predict(grid) - dm::sin(6.0 * grid));
    for (double e : err) EXPECT_LT(e, 0.06);
    EXPECT_EQ(fit.fitted().size(), 400u);
}

TEST(PSplineRegression, PredictClampsOutsideDomain) {
    dm::Vector x = {0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7};
    dm::Vector y = 2.0 * x;
    dm::PSR fit(x, y, 6, 1.0);
    dm::Vector pred = fit.predict({0.0, 0.2, 0.8, 1.0});
    EXPECT_NEAR(pred[0], pred[1], 1e-12);
    EXPECT_NEAR(pred[2], pred[3], 1e-12);
    EXPECT_NEAR(pred[1], 0.4, 1e-8);
}

TEST(PSplineRegression, KnotConstructor) {
    dm::Vector x = dm::seql(0.0, 1.0, 200);
    dm::Vector y = 3.0 * x - 1.0;
    dm::PSR fit(x, y, dm::seql(0.0, 1.0, 11));
    EXPECT_EQ(fit.basis_size(), 13u);  // 9 interior knots + degree + 1
    EXPECT_NEAR(fit.predict({0.6})[0], 0.8, 1e-6);
    dm::PSR fixed(x, y, dm::seql(0.0, 1.0, 11), 2.0, 2, 1);
    EXPECT_EQ(fixed.degree(), 2u);
    EXPECT_EQ(fixed.penalty_order(), 1u);
}

TEST(PSplineRegression, RejectsBadInputs) {
    dm::Vector x = {0.0, 0.5, 1.0};
    EXPECT_THROW(dm::PSR(x, dm::Vector{1.0, 2.0}, 6), std::invalid_argument);
    EXPECT_THROW(dm::PSR(x, x, 3), std::invalid_argument);         // < degree + 1
    EXPECT_THROW(dm::PSR(x, x, 6, -1.0), std::invalid_argument);   // negative lambda
    EXPECT_THROW(dm::PSR(x, x, 2, 1.0, 1, 2), std::invalid_argument);  // basis <= penalty order
    EXPECT_NO_THROW(dm::PSR(x, x, 20, 1.0));  // rich basis on 3 points is fine when penalized
}

TEST(PSplineRegression, DiagnosticTablesAgreeWithClosedForm) {
    const dm::PSR fit({0,1,2},{0,1,0},3,1.0,1,2);
    const auto summary=fit.summary(), details=fit.observation_details();
    EXPECT_EQ(summary.nrows(),1u);
    EXPECT_EQ(details.nrows(),3u);
    EXPECT_NEAR(summary.numeric_column("rss").values()[0],24.0/49,1e-14);
    EXPECT_NEAR(summary.numeric_column("roughness").values()[0],4.0/49,1e-14);
    EXPECT_NEAR(summary.numeric_column("loo_cv").values()[0],3,1e-12);
    EXPECT_NEAR(details.numeric_column("loo_residual").values()[1],1,1e-13);
    EXPECT_DOUBLE_EQ(details.numeric_column("observation").values()[2],3);
    EXPECT_NEAR(fit.coefficient_differences()[0],-2.0/7,1e-14);
}

TEST(PSplineRegression, BasisMatrixAndPredictionsUseIdenticalBoundaryClamping) {
    const dm::PSR fit({0,1,2},{0,1,0},3,1.0,1,2);
    const dm::Vector<> points{-100,0,.5,1,2,100};
    const auto B=fit.basis_matrix(points);
    const auto predicted=B*fit.coefficients().values();
    for (std::size_t i=0;i<points.size();++i) {
        double sum=0; for (std::size_t j=0;j<B.cols();++j) sum+=B(i,j);
        EXPECT_NEAR(sum,1,1e-14);
        EXPECT_NEAR(predicted[i],fit.predict(points)[i],1e-14);
    }
    EXPECT_EQ(fit.basis_matrix({}).rows(),0u);
}

TEST(PSplineRegression, SmoothingPathMatchesAnalyticShrinkageAndOrder) {
    const dm::Vector<> lambdas{100,0.01,1,10};
    const auto path=dm::PSR::smoothing_path({0,1,2},{0,1,0},3,lambdas,1,2);
    for (std::size_t i=0;i<lambdas.size();++i) {
        const double lambda=lambdas[i],t=1+6*lambda;
        EXPECT_DOUBLE_EQ(path.numeric_column("lambda").values()[i],lambda);
        EXPECT_NEAR(path.numeric_column("edf").values()[i],2+1/t,1e-12);
        EXPECT_NEAR(path.numeric_column("roughness").values()[i],4/(t*t),1e-12);
        EXPECT_NEAR(path.numeric_column("rss").values()[i],24*lambda*lambda/(t*t),1e-12);
        EXPECT_NEAR(path.numeric_column("gcv").values()[i],2,1e-10);
    }
    EXPECT_THROW(dm::PSR::smoothing_path({0,1,2},{0,1,0},3,{},1,2),std::invalid_argument);
    EXPECT_THROW(dm::PSR::smoothing_path({0,1,2},{0,1,0},3,{1,-1},1,2),std::invalid_argument);
}

TEST(PSplineRegression, RoughnessAndEdfDecreaseWhileRssIncreases) {
    const dm::Vector<> x=dm::seql(0,1,41);
    const auto path=dm::PSR::smoothing_path(x,dm::sin(6*x)+0.2*dm::cos(43*x),15,
                                         {0.001,0.01,0.1,1,10,100,1000});
    for (std::size_t i=1;i<path.nrows();++i) {
        EXPECT_LE(path.numeric_column("edf").values()[i],path.numeric_column("edf").values()[i-1]);
        EXPECT_LE(path.numeric_column("roughness").values()[i],path.numeric_column("roughness").values()[i-1]);
        EXPECT_GE(path.numeric_column("rss").values()[i],path.numeric_column("rss").values()[i-1]);
    }
}

TEST(PSplineRegression, ExtendedUniformCubicNullspacePreservesLineAndNonnegativeRoughness) {
    const dm::Vector<> x=dm::seql(0,3,17),y=2-x;
    for (double lambda : {0.1,100.0,1e5}) {
        const dm::PSR fit(x,y,6,lambda);
        for (double r : fit.residuals()) EXPECT_NEAR(r,0,1e-8);
        for (double d : fit.coefficient_differences()) EXPECT_NEAR(d,0,1e-8);
        EXPECT_GE(fit.summary().numeric_column("roughness").values()[0],0);
    }
    // Repeated boundary knots have a different physical nullspace; avoid claiming a line there.
    const auto clamped=dm::fda::fit_pspline(x.values(),y.values(),3,6,{100.0,2});
    EXPECT_GT(clamped.residual_sum_of_squares,1e-3);
}

TEST(PSplineRegression, NonFiniteInputsAndPredictionAreRejected) {
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const double inf=std::numeric_limits<double>::infinity();
    EXPECT_THROW(dm::PSR({0,nan,2},{0,1,0},6),std::invalid_argument);
    EXPECT_THROW(dm::PSR({0,1,2},{0,inf,0},6),std::invalid_argument);
    EXPECT_THROW(dm::PSR({0,1,2},{0,1,0},6,inf),std::invalid_argument);
    EXPECT_THROW(dm::PSR({0,1,2},{0,1,0},dm::Vector<>{0,nan,2},1.0),std::invalid_argument);
    const dm::PSR fit({0,1,2},{0,1,0},6,1.0);
    EXPECT_THROW(fit.predict({inf}),std::invalid_argument);
    EXPECT_THROW(fit.basis_matrix({nan}),std::invalid_argument);
}
