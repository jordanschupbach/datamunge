#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/stats/graphical_lasso.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fit_graphical_lasso;
using datamunge::stats::graphical_lasso;

// Constructed in R from a known sparse precision, with strict slack on unused edges.
const DenseMatrix<double> sample(5,5,{
    .75637178518412529,.29366805349382075,.01170548441328493,-.07433849238996298,0,
    .29366805349382075,.63648687469819187,-.24003422728166973,.06312808515014398,.09,
    .01170548441328493,-.24003422728166973,.84224838460771967,.25914475230651701,0,
    -.07433849238996298,.06312808515014398,.25914475230651701,.68646886202033264,-.11,
    0,.09,0,-.11,2.5});

TEST(GraphicalLasso, RecoversAUniqueOptimumWithActiveAndInactiveEdges) {
    const double penalty = .13;
    const auto fit = graphical_lasso(sample,penalty,{.tolerance=1e-11});
    ASSERT_TRUE(fit.converged);
    EXPECT_LE(fit.max_kkt_violation,1e-11);
    const DenseMatrix<double> expected(5,5,{
        1.4,-.36,0,0,0,-.36,1.7,.21,0,0,0,.21,1.25,-.23,0,0,0,-.23,1.5,0,0,0,0,0,.4});
    ASSERT_EQ(fit.edges.size(),3u);
    EXPECT_EQ(fit.edges[0].first,0u);
    EXPECT_EQ(fit.edges[0].second,1u);
    EXPECT_EQ(fit.edges[2].first,2u);
    EXPECT_EQ(fit.edges[2].second,3u);
    EXPECT_NEAR(fit.edges[0].partial_correlation,.2333533404787792,1e-11);
    EXPECT_NEAR(fit.edges[1].partial_correlation,-.1440588115197074,1e-11);
    EXPECT_NEAR(fit.edges[2].partial_correlation,.1679682509682509,1e-11);
    EXPECT_NEAR(fit.objective,4.528037330926481,1e-12);
    const auto identity = fit.covariance*fit.precision;
    for (std::size_t j = 0; j < expected.rows(); ++j) {
        EXPECT_NEAR(fit.covariance(j,j),sample(j,j),1e-11);
        for (std::size_t k = 0; k < expected.cols(); ++k) {
            EXPECT_NEAR(identity(j,k),j==k ? 1 : 0,1e-13);
            const auto difference = fit.covariance(j,k)-sample(j,k);
            if (j != k && expected(j,k) == 0) {
                EXPECT_DOUBLE_EQ(fit.precision(j,k),0);
                EXPECT_LE(std::abs(difference),.13+1e-11);
            } else {
                EXPECT_NEAR(fit.precision(j,k),expected(j,k),5e-11);
                if (j != k)
                    EXPECT_NEAR(difference,std::copysign(penalty,expected(j,k)),1e-11);
            }
        }
    }
}

TEST(GraphicalLasso, TwoVariablesReduceToCovarianceSoftThresholding) {
    const DenseMatrix<double> s(2,2,{2,-.7,-.7,1.2});
    const auto fitted = graphical_lasso(s,.18,{.tolerance=1e-11});
    ASSERT_TRUE(fitted.converged);
    EXPECT_NEAR(fitted.covariance(0,1),-.52,1e-11);
    EXPECT_NEAR(fitted.precision(0,1),.52/(2*1.2-.52*.52),1e-11);
    EXPECT_NEAR(fitted.edges.front().partial_correlation,-.52/std::sqrt(2*1.2),1e-11);

    const auto independent = graphical_lasso(s,.7);
    EXPECT_TRUE(independent.converged);
    EXPECT_TRUE(independent.edges.empty());
    EXPECT_EQ(independent.iterations,0u);
    EXPECT_DOUBLE_EQ(independent.covariance(0,1),0);
    EXPECT_DOUBLE_EQ(independent.precision(0,0),.5);

    const auto zero = graphical_lasso(s,0);
    EXPECT_TRUE(zero.converged);
    EXPECT_NEAR(zero.precision(0,1),.7/(2*1.2-.7*.7),1e-14);
    EXPECT_NEAR(zero.covariance(1,1),1.2,1e-14);
}

TEST(GraphicalLasso, RawFittingMatchesCenteredCovarianceAndHandlesMoreColumnsThanRows) {
    const DenseMatrix<double> x(4,6,{
        -1,1,-1,1,2,0, 1,-1,-1,-1,3,2, 1,1,1,-1,2,3, -1,-1,1,1,5,4});
    auto centered = x;
    for (std::size_t j = 0; j < x.cols(); ++j) {
        double mean = 0;
        for (std::size_t i = 0; i < x.rows(); ++i)
            mean += x(i,j)/x.rows();
        for (std::size_t i = 0; i < x.rows(); ++i)
            centered(i,j) -= mean;
    }
    const auto covariance = (centered.transpose()*centered)*.25;
    const auto fit = fit_graphical_lasso(x,.35);
    const auto supplied = graphical_lasso(covariance,.35);
    EXPECT_TRUE(fit.converged);
    EXPECT_TRUE(supplied.converged);
    EXPECT_TRUE(datamunge::linalg::cholesky(fit.precision).ok);
    EXPECT_EQ(fit.edges.size(),supplied.edges.size());
    for (std::size_t j = 0; j < x.cols(); ++j)
        for (std::size_t k = 0; k < x.cols(); ++k)
            EXPECT_NEAR(fit.precision(j,k),supplied.precision(j,k),1e-13);
    EXPECT_THROW((void)fit_graphical_lasso(x,0),std::domain_error);
}

TEST(GraphicalLasso, IterationLimitReturnsFeasibleSparsePrecision) {
    const auto fit = graphical_lasso(sample,.13,{.max_iterations=1,.tolerance=1e-12});
    EXPECT_FALSE(fit.converged);
    EXPECT_EQ(fit.iterations,1u);
    EXPECT_GT(fit.max_kkt_violation,.01);
    EXPECT_TRUE(datamunge::linalg::cholesky(fit.precision).ok);
    EXPECT_DOUBLE_EQ(fit.precision(0,4),0);
    EXPECT_DOUBLE_EQ(fit.covariance(0,4),0);
}

TEST(GraphicalLasso, ValidatesCovariancesAndControls) {
    EXPECT_THROW((void)graphical_lasso(sample,-.1),std::invalid_argument);
    EXPECT_THROW((void)graphical_lasso(sample,.1,{.max_iterations=0}),std::invalid_argument);
    EXPECT_THROW((void)graphical_lasso(DenseMatrix<double>(2,2,{1,.5,0,1}),.1),std::invalid_argument);
    EXPECT_THROW((void)graphical_lasso(DenseMatrix<double>(2,2,{1,2,2,1}),.1),std::domain_error);
    EXPECT_THROW((void)graphical_lasso(DenseMatrix<double>(2,2,{1,0,0,0}),.1),std::domain_error);
    EXPECT_THROW((void)fit_graphical_lasso(DenseMatrix<double>(2,1,{0,std::numeric_limits<double>::infinity()}),.1),
        std::invalid_argument);
}
} // namespace
