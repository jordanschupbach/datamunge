#include <datamunge/stats/mmrm.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::fit_mmrm;
using datamunge::stats::MMRMStatus;

TEST(MMRM, CompleteVisitMeansReduceToMultivariateNormalEstimates) {
    const std::vector<double> y{0,1, 1,3, 2,2, 3,4};
    const std::vector<std::size_t> subject{0,0,1,1,2,2,3,3}, visit{1,8,1,8,1,8,1,8};
    for (const bool reml : {false, true}) {
        const auto fit = fit_mmrm(y, subject, visit, {.reml = reml});
        ASSERT_EQ(fit.status, MMRMStatus::Converged);
        EXPECT_EQ(fit.visit_levels, (std::vector<std::size_t>{1,8}));
        EXPECT_NEAR(fit.coefficients[0], 1.5, 1e-12);
        EXPECT_NEAR(fit.coefficients[1], 2.5, 1e-12);
        const double df = reml ? 3 : 4;
        EXPECT_NEAR(fit.visit_covariance(0,0), 5 / df, 1e-8);
        EXPECT_NEAR(fit.visit_covariance(1,1), 5 / df, 1e-8);
        EXPECT_NEAR(fit.visit_covariance(1,0), 4 / df, 1e-8);
        EXPECT_NEAR(fit.covariance(0,1), 1 / df, 1e-8);
        EXPECT_NEAR(fit.covariance(1,1), 5 / (4 * df), 1e-8);
        EXPECT_NEAR(fit.fitted.at(6), 1.5, 1e-12);
        EXPECT_NEAR(fit.predict(DenseMatrix<double>(1,2,{-1,1})).at(0), 1, 1e-12);
        EXPECT_THROW((void)fit.predict(DenseMatrix<double>(1,1,{1})), std::invalid_argument);
    }
}

TEST(MMRM, IncompleteVisitsMatchIndependentGaussianProfile) {
    std::vector<std::size_t> subject, visit;
    std::vector<double> y, predictors;
    const std::array<std::size_t,3> labels{3,8,11};
    for (int s = 24; s >= 1; --s) {
        for (int v = 2; v >= 0; --v) {
            if ((s % 7 == 0 && v != 0) || (v == 1 && s % 6 == 0) || (v == 2 && s % 4 == 0))
                continue;
            const double dose = std::sin(s + .35 * v);
            y.push_back(2 + .4*v + .65*dose + .53*std::sin(.81*s)*(1 + .37*v) +
                        .31*std::sin(2*s + 3*v) + .26*std::cos(3*s - 2*v));
            subject.push_back(100 + 3*s);
            visit.push_back(labels[v]);
            predictors.insert(predictors.end(), {1,static_cast<double>(v),dose});
        }
    }
    const DenseMatrix<double> x(y.size(),3,predictors);
    // Base R optim with an independent dense observed-block Gaussian profile.
    const std::array<double,6> sigma[]{
        {.210821018464,.098779143235,.316253925296,.258471640627,.247502666042,.518054872309},
        {.223214822563,.111459378537,.333440684358,.272182687137,.269058495318,.550257925092}};
    const std::array<double,3> beta[]{
        {2.028037325205,.370545591371,.605575738294},
        {2.026045768130,.370411915055,.604433738237}};
    const double likelihood[]{-35.3575134538,-40.2571354815};
    const double slope_variance[]{.003243499195411,.00350842376980};
    for (const bool reml : {false, true}) {
        SCOPED_TRACE(reml);
        const auto fit = fit_mmrm(x, y, subject, visit, {.reml = reml});
        ASSERT_EQ(fit.status, MMRMStatus::Converged);
        EXPECT_EQ(fit.visit_levels, (std::vector<std::size_t>{3,8,11}));
        EXPECT_NEAR(fit.log_likelihood, likelihood[reml], 1e-8);
        EXPECT_NEAR(fit.covariance(1,1), slope_variance[reml], 2e-8);
        std::size_t k = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(fit.coefficients[i], beta[reml][i], 2e-7);
            for (std::size_t j = 0; j <= i; ++j)
                EXPECT_NEAR(fit.visit_covariance(i,j), sigma[reml][k++], 8e-7);
        }
    }
}

TEST(MMRM, RejectsDuplicateVisitsAndUnobservedCovariancePairs) {
    EXPECT_THROW((void)fit_mmrm({1,2,3,4}, {1,1,2,2}, {1,1,1,2}), std::invalid_argument);
    EXPECT_THROW((void)fit_mmrm({1,2,3,4}, {1,2,3,4}, {1,1,2,2}), std::invalid_argument);
    EXPECT_THROW((void)fit_mmrm({1,2,1,2}, {1,1,2,2}, {1,2,1,2}), std::invalid_argument);
    EXPECT_THROW((void)fit_mmrm({1,2,3,4}, {1,1,2,2}, {1,2,1,2}, {.tolerance = 0}),
                 std::invalid_argument);
}

TEST(MMRM, SingleVisitRemlAgreesWithOrdinaryLeastSquares) {
    const DenseMatrix<double> x(5,2,{1,0,1,1,1,2,1,3,1,4});
    const auto fit = fit_mmrm(x, {2,2,4,4,8}, {0,1,2,3,4}, {6,6,6,6,6});
    ASSERT_EQ(fit.status, MMRMStatus::Converged);
    EXPECT_NEAR(fit.coefficients[0], 1.2, 1e-12);
    EXPECT_NEAR(fit.coefficients[1], 1.4, 1e-12);
    EXPECT_NEAR(fit.visit_covariance(0,0), 4.4 / 3, 1e-8);
    EXPECT_NEAR(fit.covariance(1,1), 4.4 / 30, 1e-8);
}
} // namespace
