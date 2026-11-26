#include <datamunge/stats/likelihood_tests.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using namespace datamunge::stats;

// mtcars: vs ~ mpg + wt + hp.
const std::vector<double> vs{0,0,1,1,0,1,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,0,1,0,0,0,1};
const std::vector<double> mpg{21,21,22.8,21.4,18.7,18.1,14.3,24.4,22.8,19.2,17.8,16.4,17.3,15.2,10.4,10.4,
                              14.7,32.4,30.4,33.9,21.5,15.5,15.2,13.3,19.2,27.3,26,30.4,15.8,19.7,15,21.4};
const std::vector<double> wt{2.62,2.875,2.32,3.215,3.44,3.46,3.57,3.19,3.15,3.44,3.44,4.07,3.73,3.78,5.25,5.424,
                             5.345,2.2,1.615,1.835,2.465,3.52,3.435,3.84,3.845,1.935,2.14,1.513,3.17,2.77,3.57,2.78};
const std::vector<double> hp{110,110,93,110,175,105,245,62,95,123,123,180,180,180,205,215,230,66,52,65,97,150,150,
                             245,175,66,91,113,264,175,335,109};

std::vector<double> row(std::size_t i) {
    return {1,mpg[i],wt[i],hp[i]};
}

double logistic_loglik(const std::vector<double>& b, std::size_t i) {
    const auto x = row(i);
    double eta = 0;
    for (std::size_t j = 0; j < b.size(); ++j)
        eta += b[j]*x[j];
    return vs[i]*eta-std::log1p(std::exp(eta));
}

std::vector<double> logistic_score(const std::vector<double>& b, std::size_t i) {
    const auto x = row(i);
    double eta = 0;
    for (std::size_t j = 0; j < b.size(); ++j)
        eta += b[j]*x[j];
    const double r = vs[i]-1/(1+std::exp(-eta));
    return {r*x[0],r*x[1],r*x[2],r*x[3]};
}

// Reference values: glm(vs ~ mpg + wt + hp, binomial, mtcars) against glm(vs ~ mpg) in R 4.5.3:
// anova(test = "LRT"), anova(test = "Rao") and the joint Wald statistic from vcov().
TEST(LikelihoodTests, ThreeTestsMatchRForLogisticRegression) {
    const auto full = maximum_likelihood(vs.size(),{0,0,0,0},logistic_loglik,{.gradient_tolerance=1e-9},logistic_score);
    ASSERT_TRUE(full.converged);
    EXPECT_NEAR(full.estimate[2],3.8774934118459012,1e-5);
    EXPECT_NEAR(full.log_likelihood,-7.37411695512968,1e-9);
    const auto restricted = maximum_likelihood(vs.size(),{0,0},
        [](const std::vector<double>& b, std::size_t i) { return logistic_loglik({b[0],b[1],0,0},i); },
        {.gradient_tolerance=1e-9});
    EXPECT_NEAR(restricted.log_likelihood,-12.7666675765465,1e-9);

    const auto lr = likelihood_ratio_test(full.log_likelihood,restricted.log_likelihood,2);
    EXPECT_NEAR(lr.statistic,10.7851012428337,1e-7);
    EXPECT_EQ(lr.rank,2U);

    const auto at = evaluate_likelihood(vs.size(),{restricted.estimate[0],restricted.estimate[1],0,0},
                                        logistic_loglik,{},logistic_score);
    const auto score = score_test(at,2);
    EXPECT_NEAR(score.statistic,5.78293761349629,1e-5);

    const DenseMatrix<double> r(2,4,{0,0,1,0,0,0,0,1});
    const auto wald = wald_test(full,r,{0,0});
    EXPECT_NEAR(wald.statistic,4.84468310390233,1e-4);
    EXPECT_EQ(wald.rank,2U);
    EXPECT_NEAR(wald.p_value,.0887136464550021,1e-5);

    // A redundant third row (sum of the first two) does not change the rank or statistic.
    const DenseMatrix<double> redundant(3,4,{0,0,1,0,0,0,0,1,0,0,1,1});
    const auto again = wald_test(full,redundant,{0,0,0});
    EXPECT_EQ(again.rank,2U);
    EXPECT_NEAR(again.statistic,wald.statistic,1e-8);
    EXPECT_NE(wald_test(full,r,{0,0},true).method,wald.method);
}

TEST(LikelihoodTests, OneParameterNormalMeanHasExactForms) {
    // Normal mean with known unit variance: all three statistics equal n * ybar^2.
    const std::vector<double> y{.3,1.1,-.2,.8,.5,1.4};
    const auto ll = [&](const std::vector<double>& t, std::size_t i) { return -.5*(y[i]-t[0])*(y[i]-t[0]); };
    const auto full = maximum_likelihood(y.size(),{0.0},ll);
    double mean = 0;
    for (const double v : y)
        mean += v/y.size();
    const double expected = y.size()*mean*mean;
    EXPECT_NEAR(wald_test(full,DenseMatrix<double>(1,1,{1.0}),{0}).statistic,expected,1e-5);
    EXPECT_NEAR(score_test(evaluate_likelihood(y.size(),{0.0},ll),1).statistic,expected,1e-5);
    EXPECT_NEAR(likelihood_ratio_test(full.log_likelihood,evaluate_likelihood(y.size(),{0.0},ll).log_likelihood,1)
                    .statistic,expected,1e-8);
}

TEST(LikelihoodTests, RejectsInvalidInput) {
    MaximumLikelihoodFit fit;
    fit.estimate = {1,2};
    fit.covariance = DenseMatrix<double>(2,2,{1,0,0,1});
    EXPECT_THROW((void)wald_test(fit,DenseMatrix<double>(1,3,{1,0,0}),{0}),std::invalid_argument);
    EXPECT_THROW((void)wald_test(fit,DenseMatrix<double>(1,2,{0,0}),{0}),std::invalid_argument);
    EXPECT_THROW((void)wald_test(fit,DenseMatrix<double>(1,2,{1,0}),{0},true),std::invalid_argument);
    EXPECT_THROW((void)likelihood_ratio_test(-10,-9,1),std::invalid_argument);
    EXPECT_THROW((void)likelihood_ratio_test(-10,-11,0),std::invalid_argument);
    EXPECT_THROW((void)score_test(fit,1),std::invalid_argument);
}

} // namespace
