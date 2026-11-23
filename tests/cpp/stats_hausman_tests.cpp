#include <datamunge/stats/gls.hpp>
#include <datamunge/stats/hausman_test.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::generalized_least_squares;
using datamunge::stats::HausmanEstimates;
using datamunge::stats::hausman_test;

TEST(HausmanTest, MatchesSharedTermsByNameAcrossDifferentModels) {
    const HausmanEstimates fixed{{"speed","baseline","common"}, {2.1,.1,-.5},
        DenseMatrix<double>(3,3,{.73,0,.12,0,.5,0,.12,0,.57})};
    const HausmanEstimates random{{"common","intercept","speed"}, {.25,3,.8},
        DenseMatrix<double>(3,3,{.29,-.03,.03,-.03,.6,0,.03,0,.39})};
    const auto test = hausman_test(fixed, random);
    EXPECT_EQ(test.terms,(std::vector<std::string>{"speed","common"}));
    EXPECT_EQ(test.df,2u);
    EXPECT_NEAR(test.difference[0],1.3,1e-14);
    EXPECT_DOUBLE_EQ(test.difference[1],-.75);
    EXPECT_NEAR(test.covariance_difference(0,0),.34,1e-14);
    EXPECT_NEAR(test.covariance_difference(0,1),.09,1e-14);
    EXPECT_NEAR(test.covariance_difference(1,1),.28,1e-14);
    EXPECT_NEAR(test.statistic,9.64351320321469352,1e-12);
    EXPECT_NEAR(test.p_value,.00805262944619848,1e-14);
    EXPECT_EQ(test.untested_difference,(std::vector<double>{0,0}));
}

TEST(HausmanTest, UsesPositiveCovarianceRankAndReturnsDiscardedDifference) {
    HausmanEstimates fixed{{"speed","size"}, {5.5,4}, DenseMatrix<double>(2,2,{5,2,2,8})};
    HausmanEstimates random{{"speed","size"}, {0,0}, DenseMatrix<double>(2,2,{4,0,0,4})};
    const auto test = hausman_test(fixed,random);
    EXPECT_EQ(test.df,1u);
    EXPECT_NEAR(test.statistic,9,1e-13);
    EXPECT_NEAR(test.p_value,.00269979606326019,1e-14);
    EXPECT_NEAR(test.untested_difference[0],2.5,1e-13);
    EXPECT_NEAR(test.untested_difference[1],-2,1e-13);

    const double conversion[]{1e-100,1e100};
    for (auto* fit : {&fixed,&random})
        for (std::size_t i = 0; i < 2; ++i) {
            fit->coefficients[i] *= conversion[i];
            for (std::size_t j = 0; j < 2; ++j)
                fit->covariance(i,j) *= conversion[i] * conversion[j];
        }
    const auto rescaled = hausman_test(fixed,random);
    EXPECT_EQ(rescaled.df,test.df);
    EXPECT_NEAR(rescaled.statistic,test.statistic,1e-13);
    EXPECT_NEAR(rescaled.untested_difference[0] / conversion[0],2.5,1e-13);
    EXPECT_NEAR(rescaled.untested_difference[1] / conversion[1],-2,1e-13);
}

TEST(HausmanTest, RejectsIndefiniteOrRankZeroDifferencesAndInvalidInputs) {
    const HausmanEstimates fit{{"a","b"}, {1,2}, DenseMatrix<double>(2,2,{1,0,0,2})};
    EXPECT_THROW((void)hausman_test(fit,fit), std::domain_error);
    auto other = fit;
    other.covariance(0,0) = .95;
    other.covariance(1,1) = 2.01;
    EXPECT_THROW((void)hausman_test(fit,other), std::domain_error);
    EXPECT_EQ(hausman_test(fit,other,.01).df,1u);
    other.covariance(0,1) = other.covariance(1,0) = 2;
    EXPECT_THROW((void)hausman_test(fit,other), std::domain_error);
    other.covariance(0,1) = 0;
    EXPECT_THROW((void)hausman_test(fit,other), std::invalid_argument);
    other = fit;
    other.terms = {"c","d"};
    EXPECT_THROW((void)hausman_test(fit,other), std::invalid_argument);
    other.terms = {"a","a"};
    EXPECT_THROW((void)hausman_test(other,fit), std::invalid_argument);
    EXPECT_THROW((void)hausman_test(fit,fit,std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST(HausmanTest, KnownVariancePanelMatchesIndependentFixedAndRandomEffectsGLS) {
    constexpr std::size_t units = 12, visits = 4, n = units * visits;
    DenseMatrix<double> fixed(n,units+1), random(n,3,1), marginal(n,n);
    std::vector<double> y(n);
    std::vector<std::string> fixed_terms{"setting"};
    for (std::size_t u = 0; u < units; ++u) {
        const double s = u + 1;
        fixed_terms.push_back("site_" + std::to_string(u+1));
        const auto latent = std::sin(.8*s), layout = std::cos(.45*s);
        for (std::size_t v = 0; v < visits; ++v) {
            const double t = v + 1;
            const std::size_t i = u * visits + v;
            const auto setting = 1.5*latent + .25*std::sin(s*t) + .15*t;
            fixed(i,0) = random(i,2) = setting;
            fixed(i,u+1) = 1;
            random(i,1) = layout;
            y[i] = 3+.85*setting+.5*layout+1.2*latent+.35*std::cos(s+2*t)+.2*std::sin(2*s-t);
            for (std::size_t k = 0; k < visits; ++k)
                marginal(i,u*visits+k) = .64;
            marginal(i,i) += .16;
        }
    }
    const auto f = generalized_least_squares(fixed,y,.16 * DenseMatrix<double>::identity(n));
    const auto r = generalized_least_squares(random,y,marginal);
    const auto test = hausman_test({fixed_terms,f.coefficients,f.covariance},
        {{"intercept","layout","setting"},r.coefficients,r.covariance});
    ASSERT_EQ(test.terms,(std::vector<std::string>{"setting"}));
    EXPECT_NEAR(f.coefficients[0],.57908851443822773,1e-12);
    EXPECT_NEAR(r.coefficients[2],1.04825595167111496,1e-12);
    EXPECT_NEAR(test.covariance_difference(0,0),.03019896006834263,1e-12);
    EXPECT_NEAR(test.statistic,7.28892927642311506,1e-12);
    EXPECT_NEAR(test.p_value,.00693808143489002,1e-13);
}
} // namespace
