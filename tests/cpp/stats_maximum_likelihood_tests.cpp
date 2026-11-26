#include <datamunge/stats/maximum_likelihood.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using datamunge::stats::maximum_likelihood;

const std::vector<double> gamma_data{2.3,.8,1.9,4.1,1.2,.6,2.8,3.5,1.1,2.2,.9,5.0};

double gamma_loglik(const std::vector<double>& t, std::size_t i) {
    const double y = gamma_data[i];
    return t[0]*std::log(t[1])-std::lgamma(t[0])+(t[0]-1)*std::log(y)-t[1]*y;
}

// Reference values: profile root of the gamma likelihood, J = n [trigamma(a), -1/b; -1/b, a/b^2],
// K = crossprod(scores) and J^-1 K J^-1 computed in R 4.5.3.
TEST(MaximumLikelihood, GammaFitMatchesRInformationAndSandwich) {
    const auto fit = maximum_likelihood(gamma_data.size(),{1.0,1.0},gamma_loglik,{.lower={0.0,0.0}});
    ASSERT_TRUE(fit.converged);
    EXPECT_NEAR(fit.estimate[0],2.59640837853042,2e-6);
    EXPECT_NEAR(fit.estimate[1],1.18018562660474,1e-6);
    EXPECT_NEAR(fit.log_likelihood,-19.0699209702528,1e-10);
    EXPECT_NEAR(fit.observed_information(0,0),5.62297677786897,1e-4);
    EXPECT_NEAR(fit.observed_information(0,1),-10.1678920073977,1e-4);
    EXPECT_NEAR(fit.observed_information(1,1),22.3693624162750,1e-4);
    EXPECT_NEAR(fit.standard_errors[0],.999396704588886,1e-4);
    EXPECT_NEAR(fit.standard_errors[1],.501065216145278,1e-4);
    EXPECT_NEAR(fit.outer_product(0,0),5.21386054470631,1e-4);
    EXPECT_NEAR(fit.outer_product(0,1),-10.3039275463014,1e-4);
    EXPECT_NEAR(fit.outer_product(1,1),22.02,1e-4);
    EXPECT_NEAR(fit.robust_covariance(0,0),.395285126042676,1e-4);
    EXPECT_NEAR(fit.robust_covariance(0,1),.166510578490261,1e-4);
    EXPECT_NEAR(fit.robust_covariance(1,1),.113708591073987,1e-4);
    EXPECT_FALSE(fit.at_bound[0]);
    EXPECT_EQ(fit.score_contributions.rows(),gamma_data.size());
}

TEST(MaximumLikelihood, AnalyticScoreGivesTheSameFit) {
    // d/da = log b - digamma(a) + log y and d/db = a/b - y; digamma by its derivative of lgamma.
    const auto digamma = [](double a) {
        const double h = 1e-5*a;
        return (std::lgamma(a+h)-std::lgamma(a-h))/(2*h);
    };
    const auto score = [&](const std::vector<double>& t, std::size_t i) {
        return std::vector<double>{std::log(t[1])-digamma(t[0])+std::log(gamma_data[i]),t[0]/t[1]-gamma_data[i]};
    };
    const auto fit = maximum_likelihood(gamma_data.size(),{1.0,1.0},gamma_loglik,{.lower={0.0,0.0}},score);
    ASSERT_TRUE(fit.converged);
    EXPECT_NEAR(fit.estimate[0],2.59640837853042,1e-5);
    EXPECT_NEAR(fit.standard_errors[1],.501065216145278,1e-4);
    double total0 = 0, total1 = 0;
    for (std::size_t i = 0; i < gamma_data.size(); ++i) {
        total0 += fit.score_contributions(i,0);
        total1 += fit.score_contributions(i,1);
    }
    EXPECT_NEAR(total0,0,1e-5);
    EXPECT_NEAR(total1,0,1e-5);
}

TEST(MaximumLikelihood, NormalModelHasClosedFormEstimatesAndInformation) {
    const std::vector<double> y{4.2,5.1,3.8,6.0,5.5,4.9,5.3,4.4};
    double mean = 0, ss = 0;
    for (const double v : y)
        mean += v/y.size();
    for (const double v : y)
        ss += (v-mean)*(v-mean);
    const double sigma = std::sqrt(ss/y.size());
    const auto fit = maximum_likelihood(y.size(),{0.0,1.0},
        [&](const std::vector<double>& t, std::size_t i) {
            const double z = (y[i]-t[0])/t[1];
            return -std::log(t[1])-.5*z*z;
        },
        {.lower={-std::numeric_limits<double>::infinity(),0.0}});
    ASSERT_TRUE(fit.converged);
    EXPECT_NEAR(fit.estimate[0],mean,1e-6);
    EXPECT_NEAR(fit.estimate[1],sigma,1e-6);
    EXPECT_NEAR(fit.observed_information(0,0),y.size()/(sigma*sigma),1e-3);
    EXPECT_NEAR(fit.observed_information(1,1),2*y.size()/(sigma*sigma),1e-3);
    EXPECT_NEAR(fit.observed_information(0,1),0,1e-3);
    EXPECT_NEAR(fit.standard_errors[0],sigma/std::sqrt(8.0),1e-5);
}

TEST(MaximumLikelihood, ActiveBoundIsReportedAndTwoSidedBoundsWork) {
    // Poisson counts with mean 4.5 but the rate restricted to [0.5, 3].
    const std::vector<double> counts{3,6,4,5,7,2};
    const auto fit = maximum_likelihood(counts.size(),{1.0},
        [&](const std::vector<double>& t, std::size_t i) { return counts[i]*std::log(t[0])-t[0]; },
        {.lower={.5},.upper={3.0},.max_iterations=2000});
    EXPECT_GT(fit.estimate[0],2.999);
    EXPECT_TRUE(fit.at_bound[0]);
    // Inside a two-sided box the unconstrained maximum is recovered.
    const auto inside = maximum_likelihood(counts.size(),{1.0},
        [&](const std::vector<double>& t, std::size_t i) { return counts[i]*std::log(t[0])-t[0]; },
        {.lower={.5},.upper={10.0}});
    EXPECT_TRUE(inside.converged);
    EXPECT_NEAR(inside.estimate[0],4.5,1e-6);
    EXPECT_FALSE(inside.at_bound[0]);
    EXPECT_NEAR(inside.standard_errors[0],std::sqrt(4.5/6),1e-5);
}

TEST(MaximumLikelihood, RejectsInvalidInput) {
    const auto ll = [](const std::vector<double>& t, std::size_t) { return -t[0]*t[0]; };
    EXPECT_THROW((void)maximum_likelihood(0,{1.0},ll),std::invalid_argument);
    EXPECT_THROW((void)maximum_likelihood(1,{},ll),std::invalid_argument);
    EXPECT_THROW((void)maximum_likelihood(1,{1.0},{}),std::invalid_argument);
    EXPECT_THROW((void)maximum_likelihood(1,{1.0},ll,{.lower={2.0}}),std::invalid_argument);
    EXPECT_THROW((void)maximum_likelihood(1,{1.0},ll,{.lower={0.0,0.0}}),std::invalid_argument);
    EXPECT_THROW((void)maximum_likelihood(1,{1.0},[](const std::vector<double>&, std::size_t) { return std::nan(""); }),
                 std::invalid_argument);
    EXPECT_THROW((void)maximum_likelihood(1,{1.0},ll,{},
                     [](const std::vector<double>&, std::size_t) { return std::vector<double>{1,2}; }),
                 std::invalid_argument);
}

} // namespace
