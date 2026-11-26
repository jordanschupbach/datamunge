#include <datamunge/stats/aft.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_aft_data.inc"

Matrix columns(const std::vector<std::vector<double>>& cols) {
    Matrix x(cols[0].size(),cols.size());
    for (std::size_t j = 0; j < cols.size(); ++j)
        for (std::size_t i = 0; i < cols[j].size(); ++i)
            x(i,j) = cols[j][i];
    return x;
}

// vcov is column-major (R) over (coefficients, Log(scale)); loglik is (intercept-only, full).
void expect_fit(const AftFit& fit, const std::vector<double>& coef, const std::vector<double>& vcov, double scale,
                const std::vector<double>& loglik) {
    EXPECT_TRUE(fit.converged);
    const std::size_t m = fit.covariance.rows();
    ASSERT_EQ(vcov.size(),m*m);
    for (std::size_t a = 0; a < coef.size(); ++a)
        EXPECT_NEAR(fit.coefficients[a],coef[a],1e-7*(1+std::abs(coef[a])));
    for (std::size_t a = 0; a < m; ++a)
        for (std::size_t b = 0; b < m; ++b)
            EXPECT_NEAR(fit.covariance(a,b),vcov[b*m+a],1e-6*(std::abs(vcov[b*m+a])+1e-6));
    EXPECT_NEAR(fit.scale,scale,1e-7);
    EXPECT_NEAR(fit.null_log_likelihood,loglik[0],1e-8);
    EXPECT_NEAR(fit.log_likelihood,loglik[1],1e-8);
    EXPECT_NEAR(fit.chi_square,2*(loglik[1]-loglik[0]),1e-7);
}

// References: R survival 3.8.6 survreg.
TEST(Aft, MatchesSurvregOnLung) {
    std::vector<CensoredTime> t;
    for (std::size_t i = 0; i < l_time.size(); ++i)
        t.push_back(l_status[i] == 1 ? CensoredTime::exact(l_time[i]) : CensoredTime::right(l_time[i]));
    const auto x = columns({l_age,l_sex,l_ecog});
    expect_fit(fit_aft(t,x),l_weibull_coef,l_weibull_vcov,l_weibull_scale[0],l_weibull_loglik);
    expect_fit(fit_aft(t,x,{.distribution=AftDistribution::exponential}),l_exponential_coef,l_exponential_vcov,1,
               l_exponential_loglik);
    expect_fit(fit_aft(t,x,{.distribution=AftDistribution::lognormal}),l_lognormal_coef,l_lognormal_vcov,
               l_lognormal_scale[0],l_lognormal_loglik);
    expect_fit(fit_aft(t,x,{.distribution=AftDistribution::loglogistic}),l_loglogistic_coef,l_loglogistic_vcov,
               l_loglogistic_scale[0],l_loglogistic_loglik);
}

TEST(Aft, MatchesSurvregWithMixedCensoring) {
    std::vector<CensoredTime> t;
    for (std::size_t i = 0; i < m_lo.size(); ++i)
        t.push_back({m_lo[i],m_hi[i]});
    const auto x = columns({m_x,m_g});
    expect_fit(fit_aft(t,x),m_weibull_coef,m_weibull_vcov,m_weibull_scale[0],m_weibull_loglik);
    expect_fit(fit_aft(t,x,{.distribution=AftDistribution::lognormal}),m_lognormal_coef,m_lognormal_vcov,
               m_lognormal_scale[0],m_lognormal_loglik);
    expect_fit(fit_aft(t,x,{.distribution=AftDistribution::loglogistic}),m_loglogistic_coef,m_loglogistic_vcov,
               m_loglogistic_scale[0],m_loglogistic_loglik);
}

TEST(Aft, PredictionsAndTimeRatios) {
    std::vector<CensoredTime> t;
    for (std::size_t i = 0; i < l_time.size(); ++i)
        t.push_back(l_status[i] == 1 ? CensoredTime::exact(l_time[i]) : CensoredTime::right(l_time[i]));
    const auto fit = fit_aft(t,columns({l_age,l_sex,l_ecog}));
    const std::vector<double> profile{60,2,1};
    const double median = fit.predict_quantile(profile,.5);
    EXPECT_NEAR(fit.predict_survival(median,profile),.5,1e-12);
    // Weibull: median = exp(eta) (log 2)^sigma.
    const double eta = fit.coefficients[0]+60*fit.coefficients[1]+2*fit.coefficients[2]+fit.coefficients[3];
    EXPECT_NEAR(median,std::exp(eta)*std::pow(std::log(2.0),fit.scale),1e-9);
    EXPECT_NEAR(fit.time_ratios[1],std::exp(fit.coefficients[2]),1e-15);
    EXPECT_NEAR(std::log(fit.time_ratio_upper[1]/fit.time_ratios[1]),1.959963984540054*fit.standard_errors[2],1e-12);
}

TEST(Aft, Validation) {
    const Matrix none(2,0);
    EXPECT_THROW((void)fit_aft({},Matrix(0,0)),std::invalid_argument);
    EXPECT_THROW((void)fit_aft({CensoredTime::right(1),CensoredTime::right(2)},none),std::invalid_argument);
    EXPECT_THROW((void)fit_aft({CensoredTime::exact(-1),CensoredTime::exact(2)},none),std::invalid_argument);
    EXPECT_THROW((void)fit_aft({CensoredTime::interval(3,2),CensoredTime::exact(2)},none),std::invalid_argument);
    EXPECT_THROW((void)fit_aft({CensoredTime::exact(1),CensoredTime::exact(2)},Matrix(3,0)),std::invalid_argument);
}

} // namespace
