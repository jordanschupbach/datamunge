#include <datamunge/stats/arima_exact.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
constexpr double NA = std::numeric_limits<double>::quiet_NaN();

const std::vector<double> lake_huron{
    580.38,581.86,580.97,580.8,579.79,580.39,580.42,580.82,581.4,581.32,581.44,581.68,581.17,580.53,580.01,579.91,
    579.14,579.16,579.55,579.67,578.44,578.24,579.1,579.09,579.35,578.82,579.32,579.01,579,579.8,579.83,579.72,
    579.89,580.01,579.37,578.69,578.19,578.67,579.55,578.92,578.09,579.37,580.13,580.14,579.51,579.24,578.66,
    578.86,578.05,577.79,576.75,576.75,577.82,578.64,580.58,579.48,577.38,576.9,576.94,576.24,576.84,576.85,576.9,
    577.79,578.18,577.51,577.23,578.42,579.61,579.05,579.26,579.22,579.38,579.1,577.95,578.12,579.75,580.85,580.41,
    579.96,579.61,578.76,578.18,577.21,577.13,579.1,578.25,577.91,576.89,575.96,576.8,577.68,578.38,578.52,579.74,
    579.31,579.89,579.96};
const std::vector<double> presidents{
    NA,87,82,75,63,50,43,32,35,60,54,55,36,39,NA,NA,69,57,57,51,45,37,46,39,36,24,32,23,25,32,NA,32,59,74,75,60,
    71,61,71,57,71,68,79,73,76,71,67,75,79,62,63,57,60,49,48,52,57,62,61,66,71,62,61,57,72,83,71,78,79,71,62,74,
    76,64,62,57,80,73,69,69,71,64,69,62,63,46,56,44,44,52,38,46,36,49,35,44,59,65,65,56,66,53,61,52,51,48,54,49,
    49,61,NA,NA,68,44,40,27,28,25,24,24};
const std::vector<double> accidental_deaths{
    9007,8106,8928,9137,10017,10826,11317,10744,9713,9938,9161,8927,7750,6981,8038,8422,8714,9512,10120,9823,8743,
    9129,8710,8680,8162,7306,8124,7870,9387,9556,10093,9620,8285,8466,8160,8034,7717,7461,7767,7925,8623,8945,10078,
    9179,8037,8488,7874,8647,7792,6957,7726,8106,8890,9299,10625,9302,8314,8850,8265,8796,7836,6892,7791,8192,9115,
    9434,10484,9827,9110,9070,8633,9240};

// Reference values: arima(LakeHuron, c(2, 0, 0), method = "ML") and predict(fit, 3) in R 4.5.3.
TEST(ArimaExact, LakeHuronAr2MatchesR) {
    const auto fit = fit_arima_exact(lake_huron,{.order={.p=2}});
    EXPECT_NEAR(fit.coefficients.ar[0],1.043610749299434,2e-5);
    EXPECT_NEAR(fit.coefficients.ar[1],-.249493314353861,2e-5);
    EXPECT_NEAR(fit.mean,579.047263842206917,2e-4);
    EXPECT_NEAR(fit.log_likelihood,-103.633222538442098,1e-8);
    EXPECT_NEAR(fit.sigma2,.478820628366647,1e-6);
    EXPECT_NEAR(fit.aic,215.266445076884196,1e-7);
    EXPECT_NEAR(fit.standard_errors[0],.098282920587863,2e-4);
    EXPECT_NEAR(fit.standard_errors[2],.331875756491464,1e-3);
    EXPECT_EQ(fit.names,(std::vector<std::string>{"ar1","ar2","intercept"}));
    EXPECT_TRUE(fit.converged);
    const auto f = forecast_arima_exact(fit,3);
    EXPECT_NEAR(f.mean[0],579.789548070620,1e-4);
    EXPECT_NEAR(f.mean[2],579.432855332154,1e-4);
    EXPECT_NEAR(f.standard_error[0],.691968661405014,1e-5);
    EXPECT_NEAR(f.standard_error[2],1.156664907805542,1e-5);
}

// Reference values: arima(presidents, c(1, 0, 0), method = "ML"), which has six missing quarters.
TEST(ArimaExact, MissingObservationsMatchR) {
    const auto fit = fit_arima_exact(presidents,{.order={.p=1}});
    EXPECT_NEAR(fit.coefficients.ar[0],.824164859136208,2e-5);
    EXPECT_NEAR(fit.mean,56.150481676451754,2e-3);
    EXPECT_NEAR(fit.log_likelihood,-416.8922732940370,1e-7);
    EXPECT_NEAR(fit.sigma2,85.4685554762503,1e-4);
    EXPECT_EQ(fit.observations_used,114U);
    EXPECT_NEAR(fit.standard_errors[0],.0554620331774591,2e-4);
    EXPECT_NEAR(fit.standard_errors[1],4.6434181525274587,2e-2);
    EXPECT_TRUE(std::isnan(fit.residuals[0]));
    const auto f = forecast_arima_exact(fit,2);
    EXPECT_NEAR(f.mean[0],29.6531844744177,1e-3);
    EXPECT_NEAR(f.standard_error[1],11.98010335893793,1e-4);
}

// Reference values: arima(USAccDeaths, c(0, 1, 1), seasonal = c(0, 1, 1), method = "ML") and predict().
TEST(ArimaExact, AirlineModelMatchesR) {
    const auto fit = fit_arima_exact(accidental_deaths,{.order={.d=1,.q=1,.seasonal_d=1,.seasonal_q=1,.period=12}});
    EXPECT_NEAR(fit.coefficients.ma[0],-.430278483358636,2e-5);
    EXPECT_NEAR(fit.coefficients.seasonal_ma[0],-.552772006869988,2e-5);
    EXPECT_NEAR(fit.log_likelihood,-425.439993561082,1e-8);
    EXPECT_NEAR(fit.sigma2,99347.485299702152,1.0); // flat optimum; R stops at reltol 1e-8
    EXPECT_NEAR(fit.standard_errors[0],.122801694749295,2e-4);
    EXPECT_NEAR(fit.standard_errors[1],.178372148084654,5e-4);
    EXPECT_EQ(fit.observations_used,59U);
    const auto f = forecast_arima_exact(fit,3);
    EXPECT_NEAR(f.mean[0],8336.05991120681,5e-2);
    EXPECT_NEAR(f.mean[2],8314.64028411660,5e-2);
    EXPECT_NEAR(f.standard_error[0],315.448954536980,5e-3);
    EXPECT_NEAR(f.standard_error[2],405.015375136909,5e-3);
}

// Reference values: arima(x, c(1, 1, 1), seasonal = c(0, 1, 1), fixed = c(0.2, -0.6, -0.5),
// transform.pars = FALSE) with USAccDeaths[c(20, 50)] set to NA.
TEST(ArimaExact, FixedParameterLikelihoodWithDifferencingAndGaps) {
    auto x = accidental_deaths;
    x[19] = NA;
    x[49] = NA;
    const auto l = arima_likelihood(x,{.p=1,.d=1,.q=1,.seasonal_d=1,.seasonal_q=1,.period=12},
                                    {.ar={.2},.ma={-.6},.seasonal_ma={-.5}});
    EXPECT_NEAR(l.log_likelihood,-412.485511478715,1e-8);
    EXPECT_NEAR(l.sigma2,103891.514622889896,1e-6);
    EXPECT_NEAR(l.residuals[0],5.200192610655402,1e-6);
    EXPECT_NEAR(l.residuals[11],-6.013599311394727,1e-6);
    EXPECT_NEAR(l.residuals[12],-38.446100673891586,1e-6);
    EXPECT_NEAR(l.residuals[14],241.130664780336105,1e-6);
    EXPECT_TRUE(std::isnan(l.residuals[19]));
}

TEST(ArimaExact, Validation) {
    EXPECT_THROW((void)arima_likelihood(lake_huron,{.p=1},{.ar={1.2}}),std::invalid_argument);
    EXPECT_THROW((void)arima_likelihood(lake_huron,{.p=1},{}),std::invalid_argument);
    EXPECT_THROW((void)arima_likelihood(lake_huron,{.seasonal_q=1},{.seasonal_ma={.3}}),std::invalid_argument);
    EXPECT_THROW((void)fit_arima_exact({1,2,3},{.order={.p=2}}),std::invalid_argument);
    EXPECT_THROW((void)arima_likelihood({1,INFINITY,2},{},{}),std::invalid_argument);
}

} // namespace
