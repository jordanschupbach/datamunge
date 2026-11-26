#include <datamunge/stats/arima_exact.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_arima_regression_data.inc"

const std::vector<double> lake_huron{
    580.38,581.86,580.97,580.8,579.79,580.39,580.42,580.82,581.4,581.32,581.44,581.68,581.17,580.53,580.01,579.91,
    579.14,579.16,579.55,579.67,578.44,578.24,579.1,579.09,579.35,578.82,579.32,579.01,579,579.8,579.83,579.72,
    579.89,580.01,579.37,578.69,578.19,578.67,579.55,578.92,578.09,579.37,580.13,580.14,579.51,579.24,578.66,
    578.86,578.05,577.79,576.75,576.75,577.82,578.64,580.58,579.48,577.38,576.9,576.94,576.24,576.84,576.85,576.9,
    577.79,578.18,577.51,577.23,578.42,579.61,579.05,579.26,579.22,579.38,579.1,577.95,578.12,579.75,580.85,580.41,
    579.96,579.61,578.76,578.18,577.21,577.13,579.1,578.25,577.91,576.89,575.96,576.8,577.68,578.38,578.52,579.74,
    579.31,579.89,579.96};

// Reference values: arima(LakeHuron, c(2, 0, 0), xreg = time(LakeHuron) - 1920, method = "ML") and
// predict(fit, 3, newxreg = 53:55) in R 4.5.3.
TEST(ArimaRegression, LakeHuronTrendMatchesR) {
    Matrix year(lake_huron.size(),1);
    for (std::size_t t = 0; t < lake_huron.size(); ++t)
        year(t,0) = 1875.0+static_cast<double>(t)-1920;
    const auto fit = fit_arima_exact(lake_huron,{.order={.p=2},.regressors=year,.regressor_names={"year"}});
    EXPECT_EQ(fit.names,(std::vector<std::string>{"ar1","ar2","intercept","year"}));
    EXPECT_NEAR(fit.coefficients.ar[0],1.0048200533121874,2e-5);
    EXPECT_NEAR(fit.coefficients.ar[1],-.2913044882678641,2e-5);
    EXPECT_NEAR(fit.mean,579.0993922935654155,1e-4);
    EXPECT_NEAR(fit.regression[0],-.0215679259860622,1e-6);
    EXPECT_NEAR(fit.log_likelihood,-101.198267170235056,1e-8);
    EXPECT_NEAR(fit.sigma2,.456618330835551,1e-6);
    EXPECT_NEAR(fit.aic,212.396534340470112,1e-7);
    EXPECT_NEAR(fit.standard_errors[2],.23702511114851649,1e-3);
    EXPECT_NEAR(fit.standard_errors[3],.00809965775587126,5e-5);
    const auto f = forecast_arima_exact(fit,3,Matrix(3,1,{53,54,55}));
    EXPECT_NEAR(f.mean[0],579.397254004696,1e-4);
    EXPECT_NEAR(f.mean[2],578.368094653596,1e-4);
    EXPECT_NEAR(f.standard_error[0],.675735400016568,1e-5);
    EXPECT_NEAR(f.standard_error[2],1.073909766699956,1e-5);
}

// Reference values: arima(log(drivers), c(0, 1, 1), seasonal = c(0, 1, 1), xreg = cbind(law, log(PetrolPrice)),
// method = "ML") on Seatbelts and predict(n.ahead = 3, newxreg = cbind(1, log(c(.11, .115, .12)))).
TEST(ArimaRegression, SeatbeltsAirlineWithInterventionMatchesR) {
    Matrix x(log_drivers.size(),2);
    for (std::size_t t = 0; t < log_drivers.size(); ++t) {
        x(t,0) = law[t];
        x(t,1) = log_petrol[t];
    }
    const auto fit = fit_arima_exact(log_drivers,{.order={.d=1,.q=1,.seasonal_d=1,.seasonal_q=1,.period=12},
                                                  .regressors=x,.regressor_names={"law","petrol"}});
    EXPECT_EQ(fit.names,(std::vector<std::string>{"ma1","sma1","law","petrol"}));
    EXPECT_NEAR(fit.coefficients.ma[0],-.775712661608821,5e-5);
    EXPECT_NEAR(fit.coefficients.seasonal_ma[0],-.848178307402241,5e-5);
    EXPECT_NEAR(fit.regression[0],-.246109963176585,2e-5);
    EXPECT_NEAR(fit.regression[1],-.298355266025322,5e-5); // flat ridge; see the likelihood check
    EXPECT_NEAR(fit.log_likelihood,200.713336199933,1e-6);
    EXPECT_GT(fit.log_likelihood,200.713336199933-1e-9); // at least as high as R, which stops at reltol 1e-8
    EXPECT_NEAR(fit.sigma2,5.67931735257182e-03,1e-7);
    EXPECT_NEAR(fit.standard_errors[2],.0477870606817182,2e-4);
    EXPECT_NEAR(fit.standard_errors[3],.0983729929654518,5e-4);
    const auto f = forecast_arima_exact(fit,3,Matrix(3,2,{1,std::log(.11),1,std::log(.115),1,std::log(.12)}));
    EXPECT_NEAR(f.mean[0],7.24331960509467,1e-4);
    EXPECT_NEAR(f.mean[2],7.16981294987777,1e-4);
    EXPECT_NEAR(f.standard_error[0],.0754255293982855,1e-5);
    EXPECT_NEAR(f.standard_error[2],.0791279596543825,1e-5);
}

TEST(ArimaRegression, ValidatesRegressors) {
    EXPECT_THROW((void)fit_arima_exact(lake_huron,{.order={.p=1},.regressors=Matrix(5,1)}),std::invalid_argument);
    Matrix bad(lake_huron.size(),1);
    bad(3,0) = NAN;
    EXPECT_THROW((void)fit_arima_exact(lake_huron,{.order={.p=1},.regressors=bad}),std::invalid_argument);
    Matrix ok(lake_huron.size(),1);
    for (std::size_t t = 0; t < lake_huron.size(); ++t)
        ok(t,0) = static_cast<double>(t);
    EXPECT_THROW((void)fit_arima_exact(lake_huron,{.order={.p=1},.regressors=ok,.regressor_names={"a","b"}}),
                 std::invalid_argument);
    const auto fit = fit_arima_exact(lake_huron,{.order={.p=1},.regressors=ok});
    EXPECT_EQ(fit.names.back(),"x1");
    EXPECT_THROW((void)forecast_arima_exact(fit,2),std::invalid_argument);
}

} // namespace
