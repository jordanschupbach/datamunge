#include <datamunge/stats/structural.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_structural_data.inc"

// R stores n by c matrices column-major.
void expect_columns(const Matrix& got, const std::vector<double>& want, double tol) {
    ASSERT_EQ(got.rows()*got.cols(),want.size());
    for (std::size_t j = 0; j < got.cols(); ++j)
        for (std::size_t t = 0; t < got.rows(); ++t)
            EXPECT_NEAR(got(t,j),want[j*got.rows()+t],tol*std::max(1.0,std::abs(want[j*got.rows()+t])))
                << t << "," << j;
}

// References: R 4.5.3 StructTS(x, type, optim.control = list(factr = 10)), tsSmooth() and predict().
TEST(Structural, LocalLevelMatchesR) {
    const auto f = fit_structural(nile);
    EXPECT_GE(f.log_likelihood,level_loglik[0]-1e-8);
    EXPECT_NEAR(f.log_likelihood,level_loglik[0],1e-7);
    EXPECT_NEAR(f.level_variance,level_coef[0],1e-3*level_coef[0]);
    EXPECT_NEAR(f.irregular_variance,level_coef[1],1e-4*level_coef[1]);
    // States follow the variances to the same relative accuracy.
    expect_columns(f.filtered,level_filtered,1e-5);
    expect_columns(f.smoothed,level_smoothed,1e-5);
    for (std::size_t t = 0; t < nile.size(); ++t)
        EXPECT_NEAR(f.standardized_residuals[t],level_resid[t],1e-5);
    const auto p = forecast_structural(f,5);
    for (std::size_t h = 0; h < 5; ++h) {
        EXPECT_NEAR(p.mean[h],level_pred[h],1e-5*level_pred[h]);
        EXPECT_NEAR(p.standard_error[h],level_pred[5+h],1e-4*level_pred[5+h]);
    }
}

TEST(Structural, LocalLinearTrendMatchesR) {
    const auto f = fit_structural(nile,{.model=StructuralModel::local_linear_trend});
    EXPECT_GE(f.log_likelihood,trend_loglik[0]-1e-8);
    EXPECT_NEAR(f.log_likelihood,trend_loglik[0],1e-6);
    EXPECT_NEAR(f.level_variance,trend_coef[0],2e-3*trend_coef[0]);
    EXPECT_NEAR(f.slope_variance,trend_coef[1],1e-6);
    EXPECT_NEAR(f.irregular_variance,trend_coef[2],1e-3*trend_coef[2]);
    expect_columns(f.filtered,trend_filtered,1e-4);
    const auto p = forecast_structural(f,5);
    for (std::size_t h = 0; h < 5; ++h) {
        EXPECT_NEAR(p.mean[h],trend_pred[h],1e-4*trend_pred[h]);
        EXPECT_NEAR(p.standard_error[h],trend_pred[5+h],1e-3*trend_pred[5+h]);
    }
}

TEST(Structural, BasicStructuralModelMatchesR) {
    const auto f = fit_structural(ukgas,{.model=StructuralModel::basic_seasonal,.period=4});
    // Our optimum is slightly higher than R's L-BFGS-B stopping point (by about 6e-6), so the
    // flat directions agree to a few parts in a thousand.
    EXPECT_GE(f.log_likelihood,bsm_loglik[0]-1e-8);
    EXPECT_NEAR(f.log_likelihood,bsm_loglik[0],1e-5);
    EXPECT_NEAR(f.level_variance,bsm_coef[0],1e-7);
    EXPECT_NEAR(f.slope_variance,bsm_coef[1],5e-3*bsm_coef[1]);
    EXPECT_NEAR(f.seasonal_variance,bsm_coef[2],5e-3*bsm_coef[2]);
    EXPECT_NEAR(f.irregular_variance,bsm_coef[3],5e-3*bsm_coef[3]);
    expect_columns(f.filtered,bsm_filtered,1e-4);
    expect_columns(f.smoothed,bsm_smoothed,1e-4);
    const auto p = forecast_structural(f,8);
    for (std::size_t h = 0; h < 8; ++h) {
        EXPECT_NEAR(p.mean[h],bsm_pred[h],1e-4);
        EXPECT_NEAR(p.standard_error[h],bsm_pred[8+h],1e-3*bsm_pred[8+h]);
    }
}

TEST(Structural, MissingValuesAndValidation) {
    const auto f = fit_structural(nile_missing);
    // R's StructTS scales its per-observation objective by length(y) rather than the 80 observed values:
    // loglik_R = (n / nu) (loglik + nu/2 log 2 pi) - nu/2 log 2 pi.
    const double nu = 80, n = 100, c = .5*nu*std::log(2*M_PI);
    EXPECT_NEAR(n/nu*(f.log_likelihood+c)-c,missing_loglik[0],1e-6);
    EXPECT_NEAR(f.level_variance,missing_coef[0],2e-3*missing_coef[0]);
    EXPECT_NEAR(f.irregular_variance,missing_coef[1],1e-3*missing_coef[1]);
    expect_columns(f.smoothed,missing_smoothed,1e-4);
    EXPECT_TRUE(std::isnan(f.standardized_residuals[25]));
    EXPECT_THROW((void)fit_structural({NAN,1,2,3}),std::invalid_argument);
    EXPECT_THROW((void)fit_structural(std::vector<double>(10,2.0)),std::invalid_argument);
    EXPECT_THROW((void)fit_structural(nile,{.model=StructuralModel::basic_seasonal,.period=1}),std::invalid_argument);
    EXPECT_THROW((void)forecast_structural(f,0),std::invalid_argument);
}

} // namespace
