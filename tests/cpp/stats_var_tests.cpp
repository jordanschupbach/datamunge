#include <datamunge/stats/var.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;
using Rows = std::vector<std::vector<double>>;

#include "stats_var_data.inc"

Matrix macro() {
    Matrix y(macro_rows,3);
    for (std::size_t i = 0; i < macro_rows; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            y(i,j) = macro_growth[3*i+j];
    return y;
}

struct VarReference {
    VarTrend trend;
    Rows params, stderr_, sigma_u, sigma_mle;
    double llf, aic, bic, hqic, fpe;
    std::vector<double> inverse_root_moduli;
    Rows forecast;
    std::vector<Rows> forecast_cov;
};

// Reference values: statsmodels 0.14.6 VAR(y).fit(2, trend), forecast(y[-2:], 3), forecast_cov(3) and
// select_order(8, trend = "c"); statsmodels reports roots as inverse companion eigenvalues.
#include "stats_var_refs.inc"

void expect_matrix(const Matrix& got, const Rows& want, double tol) {
    ASSERT_EQ(got.rows(),want.size());
    for (std::size_t i = 0; i < want.size(); ++i)
        for (std::size_t j = 0; j < want[i].size(); ++j)
            EXPECT_NEAR(got(i,j),want[i][j],tol) << i << "," << j;
}

// The trend column runs to about 200, so the linear-trend design is less well conditioned; use tol = 1e-9 there.
void check(const VarReference& r, double tol = 1e-11) {
    const auto m = fit_var(macro(),{.lags=2,.trend=r.trend});
    EXPECT_EQ(m.observations,200U);
    expect_matrix(m.parameters,r.params,tol);
    expect_matrix(m.standard_errors,r.stderr_,tol);
    expect_matrix(m.sigma,r.sigma_u,tol);
    expect_matrix(m.sigma_mle,r.sigma_mle,tol);
    EXPECT_NEAR(m.log_likelihood,r.llf,1e-8);
    EXPECT_NEAR(m.aic,r.aic,1e-11);
    EXPECT_NEAR(m.bic,r.bic,1e-11);
    EXPECT_NEAR(m.hqic,r.hqic,1e-11);
    EXPECT_NEAR(m.fpe,r.fpe,1e-11);
    ASSERT_EQ(m.companion_moduli.size(),r.inverse_root_moduli.size());
    for (std::size_t i = 0; i < m.companion_moduli.size(); ++i)
        EXPECT_NEAR(m.companion_moduli[i],1/r.inverse_root_moduli[i],1e-10);
    EXPECT_TRUE(m.stable);
    const auto f = forecast_var(m,3);
    expect_matrix(f.mean,r.forecast,tol);
    for (std::size_t h = 0; h < 3; ++h)
        expect_matrix(f.covariance[h],r.forecast_cov[h],1e-10);
}

TEST(Var, ConstantMatchesStatsmodels) { check(ref_constant); }
TEST(Var, NoTrendMatchesStatsmodels) { check(ref_none); }
TEST(Var, LinearTrendMatchesStatsmodels) { check(ref_linear,1e-9); }

TEST(Var, LagLayoutAndOrderSelection) {
    const auto m = fit_var(macro(),{.lags=2});
    // A_2(1, 2): effect of investment two quarters back on consumption = params row 1 + 3 + 2, column 1.
    EXPECT_EQ(m.lag_coefficients[1](1,2),m.parameters(6,1));
    const auto s = select_var_order(macro(),8);
    ASSERT_EQ(s.aic.size(),9U);
    for (std::size_t p = 0; p <= 8; ++p) {
        EXPECT_NEAR(s.aic[p],select_aic[p],1e-11);
        EXPECT_NEAR(s.bic[p],select_bic[p],1e-11);
        EXPECT_NEAR(s.hqic[p],select_hqic[p],1e-11);
        EXPECT_NEAR(s.fpe[p],select_fpe[p],1e-11);
    }
    EXPECT_EQ(s.aic_order,1U);
    EXPECT_EQ(s.bic_order,1U);
}

TEST(Var, UnstableAndValidation) {
    Matrix walk(100,2);
    for (std::size_t t = 1; t < 100; ++t) {
        walk(t,0) = 1.08*walk(t-1,0)+std::sin(3.0*t);
        walk(t,1) = .5*walk(t-1,1)+std::cos(5.0*t);
    }
    const auto m = fit_var(walk,{.lags=1});
    EXPECT_FALSE(m.stable);
    EXPECT_GT(m.companion_moduli.front(),1.0);
    EXPECT_THROW((void)fit_var(Matrix(5,3),{.lags=2}),std::invalid_argument);
    EXPECT_THROW((void)forecast_var(m,0),std::invalid_argument);
    Matrix bad = walk;
    bad(3,1) = NAN;
    EXPECT_THROW((void)fit_var(bad),std::invalid_argument);
}

} // namespace

namespace {
using namespace datamunge::stats;

// Reference values: statsmodels 0.14.6 VARResults.test_causality(caused, causing, kind).
TEST(Granger, MatchesStatsmodels) {
    struct Case { VarTrend trend; std::vector<std::size_t> caused, causing; double f, pf, w, pw; double df2; };
    const std::vector<Case> cases{
        {VarTrend::constant,{1},{0},.5952079537740191,.5517849336760747,1.1904159075480383,.551447878275922,579},
        {VarTrend::constant,{0},{2},.8112208379058177,.4448196420743948,1.6224416758116353,.44431529801640157,579},
        {VarTrend::constant,{2},{0,1},18.1230228371798,4.8983842860660026e-14,72.4920913487192,6.754956308372163e-15,579},
        {VarTrend::constant,{0,1},{2},1.1067248090186994,.3524220041455841,4.426899236074798,.3513035513653096,579},
        {VarTrend::linear,{1},{0},.8356530625903782,.43411619581753447,1.6713061251807564,.4335912267418356,576},
        {VarTrend::linear,{2},{0,1},17.548621344633336,1.3367708105264793e-13,70.19448537853334,2.065072005451295e-14,576},
        {VarTrend::linear,{0,1},{2},1.2429769692115582,.2915143395180344,4.971907876846233,.29019208746119274,576},
    };
    for (const auto& c : cases) {
        const auto m = fit_var(macro(),{.lags=2,.trend=c.trend});
        const auto f = granger_causality(m,c.caused,c.causing);
        EXPECT_NEAR(f.statistic,c.f,1e-9*std::max(1.0,c.f));
        EXPECT_NEAR(f.p_value,c.pf,1e-9*c.pf+1e-15);
        EXPECT_EQ(f.denominator_df,c.df2);
        EXPECT_EQ(f.restrictions,2*c.caused.size()*c.causing.size());
        const auto w = granger_causality(m,c.caused,c.causing,GrangerStatistic::wald);
        EXPECT_NEAR(w.statistic,c.w,1e-9*std::max(1.0,c.w));
        EXPECT_NEAR(w.p_value,c.pw,1e-9*c.pw+1e-16);
    }
    const auto g = granger_causality(fit_var(macro(),{.lags=4}),{1},{0,2});
    EXPECT_NEAR(g.statistic,1.2659973227355552,1e-10);
    EXPECT_NEAR(g.p_value,.25871303035251897,1e-10);
    EXPECT_EQ(g.restrictions,8U);
    EXPECT_EQ(g.denominator_df,555.0);
}

TEST(Granger, Validation) {
    const auto m = fit_var(macro(),{.lags=2});
    EXPECT_THROW((void)granger_causality(m,{0},{0}),std::invalid_argument);
    EXPECT_THROW((void)granger_causality(m,{},{1}),std::invalid_argument);
    EXPECT_THROW((void)granger_causality(m,{0},{3}),std::invalid_argument);
    EXPECT_THROW((void)granger_causality(m,{0,0},{1}),std::invalid_argument);
    EXPECT_THROW((void)granger_causality(fit_var(macro(),{.lags=0}),{0},{1}),std::invalid_argument);
}

} // namespace
