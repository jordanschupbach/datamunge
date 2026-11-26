#include <datamunge/stats/cox_counting.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_cox_strata_data.inc"

Matrix columns(const std::vector<std::vector<double>>& cols) {
    Matrix x(cols[0].size(),cols.size());
    for (std::size_t j = 0; j < cols.size(); ++j)
        for (std::size_t i = 0; i < cols[j].size(); ++i)
            x(i,j) = cols[j][i];
    return x;
}

std::vector<int> labels(const std::vector<double>& v) { return {v.begin(),v.end()}; }

void expect_fit(const CoxCountingFit& fit, const std::vector<double>& coef, const std::vector<double>& se,
                const std::vector<double>& loglik, const std::vector<double>& tests) {
    EXPECT_TRUE(fit.converged);
    for (std::size_t a = 0; a < coef.size(); ++a) {
        EXPECT_NEAR(fit.coefficients[a],coef[a],1e-8);
        EXPECT_NEAR(fit.standard_errors[a],se[a],1e-8);
    }
    EXPECT_NEAR(fit.null_log_likelihood,loglik[0],1e-9);
    EXPECT_NEAR(fit.log_likelihood,loglik[1],1e-9);
    EXPECT_NEAR(fit.likelihood_ratio.statistic,tests[0],1e-7);
    EXPECT_NEAR(fit.wald.statistic,tests[1],1e-6);
    EXPECT_NEAR(fit.score.statistic,tests[2],1e-8);
}

// basehaz(centered = TRUE) lists each stratum's distinct times at coxph's centering.
void expect_baselines(const CoxCountingFit& fit, const std::vector<double>& means, const std::vector<double>& time,
                      const std::vector<double>& hazard, const std::vector<double>& stratum) {
    double shift = 0;
    for (std::size_t a = 0; a < means.size(); ++a)
        shift += (means[a]-fit.reference[a])*fit.coefficients[a];
    ASSERT_EQ(fit.baselines.size(),4u);
    std::size_t compared = 0;
    for (const auto& b : fit.baselines)
        for (std::size_t k = 0; k < b.event_times.size(); ++k) {
            std::size_t j = 0;
            while (!(stratum[j] == b.stratum && time[j] == b.event_times[k]))
                ++j;
            EXPECT_NEAR(b.cumulative_hazard[k]*std::exp(shift),hazard[j],1e-8*(1+hazard[j]));
            ++compared;
        }
    EXPECT_GT(compared,90u);
}

// References: R survival 3.8.6 coxph with strata().
TEST(CoxStrata, MatchesCoxphOnVeteranByCellType) {
    std::vector<CountingRecord> r;
    for (std::size_t i = 0; i < v_time.size(); ++i)
        r.push_back({0,v_time[i],v_status[i] == 1});
    const auto x = columns({v_trt,v_karno,v_age});
    const auto efron = fit_cox_counting(r,x,{.strata=labels(v_cell)});
    expect_fit(efron,ve_coef,ve_se,ve_loglik,ve_tests);
    expect_baselines(efron,ve_means,ve_bh_time,ve_bh,ve_bh_strata);
    const auto breslow = fit_cox_counting(r,x,{.ties=CoxTies::breslow,.strata=labels(v_cell)});
    expect_fit(breslow,vb_coef,vb_se,vb_loglik,vb_tests);
    expect_baselines(breslow,vb_means,vb_bh_time,vb_bh,vb_bh_strata);
    // Profiles differ across strata only through the baselines.
    EXPECT_NE(efron.predict_survival(50,efron.reference,1),efron.predict_survival(50,efron.reference,2));
    EXPECT_THROW((void)efron.predict_survival(50,efron.reference,9),std::invalid_argument);
}

TEST(CoxStrata, MatchesCoxphWithStartStop) {
    std::vector<CountingRecord> r;
    for (std::size_t i = 0; i < h_start.size(); ++i)
        r.push_back({h_start[i],h_stop[i],h_event[i] == 1});
    const auto fit = fit_cox_counting(r,columns({h_age,h_year,h_transplant}),{.strata=labels(h_surgery)});
    expect_fit(fit,hs_coef,hs_se,hs_loglik,hs_tests);
    EXPECT_EQ(fit.baselines.size(),2u);
}

TEST(CoxStrata, Validation) {
    EXPECT_THROW((void)fit_cox_counting({{0,1,true},{0,2,true}},Matrix(2,1,{0,1}),{.strata={1}}),
                 std::invalid_argument);
}

} // namespace
