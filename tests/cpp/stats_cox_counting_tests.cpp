#include <datamunge/stats/cox_counting.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_cox_counting_data.inc"

std::vector<CountingRecord> records(const std::vector<double>& start, const std::vector<double>& stop,
                                    const std::vector<double>& event) {
    std::vector<CountingRecord> r;
    for (std::size_t i = 0; i < start.size(); ++i)
        r.push_back({start[i],stop[i],event[i] == 1});
    return r;
}

Matrix columns(const std::vector<std::vector<double>>& cols) {
    Matrix x(cols[0].size(),cols.size());
    for (std::size_t j = 0; j < cols.size(); ++j)
        for (std::size_t i = 0; i < cols[j].size(); ++i)
            x(i,j) = cols[j][i];
    return x;
}

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

// basehaz(centered = TRUE) reports every distinct stop time at coxph's centering, which is 0
// for 0/1 columns in survival 3.x; rescale ours from the column means to that profile.
void expect_baseline(const CoxCountingFit& fit, const std::vector<double>& means, const std::vector<double>& time,
                     const std::vector<double>& hazard) {
    double shift = 0;
    for (std::size_t a = 0; a < means.size(); ++a)
        shift += (means[a]-fit.reference[a])*fit.coefficients[a];
    const auto& b = fit.baselines.at(0);
    for (std::size_t k = 0; k < b.event_times.size(); ++k) {
        std::size_t j = 0;
        while (time[j] != b.event_times[k])
            ++j;
        EXPECT_NEAR(b.cumulative_hazard[k]*std::exp(shift),hazard[j],1e-8*(1+hazard[j]));
    }
}

// References: R survival 3.8.6 coxph on the Stanford heart transplant data, where transplant
// status changes during follow-up, and on a simulated left-truncated sample.
TEST(CoxCounting, MatchesCoxphOnHeartTransplant) {
    const auto r = records(h_start,h_stop,h_event);
    const auto x = columns({h_age,h_year,h_surgery,h_transplant});
    const auto efron = fit_cox_counting(r,x);
    expect_fit(efron,e_coef,e_se,e_loglik,e_tests);
    expect_baseline(efron,e_means,e_bh_time,e_bh);
    const auto breslow = fit_cox_counting(r,x,{.ties=CoxTies::breslow});
    expect_fit(breslow,b_coef,b_se,b_loglik,b_tests);
    expect_baseline(breslow,b_means,b_bh_time,b_bh);
    // Prediction at the reference profile is exp(-H0).
    EXPECT_NEAR(efron.predict_survival(100,efron.reference),
                std::exp(-[&] { const auto& b = efron.baselines[0]; double h = 0;
                    for (std::size_t k = 0; k < b.event_times.size(); ++k)
                        if (b.event_times[k] <= 100) h = b.cumulative_hazard[k];
                    return h; }()),1e-14);
}

TEST(CoxCounting, MatchesCoxphWithDelayedEntry) {
    const auto fit = fit_cox_counting(records(lt_entry,lt_stop,lt_event),columns({lt_x1,lt_x2}));
    expect_fit(fit,lt_coef,lt_se,lt_loglik,lt_tests);
}

TEST(CoxCounting, SplittingRecordsLeavesFitUnchanged) {
    // Splitting a record at a non-event time with unchanged covariates does not change the likelihood.
    const std::vector<CountingRecord> whole{{0,3,true},{0,5,false},{0,2,true},{1,4,true},{0,6,true}};
    const std::vector<CountingRecord> split{{0,1.5,false},{1.5,3,true},{0,5,false},{0,2,true},
                                            {1,2.5,false},{2.5,4,true},{0,6,true}};
    const Matrix xw(5,1,{1,0,1,0,1});
    const Matrix xs(7,1,{1,1,0,1,0,0,1});
    const auto a = fit_cox_counting(whole,xw);
    const auto b = fit_cox_counting(split,xs);
    EXPECT_NEAR(a.coefficients[0],b.coefficients[0],1e-10);
    EXPECT_NEAR(a.log_likelihood,b.log_likelihood,1e-10);
}

TEST(CoxCounting, Validation) {
    EXPECT_THROW((void)fit_cox_counting({},Matrix(0,1)),std::invalid_argument);
    EXPECT_THROW((void)fit_cox_counting({{1,1,true}},Matrix(1,1,{0})),std::invalid_argument);
    EXPECT_THROW((void)fit_cox_counting({{0,1,false},{0,2,false}},Matrix(2,1,{0,1})),std::invalid_argument);
    EXPECT_THROW((void)fit_cox_counting({{0,1,true}},Matrix(2,1,{0,1})),std::invalid_argument);
    EXPECT_THROW((void)fit_cox_counting({{0,1,true},{0,2,true},{0,3,false}},Matrix(3,1,{1,1,1})),
                 std::runtime_error);
}

} // namespace
