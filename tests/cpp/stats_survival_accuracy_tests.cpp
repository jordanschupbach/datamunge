#include <datamunge/stats/survival_accuracy.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_survival_accuracy_data.inc"

std::vector<SurvivalObservation> sample() {
    std::vector<SurvivalObservation> r;
    for (std::size_t i = 0; i < a_time.size(); ++i)
        r.push_back({a_time[i],a_event[i] != 0});
    return r;
}

// R stores matrices by column.
Matrix from_r(const std::vector<double>& v, std::size_t n, std::size_t m) {
    Matrix x(n,m);
    for (std::size_t k = 0; k < m; ++k)
        for (std::size_t i = 0; i < n; ++i)
            x(i,k) = v[k*n+i];
    return x;
}

// References: riskRegression Score (AUC) on pbc with a tied marker and Cox risks.
TEST(SurvivalAccuracy, AucMatchesScore) {
    const auto obs = sample();
    const std::size_t n = obs.size();
    Matrix marker(n,3);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            marker(i,k) = a_marker[i];
    const auto a = time_dependent_auc(obs,marker,auc_times);
    Matrix risk = from_r(auc_cox_surv,n,3);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            risk(i,k) = 1-risk(i,k);
    const auto b = time_dependent_auc(obs,risk,auc_times);
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(a.auc[k],auc_marker[k],1e-12);
        EXPECT_NEAR(b.auc[k],auc_cox[k],1e-12);
        EXPECT_GT(a.cases[k],0u);
        EXPECT_EQ(a.cases[k]+a.controls[k]<=n,true);
    }
}

// References: pec pec() and crps() with marginal censoring weights and the Kaplan–Meier reference.
TEST(SurvivalAccuracy, BrierMatchesPec) {
    const auto obs = sample();
    const std::size_t n = obs.size(), m = b_grid.size();
    const auto b = brier_score(obs,from_r(b_surv,n,m),b_grid);
    for (std::size_t k = 0; k < m; ++k) {
        EXPECT_NEAR(b.brier[k],b_brier[k],1e-12);
        EXPECT_NEAR(b.null_brier[k],b_null[k],1e-12);
    }
    EXPECT_NEAR(b.null_integrated,b_ibs[0],1e-12);
    EXPECT_NEAR(b.integrated,b_ibs[1],1e-12);
    EXPECT_NEAR(b.explained,1-b_ibs[1]/b_ibs[0],1e-12);
    // Grid starting at 500: same columns from 500 onwards.
    std::vector<double> grid(b_grid.begin()+2,b_grid.end());
    Matrix surv(n,grid.size());
    const Matrix full = from_r(b_surv,n,m);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < grid.size(); ++k)
            surv(i,k) = full(i,k+2);
    const auto b2 = brier_score(obs,surv,grid);
    EXPECT_NEAR(b2.null_integrated,b2_ibs[0],1e-12);
    EXPECT_NEAR(b2.integrated,b2_ibs[1],1e-12);
}

TEST(SurvivalAccuracy, Validation) {
    const std::vector<SurvivalObservation> obs{{1,true},{2,false},{3,true},{4,false}};
    EXPECT_THROW((void)brier_score(obs,Matrix(4,2,.5),{1}),std::invalid_argument);
    EXPECT_THROW((void)brier_score(obs,Matrix(4,1,1.5),{1}),std::invalid_argument);
    EXPECT_THROW((void)brier_score(obs,Matrix(4,2,.5),{2,1}),std::invalid_argument);
    EXPECT_THROW((void)brier_score(obs,Matrix(4,1,.5),{4}),std::invalid_argument); // G(4) = 0
    EXPECT_THROW((void)time_dependent_auc(obs,Matrix(4,1,1),{0.5}),std::invalid_argument); // no cases
    EXPECT_THROW((void)time_dependent_auc({},Matrix(0,1),{1}),std::invalid_argument);
}

} // namespace
