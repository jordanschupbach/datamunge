#include <datamunge/stats/fine_gray.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_fine_gray_data.inc"

std::vector<CompetingRiskObservation> records() {
    std::vector<CompetingRiskObservation> r;
    for (std::size_t i = 0; i < f_time.size(); ++i)
        r.push_back({f_time[i],static_cast<int>(f_cause[i])});
    return r;
}

Matrix design() {
    Matrix x(f_time.size(),3);
    for (std::size_t i = 0; i < f_time.size(); ++i) {
        x(i,0) = f_age[i];
        x(i,1) = f_male[i];
        x(i,2) = f_hgb[i];
    }
    return x;
}

void expect_fit(const FineGrayFit& f, const std::vector<double>& coef, const std::vector<double>& var,
                const std::vector<double>& inf, const std::vector<double>& loglik, const std::vector<double>& times,
                const std::vector<double>& bfit, const std::vector<double>& pred) {
    EXPECT_TRUE(f.converged);
    for (std::size_t a = 0; a < 3; ++a) {
        EXPECT_NEAR(f.coefficients[a],coef[a],1e-8);
        for (std::size_t b = 0; b < 3; ++b) {
            EXPECT_NEAR(f.covariance(a,b),var[b*3+a],1e-7*std::abs(var[b*3+a])+1e-14);
            EXPECT_NEAR(f.information(a,b),inf[b*3+a],1e-6*std::abs(inf[b*3+a])+1e-10);
        }
    }
    EXPECT_NEAR(f.null_log_likelihood,loglik[0],1e-8);
    EXPECT_NEAR(f.log_likelihood,loglik[1],1e-8);
    ASSERT_EQ(f.event_times.size(),times.size());
    for (std::size_t k = 0; k < times.size(); ++k) {
        EXPECT_EQ(f.event_times[k],times[k]);
        EXPECT_NEAR(f.baseline_cumulative_hazard[k],bfit[k],1e-7*bfit[k]);
        EXPECT_NEAR(f.predict_incidence(times[k],{70,1,12}),pred[k],1e-7);
    }
}

// References: cmprsk 2.2.12 crr on 500 mgus2 subjects (age, male, haemoglobin).
TEST(FineGray, MatchesCrr) {
    const auto obs = records();
    const auto x = design();
    expect_fit(fit_fine_gray(obs,x),c1_coef,c1_var,c1_inf,c1_loglik,c1_times,c1_bfit,c1_pred);
    expect_fit(fit_fine_gray(obs,x,{.cause=2}),c2_coef,c2_var,c2_inf,c2_loglik,c2_times,c2_bfit,c2_pred);
    std::vector<int> groups(f_male.begin(),f_male.end());
    expect_fit(fit_fine_gray(obs,x,{.censoring_groups=groups}),cg_coef,cg_var,cg_inf,cg_loglik,cg_times,cg_bfit,
               cg_pred);
}

TEST(FineGray, Validation) {
    const std::vector<CompetingRiskObservation> obs{{1,1},{2,2},{3,0}};
    EXPECT_THROW((void)fit_fine_gray(obs,Matrix(2,1)),std::invalid_argument);
    EXPECT_THROW((void)fit_fine_gray(obs,Matrix(3,1),{.cause=3}),std::invalid_argument);
    EXPECT_THROW((void)fit_fine_gray(obs,Matrix(3,1),{.censoring_groups={0}}),std::invalid_argument);
    const auto f = fit_fine_gray({{1,1},{2,2},{3,1},{4,0},{5,1}},Matrix(5,1,{1,0,0,1,1}));
    EXPECT_THROW((void)f.predict_incidence(1,{1,2}),std::invalid_argument);
}

} // namespace
