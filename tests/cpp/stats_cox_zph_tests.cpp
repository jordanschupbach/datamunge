#include <datamunge/stats/cox_counting.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_cox_zph_data.inc"

Matrix columns(const std::vector<std::vector<double>>& cols) {
    Matrix x(cols[0].size(),cols.size());
    for (std::size_t j = 0; j < cols.size(); ++j)
        for (std::size_t i = 0; i < cols[j].size(); ++i)
            x(i,j) = cols[j][i];
    return x;
}

// table holds (chisq, p) for each term then GLOBAL.
void expect_table(const ProportionalHazardsTest& t, const std::vector<double>& table) {
    const std::size_t p = t.terms.size();
    ASSERT_EQ(table.size(),2*(p+1));
    for (std::size_t a = 0; a < p; ++a) {
        EXPECT_NEAR(t.terms[a].statistic,table[2*a],1e-7*(1+table[2*a]));
        EXPECT_NEAR(t.terms[a].p_value,table[2*a+1],1e-8);
    }
    EXPECT_NEAR(t.global.statistic,table[2*p],1e-7*(1+table[2*p]));
    EXPECT_NEAR(t.global.p_value,table[2*p+1],1e-8);
}

// R orders tied rows differently; compare rows sorted by (stratum, time, residuals).
void expect_residuals(const ProportionalHazardsTest& t, const std::vector<double>& x, const std::vector<double>& time,
                      const std::vector<double>& y, const std::vector<double>& strata = {}) {
    const std::size_t p = t.terms.size(), rows = time.size();
    ASSERT_EQ(t.event_times.size(),rows);
    std::vector<std::vector<double>> ours, theirs;
    for (std::size_t r = 0; r < rows; ++r) {
        std::vector<double> a{strata.empty() ? 0.0 : double(t.event_strata[r]),t.event_times[r],
                              t.transformed_times[r]};
        std::vector<double> b{strata.empty() ? 0.0 : strata[r],time[r],x[r]};
        for (std::size_t k = 0; k < p; ++k) {
            a.push_back(t.scaled_schoenfeld(r,k));
            b.push_back(y[r*p+k]);
        }
        ours.push_back(a);
        theirs.push_back(b);
    }
    std::ranges::sort(ours);
    std::ranges::sort(theirs);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t k = 0; k < p+3; ++k)
            EXPECT_NEAR(ours[r][k],theirs[r][k],1e-7*(1+std::abs(theirs[r][k]))) << r << ' ' << k;
}

// References: R survival 3.8.6 cox.zph.
TEST(CoxZph, MatchesCoxZphOnVeteran) {
    std::vector<CountingRecord> r;
    for (std::size_t i = 0; i < v_time.size(); ++i)
        r.push_back({0,v_time[i],v_status[i] == 1});
    const auto x = columns({v_trt,v_karno,v_age,v_prior});
    for (const auto ties : {CoxTies::efron,CoxTies::breslow}) {
        const CoxCountingOptions options{.ties=ties};
        const auto fit = fit_cox_counting(r,x,options);
        const bool efron = ties == CoxTies::efron;
        expect_table(test_proportional_hazards(r,x,fit,options),efron ? ve_km_table : vb_km_table);
        expect_table(test_proportional_hazards(r,x,fit,options,TimeTransform::rank),
                     efron ? ve_rank_table : vb_rank_table);
        expect_table(test_proportional_hazards(r,x,fit,options,TimeTransform::identity),
                     efron ? ve_identity_table : vb_identity_table);
        expect_table(test_proportional_hazards(r,x,fit,options,TimeTransform::log),
                     efron ? ve_log_table : vb_log_table);
        const auto km = test_proportional_hazards(r,x,fit,options);
        expect_residuals(km,efron ? ve_x : vb_x,efron ? ve_t : vb_t,efron ? ve_y : vb_y);
    }
}

TEST(CoxZph, MatchesCoxZphWithStrataAndStartStop) {
    std::vector<CountingRecord> r;
    for (std::size_t i = 0; i < h_start.size(); ++i)
        r.push_back({h_start[i],h_stop[i],h_event[i] == 1});
    std::vector<int> strata(h_surgery.begin(),h_surgery.end());
    // cox.zph numbers strata from 1; our labels are the surgery codes 0 and 1.
    std::vector<double> r_strata = hs_strata;
    for (double& s : r_strata)
        s -= 1;
    const auto x = columns({h_age,h_year,h_transplant});
    const CoxCountingOptions options{.strata=strata};
    const auto t = test_proportional_hazards(r,x,fit_cox_counting(r,x,options),options);
    expect_table(t,hs_table);
    expect_residuals(t,hs_x,hs_t,hs_y,r_strata);
}

TEST(CoxZph, Validation) {
    const std::vector<CountingRecord> r{{0,1,true},{0,2,true},{0,3,false},{0,4,true}};
    const Matrix x(4,1,{1,0,1,0});
    const auto fit = fit_cox_counting(r,x);
    EXPECT_THROW((void)test_proportional_hazards(r,Matrix(4,2),fit),std::invalid_argument);
    EXPECT_THROW((void)test_proportional_hazards({{-1,0,true},{-1,1,true},{0,3,false},{0,4,true}},x,fit,{},
                                                 TimeTransform::log),std::invalid_argument);
}

} // namespace
