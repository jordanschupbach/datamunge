#include <datamunge/stats/cox_counting.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_cox_cluster_data.inc"

std::vector<CountingRecord> records(const std::vector<double>& start, const std::vector<double>& stop,
                                    const std::vector<double>& event) {
    std::vector<CountingRecord> r;
    for (std::size_t i = 0; i < start.size(); ++i)
        r.push_back({start[i],stop[i],event[i] != 0});
    return r;
}

Matrix columns(const std::vector<std::vector<double>>& cols) {
    Matrix x(cols[0].size(),cols.size());
    for (std::size_t a = 0; a < cols.size(); ++a)
        for (std::size_t i = 0; i < cols[a].size(); ++i)
            x(i,a) = cols[a][i];
    return x;
}

std::vector<int> labels(const std::vector<double>& v) { return {v.begin(),v.end()}; }

void expect_fit(const CoxCountingFit& f, const std::vector<double>& coef, const std::vector<double>& var,
                const std::vector<double>& naive, const std::vector<double>& wald, std::size_t clusters) {
    const std::size_t p = coef.size();
    EXPECT_EQ(f.clusters,clusters);
    for (std::size_t a = 0; a < p; ++a) {
        EXPECT_NEAR(f.coefficients[a],coef[a],1e-8);
        EXPECT_NEAR(f.robust_standard_errors[a],std::sqrt(var[a*p+a]),1e-8*std::sqrt(var[a*p+a]));
        for (std::size_t b = 0; b < p; ++b) {
            EXPECT_NEAR(f.robust_covariance(a,b),var[b*p+a],1e-8*std::sqrt(var[a*p+a]*var[b*p+b]));
            EXPECT_NEAR(f.covariance(a,b),naive[b*p+a],1e-8*std::sqrt(naive[a*p+a]*naive[b*p+b]));
        }
    }
    EXPECT_NEAR(f.robust_wald.statistic,wald[0],1e-7*wald[0]);
    EXPECT_EQ(f.robust_wald.degrees_of_freedom,static_cast<std::size_t>(wald[1]));
    EXPECT_NEAR(f.robust_wald.p_value,wald[2],1e-10);
}

// References: survival 3.8 coxph(... + cluster(id)) on bladder2 (Andersen–Gill and stratified by
// event number) and cgd.
TEST(CoxCluster, MatchesCoxphBladder) {
    const auto r = records(b_start,b_stop,b_event);
    const auto x = columns({b_rx,b_number,b_size});
    const auto id = labels(b_id);
    expect_fit(fit_cox_counting(r,x,{.clusters=id}),ag_efron_coef,ag_efron_var,ag_efron_naive,ag_efron_wald,85);
    expect_fit(fit_cox_counting(r,x,{.ties=CoxTies::breslow,.clusters=id}),ag_breslow_coef,ag_breslow_var,
               ag_breslow_naive,ag_breslow_wald,85);
    expect_fit(fit_cox_counting(r,x,{.strata=labels(b_enum),.clusters=id}),pwp_coef,pwp_var,pwp_naive,pwp_wald,85);
}

TEST(CoxCluster, MatchesCoxphCgd) {
    expect_fit(fit_cox_counting(records(g_start,g_stop,g_event),columns({g_treat,g_steroids}),{.clusters=labels(g_id)}),
               cgd_coef,cgd_var,cgd_naive,cgd_wald,128);
}

TEST(CoxCluster, WithoutClustersLeavesRobustEmpty) {
    const auto f = fit_cox_counting(records(b_start,b_stop,b_event),columns({b_rx}));
    EXPECT_EQ(f.clusters,0u);
    EXPECT_TRUE(f.robust_standard_errors.empty());
    EXPECT_THROW((void)fit_cox_counting(records(b_start,b_stop,b_event),columns({b_rx}),{.clusters={1,2}}),
                 std::invalid_argument);
}

} // namespace
