#include <datamunge/stats/garch.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;

struct GarchReference {
    GarchMean mean;
    std::size_t p, q;
    std::vector<double> params;
    double llf;
    std::vector<double> se, se_classic, vol_head, vol_tail, forecast;
    double aic, bic;
};

#include "stats_garch_data.inc"

std::vector<double> demeaned() {
    double m = 0;
    for (const double v : garch_returns)
        m += v/static_cast<double>(garch_returns.size());
    auto e = garch_returns;
    for (auto& v : e)
        v -= m;
    return e;
}

TEST(ArchLm, MatchesStatsmodels) {
    const auto e = demeaned();
    for (const auto& [lags,ref] : std::vector<std::pair<std::size_t,std::vector<double>>>{{1,ref_lm1},{4,ref_lm4},{12,ref_lm12}}) {
        const auto t = arch_lm_test(e,lags);
        EXPECT_NEAR(t.lm,ref[0],1e-9*ref[0]);
        EXPECT_NEAR(t.lm_p_value,ref[1],1e-8*ref[1]);
        EXPECT_NEAR(t.f,ref[2],1e-9*ref[2]);
        EXPECT_NEAR(t.f_p_value,ref[3],1e-8*ref[3]);
        EXPECT_EQ(t.observations,1000-lags);
    }
    EXPECT_THROW((void)arch_lm_test(e,0),std::invalid_argument);
    EXPECT_THROW((void)arch_lm_test({1,2,3},2),std::invalid_argument);
}

TEST(Garch, MatchesArchAtInteriorOptima) {
    for (const auto& ref : garch_refs) {
        SCOPED_TRACE(testing::Message() << "p " << ref.p << " q " << ref.q << " mean " << int(ref.mean));
        const auto f = fit_garch(garch_returns,{.arch_order=ref.p,.garch_order=ref.q,.mean=ref.mean});
        // At interior optima our log likelihood is at least arch's (up to rounding); at a boundary
        // (arch's beta_2 = 0) our interior parameterization gets within 1e-6.
        const bool boundary = std::ranges::any_of(ref.params,[](double v) { return v < 1e-6; });
        if (!boundary)
            EXPECT_GT(f.log_likelihood,ref.llf-1e-7);
        EXPECT_NEAR(f.log_likelihood,ref.llf,1e-6);
        ASSERT_EQ(f.parameters.size(),ref.params.size());
        for (std::size_t i = 0; i < ref.params.size(); ++i)
            EXPECT_NEAR(f.parameters[i],ref.params[i],boundary ? 2e-3 : 2e-5);
        if (boundary)
            continue;
        EXPECT_TRUE(f.converged);
        for (std::size_t i = 0; i < ref.se.size(); ++i) {
            EXPECT_NEAR(f.standard_errors[i],ref.se[i],2e-3*ref.se[i]);
            EXPECT_NEAR(f.classical_standard_errors[i],ref.se_classic[i],2e-3*ref.se_classic[i]);
        }
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(std::sqrt(f.conditional_variance[i]),ref.vol_head[i],1e-5);
            EXPECT_NEAR(std::sqrt(f.conditional_variance[997+i]),ref.vol_tail[i],1e-5);
        }
        const auto fc = forecast_garch_variance(f,5);
        for (std::size_t h = 0; h < 5; ++h)
            EXPECT_NEAR(fc[h],ref.forecast[h],1e-5);
        EXPECT_NEAR(f.aic,ref.aic,1e-5);
        EXPECT_NEAR(f.bic,ref.bic,1e-5);
    }
}

TEST(Garch, StandardizedResidualsAndValidation) {
    const auto f = fit_garch(garch_returns);
    const auto t = arch_lm_test(f.standardized_residuals,4);
    EXPECT_NEAR(t.lm,ref_lm_std[0],1e-3);
    EXPECT_NEAR(t.lm_p_value,ref_lm_std[1],1e-4);
    EXPECT_LT(f.persistence,1.0);
    const auto far = forecast_garch_variance(f,2000);
    EXPECT_NEAR(far.back(),f.unconditional_variance,1e-8);
    EXPECT_THROW((void)fit_garch(garch_returns,{.arch_order=0}),std::invalid_argument);
    EXPECT_THROW((void)fit_garch(std::vector<double>(50,1.0)),std::invalid_argument);
    EXPECT_THROW((void)forecast_garch_variance(f,0),std::invalid_argument);
}

} // namespace
