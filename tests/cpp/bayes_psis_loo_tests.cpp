#include <datamunge/bayes/psis_loo.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace datamunge::bayes;
#include "bayes_psis_loo_data.inc"

std::vector<std::vector<double>> unflatten(const std::vector<double>& flat, std::size_t n) {
    std::vector<std::vector<double>> out;
    for (std::size_t i = 0; i < flat.size(); i += n)
        out.emplace_back(flat.begin() + static_cast<std::ptrdiff_t>(i), flat.begin() + static_cast<std::ptrdiff_t>(i + n));
    return out;
}

void expect_loo(const PsisLooResult& r, const std::vector<double>& totals, const std::vector<double>& elpd,
                const std::vector<double>& k, const std::vector<double>& neff, const std::string& what) {
    EXPECT_NEAR(r.elpd_loo, totals[0], 1e-9) << what;
    EXPECT_NEAR(r.se_elpd_loo, totals[1], 1e-9) << what;
    EXPECT_NEAR(r.p_loo, totals[2], 1e-9) << what;
    EXPECT_NEAR(r.se_p_loo, totals[3], 1e-9) << what;
    EXPECT_NEAR(r.looic, totals[4], 1e-9) << what;
    EXPECT_NEAR(r.se_looic, totals[5], 1e-9) << what;
    for (std::size_t i = 0; i < elpd.size(); ++i) {
        EXPECT_NEAR(r.pointwise[i].elpd, elpd[i], 1e-10) << what << " obs " << i;
        EXPECT_NEAR(r.pointwise[i].pareto_k, k[i], 1e-9) << what << " obs " << i;
        EXPECT_NEAR(r.pointwise[i].effective_sample_size, neff[i], 1e-7 * neff[i]) << what << " obs " << i;
    }
}
} // namespace

TEST(PsisLoo, MatchesLooPackage) {
    const auto ll = unflatten(ll_flat, 20);
    expect_loo(psis_loo(ll), ref_unit_totals, ref_unit_elpd, ref_unit_k, ref_unit_neff, "r_eff 1");
    expect_loo(psis_loo(ll, reff), ref_reff_totals, ref_reff_elpd, ref_reff_k, ref_reff_neff, "varying r_eff");
    const auto r = psis_loo(ll);
    EXPECT_NEAR(r.k_threshold, 2.0 / 3.0, 1e-15);
    EXPECT_EQ(r.pointwise.size(), 20u);
}

TEST(PsisLoo, SingleVectorWeightsMatchPsis) {
    const auto w = pareto_smoothed_importance_sampling(lr_heavy, 0.8);
    ASSERT_EQ(w.log_weights.size(), ref_heavy_lw.size());
    for (std::size_t s = 0; s < ref_heavy_lw.size(); ++s)
        EXPECT_NEAR(w.log_weights[s], ref_heavy_lw[s], 1e-10) << s;
    EXPECT_NEAR(w.pareto_k, ref_heavy_k[0], 1e-9);
    EXPECT_NEAR(w.effective_sample_size, ref_heavy_k[1], 1e-7 * ref_heavy_k[1]);
    EXPECT_EQ(w.tail_length, static_cast<std::size_t>(std::ceil(3 * std::sqrt(2000 / 0.8))));
}

TEST(PsisLoo, CompareMatchesLooCompare) {
    const auto a = psis_loo(unflatten(ll_flat, 20));
    const auto b = psis_loo(unflatten(ll2_flat, 20));
    const auto c = loo_compare(a, b);
    EXPECT_NEAR(c.elpd_diff, ref_compare[0], 1e-9);
    EXPECT_NEAR(c.se_diff, ref_compare[1], 1e-9);
}

TEST(PsisLoo, EdgeCasesAndValidation) {
    const auto flat = pareto_smoothed_importance_sampling(std::vector<double>(10, 0.0));
    EXPECT_TRUE(std::isinf(flat.pareto_k));
    EXPECT_NEAR(flat.effective_sample_size, 10, 1e-12);
    EXPECT_DOUBLE_EQ(pareto_k_threshold(100), 0.5);
    EXPECT_THROW(pareto_smoothed_importance_sampling({1.0}), std::invalid_argument);
    EXPECT_THROW(pareto_smoothed_importance_sampling({1.0, 2.0}, 0), std::invalid_argument);
    EXPECT_THROW(psis_loo({{1.0, 2.0}, {1.0}}), std::invalid_argument);
    EXPECT_THROW(psis_loo({{-1.0}, {-2.0}}, {1.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(loo_compare(PsisLooResult{}, PsisLooResult{}), std::invalid_argument);
}
