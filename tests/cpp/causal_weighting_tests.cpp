#include <datamunge/causal/weighting.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using datamunge::causal::entropy_balance;
using datamunge::causal::EntropyBalanceOptions;
using datamunge::causal::weighted_effect;
using datamunge::causal::WeightingEstimand;
using Matrix = datamunge::linalg::DenseMatrix<double>;
const double na = std::numeric_limits<double>::quiet_NaN();
#include "causal_matching_data.inc"
#include "causal_weighting_data.inc"

struct Lalonde {
    Matrix x;
    std::vector<int> a;
};

Lalonde lalonde() {
    const std::size_t n = k_treat.size(), p = k_x.size() / n;
    Lalonde d{Matrix(n, p), {}};
    for (std::size_t i = 0; i < n; ++i) {
        d.a.push_back(static_cast<int>(k_treat[i]));
        for (std::size_t j = 0; j < p; ++j)
            d.x(i, j) = k_x[i * p + j];
    }
    return d;
}

void expect_close(const std::vector<double>& got, const std::vector<double>& want, double rel, double abs,
                  const std::string& what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (std::isnan(want[i]))
            EXPECT_TRUE(std::isnan(got[i])) << what << " " << i;
        else
            EXPECT_NEAR(got[i], want[i], abs + rel * std::abs(want[i])) << what << " " << i;
    }
}

struct Reference {
    const char* name;
    double tolerance; // relative tolerance for weights and adjusted balance
    EntropyBalanceOptions options;
    const std::vector<double>& weights;
    const std::vector<double>& effect;
    const std::vector<double>& ess;
    const std::vector<double>& smd_before;
    const std::vector<double>& vr_after;
    const std::vector<double>& ks_after;
    const std::vector<double>& ks_before;
};
} // namespace

TEST(CausalWeighting, EntropyBalanceMatchesWeightIt) {
    const auto d = lalonde();
    const std::vector<Reference> refs{
        {"att", 1e-7, {}, e_att_weights, e_att_effect, e_att_ess, e_att_smd_before, e_att_vr_after, e_att_ks_after,
         e_att_ks_before},
        {"att2", 1e-7, {.moments = 2}, e_att2_weights, e_att2_effect, e_att2_ess, e_att2_smd_before, e_att2_vr_after,
         e_att2_ks_after, e_att2_ks_before},
        {"ate", 1e-7, {.estimand = WeightingEstimand::ate}, e_ate_weights, e_ate_effect, e_ate_ess, e_ate_smd_before,
         e_ate_vr_after, e_ate_ks_after, e_ate_ks_before},
        {"atc", 2e-3, {.estimand = WeightingEstimand::atc}, e_atc_weights, e_atc_effect, e_atc_ess, e_atc_smd_before,
         e_atc_vr_after, e_atc_ks_after, e_atc_ks_before},
        {"sampling weights", 1e-7, {.base_weights = e_base}, e_sw_weights, e_sw_effect, e_sw_ess, e_sw_smd_before,
         e_sw_vr_after, e_sw_ks_after, e_sw_ks_before},
    };
    for (const auto& r : refs) {
        SCOPED_TRACE(r.name);
        const auto fit = entropy_balance(d.a, d.x, r.options);
        EXPECT_TRUE(fit.converged);
        std::vector<double> want = r.weights;
        if (!r.options.base_weights.empty())
            for (std::size_t i = 0; i < want.size(); ++i)
                want[i] *= r.options.base_weights[i];
        // Compare within-group shares: WeightIt scales sampling-weighted groups differently, and the
        // effect and its HC0 error do not depend on a group's weight scale.
        const auto shares = [&](std::vector<double> v) {
            double s[2]{};
            for (std::size_t i = 0; i < v.size(); ++i)
                s[d.a[i]] += v[i];
            for (std::size_t i = 0; i < v.size(); ++i)
                v[i] /= s[d.a[i]];
            return v;
        };
        expect_close(shares(fit.weights), shares(want), r.tolerance, 1e-12, "weights");
        const auto e = weighted_effect(k_re78, d.a, fit.weights);
        const double tol = std::max(1e-6, r.tolerance);
        EXPECT_NEAR(e.estimate, r.effect[0], tol * std::abs(r.effect[0]));
        EXPECT_NEAR(e.standard_error, r.effect[1], tol * r.effect[1]);
        expect_close({fit.effective_sample_size[0], fit.effective_sample_size[1]}, r.ess, tol, 0, "ess");
        expect_close(fit.before.standardized_difference, r.smd_before, 1e-9, 1e-12, "smd before");
        expect_close(fit.after.standardized_difference, std::vector<double>(d.x.cols(), 0.0), 0, 1e-8, "smd after");
        expect_close(fit.after.variance_ratio, r.vr_after, tol, 0, "variance ratio after");
        expect_close(fit.after.ecdf_max, r.ks_after, tol, tol > 1e-6 ? 1e-4 : 1e-9, "ks after");
        expect_close(fit.before.ecdf_max, r.ks_before, 1e-9, 1e-12, "ks before");
    }
}

TEST(CausalWeighting, CovariateAdjustedEffectMatchesLmWeightit) {
    const auto d = lalonde();
    const auto one = entropy_balance(d.a, d.x);
    const auto e1 = weighted_effect(k_re78, d.a, one.weights, d.x);
    EXPECT_NEAR(e1.estimate, e_att_adjusted[0], 1e-6 * std::abs(e_att_adjusted[0]));
    EXPECT_NEAR(e1.standard_error, e_att_adjusted[1], 1e-6 * e_att_adjusted[1]);
    // Exact balance: the adjusted coefficient equals the unadjusted weighted contrast.
    EXPECT_NEAR(e1.estimate, weighted_effect(k_re78, d.a, one.weights).estimate, 1e-6);
    const auto two = entropy_balance(d.a, d.x, {.moments = 2});
    const auto e2 = weighted_effect(k_re78, d.a, two.weights, d.x);
    EXPECT_NEAR(e2.estimate, e_att2_adjusted[0], 1e-6 * std::abs(e_att2_adjusted[0]));
    EXPECT_NEAR(e2.standard_error, e_att2_adjusted[1], 1e-6 * e_att2_adjusted[1]);
    EXPECT_EQ(e2.df, static_cast<double>(d.a.size() - 10));
}

TEST(CausalWeighting, ExactBalancePropertiesAndValidation) {
    // Second moments: weighted control means of x and x^2 equal the treated ones.
    const Matrix x(8, 1, {0.1, 0.9, 1.4, 2.2, 0.0, 0.5, 1.1, 1.8});
    const std::vector<int> a{1, 1, 1, 1, 0, 0, 0, 0};
    const auto fit = entropy_balance(a, x, {.moments = 2});
    double m1[2]{}, m2[2]{}, s[2]{};
    for (std::size_t i = 0; i < 8; ++i) {
        s[a[i]] += fit.weights[i];
        m1[a[i]] += fit.weights[i] * x(i, 0);
        m2[a[i]] += fit.weights[i] * x(i, 0) * x(i, 0);
    }
    EXPECT_NEAR(s[0], 4.0, 1e-10);
    EXPECT_NEAR(m1[0] / s[0], m1[1] / s[1], 1e-9);
    EXPECT_NEAR(m2[0] / s[0], m2[1] / s[1], 1e-9);
    // A treated mean outside the control range cannot be balanced.
    const Matrix far(4, 1, {5.0, 6.0, 0.0, 1.0});
    EXPECT_FALSE(entropy_balance({1, 1, 0, 0}, far).converged);
    EXPECT_THROW((void)entropy_balance({1, 1, 0, 0}, far, {.moments = 3}), std::invalid_argument);
    EXPECT_THROW((void)entropy_balance({1, 1, 1, 1}, far), std::invalid_argument);
    EXPECT_THROW((void)weighted_effect({1, 2, 3}, {0, 1, 1}, {1, -1, 1}), std::invalid_argument);
}
