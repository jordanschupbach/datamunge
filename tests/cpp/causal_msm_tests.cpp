#include <datamunge/causal/msm.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <set>
#include <vector>

namespace {
using datamunge::causal::fit_marginal_structural_model;
using datamunge::causal::longitudinal_weights;
using datamunge::causal::MsmFamily;
using datamunge::causal::WeightRiskSet;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "causal_msm_data.inc"

struct Haart {
    std::vector<std::size_t> id;
    std::vector<int> treat, dropout;
    Matrix numerator, denominator;
};

Haart haart() {
    const std::size_t n = h_id.size();
    Haart h{{}, {}, {}, Matrix(n, 3), Matrix(n, 4)};
    for (std::size_t i = 0; i < n; ++i) {
        h.id.push_back(static_cast<std::size_t>(h_id[i]));
        h.treat.push_back(static_cast<int>(h_treat[i]));
        h.dropout.push_back(static_cast<int>(h_dropout[i]));
        const double row[4]{h_sex[i], h_age[i], h_t100[i], h_cd4[i]};
        for (std::size_t j = 0; j < 4; ++j) {
            if (j < 3)
                h.numerator(i, j) = row[j];
            h.denominator(i, j) = row[j];
        }
    }
    return h;
}

void expect_close(const std::vector<double>& got, const std::vector<double>& want, double rel, const std::string& what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    for (std::size_t i = 0; i < want.size(); ++i)
        EXPECT_NEAR(got[i], want[i], rel * (1 + std::abs(want[i]))) << what << " " << i;
}
} // namespace

TEST(CausalMsm, WeightsMatchIpwtm) {
    const auto h = haart();
    const auto treatment = longitudinal_weights(h.id, {.event = h.treat, .numerator = h.numerator,
                                                       .denominator = h.denominator});
    expect_close(treatment.weights, h_treat_weights, 1e-7, "treatment weights");
    const auto censoring = longitudinal_weights(h.id, {.event = h.dropout, .numerator = h.numerator,
                                                       .denominator = h.denominator});
    expect_close(censoring.weights, h_dropout_weights, 1e-7, "censoring weights");
    const auto truncated = longitudinal_weights(
        h.id, {.event = h.treat, .numerator = h.numerator, .denominator = h.denominator}, {.truncation = 0.01});
    expect_close(truncated.truncated, h_trunc_weights, 1e-7, "truncated weights");
    // Rows after treatment starts contribute factor 1: weights stay constant there.
    for (std::size_t i = 1; i < h.id.size(); ++i)
        if (h.id[i] == h.id[i - 1] && !treatment.at_risk[i])
            EXPECT_DOUBLE_EQ(treatment.weights[i], treatment.weights[i - 1]);

    std::vector<double> w(h.id.size());
    Matrix design(h.id.size(), 2);
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = treatment.weights[i] * censoring.weights[i];
        design(i, 0) = h_treat[i];
        design(i, 1) = h_t100[i];
    }
    const auto fit = fit_marginal_structural_model(h_event, design, w, h.id, MsmFamily::logistic);
    EXPECT_TRUE(fit.converged);
    expect_close(fit.coefficients, {h_msm[0], h_msm[1], h_msm[2]}, 1e-6, "msm coefficients");
    expect_close(fit.standard_errors, {h_msm[3], h_msm[4], h_msm[5]}, 1e-6, "msm standard errors");
    EXPECT_EQ(fit.clusters, static_cast<std::size_t>(std::set<double>(h_id.begin(), h_id.end()).size()));
}

TEST(CausalMsm, UnstabilizedAllRowsAndValidation) {
    // All-rows risk set with a covariate-free numerator: unstabilized weights are 1 / P(observed history).
    const std::vector<std::size_t> id{1, 1, 2, 2, 3, 3, 4, 4};
    const std::vector<int> a{0, 1, 1, 1, 0, 0, 1, 0};
    const Matrix l(8, 1, {0, 1, 1, 0, 0, 1, 1, 0});
    const auto w = longitudinal_weights(id, {.event = a, .denominator = l},
                                        {.risk_set = WeightRiskSet::all_rows, .stabilized = false});
    for (std::size_t i = 0; i < id.size(); ++i) {
        const double p = w.denominator_probability[i];
        const double f = a[i] ? 1 / p : 1 / (1 - p);
        const double previous = i > 0 && id[i] == id[i - 1] ? w.weights[i - 1] : 1.0;
        EXPECT_NEAR(w.weights[i], previous * f, 1e-12);
    }
    EXPECT_TRUE(w.numerator_coefficients.empty());
    EXPECT_THROW((void)longitudinal_weights({1, 2, 1}, {.event = {0, 1, 0}, .denominator = Matrix(3, 0)}),
                 std::invalid_argument);
    EXPECT_THROW((void)longitudinal_weights(id, {.event = a, .denominator = l}, {.truncation = 0.5}),
                 std::invalid_argument);
    // Linear MSM: unweighted equals OLS.
    const Matrix x(6, 1, {0, 0, 0, 1, 1, 1});
    const auto f = fit_marginal_structural_model({1, 2, 3, 3, 4, 5}, x, std::vector<double>(6, 1.0),
                                                 {1, 2, 3, 4, 5, 6}, MsmFamily::linear);
    EXPECT_NEAR(f.coefficients[1], 2.0, 1e-12);
    EXPECT_THROW((void)fit_marginal_structural_model({0, 2}, Matrix(2, 0), {1, 1}, {1, 2}, MsmFamily::logistic),
                 std::invalid_argument);
}
