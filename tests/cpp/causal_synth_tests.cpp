#include <datamunge/causal/synthetic_control.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using datamunge::causal::synthetic_control;
using datamunge::causal::synthetic_control_placebos;
using datamunge::causal::SyntheticControlData;
using Matrix = datamunge::linalg::DenseMatrix<double>;
#include "causal_synth_data.inc"

SyntheticControlData basque() {
    SyntheticControlData d;
    d.treated_predictors = basque_x1;
    d.donor_predictors = Matrix(basque_k, basque_j, basque_x0);
    d.treated_outcome = basque_y1;
    d.donor_outcomes = Matrix(basque_t, basque_j, basque_y0);
    for (std::size_t t = 5; t < 15; ++t) // 1960-1969
        d.fit_periods.push_back(t);
    d.treatment_period = 15; // 1970
    return d;
}
} // namespace

TEST(SyntheticControl, ExactDonorWeightsAtFixedV) {
    const auto d = basque();
    const auto equal = synthetic_control(d, {.predictor_weights = std::vector<double>(basque_k, 1.0)});
    for (std::size_t j = 0; j < basque_j; ++j)
        EXPECT_NEAR(equal.weights[j], exact_w_equal_v[j], 1e-7) << j;
    EXPECT_NEAR(equal.fit_mspe, exact_loss_equal_v[0], 1e-7 * exact_loss_equal_v[0]);
    const auto fixed = synthetic_control(d, {.predictor_weights = synth_v});
    for (std::size_t j = 0; j < basque_j; ++j)
        EXPECT_NEAR(fixed.weights[j], exact_w_synth_v[j], 1e-7) << j;
    EXPECT_NEAR(fixed.fit_mspe, exact_loss_synth_v[0], 1e-6 * exact_loss_synth_v[0]);
    // Synth's interior-point weights agree to its own tolerance.
    for (std::size_t j = 0; j < basque_j; ++j)
        EXPECT_NEAR(fixed.weights[j], synth_w[j], 1e-5) << j;
    EXPECT_NEAR(std::accumulate(fixed.weights.begin(), fixed.weights.end(), 0.0), 1.0, 1e-12);
}

TEST(SyntheticControl, OptimisedVFitsAtLeastAsWellAsSynth) {
    const auto d = basque();
    const auto f = synthetic_control(d);
    EXPECT_LE(f.fit_mspe, synth_loss_v[0] * (1 + 1e-4));
    EXPECT_NEAR(std::accumulate(f.predictor_weights.begin(), f.predictor_weights.end(), 0.0), 1.0, 1e-12);
    // Cataluna (donor 9) and Madrid (donor 13) dominate, as in Abadie and Gardeazabal.
    EXPECT_GT(f.weights[8] + f.weights[12], 0.95);
    // Gap definitions.
    double pre = 0;
    for (std::size_t t = 0; t < 15; ++t)
        pre += f.gap[t] * f.gap[t];
    EXPECT_NEAR(f.pre_rmspe, std::sqrt(pre / 15), 1e-12);
    EXPECT_NEAR(f.gap[20], basque_y1[20] - f.synthetic[20], 1e-12);
}

TEST(SyntheticControl, PlacebosRefitEachDonor) {
    const auto d = basque();
    const std::vector<double> v(basque_k, 1.0);
    const auto p = synthetic_control_placebos(d, {.predictor_weights = v});
    ASSERT_EQ(p.placebos.size(), basque_j);
    EXPECT_EQ(p.included.size(), basque_j);
    std::size_t exceed = 0;
    for (const auto& f : p.placebos) {
        EXPECT_EQ(f.weights.size(), basque_j - 1);
        exceed += f.rmspe_ratio >= p.treated.rmspe_ratio;
    }
    EXPECT_DOUBLE_EQ(p.p_value, (1.0 + exceed) / (1.0 + basque_j));
    // Placebo 0 equals a direct fit with donor 0 as the treated unit.
    SyntheticControlData pd = d;
    pd.treated_predictors.clear();
    pd.treated_outcome.clear();
    pd.donor_predictors = Matrix(basque_k, basque_j - 1, 0.0);
    pd.donor_outcomes = Matrix(basque_t, basque_j - 1, 0.0);
    for (std::size_t r = 0; r < basque_k; ++r) {
        pd.treated_predictors.push_back(d.donor_predictors(r, 0));
        for (std::size_t j = 1; j < basque_j; ++j)
            pd.donor_predictors(r, j - 1) = d.donor_predictors(r, j);
    }
    for (std::size_t t = 0; t < basque_t; ++t) {
        pd.treated_outcome.push_back(d.donor_outcomes(t, 0));
        for (std::size_t j = 1; j < basque_j; ++j)
            pd.donor_outcomes(t, j - 1) = d.donor_outcomes(t, j);
    }
    const auto direct = synthetic_control(pd, {.predictor_weights = v});
    EXPECT_NEAR(direct.rmspe_ratio, p.placebos[0].rmspe_ratio, 1e-12);
    // A strict pre-fit filter keeps only placebos fitting at least as well as the treated unit.
    const auto strict = synthetic_control_placebos(d, {.predictor_weights = v}, 1.0);
    for (const std::size_t u : strict.included)
        EXPECT_LE(strict.placebos[u].pre_rmspe, strict.treated.pre_rmspe * (1 + 1e-12));
}

TEST(SyntheticControl, ValidatesInputs) {
    auto d = basque();
    EXPECT_THROW(synthetic_control(d, {.predictor_weights = {1, 2}}), std::invalid_argument);
    EXPECT_THROW(synthetic_control(d, {.predictor_weights = std::vector<double>(basque_k, 0.0)}),
                 std::invalid_argument);
    auto late = d;
    late.fit_periods.push_back(20);
    EXPECT_THROW(synthetic_control(late), std::invalid_argument);
    auto flat = d;
    flat.treated_predictors[0] = 1;
    for (std::size_t j = 0; j < basque_j; ++j)
        flat.donor_predictors(0, j) = 1;
    EXPECT_THROW(synthetic_control(flat), std::invalid_argument);
    auto one = d;
    one.donor_predictors = Matrix(basque_k, 1, 1.0);
    EXPECT_THROW(synthetic_control(one), std::invalid_argument);
}
