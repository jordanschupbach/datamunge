#include <datamunge/causal/rdd.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using datamunge::causal::rd_bandwidths;
using datamunge::causal::RdKernel;
using datamunge::causal::RdOptions;
using datamunge::causal::RdResult;
using datamunge::causal::RdVariance;
using datamunge::causal::regression_discontinuity;
#include "causal_rdd_data.inc"

void near(double got, double want, const std::string& what, double tol = 1e-7) {
    EXPECT_NEAR(got, want, tol * (1 + std::abs(want))) << what;
}

// Reference layout: h, b, tau_cl, tau_bc, se_cl, se_rb, N_h_l, N_h_r, robust CI lower, upper,
// then for fuzzy designs first-stage tau_cl, tau_bc, se_cl, se_rb.
void expect_rd(const RdResult& r, const std::vector<double>& ref) {
    near(r.h_left, ref[0], "h");
    near(r.b_left, ref[1], "b");
    near(r.conventional.estimate, ref[2], "tau conventional");
    near(r.robust.estimate, ref[3], "tau bias-corrected");
    near(r.bias_corrected.estimate, ref[3], "bias-corrected estimate");
    near(r.conventional.standard_error, ref[4], "se conventional");
    near(r.bias_corrected.standard_error, ref[4], "bias-corrected uses conventional se");
    near(r.robust.standard_error, ref[5], "se robust");
    EXPECT_EQ(r.effective_left, static_cast<std::size_t>(ref[6]));
    EXPECT_EQ(r.effective_right, static_cast<std::size_t>(ref[7]));
    near(r.robust.lower, ref[8], "robust lower");
    near(r.robust.upper, ref[9], "robust upper");
    if (ref.size() > 10) {
        ASSERT_TRUE(r.fuzzy);
        near(r.first_stage_conventional.estimate, ref[10], "first stage conventional");
        near(r.first_stage_robust.estimate, ref[11], "first stage bias-corrected");
        near(r.first_stage_conventional.standard_error, ref[12], "first stage se conventional");
        near(r.first_stage_robust.standard_error, ref[13], "first stage se robust");
    }
}
} // namespace

TEST(RegressionDiscontinuity, SharpMatchesRdrobust) {
    expect_rd(regression_discontinuity(senate_y, senate_x), ref_default);
    expect_rd(regression_discontinuity(senate_y, senate_x,
                                       {.order = 2, .kernel = RdKernel::uniform, .variance = RdVariance::hc1}),
              ref_uniform_hc1);
    expect_rd(regression_discontinuity(senate_y, senate_x,
                                       {.kernel = RdKernel::epanechnikov,
                                        .variance = RdVariance::hc2,
                                        .bandwidth = 10,
                                        .bias_bandwidth = 20}),
              ref_manual);
    expect_rd(regression_discontinuity(senate_y, senate_x, {.order = 2, .derivative = 1}), ref_kink);
}

TEST(RegressionDiscontinuity, MassPointsMatchRdrobust) {
    std::vector<double> rounded;
    for (const double v : senate_x)
        rounded.push_back(std::round(v));
    const auto bw = rd_bandwidths(senate_y, rounded, {.variance = RdVariance::hc3});
    EXPECT_TRUE(bw.mass_points);
    expect_rd(regression_discontinuity(senate_y, rounded, {.variance = RdVariance::hc3}), ref_mass);
}

TEST(RegressionDiscontinuity, FuzzyMatchesRdrobust) {
    expect_rd(regression_discontinuity(fuzzy_y, fuzzy_x, {.cutoff = 0.1}, fuzzy_t), ref_fuzzy);
    expect_rd(regression_discontinuity(fuzzy_y, fuzzy_x,
                                       {.cutoff = 0.1, .kernel = RdKernel::epanechnikov, .variance = RdVariance::hc0},
                                       fuzzy_t),
              ref_fuzzy_hc0);
}

TEST(RegressionDiscontinuity, BandwidthsAgreeWithEstimate) {
    const auto bw = rd_bandwidths(senate_y, senate_x, {});
    near(bw.h, ref_default[0], "h");
    near(bw.b, ref_default[1], "b");
    EXPECT_FALSE(bw.mass_points);
    EXPECT_GT(bw.pilot, 0);
}

TEST(RegressionDiscontinuity, ValidatesInputs) {
    const std::vector<double> x{-2, -1, 0, 1, 2}, y{1, 2, 3, 4, 5};
    EXPECT_THROW(regression_discontinuity({1, 2}, {1}), std::invalid_argument);
    EXPECT_THROW(regression_discontinuity(y, x), std::invalid_argument); // too few per side
    EXPECT_THROW(regression_discontinuity(senate_y, senate_x, {.order = 1, .bias_order = 1}),
                 std::invalid_argument);
    EXPECT_THROW(regression_discontinuity(senate_y, senate_x, {.derivative = 2}), std::invalid_argument);
    EXPECT_THROW(regression_discontinuity(senate_y, senate_x, {.matches = 0}), std::invalid_argument);
    EXPECT_THROW(regression_discontinuity(senate_y, senate_x, {.bandwidth = -1}), std::invalid_argument);
    EXPECT_THROW(regression_discontinuity(senate_y, senate_x, {.confidence = 1}), std::invalid_argument);
    auto bad = senate_y;
    bad[3] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(regression_discontinuity(bad, senate_x), std::invalid_argument);
}
