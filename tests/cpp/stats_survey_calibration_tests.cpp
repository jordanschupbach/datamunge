#include <datamunge/stats/survey_calibration.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {
using namespace datamunge::stats;
using Matrix = datamunge::linalg::DenseMatrix<double>;

#include "stats_survey_calibration_data.inc"

// Model matrix of ~stype + api99 with treatment coding (E is the reference level).
Matrix model() {
    Matrix x(k_pw.size(),4);
    for (std::size_t i = 0; i < k_pw.size(); ++i) {
        x(i,0) = 1;
        x(i,1) = k_stype[i] == 2;
        x(i,2) = k_stype[i] == 3;
        x(i,3) = k_api99[i];
    }
    return x;
}

void expect_weights(const CalibrationResult& r, const std::vector<double>& ref, double tol) {
    EXPECT_TRUE(r.converged);
    ASSERT_EQ(r.weights.size(),ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i)
        EXPECT_NEAR(r.weights[i],ref[i],tol*ref[i]) << i;
}

// References: survey calibrate() on apiclus1 with the population totals of its documentation.
TEST(SurveyCalibration, MatchesSurveyCalibrate) {
    const auto x = model();
    expect_weights(calibrate_weights(k_pw,x,k_targets),ref_linear,1e-10);
    expect_weights(calibrate_weights(k_pw,x,k_targets,{.distance=CalibrationDistance::raking}),ref_raking,1e-9);
    expect_weights(calibrate_weights(k_pw,x,k_targets,{.distance=CalibrationDistance::logit,.lower=.5,.upper=1.9}),
                   ref_logit,1e-9);
    expect_weights(calibrate_weights(k_pw,x,k_targets,{.lower=.5,.upper=1.7}),ref_bounded_linear,1e-9);
    // Truncated raking converges linearly; the reference also used 200 iterations.
    expect_weights(calibrate_weights(k_pw,x,k_targets,{.distance=CalibrationDistance::raking,.lower=.6,.upper=1.6,
                                                       .tolerance=1e-13,.max_iterations=200}),
                   ref_bounded_raking,1e-9);
    const auto r = calibrate_weights(k_pw,x,k_targets,{.distance=CalibrationDistance::logit,.lower=.5,.upper=1.9});
    for (std::size_t a = 0; a < 4; ++a)
        EXPECT_NEAR(r.achieved[a],k_targets[a],1e-8*k_targets[a]);
    for (const double g : r.g) {
        EXPECT_GT(g,.5);
        EXPECT_LT(g,1.9);
    }
}

// Reference: survey rake() to school-type and school-wide-target margins.
TEST(SurveyCalibration, RakingMatchesSurveyRake) {
    std::vector<std::size_t> type, wide;
    for (std::size_t i = 0; i < k_pw.size(); ++i) {
        type.push_back(static_cast<std::size_t>(k_stype[i])-1);
        wide.push_back(static_cast<std::size_t>(k_schwide[i])-1);
    }
    const auto r = rake_weights(k_pw,{{type,{4421,755,1018}},{wide,{1072,5122}}});
    expect_weights(r,ref_rake,1e-8);
    ASSERT_EQ(r.residual.size(),5u);
    for (const double v : r.residual)
        EXPECT_LT(std::abs(v),1e-6);
    // Raking calibration on a non-collinear set of indicators gives the same weights.
    Matrix x(k_pw.size(),4);
    for (std::size_t i = 0; i < k_pw.size(); ++i) {
        x(i,0) = 1;
        x(i,1) = type[i] == 1;
        x(i,2) = type[i] == 2;
        x(i,3) = wide[i] == 1;
    }
    const auto c = calibrate_weights(k_pw,x,{6194,755,1018,5122},{.distance=CalibrationDistance::raking});
    for (std::size_t i = 0; i < k_pw.size(); ++i)
        EXPECT_NEAR(c.weights[i],r.weights[i],1e-7*r.weights[i]);
}

TEST(SurveyCalibration, ReportsInfeasibleBoundsAndValidates) {
    const Matrix x(4,2,{1,0,1,0,1,1,1,1}); // row-major: intercept and group indicator
    const auto r = calibrate_weights({10,10,10,10},x,{50,45},{.distance=CalibrationDistance::logit,.lower=.5,.upper=1.5});
    EXPECT_FALSE(r.converged); // 45 would need g = 2.25 on the second group
    EXPECT_GT(r.max_relative_residual,.01);
    EXPECT_THROW((void)calibrate_weights({10,10},Matrix(2,1,1.0),{20},{.lower=1.2}),std::invalid_argument);
    EXPECT_THROW((void)calibrate_weights({10,10},Matrix(2,1,1.0),{20},{.distance=CalibrationDistance::logit}),
                 std::invalid_argument);
    EXPECT_THROW((void)calibrate_weights({10,10},Matrix(2,2,1.0),{30,30}),std::runtime_error); // dependent columns
    EXPECT_THROW((void)calibrate_weights({10,0},Matrix(2,1,1.0),{20}),std::invalid_argument);
    EXPECT_THROW((void)rake_weights({1,1},{{{0,1},{1,1}},{{0,1},{2,2}}}),std::invalid_argument); // grand totals differ
    EXPECT_THROW((void)rake_weights({1,1},{{{0,0},{1,1}}}),std::invalid_argument); // empty category with a target
}

} // namespace
