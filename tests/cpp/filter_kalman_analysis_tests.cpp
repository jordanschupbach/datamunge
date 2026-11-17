#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/filter/kalman_analysis.hpp>
#include <limits>
using namespace datamunge::filter;
TEST(KalmanStudy, TinyIndependentFractionControlAndActualStages) {
  const auto s = kalman_tiny_step();
  ASSERT_EQ(s.prior.x.size(), 1U);
  EXPECT_EQ(s.prior.x[0], 0);
  EXPECT_EQ(s.prior.P[0][0], 4);
  EXPECT_EQ(s.predicted.x[0], 0);
  EXPECT_EQ(s.predicted.P[0][0], 5);
  EXPECT_EQ(s.filtered.x[0], 1.25);
  EXPECT_EQ(s.filtered.P[0][0], 1.875);
}
TEST(KalmanStudy, OriginalFamilyCompleteOutputsAndInformationEquivalence) {
  const auto d = kalman_demo_data();
  const auto s = kalman_family_study(d);
  ASSERT_EQ(s.linear.filtered.size(), 50U);
  ASSERT_EQ(s.linear.smoothed.size(), 50U);
  ASSERT_EQ(s.ekf.size(), 50U);
  ASSERT_EQ(s.ukf.size(), 50U);
  ASSERT_EQ(s.ensemble.size(), 50U);
  ASSERT_EQ(s.particle.size(), 50U);
  ASSERT_EQ(s.alpha_beta.size(), 50U);
  ASSERT_EQ(s.alpha_beta_gamma.size(), 40U);
  EXPECT_EQ(s.linear.filtered.back().x, s.linear.smoothed.back().x);
  EXPECT_EQ(s.linear.filtered.back().P, s.linear.smoothed.back().P);
  for (std::size_t i = 0; i < 50; ++i) {
    for (int j = 0; j < 2; ++j)
      EXPECT_NEAR(s.linear.filtered[i].x[j], s.information[i].x[j], 1e-9);
    auto diff = s.linear.filtered[i].P;
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < 2; ++c)
        diff[r][c] -= s.linear.smoothed[i].P[r][c];
    EXPECT_GE(kalman_min_eigenvalue2(diff), -1e-9);
  }
  EXPECT_TRUE(std::isfinite(kalman_position_mse(s.ukf, d.nonlinear_truth)));
}
TEST(KalmanStudy, CompleteAdmissionAndMalformedDataBeforeExecution) {
  auto d = kalman_demo_data();
  EXPECT_THROW((void)kalman_family_study(d, {300, 300, .001, 139, 600}), std::length_error);
  EXPECT_THROW((void)kalman_family_study(d, {300, 300, .001, 140, 599}), std::length_error);
  EXPECT_THROW((void)kalman_family_study(d, {1, 300, .001, 140, 600}), std::invalid_argument);
  EXPECT_THROW((void)kalman_demo_ukf(d, 0), std::invalid_argument);
  EXPECT_THROW((void)kalman_demo_ukf(d, 1e-300), std::invalid_argument);
  d.linear_measurements[0].push_back(1);
  EXPECT_THROW((void)kalman_family_study(d), std::invalid_argument);
  d                              = kalman_demo_data();
  d.nonlinear_measurements[0][0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)kalman_family_study(d), std::invalid_argument);
}
TEST(KalmanStudy, CheckedDiagnosticsAndCaptureFailureSemantics) {
  EXPECT_EQ(kalman_position_mse({{{2}, {{1}}}}, {1}), 1);
  EXPECT_EQ(kalman_min_eigenvalue2({{2, 0}, {0, 1}}), 1);
  EXPECT_THROW((void)kalman_position_mse({}, {}), std::invalid_argument);
  EXPECT_THROW((void)kalman_state_column({{{1}, {{1}}}}, 1), std::invalid_argument);
  KalmanFilter k({{1}}, {{1}}, {{1}}, {{3}}, {0}, {{4}});
  EXPECT_THROW((void)kalman_capture_step(k, {1, 2}), std::invalid_argument);
  EXPECT_EQ(k.covariance()[0][0], 5);
}
