#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/lanczos_analysis.hpp>
#include <limits>
#include <numbers>
using namespace datamunge::algorithms;
TEST(LanczosAnalysis, ExactTinyStagesProjectionAndExplicitRitzResiduals) {
  const std::vector<std::vector<double>> A = {{2, 1}, {1, 2}};
  const auto                             t = lanczos_trace(A, {1, 0}, 2);
  ASSERT_EQ(t.steps.size(), 2U);
  EXPECT_EQ(t.result.alpha, (std::vector<double>{2, 2}));
  EXPECT_EQ(t.result.beta, (std::vector<double>{1}));
  EXPECT_EQ(t.result.V, (std::vector<std::vector<double>>{{1, 0}, {0, 1}}));
  EXPECT_EQ(t.steps[0].matrix_product, (std::vector<double>{2, 1}));
  EXPECT_EQ(t.steps[0].three_term, (std::vector<double>{0, 1}));
  EXPECT_EQ(t.steps[0].residual, (std::vector<double>{0, 1}));
  EXPECT_EQ(t.steps[0].residual_norm, 1);
  EXPECT_EQ(t.steps[1].residual_norm, 0);
  EXPECT_EQ(t.stop, LanczosStop::InvariantSubspace);
  EXPECT_EQ(t.work.matvecs, 2U);
  EXPECT_EQ(t.work.matrix_products, 8U);
  EXPECT_EQ(t.work.dot_products, 5U);
  EXPECT_EQ(t.work.reorth_projections, 3U);
  EXPECT_EQ(t.work.norm_evaluations, 3U);
  EXPECT_EQ(t.captured_entries, 4 + 2 + 2 * 7 + 3);
  const auto d = lanczos_ritz_analysis(A, t.result);
  ASSERT_EQ(d.values.size(), 2U);
  EXPECT_NEAR(d.values[0], 3, 1e-12);
  EXPECT_NEAR(d.values[1], 1, 1e-12);
  EXPECT_LT(d.residual_norms[0], 1e-12);
  EXPECT_EQ(d.projection_error, 0);
  EXPECT_EQ(d.orthogonality_error, 0);
}
TEST(LanczosAnalysis, AnalyticTridiagonalFullSpectrumAndProjectionCertificate) {
  for (int n = 2; n <= 12; ++n) {
    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
      A[i][i] = 2;
      if (i + 1 < n)
        A[i][i + 1] = A[i + 1][i] = -1;
    }
    std::vector<double> b(n);
    b[0]         = 1;
    const auto t = lanczos_trace(A, b, n);
    ASSERT_EQ(t.result.alpha.size(), n);
    const auto d = lanczos_ritz_analysis(A, t.result);
    for (int i = 0; i < n; ++i) {
      const double expected = 2 - 2 * std::cos((n - i) * std::numbers::pi / (n + 1));
      EXPECT_NEAR(d.values[i], expected, 1e-10);
      EXPECT_LT(d.residual_norms[i], 1e-10);
    }
    EXPECT_LT(d.projection_error, 1e-12);
    EXPECT_LT(d.orthogonality_error, 1e-12);
  }
}
TEST(LanczosAnalysis, OriginalFullBasisAndActualWorkShape) {
  const auto A = lanczos_demo_matrix();
  const auto t = lanczos_trace(A, std::vector<double>(30, 1), 30);
  const auto d = t.result.alpha.size();
  EXPECT_EQ(d, 30U);
  EXPECT_EQ(t.work.matvecs, d);
  EXPECT_EQ(t.work.matrix_products, 30 * 30 * d);
  EXPECT_EQ(t.work.dot_products, d + d * (d + 1) / 2);
  EXPECT_EQ(t.work.reorth_projections, d * (d + 1) / 2);
  EXPECT_EQ(t.work.norm_evaluations, d + 1);
  EXPECT_EQ(t.captured_entries, 30 * 30 + 30 + d * (1 + 3 * 30) + d * (d + 1) / 2);
  const auto r = lanczos_ritz_analysis(A, t.result);
  EXPECT_LT(r.orthogonality_error, 1e-12);
  EXPECT_LT(r.projection_error, 1e-11);
  for (auto residual : r.residual_norms)
    EXPECT_LT(residual, 1e-10);
}
TEST(LanczosAnalysis, MissingStartComponentsAbsoluteToleranceAndTinyNonzeroStart) {
  const std::vector<std::vector<double>> A    = {{1, 0, 0}, {0, 2, 0}, {0, 0, 9}};
  const auto                             axis = lanczos_trace(A, {1, 0, 0}, 10);
  ASSERT_EQ(axis.result.alpha.size(), 1U);
  EXPECT_EQ(axis.result.beta.size(), 1U);
  const auto one = lanczos_ritz_analysis(A, axis.result);
  EXPECT_EQ(one.values[0], 1);
  EXPECT_EQ(one.residual_norms[0], 0);
  auto tinyA = A;
  for (auto& row : tinyA)
    for (auto& x : row)
      x *= 1e-14;
  const auto small = lanczos_trace(tinyA, {1, 1, 1}, 3);
  EXPECT_EQ(small.result.alpha.size(), 1U);
  EXPECT_EQ(small.stop, LanczosStop::InvariantSubspace);
  EXPECT_GT(small.steps[0].residual_norm, 0);
  const auto tiny =
      lanczos_trace({{2, 0}, {0, 3}}, {std::numeric_limits<double>::denorm_min(), 0}, 2);
  EXPECT_EQ(tiny.result.alpha, (std::vector<double>{2}));
  EXPECT_EQ(lanczos_trace(A, {0, 0, 0}, 3).stop, LanczosStop::ZeroStart);
  EXPECT_EQ(lanczos_trace({}, {}, 3).stop, LanczosStop::Empty);
  EXPECT_EQ(lanczos_trace(A, {1, 1, 1}, 0).stop, LanczosStop::StepLimit);
}
TEST(LanczosAnalysis, CompleteValidationQuotasAndOwnedStages) {
  EXPECT_THROW((void)lanczos_iteration({{1}}, {1}, -1), std::invalid_argument);
  EXPECT_THROW((void)lanczos_iteration({{1, 2}, {0, 1}}, {1, 1}, 2), std::invalid_argument);
  EXPECT_THROW((void)lanczos_iteration({{1}}, {1, 2}, 2), std::invalid_argument);
  EXPECT_THROW((void)lanczos_iteration({{1, 2}, {2}}, {1, 2}, 2), std::invalid_argument);
  EXPECT_THROW(
      (void)lanczos_iteration({{std::numeric_limits<double>::infinity()}}, {1}, 0),
      std::invalid_argument);
  const auto t = lanczos_trace({{2, 1}, {1, 2}}, {1, 0}, 2);
  EXPECT_NO_THROW((void)lanczos_trace(t.matrix, t.start, 2, t.captured_entries));
  EXPECT_THROW(
      (void)lanczos_trace(t.matrix, t.start, 2, t.captured_entries - 1), std::length_error);
  EXPECT_NO_THROW((void)lanczos_trace(t.matrix, t.start, 2, 0));
  EXPECT_THROW((void)lanczos_ritz_analysis(t.matrix, t.result, 1), std::length_error);
  auto bad = t.result;
  bad.V[0].clear();
  EXPECT_THROW((void)lanczos_ritz_analysis(t.matrix, bad), std::invalid_argument);
  auto c                       = t;
  c.matrix[0][0]               = 99;
  c.start[0]                   = 99;
  c.steps[0].matrix_product[0] = 99;
  c.projections[0].coefficient = 99;
  c.result.V[0][0]             = 99;
  EXPECT_EQ(t.matrix[0][0], 2);
  EXPECT_EQ(t.result.V[0][0], 1);
  EXPECT_EQ(t.steps[0].matrix_product[0], 2);
}
