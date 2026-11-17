#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/kabsch.hpp>
#include <limits>
using namespace datamunge::algorithms;
namespace {
void rotation_checks(const KabschResult& r) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double v = 0;
      for (int k = 0; k < 3; ++k)
        v += r.rotation[k][i] * r.rotation[k][j];
      EXPECT_NEAR(v, i == j ? 1 : 0, 1e-12);
    }
  const auto&  m   = r.rotation;
  const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                     m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                     m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  EXPECT_NEAR(det, 1, 1e-12);
}
} // namespace
TEST(KabschAnalysis, KnownProperRigidTransform) {
  const std::vector<Point3> s = {{0, 0, 0}, {2, 0, 0}, {0, 3, 0}, {0, 0, 4}, {1, 2, 3}};
  std::vector<Point3>       t;
  for (auto p : s)
    t.push_back({-p[1] + 5, p[0] - 2, p[2] + 1});
  const auto a = kabsch_analyze(s, t);
  rotation_checks(a.result);
  EXPECT_LT(a.result.rmsd, 1e-12);
  EXPECT_NEAR(a.result.translation[0], 5, 1e-12);
  EXPECT_NEAR(a.result.translation[1], -2, 1e-12);
  EXPECT_NEAR(a.result.translation[2], 1, 1e-12);
  EXPECT_LT(a.eigen_residual, 1e-11);
  ASSERT_EQ(a.aligned.size(), s.size());
  for (std::size_t i = 0; i < s.size(); ++i)
    for (int d = 0; d < 3; ++d)
      EXPECT_NEAR(a.aligned[i][d], t[i][d], 1e-12);
  const auto r = kabsch(s, t);
  EXPECT_DOUBLE_EQ(r.rmsd, a.result.rmsd);
}
TEST(KabschAnalysis, ReflectionCannotBecomeAnImproperRotation) {
  const std::vector<Point3> s = {{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};
  std::vector<Point3>       t = s;
  for (auto& p : t)
    p[0] = -p[0];
  const auto a = kabsch_analyze(s, t);
  rotation_checks(a.result);
  EXPECT_NEAR(a.result.rmsd, 2, 1e-12);
}
TEST(KabschAnalysis, DegeneratePointLineAndRepeatedCloud) {
  for (auto s : std::vector<std::vector<Point3>>{
           {{0, 0, 0}}, {{1, 2, 3}, {1, 2, 3}}, {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}}}) {
    std::vector<Point3> t;
    for (auto p : s)
      t.push_back({p[0] + 1, p[1] + 2, p[2] + 3});
    const auto a = kabsch_analyze(s, t);
    rotation_checks(a.result);
    EXPECT_LT(a.result.rmsd, 1e-12);
  }
}
TEST(KabschAnalysis, ScaleAndOffsetControls) {
  const std::vector<Point3> base = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (double scale : {1e-150, 1., 1e150}) {
    std::vector<Point3> s, t;
    for (auto p : base) {
      s.push_back({scale * p[0], scale * p[1], scale * p[2]});
      t.push_back({-scale * p[1], scale * p[0], scale * p[2]});
    }
    const auto r = kabsch(s, t);
    rotation_checks(r);
    EXPECT_LT(r.rmsd, scale * 1e-12);
  }
}
TEST(KabschAnalysis, InputGuardsCompleteQuotaAndOwnedData) {
  EXPECT_THROW((void)kabsch({}, {}), std::invalid_argument);
  EXPECT_THROW((void)kabsch({{0, 0, 0}}, {}), std::invalid_argument);
  EXPECT_THROW(
      (void)kabsch({{0, 0, 0}}, {{std::numeric_limits<double>::infinity(), 0, 0}}),
      std::invalid_argument);
  const std::vector<Point3> s = {{0, 0, 0}, {1, 0, 0}};
  EXPECT_THROW((void)kabsch_analyze(s, s, 1), std::length_error);
  auto a         = kabsch_analyze(s, s, 2);
  auto b         = a;
  a.source[0][0] = 99;
  EXPECT_EQ(b.source[0][0], 0);
  EXPECT_NO_THROW((void)kabsch_analyze(s, s, 0));
}
