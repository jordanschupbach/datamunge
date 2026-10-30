#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/kernel_pca.hpp>
#include <limits>
using datamunge::dstruct::DataFrame;
using datamunge::stats::KernelPCA;
using datamunge::stats::KernelPCAOptions;
namespace {
DataFrame line(std::vector<double> x) {
  DataFrame d;
  d.add_column("x", x);
  return d;
}
KernelPCAOptions options(
    std::size_t d,
    std::string k      = "linear",
    double      g      = 1,
    double      degree = 3,
    double      offset = 1) {
  return {d, k, g, degree, offset};
}
} // namespace
TEST(KernelPcaDiagnostics, ThreePointLinearCenteringAndNormalizationAreExact) {
  const auto      d = line({0, 1, 2});
  const KernelPCA m(d, {"x"}, options(1));
  EXPECT_NEAR(m.eigenvalues()[0], 2, 1e-14);
  EXPECT_EQ(m.numerical_rank(), 1);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(m.kernel_matrix()(i, j), double(i * j));
      EXPECT_NEAR(m.centered_kernel_matrix()(i, j), (double(i) - 1) * (double(j) - 1), 1e-15);
      EXPECT_NEAR(
          m.embedding()(i, 0) * m.embedding()(j, 0), m.centered_kernel_matrix()(i, j), 1e-14);
    }
  EXPECT_NEAR(m.spectrum().double_at("sample_variance", 0), 1, 1e-14);
  EXPECT_NEAR(std::abs(m.transform(line({3}))(0, 0)), 2, 1e-14);
  EXPECT_NEAR(m.projection_details(line({3})).double_at("residual_energy", 0), 0, 1e-13);
}
TEST(KernelPcaDiagnostics, QuadraticKernelMatchesExplicitTwoFeatureCovariance) {
  const auto      d = line({-2, -1, 1, 2});
  const KernelPCA m(d, {"x"}, options(2, "polynomial", 1, 2, 1));
  EXPECT_NEAR(m.eigenvalues()[0], 20, 1e-12);
  EXPECT_NEAR(m.eigenvalues()[1], 9, 1e-12);
  const auto x = d.numeric_column("x").values();
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t j = 0; j < 4; ++j) {
      const double a = x[i] * x[i] - 2.5, b = x[j] * x[j] - 2.5;
      EXPECT_NEAR(m.centered_kernel_matrix()(i, j), 2 * x[i] * x[j] + a * b, 1e-13);
    }
  const auto q = m.projection_details(line({3}));
  EXPECT_NEAR(q.double_at("centered_norm_squared", 0), 60.25, 1e-12);
  EXPECT_NEAR(q.double_at("residual_energy", 0), 0, 1e-12);
}
TEST(KernelPcaDiagnostics, ThreePointRbfEigenvaluesHaveClosedForms) {
  const auto d = line({-1, 0, 1});
  for (double g : {.01, .3, 1., 10.}) {
    const KernelPCA m(d, {"x"}, options(2, "rbf", g));
    const double    a = std::exp(-g), b = std::exp(-4 * g);
    EXPECT_NEAR(m.eigenvalues()[0], 1 - b, 1e-14);
    EXPECT_NEAR(m.eigenvalues()[1], 1 - 4 * a / 3 + b / 3, 1e-14);
  }
}
TEST(KernelPcaDiagnostics, TinyGammaRetainsTheLinearLimitWhenRawKernelRoundsToOnes) {
  const auto      d = line({-1, 0, 1});
  const KernelPCA m(d, {"x"}, options(1, "rbf", 1e-100));
  EXPECT_DOUBLE_EQ(m.kernel_matrix()(0, 2), 1);
  EXPECT_NEAR(m.eigenvalues()[0] / 4e-100, 1, 1e-14);
  EXPECT_NEAR(m.centered_kernel_matrix()(0, 2) / (-2e-100), 1, 1e-14);
  EXPECT_NEAR(std::abs(m.transform(line({2}))(0, 0)) / std::sqrt(8e-100), 1, 1e-14);
}
TEST(KernelPcaDiagnostics, LinearSmallUnitsAndLargeTranslationsPreserveTheGeometry) {
  const KernelPCA tiny(line({-1e-100, 0, 1e-100}), {"x"}, options(1));
  EXPECT_NEAR(tiny.eigenvalues()[0] / 2e-200, 1, 1e-14);
  const KernelPCA shifted(line({1e12 - 1, 1e12, 1e12 + 1}), {"x"}, options(1));
  EXPECT_NEAR(shifted.eigenvalues()[0], 2, 1e-14);
  EXPECT_NEAR(std::abs(shifted.transform(line({1e12 + 2}))(0, 0)), 2, 1e-14);
}
TEST(KernelPcaDiagnostics, RbfUnitsAndGammaMustChangeTogether) {
  const KernelPCA a(line({-2, -1, 0, 1, 3}), {"x"}, options(2, "rbf", .2));
  const KernelPCA b(line({-20, -10, 0, 10, 30}), {"x"}, options(2, "rbf", .002));
  for (std::size_t i = 0; i < 5; ++i)
    for (std::size_t j = 0; j < 5; ++j)
      EXPECT_NEAR(a.centered_kernel_matrix()(i, j), b.centered_kernel_matrix()(i, j), 1e-15);
}
TEST(KernelPcaDiagnostics, NumericalRankRejectsNullAxesInsteadOfPromotingRoundoff) {
  const auto d = line({-2, -1, 0, 1, 2});
  EXPECT_THROW(KernelPCA(d, {"x"}, options(2)), std::invalid_argument);
  EXPECT_THROW(KernelPCA(line({1, 1, 1}), {"x"}, options(1, "rbf")), std::invalid_argument);
  const KernelPCA duplicate(line({0, 0, 1, 1}), {"x"}, options(1, "rbf"));
  EXPECT_EQ(duplicate.numerical_rank(), 1);
  EXPECT_NEAR(duplicate.dimension(0)[0], duplicate.dimension(0)[1], 1e-14);
}
TEST(KernelPcaDiagnostics, TrainingReprojectionAndFeatureEnergyMatchTheSpectrum) {
  const auto      d = datamunge::datasets::iris();
  const KernelPCA m(
      d, {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"}, options(2, "rbf", .13));
  const auto t = m.transform(d);
  const auto details = m.projection_details(d);
  double     residual = 0;
  for (std::size_t i = 0; i < d.nrows(); ++i) {
    for (std::size_t c = 0; c < 2; ++c)
      EXPECT_NEAR(t(i, c), m.embedding()(i, c), 1e-13);
    residual += details.double_at("residual_energy", i);
  }
  EXPECT_NEAR(residual / d.nrows(), m.diagnostics().double_at("mean_residual_energy", 0), 1e-12);
  EXPECT_LT(m.diagnostics().double_at("relative_eigen_residual", 0), 1e-13);
  EXPECT_LT(m.diagnostics().double_at("max_centered_row_mean", 0), 1e-14);
}
TEST(KernelPcaDiagnostics, QueryBatchNeverChangesItsTrainingCentre) {
  const KernelPCA m(line({0, 1, 2, 3}), {"x"}, options(2, "rbf", .4));
  const auto      single = m.transform(line({10})), batch = m.transform(line({10, -100, 0}));
  for (std::size_t c = 0; c < 2; ++c)
    EXPECT_DOUBLE_EQ(single(0, c), batch(0, c));
  EXPECT_GT(std::abs(batch(0, 0) + batch(1, 0) + batch(2, 0)), 1e-3);
}
TEST(KernelPcaDiagnostics, JointTrainingOmissionAndQueryRowsRemainAligned) {
  DataFrame d;
  d.add_column("x", std::vector<std::optional<double>>{0, std::nullopt, 1, 2});
  const KernelPCA m(d, {"x"}, options(1));
  EXPECT_EQ(m.kept_row_indices(), (std::vector<std::size_t>{0, 2, 3}));
  DataFrame q;
  q.add_column(
      "x",
      std::vector<std::optional<double>>{3, std::nullopt, std::numeric_limits<double>::infinity()});
  const auto t = m.transform(q);
  const auto details = m.projection_details(q);
  EXPECT_EQ(t.rows(), 3);
  EXPECT_TRUE(std::isfinite(t(0, 0)));
  EXPECT_TRUE(std::isnan(t(1, 0)));
  EXPECT_TRUE(std::isnan(t(2, 0)));
  EXPECT_EQ(details.string_at("status", 1), "missing_predictor");
  EXPECT_EQ(details.string_at("status", 2), "nonfinite_predictor");
  EXPECT_EQ(m.transform(DataFrame{}).rows(), 0);
  EXPECT_EQ(m.projection_details(DataFrame{}).nrows(), 0);
}
TEST(KernelPcaDiagnostics, FinitePsdKernelControlsAreEnforced) {
  const auto d = line({-2, -1, 0, 1, 2});
  for (double degree : {0., -1., .5, std::numeric_limits<double>::infinity()})
    EXPECT_THROW(KernelPCA(d, {"x"}, options(1, "polynomial", 1, degree)), std::invalid_argument);
  EXPECT_THROW(KernelPCA(d, {"x"}, options(1, "polynomial", 1, 2, -1)), std::invalid_argument);
  EXPECT_THROW(
      KernelPCA(d, {"x"}, options(1, "rbf", std::numeric_limits<double>::quiet_NaN())),
      std::invalid_argument);
  EXPECT_THROW(
      KernelPCA(line({0, 1, std::numeric_limits<double>::infinity()}), {"x"}, options(1)),
      std::invalid_argument);
  EXPECT_THROW(KernelPCA(d, {"x", "x"}, options(1)), std::invalid_argument);
  EXPECT_THROW(KernelPCA(line({-1e200, 0, 1e200}), {"x"}, options(1)), std::invalid_argument);
  const KernelPCA m(d, {"x"}, options(1));
  DataFrame       wrong;
  wrong.add_column("x", std::vector<std::string>{"a"});
  EXPECT_THROW((void)m.transform(wrong), std::invalid_argument);
}
