#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/stats/pca.hpp>
#include <limits>
using datamunge::dstruct::DataFrame;
using datamunge::stats::PCA;
using datamunge::stats::PCAOptions;
namespace {
DataFrame four(double factor = 1) {
  DataFrame d;
  d.add_column("x", std::vector<double>{-3, -1, 1, 3});
  d.add_column("y", std::vector<double>{-factor, -3 * factor, 3 * factor, factor});
  return d;
}
} // namespace
TEST(PcaDiagnostics, FourPointCovarianceAndResidualTailAgree) {
  const auto d = four();
  const PCA  p(d, {"x", "y"}, PCAOptions{true, false});
  EXPECT_NEAR(p.covariance_matrix()(0, 0), 20. / 3, 1e-14);
  EXPECT_NEAR(p.covariance_matrix()(0, 1), 4, 1e-14);
  EXPECT_NEAR(p.explained_variance()[0], 32. / 3, 1e-13);
  EXPECT_NEAR(p.explained_variance()[1], 8. / 3, 1e-13);
  const auto curve = p.reconstruction_curve(d);
  EXPECT_NEAR(curve.double_at("preprocessed_sse", 0), 40, 1e-13);
  EXPECT_NEAR(curve.double_at("preprocessed_sse", 1), 8, 1e-13);
  EXPECT_NEAR(curve.double_at("preprocessed_rmse", 1), 1, 1e-14);
  EXPECT_LT(curve.double_at("preprocessed_sse", 2), 1e-26);
  EXPECT_NEAR(p.spectrum().double_at("tail_sse", 0), 8, 1e-13);
  const auto rebuilt = p.reconstruct(d, 1);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(rebuilt(i, 0), i < 2 ? -2 : 2, 1e-14);
    EXPECT_NEAR(rebuilt(i, 1), i < 2 ? -2 : 2, 1e-14);
  }
  EXPECT_EQ(p.loadings_frame().nrows(), 2);
}
TEST(PcaDiagnostics, FrozenPreprocessingWorksForSingletonAndBatchQueries) {
  const auto d = four(10);
  const PCA  p(d, {"x", "y"});
  DataFrame  one;
  one.add_column("x", std::vector<double>{p.scales()[0]});
  one.add_column("y", std::vector<double>{0});
  DataFrame batch;
  batch.add_column("x", std::vector<double>{p.scales()[0], 100});
  batch.add_column("y", std::vector<double>{0, -100});
  const auto t = p.transform(one), both = p.transform(batch);
  for (std::size_t c = 0; c < 2; ++c)
    EXPECT_DOUBLE_EQ(t(0, c), both(0, c));
  const auto curve = p.reconstruction_curve(one);
  EXPECT_GT(curve.double_at("original_rmse", 1), curve.double_at("original_rmse", 0));
  EXPECT_LT(curve.double_at("preprocessed_rmse", 1), curve.double_at("preprocessed_rmse", 0));
  EXPECT_NEAR(p.reconstruct(one, 2)(0, 0), p.scales()[0], 1e-14);
  EXPECT_NEAR(p.reconstruct(one, 2)(0, 1), 0, 1e-13);
}
TEST(PcaDiagnostics, StandardizationIsInvariantToPositiveFeatureUnitChanges) {
  const PCA a(four(), {"x", "y"}), b(four(100), {"x", "y"});
  EXPECT_NEAR(b.scales()[1] / a.scales()[1], 100, 1e-13);
  for (std::size_t j = 0; j < 2; ++j)
    for (std::size_t k = 0; k < 2; ++k)
      EXPECT_NEAR(a.covariance_matrix()(j, k), b.covariance_matrix()(j, k), 1e-14);
  for (std::size_t k = 0; k < 2; ++k)
    EXPECT_NEAR(a.explained_variance()[k], b.explained_variance()[k], 1e-14);
}
TEST(PcaDiagnostics, ZeroEnergyAndWideDataKeepFullBasisButReportRank) {
  DataFrame zero;
  zero.add_column("x", std::vector<double>{7, 7, 7});
  const PCA a(zero, {"x"});
  EXPECT_EQ(a.numerical_rank(), 0);
  EXPECT_EQ(a.num_components(), 1);
  EXPECT_EQ(a.scales()[0], 1);
  EXPECT_EQ(a.explained_variance_ratio()[0], 0);
  EXPECT_EQ(a.cumulative_explained_variance_ratio()[0], 0);
  EXPECT_EQ(a.reconstruct(zero, 0)(0, 0), 7);
  DataFrame wide;
  wide.add_column("x", std::vector<double>{0, 1});
  wide.add_column("y", std::vector<double>{0, 2});
  wide.add_column("z", std::vector<double>{5, 5});
  const PCA b(wide, {"x", "y", "z"});
  EXPECT_EQ(b.num_components(), 3);
  EXPECT_EQ(b.numerical_rank(), 1);
  EXPECT_EQ(b.scores().cols(), 3);
  EXPECT_EQ(b.scales()[2], 1);
  const auto restored = b.reconstruct(wide, 3);
  for (int i = 0; i < 2; ++i)
    EXPECT_NEAR(restored(i, 2), 5, 1e-14);
}
TEST(PcaDiagnostics, UncenteredScalingUsesRmsAboutZero) {
  DataFrame d;
  d.add_column("x", std::vector<double>{1, 2, 3});
  const PCA p(d, {"x"}, PCAOptions{false, true});
  EXPECT_EQ(p.means()[0], 0);
  EXPECT_NEAR(p.scales()[0], std::sqrt(7.), 1e-14);
  EXPECT_NEAR(p.explained_variance()[0], 1, 1e-14);
  EXPECT_NE(p.summary().find("Uncentered second moment"), std::string::npos);
}
TEST(PcaDiagnostics, NormalizedGramHandlesTinyAndLargeRepresentableCovariance) {
  for (double a : {1e-100, 1e100}) {
    auto d = four();
    d = d.mutate_with("x", [&](const DataFrame::Row& r) { return r.get_double("x") * a; });
    d = d.mutate_with("y", [&](const DataFrame::Row& r) { return r.get_double("y") * a; });
    const PCA p(d, {"x", "y"}, PCAOptions{true, false});
    EXPECT_NEAR(p.explained_variance()[0] / (a * a), 32. / 3, 1e-12);
    EXPECT_EQ(p.numerical_rank(), 2);
    EXPECT_NEAR(p.explained_variance_ratio()[0], .8, 1e-13);
  }
}
TEST(PcaDiagnostics, SchemaMissingValuesAndNumericOverflowAreRejected) {
  const auto d = four();
  EXPECT_THROW(PCA(d, {"x", "x"}), std::invalid_argument);
  DataFrame bad;
  bad.add_column("x", std::vector<double>{0, std::numeric_limits<double>::infinity()});
  EXPECT_THROW(PCA(bad, {"x"}), std::invalid_argument);
  const PCA p(d, {"x", "y"});
  EXPECT_THROW((void)p.reconstruct(d, 3), std::invalid_argument);
  EXPECT_THROW((void)p.transform(bad), std::invalid_argument);
  DataFrame missing;
  missing.add_column("x", std::vector<std::optional<double>>{0, std::nullopt});
  missing.add_column("y", std::vector<double>{0, 1});
  EXPECT_THROW((void)p.transform(missing), std::invalid_argument);
  DataFrame strings;
  strings.add_column("x", std::vector<std::string>{"x"});
  strings.add_column("y", std::vector<double>{0});
  EXPECT_THROW((void)p.transform(strings), std::invalid_argument);
  DataFrame huge;
  huge.add_column("x", std::vector<double>{-1e308, 1e308});
  EXPECT_THROW(PCA(huge, {"x"}, PCAOptions{true, false}), std::invalid_argument);
  DataFrame empty;
  empty.add_column("x", std::vector<double>{});
  empty.add_column("y", std::vector<double>{});
  EXPECT_EQ(p.transform(empty).rows(), 0);
  EXPECT_EQ(p.reconstruct(empty, 1).rows(), 0);
  EXPECT_THROW((void)p.reconstruction_curve(empty), std::invalid_argument);
}
