#include <gtest/gtest.h>

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/image_explainers.hpp>

#include <cmath>
#include <stdexcept>

using datamunge::linalg::DenseMatrix;
using datamunge::stats::ImageExplanationOptions;
using datamunge::stats::LimeImageExplainer;
using datamunge::stats::ShapImageExplainer;

namespace {

constexpr std::size_t H = 12, W = 12;

// A synthetic "model" whose score is the mean intensity of the top-left 3x3 region -- i.e. it
// depends only on superpixel 0 of a 4x4 grid. The explainers should attribute everything there.
double top_left_mean(const DenseMatrix<double>& im) {
  double s = 0.0;
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) s += im(i, j);
  return s / 9.0;
}

DenseMatrix<double> ones() { return DenseMatrix<double>(H, W, 1.0); }

} // namespace

TEST(ShapImageExplainer, EfficiencyAxiomHoldsExactly) {
  const auto img = ones();
  ImageExplanationOptions opt;
  opt.patch_rows = 4;
  opt.patch_cols = 4;
  ShapImageExplainer shap(img, top_left_mean, opt);

  double sum = 0.0;
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t j = 0; j < 4; ++j) sum += shap.patch_shapley()(i, j);
  // Sum of Shapley values equals f(full) - f(empty), exactly.
  EXPECT_NEAR(sum, shap.full_value() - shap.base_value(), 1e-9);
  EXPECT_NEAR(shap.full_value(), 1.0, 1e-9);   // all-ones -> mean 1
  EXPECT_NEAR(shap.base_value(), 0.0, 1e-9);   // top-left masked to baseline 0 -> mean 0
}

TEST(ShapImageExplainer, AttributesToTheRelevantSuperpixel) {
  const auto img = ones();
  ImageExplanationOptions opt;
  opt.patch_rows = 4;
  opt.patch_cols = 4;
  ShapImageExplainer shap(img, top_left_mean, opt);
  // Superpixel (0,0) drives the model; it should carry essentially all the attribution.
  EXPECT_NEAR(shap.patch_shapley()(0, 0), 1.0, 1e-9);
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t j = 0; j < 4; ++j)
      if (!(i == 0 && j == 0)) EXPECT_NEAR(shap.patch_shapley()(i, j), 0.0, 1e-9);

  const auto px = shap.pixel_relevance();
  EXPECT_EQ(px.rows(), H);
  EXPECT_EQ(px.cols(), W);
  EXPECT_GT(px(0, 0), 0.5);   // top-left pixel gets superpixel 0's value
  EXPECT_NEAR(px(H - 1, W - 1), 0.0, 1e-9);
}

TEST(LimeImageExplainer, RecoversTheRelevantSuperpixel) {
  const auto img = ones();
  ImageExplanationOptions opt;
  opt.patch_rows = 4;
  opt.patch_cols = 4;
  opt.n_samples = 1000;
  opt.seed = 3;
  LimeImageExplainer lime(img, top_left_mean, opt);

  EXPECT_GT(lime.local_r_squared(), 0.9);  // the model is linear in the on/off features here
  const double w00 = lime.patch_weights()(0, 0);
  EXPECT_GT(w00, 0.5);
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t j = 0; j < 4; ++j)
      if (!(i == 0 && j == 0)) EXPECT_LT(std::abs(lime.patch_weights()(i, j)), 0.2 * w00);
}

TEST(LimeImageExplainer, DeterministicGivenSeed) {
  const auto img = ones();
  ImageExplanationOptions opt;
  opt.n_samples = 400;
  opt.seed = 7;
  LimeImageExplainer a(img, top_left_mean, opt);
  LimeImageExplainer b(img, top_left_mean, opt);
  for (std::size_t i = 0; i < opt.patch_rows; ++i)
    for (std::size_t j = 0; j < opt.patch_cols; ++j)
      EXPECT_NEAR(a.patch_weights()(i, j), b.patch_weights()(i, j), 1e-12);
}

TEST(ImageExplainers, RejectInvalidOptions) {
  const auto img = ones();
  ImageExplanationOptions zero_grid;
  zero_grid.patch_rows = 0;
  EXPECT_THROW(ShapImageExplainer(img, top_left_mean, zero_grid), std::invalid_argument);
  EXPECT_THROW(LimeImageExplainer(img, top_left_mean, zero_grid), std::invalid_argument);

  ImageExplanationOptions huge;
  huge.patch_rows = 6;
  huge.patch_cols = 6;  // 36 superpixels -> 2^36 coalitions, refused by exact SHAP
  EXPECT_THROW(ShapImageExplainer(img, top_left_mean, huge), std::invalid_argument);
}
