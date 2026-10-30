/// @file nn_normalization_variants_tests.cpp
/// @brief Tests for normalize(), the layer-norm and RMSNorm Jacobians, ScaleNorm, Dynamic Tanh and weight
///        standardisation.

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cmath>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {
const auto F64 = nn::DType::Float64;

double max_abs_diff(const Tensor& a, const Tensor& b) { return nn::max(nn::abs(a - b)).item(); }
}  // namespace

TEST(NnNormalizationVariants, NormalizeMatchesTheLayers) {
  nn::Generator g(3);
  nn::ParameterStore p(1, F64);
  const Tensor x = nn::normal({3, 7}, 1.5, 2.0, g, F64);
  EXPECT_LT(max_abs_diff(nn::normalize(x, {-1}), nn::layer_norm(p, "ln", x, 1e-5)), 1e-13);
  EXPECT_LT(max_abs_diff(nn::normalize(x, {1}, false, 1e-6), nn::rms_norm(p, "rms", x, 1e-6)), 1e-13);
  const Tensor img = nn::normal({2, 3, 3, 4}, 0.0, 1.0, g, F64);
  EXPECT_LT(max_abs_diff(nn::normalize(img, {1, 2}), nn::instance_norm(p, "in", img)), 1e-13);
  EXPECT_LT(max_abs_diff(nn::normalize(img, {1, 2, 3}), nn::group_norm(p, "gn", img, 1)), 1e-13);
  // Hand computation of one row.
  double m = 0, v = 0;
  for (std::int64_t j = 0; j < 7; ++j) m += x.at({1, j}) / 7;
  for (std::int64_t j = 0; j < 7; ++j) v += (x.at({1, j}) - m) * (x.at({1, j}) - m) / 7;
  EXPECT_NEAR(nn::normalize(x, {1}).at({1, 2}), (x.at({1, 2}) - m) / std::sqrt(v + 1e-5), 1e-13);
  EXPECT_THROW((void)nn::normalize(x, {}), std::invalid_argument);
  EXPECT_THROW((void)nn::normalize(x, {1}, true, -1.0), std::invalid_argument);
}

TEST(NnNormalizationVariants, JacobiansMatchAutodiff) {
  nn::Generator g(4);
  const Tensor x = nn::normal({6}, 0.7, 1.3, g, F64);
  const Tensor j_ln = nn::layer_norm_jacobian(x, 1e-3);
  const Tensor j_ad = nn::jacobian([](const Tensor& v) { return nn::normalize(v, {0}, true, 1e-3); }, x);
  EXPECT_LT(max_abs_diff(j_ln, j_ad), 1e-12);
  const Tensor j_rms = nn::rms_norm_jacobian(x, 1e-3);
  const Tensor j_rad = nn::jacobian([](const Tensor& v) { return nn::normalize(v, {0}, false, 1e-3); }, x);
  EXPECT_LT(max_abs_diff(j_rms, j_rad), 1e-12);
  // Null directions: J 1 = 0 for layer norm, and J x -> 0 as eps -> 0 for RMSNorm.
  EXPECT_LT(nn::max(nn::abs(nn::matmul(j_ln, nn::reshape(nn::ones({6}, F64), {6, 1})))).item(), 1e-14);
  EXPECT_LT(nn::max(nn::abs(nn::matmul(nn::rms_norm_jacobian(x, 0.0), nn::reshape(x, {6, 1})))).item(), 1e-14);
  EXPECT_THROW((void)nn::layer_norm_jacobian(nn::zeros({2, 2}, F64)), std::invalid_argument);
}

TEST(NnNormalizationVariants, ScaleNormDynamicTanhWeightStandardize) {
  nn::Generator g(5);
  nn::ParameterStore p(1, F64);
  const Tensor x = nn::normal({4, 9}, 0.0, 3.0, g, F64);
  // ScaleNorm with g = sqrt(d) is RMSNorm with eps / d.
  EXPECT_LT(max_abs_diff(nn::scale_norm(p, "sn", x, 9e-6), nn::normalize(x, {-1}, false, 1e-6)), 1e-13);
  EXPECT_DOUBLE_EQ(p.at("sn/g").at({0}), 3.0);
  const Tensor y = nn::dynamic_tanh(p, "dyt", x, 0.25);
  EXPECT_NEAR(y.at({2, 5}), std::tanh(0.25 * x.at({2, 5})), 1e-15);
  EXPECT_EQ(p.at("dyt/gamma").size(0), 9);
  const Tensor w = nn::normal({3, 3, 2, 5}, 0.4, 2.0, g, F64);
  const Tensor ws = nn::weight_standardize(w, 0.0);
  for (std::int64_t o = 0; o < 5; ++o) {
    double s = 0, ss = 0;
    for (std::int64_t i = 0; i < 18; ++i) {
      const double v = ws.at({i / 6, (i / 2) % 3, i % 2, o});
      s += v, ss += v * v;
    }
    EXPECT_NEAR(s / 18, 0.0, 1e-13);
    EXPECT_NEAR(ss / 18, 1.0, 1e-12);
  }
  EXPECT_THROW((void)nn::weight_standardize(nn::zeros({4}, F64)), std::invalid_argument);
  EXPECT_THROW((void)nn::scale_norm(p, "bad", nn::scalar(1.0, F64)), std::invalid_argument);
}
