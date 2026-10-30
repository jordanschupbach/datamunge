/// @file nn_dropout_variants_tests.cpp
/// @brief Tests for Gaussian, alpha, spatial and drop-path dropout.

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {
const auto F64 = nn::DType::Float64;
double mean_of(const Tensor& t) { return nn::mean(t).item(); }
double var_of(const Tensor& t) { return nn::mean(nn::square(t - nn::mean(t))).item(); }
}  // namespace

TEST(NnDropoutVariants, InferenceIsIdentity) {
  nn::Generator g(1);
  const Tensor x = nn::normal({4, 3, 5}, 0, 1, g, F64);
  for (auto f : {nn::gaussian_dropout, nn::alpha_dropout, nn::spatial_dropout, nn::drop_path}) {
    EXPECT_EQ(nn::max(nn::abs(f(x, 0.3, false, g) - x)).item(), 0.0);
    EXPECT_EQ(nn::max(nn::abs(f(x, 0.0, true, g) - x)).item(), 0.0);
    EXPECT_THROW((void)f(x, 1.0, true, g), std::invalid_argument);
  }
}

TEST(NnDropoutVariants, GaussianMoments) {
  nn::Generator g(2);
  const Tensor y = nn::gaussian_dropout(nn::ones({400000}, F64), 0.25, true, g);
  EXPECT_NEAR(mean_of(y), 1.0, 0.005);
  EXPECT_NEAR(var_of(y), 0.25 / 0.75, 0.005);
}

TEST(NnDropoutVariants, AlphaDropoutKeepsStandardMoments) {
  nn::Generator g(3);
  const Tensor x = nn::normal({400000}, 0, 1, g, F64);
  for (double p : {0.05, 0.2, 0.5}) {
    const Tensor y = nn::alpha_dropout(x, p, true, g);
    EXPECT_NEAR(mean_of(y), 0.0, 0.01) << p;
    EXPECT_NEAR(var_of(y), 1.0, 0.01) << p;
  }
  // Dropped units all take one value: a alpha' + b.
  const Tensor y = nn::alpha_dropout(nn::zeros({1000}, F64) + 7.0, 0.5, true, g);
  const double ap = -1.0507009873554805 * 1.6732632423543772, a = 1 / std::sqrt(0.5 * (1 + 0.5 * ap * ap)), b = -a * 0.5 * ap;
  int dropped = 0;
  for (std::int64_t i = 0; i < 1000; ++i) dropped += std::abs(y.at({i}) - (a * ap + b)) < 1e-12;
  EXPECT_GT(dropped, 400);
  EXPECT_LT(dropped, 600);
}

TEST(NnDropoutVariants, SpatialAndDropPathShareMasks) {
  nn::Generator g(4);
  const Tensor x = nn::ones({8, 5, 6, 3}, F64);
  const Tensor s = nn::spatial_dropout(x, 0.5, true, g);
  for (std::int64_t n = 0; n < 8; ++n)
    for (std::int64_t c = 0; c < 3; ++c) {
      const double v = s.at({n, 0, 0, c});
      EXPECT_TRUE(v == 0.0 || v == 2.0);
      for (std::int64_t i = 0; i < 5; ++i)
        for (std::int64_t j = 0; j < 6; ++j) EXPECT_EQ(s.at({n, i, j, c}), v);
    }
  const Tensor d = nn::drop_path(x, 0.25, true, g);
  for (std::int64_t n = 0; n < 8; ++n) {
    const double v = d.at({n, 0, 0, 0});
    EXPECT_TRUE(v == 0.0 || std::abs(v - 4.0 / 3.0) < 1e-15);
    EXPECT_EQ(nn::max(nn::abs(nn::slice(d, {n, 0, 0, 0}, {1, 5, 6, 3}) - v)).item(), 0.0);
  }
  EXPECT_THROW((void)nn::spatial_dropout(nn::ones({3}, F64), 0.5, true, g), std::invalid_argument);
}
