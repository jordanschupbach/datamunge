/// @file nn_pooling_variants_tests.cpp
/// @brief Tests for power-mean, GeM, log-sum-exp, adaptive and blur pooling.

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cmath>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {
const auto F64 = nn::DType::Float64;

double max_abs_diff(const Tensor& a, const Tensor& b) { return nn::max(nn::abs(a - b)).item(); }
}  // namespace

TEST(NnPoolingVariants, PowerMeanAndGem) {
  nn::Generator g(1);
  const Tensor x = nn::normal({2, 6, 6, 3}, 0.0, 1.0, g, F64);
  EXPECT_LT(max_abs_diff(nn::power_mean_pool2d(x, 1.0, 2, 2), nn::avg_pool2d(nn::abs(x), 2, 2)), 1e-14);
  // Hand value of one window with p = 3.
  double s = 0;
  for (std::int64_t i = 2; i < 4; ++i)
    for (std::int64_t j = 0; j < 2; ++j) s += std::pow(std::abs(x.at({1, i, j, 2})), 3) / 4;
  EXPECT_NEAR(nn::power_mean_pool2d(x, 3.0, 2, 2).at({1, 1, 0, 2}), std::cbrt(s), 1e-13);
  // Sandwich max |x| * n^(-1/p) <= M_p <= max |x| for n = 4 entries per window.
  const Tensor mp = nn::power_mean_pool2d(x, 200.0, 2, 2), mx_abs = nn::max_pool2d(nn::abs(x), 2, 2);
  EXPECT_LE(nn::max(mp - mx_abs).item(), 1e-12);
  EXPECT_GE(nn::min(mp - mx_abs * std::pow(4.0, -1.0 / 200)).item(), -1e-12);
  const Tensor pos = nn::abs(x) + 0.1;
  EXPECT_LT(max_abs_diff(nn::gem_pool2d(pos, 1.0), nn::global_avg_pool2d(pos)), 1e-14);
  const Tensor gm = nn::gem_pool2d(pos, 300.0), gmax = nn::global_max_pool2d(pos);
  EXPECT_LE(nn::max(gm - gmax).item(), 1e-12);
  EXPECT_GE(nn::min(gm - gmax * std::pow(36.0, -1.0 / 300)).item(), -1e-12);
  // Learnable exponent: gradient against a central difference.
  Tensor p = nn::scalar(2.5, F64);
  p.requires_grad_();
  const Tensor gp = nn::grad(nn::sum(nn::gem_pool2d(pos, p)), {p})[0];
  const double h = 1e-6;
  const double fd = (nn::sum(nn::gem_pool2d(pos, 2.5 + h)).item() - nn::sum(nn::gem_pool2d(pos, 2.5 - h)).item()) / (2 * h);
  EXPECT_NEAR(gp.item(), fd, 1e-6);
  // Zero windows (as after ReLU) give 0 and finite gradients, for p above and below 1.
  Tensor rx = nn::relu(x - 0.8);
  rx.requires_grad_();
  for (double pp : {0.5, 3.0}) {
    const Tensor gr = nn::grad(nn::sum(nn::power_mean_pool2d(rx, pp, 2, 2)), {rx})[0];
    EXPECT_TRUE(std::isfinite(nn::sum(nn::abs(gr)).item())) << "p = " << pp;
  }
  EXPECT_DOUBLE_EQ(nn::power_mean_pool2d(nn::zeros({1, 2, 2, 1}, F64), 3.0, 2, 2).item(), 0.0);
  EXPECT_THROW((void)nn::power_mean_pool2d(x, 0.0, 2, 2), std::invalid_argument);
  EXPECT_THROW((void)nn::gem_pool2d(nn::zeros({3, 3}, F64), 2.0), std::invalid_argument);
}

TEST(NnPoolingVariants, LogSumExpBetweenMeanAndMax) {
  nn::Generator g(2);
  const Tensor x = nn::normal({1, 4, 4, 2}, 0.0, 2.0, g, F64);
  const Tensor avg = nn::avg_pool2d(x, 2, 2), mx = nn::max_pool2d(x, 2, 2);
  const Tensor l = nn::logsumexp_pool2d(x, 1.0, 2, 2);
  EXPECT_GE(nn::min(l - avg).item(), 0.0);
  EXPECT_LE(nn::max(l - mx).item(), 0.0);
  EXPECT_LT(max_abs_diff(nn::logsumexp_pool2d(x, 1e-6, 2, 2), avg), 1e-5);
  EXPECT_LT(max_abs_diff(nn::logsumexp_pool2d(x, 1e4, 2, 2), mx), 1e-3);
  // SAME padding on an odd size: the corner window holds one real entry, so the result is that entry.
  const Tensor odd = nn::normal({1, 5, 5, 1}, 0.0, 1.0, g, F64);
  EXPECT_NEAR(nn::logsumexp_pool2d(odd, 2.0, 2, 2, nn::Padding::Same).at({0, 2, 2, 0}), odd.at({0, 4, 4, 0}), 1e-14);
  EXPECT_LT(max_abs_diff(nn::logsumexp_pool2d(odd, 1e-7, 3, 2, nn::Padding::Same), nn::avg_pool2d(odd, 3, 2, nn::Padding::Same)), 1e-6);
  // Gradient of one output is the softmax of beta x over its window.
  Tensor xv = x.clone();
  xv.requires_grad_();
  const Tensor gx = nn::grad(nn::sum(nn::slice(nn::logsumexp_pool2d(xv, 0.7, 2, 2), {0, 0, 1, 0}, {1, 1, 1, 1})), {xv})[0];
  double z = 0;
  for (std::int64_t i = 0; i < 2; ++i)
    for (std::int64_t j = 2; j < 4; ++j) z += std::exp(0.7 * x.at({0, i, j, 0}));
  EXPECT_NEAR(gx.at({0, 1, 3, 0}), std::exp(0.7 * x.at({0, 1, 3, 0})) / z, 1e-13);
  EXPECT_THROW((void)nn::logsumexp_pool2d(x, 0.0, 2, 2), std::invalid_argument);
}

TEST(NnPoolingVariants, AdaptiveAndBlur) {
  nn::Generator g(3);
  const Tensor x = nn::normal({2, 6, 8, 3}, 0.0, 1.0, g, F64);
  EXPECT_LT(max_abs_diff(nn::adaptive_avg_pool2d(x, 3, 4), nn::avg_pool2d(x, 2, 2)), 1e-14);
  EXPECT_LT(max_abs_diff(nn::adaptive_max_pool2d(x, 3, 4), nn::max_pool2d(x, 2, 2)), 1e-14);
  EXPECT_LT(max_abs_diff(nn::adaptive_avg_pool2d(x, 1, 1), nn::global_avg_pool2d(x, true)), 1e-14);
  // 5 rows into 3 cells: rows [0, 2), [1, 4), [3, 5).
  const Tensor r = nn::reshape(nn::arange(0, 5, 1, F64), {1, 5, 1, 1});
  const Tensor a = nn::adaptive_avg_pool2d(r, 3, 1), m = nn::adaptive_max_pool2d(r, 3, 1);
  EXPECT_DOUBLE_EQ(a.at({0, 0, 0, 0}), 0.5);
  EXPECT_DOUBLE_EQ(a.at({0, 1, 0, 0}), 2.0);
  EXPECT_DOUBLE_EQ(a.at({0, 2, 0, 0}), 3.5);
  EXPECT_DOUBLE_EQ(m.at({0, 1, 0, 0}), 3.0);
  EXPECT_THROW((void)nn::adaptive_avg_pool2d(x, 7, 2), std::invalid_argument);
  // Blur pool: output size, constants preserved, filter size 1 is plain subsampling, and an interior hand value.
  const Tensor b = nn::blur_pool2d(x, 2, 3);
  EXPECT_EQ(b.size(1), 3);
  EXPECT_EQ(b.size(2), 4);
  EXPECT_LT(max_abs_diff(nn::blur_pool2d(nn::full_like(x, 2.5), 2, 5), nn::full({2, 3, 4, 3}, 2.5, F64)), 1e-14);
  const Tensor odd = nn::normal({1, 7, 7, 1}, 0.0, 1.0, g, F64);
  const Tensor sub = nn::blur_pool2d(odd, 2, 1);
  EXPECT_DOUBLE_EQ(sub.at({0, 1, 2, 0}), odd.at({0, 2, 4, 0}));
  const Tensor b3 = nn::blur_pool2d(odd, 2, 3);
  const double w[3] = {0.25, 0.5, 0.25};
  double hand = 0;
  for (int u = 0; u < 3; ++u)
    for (int v = 0; v < 3; ++v) hand += w[u] * w[v] * odd.at({0, 1 + u, 3 + v, 0});
  EXPECT_NEAR(b3.at({0, 1, 2, 0}), hand, 1e-14);
  EXPECT_THROW((void)nn::blur_pool2d(x, 2, 8), std::invalid_argument);
}
