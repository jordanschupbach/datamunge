/// @file nn_differential_privacy_tests.cpp
/// @brief Tests for per-example dense gradient norms, clipping factors and the DP-SGD Rényi accountant.

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cmath>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

/// Per-example gradient norms of a dense layer by one autograd pass per example.
std::vector<double> brute_force_norms(const Tensor& x, const Tensor& W, const Tensor& b, const Tensor& target) {
  std::vector<double> out;
  for (std::int64_t i = 0; i < x.size(0); ++i) {
    Tensor Wi = W.detach().clone(), bi = b.detach().clone();
    Wi.requires_grad_();
    bi.requires_grad_();
    auto row = [i](const Tensor& t) {
      nn::Shape start(t.shape().size(), 0), size = t.shape();
      start[0] = i, size[0] = 1;
      return nn::slice(t, start, size);
    };
    const Tensor xi = row(x), ti = row(target);
    const Tensor y = nn::matmul(xi, Wi) + bi;
    const auto   g = nn::grad(nn::sum(nn::square(nn::tanh(y) - ti)), {Wi, bi});
    out.push_back(std::sqrt(nn::sum(nn::square(g[0])).item() + nn::sum(nn::square(g[1])).item()));
  }
  return out;
}

}  // namespace

TEST(NnDifferentialPrivacy, PerExampleNormsMatchBruteForce) {
  nn::Generator g(3);
  const auto F64 = nn::DType::Float64;
  const Tensor W = nn::normal({5, 4}, 0, 0.5, g, F64), b = nn::normal({4}, 0, 0.5, g, F64);
  // Plain [N, in] inputs, sequences with T^2 <= in out (Gram route) and T^2 > in out (direct route).
  for (const nn::Shape& shape : {nn::Shape{6, 5}, nn::Shape{3, 2, 5}, nn::Shape{3, 7, 5}}) {
    const Tensor x = nn::normal(shape, 0, 1, g, F64);
    nn::Shape ts = shape;
    ts.back() = 4;
    const Tensor target = nn::normal(ts, 0, 1, g, F64);
    Tensor y = nn::matmul(x, W) + b;
    y.requires_grad_();
    const Tensor dy = nn::grad(nn::sum(nn::square(nn::tanh(y) - target)), {y})[0];
    const Tensor fast = nn::dense_per_example_grad_norms(x, dy);
    const auto slow = brute_force_norms(x, W, b, target);
    for (std::int64_t i = 0; i < x.size(0); ++i) EXPECT_NEAR(fast.at({i}), slow[static_cast<std::size_t>(i)], 1e-12) << "shape rank " << shape.size();
    // Without the bias term the norm can only shrink.
    const Tensor wonly = nn::dense_per_example_grad_norms(x, dy, false);
    for (std::int64_t i = 0; i < x.size(0); ++i) EXPECT_LE(wonly.at({i}), fast.at({i}));
  }
  EXPECT_THROW((void)nn::dense_per_example_grad_norms(nn::zeros({3}), nn::zeros({3})), std::invalid_argument);
  EXPECT_THROW((void)nn::dense_per_example_grad_norms(nn::zeros({3, 2}), nn::zeros({4, 2})), std::invalid_argument);
}

TEST(NnDifferentialPrivacy, ClipFactors) {
  const Tensor f = nn::clip_factors(nn::from_vector(std::vector<double>{0.0, 0.5, 1.0, 4.0}, {4}), 1.0);
  EXPECT_DOUBLE_EQ(f.at({0}), 1.0);
  EXPECT_DOUBLE_EQ(f.at({1}), 1.0);
  EXPECT_DOUBLE_EQ(f.at({2}), 1.0);
  EXPECT_DOUBLE_EQ(f.at({3}), 0.25);
  EXPECT_THROW((void)nn::clip_factors(f, 0.0), std::invalid_argument);
}

TEST(NnDifferentialPrivacy, RenyiAccountant) {
  // q = 1: the Gaussian mechanism, alpha / (2 sigma^2).
  EXPECT_NEAR(nn::rdp_subsampled_gaussian(1.0, 2.0, 7), 7.0 / 8.0, 1e-15);
  EXPECT_EQ(nn::rdp_subsampled_gaussian(0.0, 1.0, 5), 0.0);
  // Order 2 in closed form: log(1 + q^2 (e^{1/sigma^2} - 1)).
  const double q = 0.03, s = 0.9;
  EXPECT_NEAR(nn::rdp_subsampled_gaussian(q, s, 2), std::log1p(q * q * std::expm1(1 / (s * s))), 1e-14);
  // Numerical integration of the divergence at order 4.
  double integral = 0;
  const double h = 1e-3;
  for (double x = -20; x <= 21; x += h) {
    const double p0 = std::exp(-x * x / (2 * s * s)), p1 = std::exp(-(x - 1) * (x - 1) / (2 * s * s));
    integral += h * p0 * std::pow((1 - q) + q * p1 / p0, 4) / (s * std::sqrt(2 * M_PI));
  }
  EXPECT_NEAR(nn::rdp_subsampled_gaussian(q, s, 4), std::log(integral) / 3, 1e-9);
  // Composition: epsilon grows with steps and shrinks with noise.
  const auto e1 = nn::dp_sgd_epsilon(0.01, 1.0, 1000, 1e-5), e2 = nn::dp_sgd_epsilon(0.01, 1.0, 4000, 1e-5);
  const auto e3 = nn::dp_sgd_epsilon(0.01, 2.0, 1000, 1e-5);
  EXPECT_LT(e1.epsilon, e2.epsilon);
  EXPECT_LT(e3.epsilon, e1.epsilon);
  EXPECT_GE(e1.order, 2);
  EXPECT_NEAR(nn::dp_sgd_epsilon(0.01, 1.0, 0, 1e-5).epsilon, std::log(1e5) / 255, 1e-12);
  EXPECT_THROW((void)nn::rdp_subsampled_gaussian(1.5, 1.0, 2), std::invalid_argument);
  EXPECT_THROW((void)nn::rdp_subsampled_gaussian(0.5, 1.0, 1), std::invalid_argument);
  EXPECT_THROW((void)nn::dp_sgd_epsilon(0.5, 1.0, 10, 0.0), std::invalid_argument);
}
