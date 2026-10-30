/// @file nn_weight_reparam_tests.cpp
/// @brief Tests for weight normalisation, power-iteration spectral normalisation and lipschitz_bound().

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {
const auto F64 = nn::DType::Float64;

double max_abs_diff(const Tensor& a, const Tensor& b) { return nn::max(nn::abs(a - b)).item(); }
}  // namespace

TEST(NnWeightReparam, WeightNormColumnsAndGradients) {
  nn::Generator g(1);
  const Tensor x = nn::normal({4, 5}, 0.0, 1.0, g, F64), r = nn::normal({4, 3}, 0.0, 1.0, g, F64);
  nn::ParameterStore p(1, F64);
  const auto wn = nn::weight_norm_dense(p, "d", x, 3);
  // At creation g equals the column norms of v, so W = v.
  EXPECT_LT(max_abs_diff(wn.W, p.at("d/v")), 1e-15);
  EXPECT_FALSE(wn.sigma.defined());
  {  // ||W_j|| = g_j for any gains.
    nn::NoGradGuard ng;
    p.set("d/g", nn::from_vector(std::vector<double>{0.5, -2.0, 3.0}, {3}));
  }
  const auto wn2 = nn::weight_norm_dense(p, "d", x, 3);
  EXPECT_LT(max_abs_diff(nn::sqrt(nn::sum(nn::square(wn2.W), {0})), nn::abs(p.at("d/g"))), 1e-14);
  // Closed-form gradients, and dL/dv orthogonal to v.
  const Tensor v = p.at("d/v"), gg = p.at("d/g");
  const auto grads = nn::grad(nn::sum(wn2.y * r), {v, gg, wn2.W});
  nn::NoGradGuard ng;
  const Tensor norm = nn::sqrt(nn::sum(nn::square(v), {0}));
  const Tensor dg = nn::sum(grads[2] * v, {0}) / norm;
  EXPECT_LT(max_abs_diff(grads[1], dg), 1e-13);
  const Tensor dv = gg / norm * (grads[2] - dg * v / norm);
  EXPECT_LT(max_abs_diff(grads[0], dv), 1e-13);
  EXPECT_LT(nn::max(nn::abs(nn::sum(grads[0] * v, {0}))).item(), 1e-13);
  EXPECT_THROW((void)nn::weight_normalize(v, nn::ones({2}, F64)), std::invalid_argument);
}

TEST(NnWeightReparam, PowerIterationMatchesClosedFormOnDiagonal) {
  // W = diag(3, 2, 1) from u0 = (1, 1, 1): u_k ~ s^{2k}, v_k ~ s^{2k-1}, sigma_k = sum s^{4k} / (||s^{2k-1}|| ||s^{2k}||).
  const Tensor W = nn::from_vector(std::vector<double>{3, 0, 0, 0, 2, 0, 0, 0, 1}, {3, 3});
  const Tensor u0 = nn::ones({1, 3}, F64);
  const std::vector<double> s{3, 2, 1};
  double prev = 0;
  for (int k = 1; k <= 6; ++k) {
    double num = 0, a = 0, b = 0;
    for (double si : s) num += std::pow(si, 4 * k), a += std::pow(si, 4 * k - 2), b += std::pow(si, 4 * k);
    const double expected = num / std::sqrt(a * b);
    const double got = nn::spectral_norm_power_iteration(W, u0, k).sigma.item();
    EXPECT_NEAR(got, expected, 1e-11 * expected) << "k = " << k;  // the 1e-12 guard in the normalisation shifts it ~1e-13
    EXPECT_LE(got, 3.0 + 1e-14);
    EXPECT_GE(got, prev - 1e-14);
    prev = got;
  }
  EXPECT_THROW((void)nn::spectral_norm_power_iteration(W, nn::ones({1, 2}, F64)), std::invalid_argument);
  EXPECT_THROW((void)nn::spectral_norm_power_iteration(W, u0, 0), std::invalid_argument);
}

TEST(NnWeightReparam, SpectralNormDenseAndLipschitzBound) {
  nn::Generator g(2);
  const Tensor x = nn::normal({200, 16}, 0.0, 1.0, g, F64);
  nn::ParameterStore p(3, F64), state(4, F64);
  const Tensor u_before = state.get("d/u", {1, 8}, nn::init::normal(1.0)).clone();
  (void)nn::spectral_norm_dense(p, state, "d", x, 8, 1);
  EXPECT_GT(max_abs_diff(state.at("d/u"), u_before), 1e-3);  // the refined vector was stored
  for (int i = 0; i < 50; ++i) (void)nn::spectral_norm_dense(p, state, "d", x, 8, 1);
  const auto sn = nn::spectral_norm_dense(p, state, "d", x, 8, 1, nn::Activation::Relu);
  const double s1 = nn::svd(p.at("d/W"), false).s.to_vector<double>()[0];
  EXPECT_NEAR(sn.sigma.item(), s1, 1e-10 * s1);
  EXPECT_NEAR(nn::lipschitz_bound({sn.W}), 1.0, 1e-10);
  // The layer (ReLU is 1-Lipschitz) shrinks every difference of inputs.
  nn::NoGradGuard ng;
  const Tensor x2 = x + nn::normal({200, 16}, 0.0, 0.1, g, F64);
  const auto sn2 = nn::spectral_norm_dense(p, state, "d", x2, 8, 1, nn::Activation::Relu, false);
  const Tensor ratio = nn::sqrt(nn::sum(nn::square(sn2.y - sn.y), {1})) / nn::sqrt(nn::sum(nn::square(x2 - x), {1}));
  EXPECT_LE(nn::max(ratio).item(), 1.0 + 1e-9);
  // lipschitz_bound is the product of spectral norms.
  const Tensor A = nn::normal({5, 4}, 0.0, 1.0, g, F64), B = nn::normal({4, 3}, 0.0, 1.0, g, F64);
  const double sa = nn::svd(A, false).s.to_vector<double>()[0], sb = nn::svd(B, false).s.to_vector<double>()[0];
  EXPECT_NEAR(nn::lipschitz_bound({A, B}), sa * sb, 1e-12 * sa * sb);
  EXPECT_DOUBLE_EQ(nn::lipschitz_bound({}), 1.0);
  EXPECT_THROW((void)nn::lipschitz_bound({nn::ones({3}, F64)}), std::invalid_argument);
}
