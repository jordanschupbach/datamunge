#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cmath>
#include <numbers>

namespace nn = datamunge::nn;
using nn::Activation;

TEST(NnActivations, HardVariantsAndLogSigmoid) {
  const auto x = nn::tensor({-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0}, {7}, nn::DType::Float64);
  EXPECT_EQ(nn::activate(x, Activation::HardSigmoid).to_vector<double>(), (std::vector<double>{0, 0, 1.0 / 3, 0.5, 2.0 / 3, 1, 1}));
  const auto hs = nn::activate(x, Activation::HardSwish).to_vector<double>();
  EXPECT_DOUBLE_EQ(hs[2], -1.0 / 3);
  EXPECT_DOUBLE_EQ(hs[5], 3.0);
  EXPECT_EQ(nn::activate(x, Activation::HardTanh).to_vector<double>(), (std::vector<double>{-1, -1, -1, 0, 1, 1, 1}));
  const auto ls = nn::activate(x, Activation::LogSigmoid).to_vector<double>();
  EXPECT_NEAR(ls[3], -std::log(2.0), 1e-15);
  EXPECT_EQ(nn::activation_name(Activation::HardSwish), "hard_swish");
}

TEST(NnActivations, GluHalvesAndGates) {
  const auto x = nn::tensor({{1.0, 2.0, 0.0, 100.0}, {3.0, -1.0, -100.0, 0.0}}, nn::DType::Float64);
  const auto g = nn::glu(x).to_vector<double>();  // a * sigmoid(b)
  EXPECT_NEAR(g[0], 0.5, 1e-12);
  EXPECT_NEAR(g[1], 2.0, 1e-12);
  EXPECT_NEAR(g[2], 0.0, 1e-12);
  EXPECT_NEAR(g[3], -0.5, 1e-12);
  EXPECT_EQ(nn::glu(x, Activation::Relu, 0).shape(), (nn::Shape{1, 4}));
  EXPECT_THROW((void)nn::glu(nn::ones({2, 3})), std::invalid_argument);
  nn::Generator gen(3);
  auto z = nn::normal({3, 6}, 0, 1, gen, nn::DType::Float64).requires_grad_(true);
  EXPECT_TRUE(nn::gradcheck([](const std::vector<nn::Tensor>& in) { return nn::glu(in[0], Activation::Silu); }, {z}).ok);
}

TEST(NnActivations, GaussianMoments) {
  // ReLU: E f = 1/sqrt(2 pi), E f^2 = 1/2, E f' = E f'^2 = 1/2.
  const auto r = nn::activation_moments(Activation::Relu);
  EXPECT_NEAR(r.mean, 1 / std::sqrt(2 * std::numbers::pi), 1e-7);
  EXPECT_NEAR(r.second_moment, 0.5, 1e-7);
  EXPECT_NEAR(r.derivative_mean, 0.5, 1e-12);
  EXPECT_NEAR(r.derivative_second_moment, 0.5, 1e-12);
  // The kinks never bias f' moments, whatever q: ReLU keeps E f' = 1/2, and hard_tanh has E f' = P(|Z| < 1).
  for (double q : {1e-6, 0.37, 7.3}) {
    EXPECT_NEAR(nn::activation_moments(Activation::Relu, q).derivative_mean, 0.5, 1e-9) << q;
    const double inside = std::erf(1 / std::sqrt(2 * q));
    EXPECT_NEAR(nn::activation_moments(Activation::HardTanh, q).derivative_mean, inside, 1e-7) << q;
  }
  // Identity under N(0, q): E f^2 = q, E f'^2 = 1.
  const auto id = nn::activation_moments(Activation::None, 2.5);
  EXPECT_NEAR(id.second_moment, 2.5, 1e-9);
  EXPECT_NEAR(id.derivative_second_moment, 1.0, 1e-9);
  // SELU's constants make N(0, 1) a fixed point: mean 0, variance 1.
  const auto s = nn::activation_moments(Activation::Selu);
  EXPECT_NEAR(s.mean, 0.0, 1e-6);
  EXPECT_NEAR(s.variance, 1.0, 1e-6);
  EXPECT_THROW((void)nn::activation_moments(Activation::Relu, 0.0), std::invalid_argument);
}
