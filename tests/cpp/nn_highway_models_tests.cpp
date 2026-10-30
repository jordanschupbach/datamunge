#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnHighwayModels, InputDependentGateHasCompleteJacobianAndPointwiseEquivalent) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), q(17, type);
  const auto x = n::from_vector(std::vector<double>{.8, -.5}, {1, 2}).requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::HighwayDense(p, "h", x, -1);
    n::models::HighwayConv2D(q, "h", x.reshape({1, 1, 1, 2}), 1, -1);
  }
  const std::vector<std::pair<std::string, std::vector<double>>> values{
      {"transform/W", {.7, -.2, .3, .8}},
      {"transform/b", {.2, .1}},
      {"gate/W", {.4, -.1, -.2, .3}},
      {"gate/b", {-1, -1}}};
  for (const auto &[part, value] : values) {
    p.set("h/" + part, n::from_vector(value, p.at("h/" + part).shape()));
    q.set("h/" + part, n::from_vector(value, q.at("h/" + part).shape()));
  }
  const auto net = n::models::HighwayDense(p, "h", x, -1);
  const auto jac = n::jacobian(
                       [&](const n::Tensor &z) {
                         return n::models::HighwayDense(p, "h", z, -1).output;
                       },
                       x)
                       .reshape({2, 2});
  const auto h = net.transformed.to_vector<double>(), t = net.gate.to_vector<double>(),
             input = x.to_vector<double>();
  for (int j = 0; j < 2; ++j)
    for (int k = 0; k < 2; ++k) {
      const double expected =
          (j == k ? 1 - t[j] : 0) + t[j] * (h[j] > 0 ? values[0].second[2 * k + j] : 0) +
          (h[j] - input[j]) * t[j] * (1 - t[j]) * values[2].second[2 * k + j];
      EXPECT_NEAR(jac.at({j, k}), expected, 1e-12);
    }
  const auto conv = n::models::HighwayConv2D(q, "h", x.reshape({1, 1, 1, 2}), 1, -1);
  EXPECT_LT(n::max(n::abs(conv.output.reshape({1, 2}) - net.output)).item(), 1e-12);
  EXPECT_LT(n::max(n::abs(n::models::HighwayDense(p, "h", x.reshape({1, 1, 1, 2}), -1).output -
                          conv.output))
                .item(),
            1e-12);
  EXPECT_EQ(
      n::max(n::abs(n::models::HighwayMix(x, net.transformed, n::scalar(0, type)) - x)).item(),
      0);
  EXPECT_EQ(n::max(n::abs(n::models::HighwayMix(x, net.transformed, n::scalar(1, type)) -
                          net.transformed))
                .item(),
            0);
  for (const auto &g : n::grad(n::sum(net.output), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
}
TEST(NnHighwayModels, ConstantScalarGatesMatchHundredLayerAttenuation) {
  const auto type = n::DType::Float64;
  const auto x = n::ones({1, 1}, type).requires_grad_();
  auto plain = x;
  for (int i = 0; i < 100; ++i)
    plain = .5 * plain;
  EXPECT_NEAR(plain.item() / std::pow(.5, 100), 1, 1e-12);
  EXPECT_NEAR(n::grad(plain, {x})[0].item() / std::pow(.5, 100), 1, 1e-12);
  for (double bias : {0., -3.}) {
    n::ParameterStore p(17, type);
    auto h = x;
    for (int i = 0; i < 100; ++i) {
      const auto name = "b" + std::to_string(i);
      p.set(name + "/transform/W", n::full({1, 1}, .5, type));
      p.set(name + "/transform/b", n::zeros({1}, type));
      p.set(name + "/gate/W", n::zeros({1, 1}, type));
      p.set(name + "/gate/b", n::full({1}, bias, type));
      h = n::models::HighwayDense(p, name, h, bias).output;
    }
    const double expected = std::pow(1 - .5 / (1 + std::exp(-bias)), 100);
    EXPECT_NEAR(h.item() / expected, 1, 1e-12);
    EXPECT_NEAR(n::grad(h, {x})[0].item() / expected, 1, 1e-12);
  }
}
TEST(NnHighwayModels, CompleteClassifierCountGradientsAndGateBiasReuse) {
  n::ParameterStore p(17);
  const auto out = n::models::HighwayClassifier(p, n::ones({2, 784}));
  EXPECT_EQ(p.count(), 136660);
  EXPECT_EQ(out.layers.size(), 19);
  EXPECT_EQ(out.logits.shape(), (n::Shape{2, 10}));
  for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  p.set("highway0/gate/b", n::full({50}, .25));
  n::NoGradGuard guard;
  n::models::HighwayClassifier(p, n::ones({2, 784}), 20, 50, 10, -10);
  EXPECT_EQ(n::max(n::abs(p.at("highway0/gate/b") - .25)).item(), 0);
}
TEST(NnHighwayModels, InvalidDimensionsAndBiasAreRejected) {
  n::ParameterStore p(17);
  EXPECT_THROW(n::models::HighwayDense(p, "bad", n::scalar(1)), std::invalid_argument);
  EXPECT_THROW(n::models::HighwayConv2D(p, "bad", n::ones({1, 3, 3, 2}), 2),
               std::invalid_argument);
  EXPECT_THROW(n::models::HighwayClassifier(p, n::ones({1, 4}), 0), std::invalid_argument);
  EXPECT_THROW(n::models::HighwayDense(p, "bad", n::ones({1, 2}),
                                       std::numeric_limits<double>::infinity()),
               std::invalid_argument);
}
