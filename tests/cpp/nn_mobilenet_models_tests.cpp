
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnMobileNetModels, LoggedDenseDepthwiseValuesAndArithmetic) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  std::vector<n::models::MobileLayer> layers;
  n::models::MobileContext c{p, state, layers, false};
  const auto x = n::linspace(-1, 1, 100, type).reshape({1, 5, 5, 4});
  for (bool depthwise : {false, true}) {
    const auto kind =
        depthwise ? n::models::MobileConv::kDepthwise : n::models::MobileConv::kDense;
    const std::string name = depthwise ? "d" : "c";
    const int out = depthwise ? 4 : 8;
    const auto y = n::models::ConvBN(c, name, x, out, 3, 2, n::Activation::Relu6, kind);
    const auto raw =
        depthwise
            ? n::depthwise_conv2d(x, p.at(name + "/W"), n::conv_options(2, n::Padding::Same))
            : n::conv2d(x, p.at(name + "/W"), n::conv_options(2, n::Padding::Same));
    EXPECT_LT(n::max(n::abs(y - n::relu6(raw / std::sqrt(1.001)))).item(), 1e-12);
    EXPECT_EQ(y.shape(), n::Shape({1, 3, 3, out}));
    EXPECT_EQ(layers.back().Macs(), depthwise ? 324 : 2592);
  }
  EXPECT_EQ(p.count(), 348);
  EXPECT_EQ(state.count(), 24);
}
TEST(NnMobileNetModels, InvertedResidualSignedIdentityAndStridedGeometry) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  std::vector<n::models::MobileLayer> layers;
  n::models::MobileContext c{p, state, layers, false};
  const auto x = n::linspace(-1, 1, 256, type).reshape({1, 8, 8, 4}).detach().requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::InvertedResidual(c, "b", x, 4, 6, 1);
  }
  p.set("b/project/bn/gamma", n::zeros({4}, type));
  p.set("b/project/bn/beta", n::zeros({4}, type));
  const auto y = n::models::InvertedResidual(c, "b", x, 4, 6, 1);
  EXPECT_EQ(n::max(n::abs(y.output - x)).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::grad(n::sum(y.output), {x})[0] - 1)).item(), 0);
  EXPECT_EQ(y.expanded.size(3), 24);
  const auto down = n::models::InvertedResidual(c, "down", x, 6, 6, 2);
  EXPECT_EQ(down.output.shape(), n::Shape({1, 4, 4, 6}));
}
TEST(NnMobileNetModels, HardActivationsValuesAndInteriorDerivatives) {
  const auto x = n::from_vector(std::vector<double>{-4, -2, -1, 0, 1, 2, 4}).requires_grad_();
  const auto y = n::models::HardSwish(x), g = n::models::HardSigmoid(x);
  const auto dx = n::grad(n::sum(y), {x})[0];
  for (int64_t i = 0; i < 7; ++i) {
    const double v = x.at({i}), gate = std::clamp((v + 3) / 6, 0., 1.);
    EXPECT_NEAR(g.at({i}), gate, 1e-12);
    EXPECT_NEAR(y.at({i}), v * gate, 1e-12);
    EXPECT_NEAR(dx.at({i}), v < -3 ? 0 : v > 3 ? 1 : v / 3 + .5, 1e-12);
  }
}
TEST(NnMobileNetModels, SqueezeExcitationIncludesInputDependentGateDerivative) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const auto x = n::full({1, 2, 2, 2}, .5, type).requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::SqueezeExcite(p, "se", x, 2);
  }
  p.set("se/reduce/W", n::full({1, 1, 2, 2}, .2, type));
  p.set("se/reduce/b", n::full({2}, .1, type));
  p.set("se/expand/W", n::full({1, 1, 2, 2}, .1, type));
  p.set("se/expand/b", n::full({2}, -.1, type));
  const auto out = n::models::SqueezeExcite(p, "se", x, 2);
  EXPECT_NEAR(out.gate.at({0, 0, 0, 0}), 2.96 / 6, 1e-12);
  const auto dx = n::grad(n::sum(out.output), {x})[0];
  EXPECT_LT(n::max(n::abs(dx - .5)).item(), 1e-12);
  p.set("se/expand/W", n::zeros({1, 1, 2, 2}, type));
  p.set("se/expand/b", n::from_vector(std::vector<double>{-4, 4}));
  const auto controlled = n::models::SqueezeExcite(p, "se", x, 2);
  EXPECT_EQ(controlled.gate.at({0, 0, 0, 0}), 0);
  EXPECT_EQ(controlled.gate.at({0, 0, 0, 1}), 1);
}
TEST(NnMobileNetModels, FullReleasedCountsAndMeasuredMacs) {
  n::set_num_threads(2);
  n::NoGradGuard guard;
  for (int version : {1, 2, 3, 4}) {
    n::ParameterStore p(17), state(17);
    const auto x = n::zeros({1, 224, 224, 3});
    n::models::MobileNetOutputs out;
    if (version == 1)
      out = n::models::MobileNetV1(p, state, x);
    else if (version == 2)
      out = n::models::MobileNetV2(p, state, x);
    else {
      n::models::MobileNetV3Config cfg;
      cfg.large = version == 3;
      out = n::models::MobileNetV3(p, state, x, cfg);
    }
    const std::vector<int64_t> counts = {4231976, 3504872, 5483032, 2542856},
                               moving = {21888, 34112, 24400, 12112},
                               macs = {568740352, 300774272, 216589760, 56510400};
    EXPECT_EQ(p.count(), counts[version - 1]);
    EXPECT_EQ(state.count(), moving[version - 1]);
    EXPECT_EQ(out.Macs(), macs[version - 1]);
    EXPECT_EQ(out.logits.shape(), n::Shape({1, 1000}));
  }
}
TEST(NnMobileNetModels, CompleteCifarGradientsAndMovingStateSemantics) {
  n::set_num_threads(2);
  for (int version : {1, 2, 3, 4}) {
    n::ParameterStore p(17), state(17);
    n::Generator gen(29);
    const auto x = n::normal({2, 32, 32, 3}, 0, .2, gen);
    bool update = false;
    auto model = [&](bool train) {
      if (version == 1) {
        n::models::MobileNetV1Config cfg;
        cfg.classes = 10;
        cfg.small_stem = true;
        cfg.update_stats = update;
        return n::models::MobileNetV1(p, state, x, cfg, train);
      }
      if (version == 2) {
        n::models::MobileNetV2Config cfg;
        cfg.classes = 10;
        cfg.small_stem = true;
        cfg.keep_probability = 1;
        cfg.update_stats = update;
        return n::models::MobileNetV2(p, state, x, cfg, train);
      }
      n::models::MobileNetV3Config cfg;
      cfg.classes = 10;
      cfg.small_stem = true;
      cfg.large = version == 3;
      cfg.keep_probability = 1;
      cfg.update_stats = update;
      return n::models::MobileNetV3(p, state, x, cfg, train);
    };
    const auto out = model(true);
    const auto gradients = n::grad(n::mean(n::square(out.logits)), p.parameters());
    EXPECT_EQ(gradients.size(), p.parameters().size());
    for (const auto &g : gradients)
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    const std::vector<int64_t> counts = {3217226, 2236682, 4214842, 1528106};
    EXPECT_EQ(p.count(), counts[version - 1]);
    const auto saved = state.flatten();
    n::NoGradGuard guard;
    model(false);
    EXPECT_EQ(saved, state.flatten());
    update = true;
    model(true);
    EXPECT_NE(saved, state.flatten());
  }
}
TEST(NnMobileNetModels, AlignmentAndInvalidMetadataShapesAndWidths) {
  EXPECT_EQ(n::models::MakeDivisible(9), 16);
  EXPECT_EQ(n::models::MakeDivisible(17), 16);
  EXPECT_THROW(n::models::MakeDivisible(-1), std::invalid_argument);
  EXPECT_THROW(n::models::MakeDivisible(4, 0), std::invalid_argument);
  const n::models::MobileLayer bad{"bad", {}, 1};
  EXPECT_THROW(bad.Macs(), std::invalid_argument);
  n::ParameterStore p(17), state(17);
  std::vector<n::models::MobileLayer> layers;
  n::models::MobileContext c{p, state, layers, false};
  EXPECT_THROW(n::models::InvertedResidual(c, "b", n::ones({1, 5, 5, 4}), 4, 0, 1),
               std::invalid_argument);
  EXPECT_THROW(n::models::SqueezeExcite(p, "s", n::ones({2, 3}), 2), std::invalid_argument);
  n::models::MobileNetV1Config cfg;
  cfg.alpha = std::numeric_limits<double>::infinity();
  EXPECT_THROW(n::models::MobileNetV1(p, state, n::ones({1, 8, 8, 3}), cfg),
               std::invalid_argument);
}
