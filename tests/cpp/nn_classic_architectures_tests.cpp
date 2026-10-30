#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnClassicArchitectures, LrnClipsEdgesAndSupportsSecondDerivatives) {
  auto x = n::tensor({1., 2., 3.}).to(n::DType::Float64).detach().requires_grad_();
  const auto y = n::models::LocalResponseNorm(x, 1, 2., .1, .75);
  const auto expected =
      x * n::pow(n::from_vector(std::vector<double>{2.5, 3.4, 3.3}, {3}), -.75);
  EXPECT_LT(n::max(n::abs(y - expected)).item(), 1e-12);
  const auto first = n::grad(n::sum(y), {x}, {.retain_graph = true, .create_graph = true})[0];
  EXPECT_TRUE(std::isfinite(n::sum(n::grad(n::sum(first), {x})[0]).item()));
  EXPECT_THROW(n::models::LocalResponseNorm(x, -1, 2., .1, .75), std::invalid_argument);
  EXPECT_THROW(n::models::LocalResponseNorm(x, 1, 0., .1, .75), std::invalid_argument);
  EXPECT_THROW(n::models::LocalResponseNorm(x, 1, 2., -.1, .75), std::invalid_argument);
}
TEST(NnClassicArchitectures, ExplicitAndFusedGroupsAgreeInValuesAndEveryGradient) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  n::Generator rng(17);
  auto x = n::normal({2, 5, 6, 4}, 0, .2, rng, type).requires_grad_();
  const auto a = n::models::GroupedConv2D(p, "g", x, 6, 3, 2);
  const auto b = n::models::FusedGroupedConv2D(p, "g", x, 6, 3, 2);
  EXPECT_LT(n::max(n::abs(a - b)).item(), 1e-12);
  auto leaves = p.parameters();
  leaves.push_back(x);
  const auto ga = n::grad(n::sum(a), leaves), gb = n::grad(n::sum(b), leaves);
  for (size_t i = 0; i < ga.size(); ++i)
    EXPECT_LT(n::max(n::abs(ga[i] - gb[i])).item(), 1e-12);
  const auto y = n::models::GroupedConv2D(p, "g", x, 6, 3, 2);
  const auto dx = n::grad(n::sum(n::slice_axis(y, 3, 0, 3)), {x})[0];
  EXPECT_EQ(n::max(n::abs(n::slice_axis(dx, 3, 2, 4))).item(), 0);
  EXPECT_THROW(n::models::GroupedConv2D(p, "bad", x, 5, 3, 2), std::invalid_argument);
}
TEST(NnClassicArchitectures, InceptionBranchOrderingAndReductionCounts) {
  n::ParameterStore p(17, n::DType::Float64);
  const n::models::InceptionWidths w{1, 1, 1, 1, 1, 1};
  {
    n::NoGradGuard guard;
    n::models::InceptionV1(p, "i", n::ones({1, 3, 3, 1}, p.dtype()), w);
  }
  for (const auto &name : p.names())
    p.set(name, name.ends_with("/W") ? n::ones(p.at(name).shape(), p.dtype())
                                     : n::zeros(p.at(name).shape(), p.dtype()));
  const auto out = n::models::InceptionV1(p, "i", n::ones({1, 3, 3, 1}, p.dtype()), w);
  EXPECT_EQ(out.concatenated.shape(), (n::Shape{1, 3, 3, 4}));
  EXPECT_EQ(n::slice(out.concatenated, {0, 1, 1, 0}, {1, 1, 1, 4}).to_vector<double>(),
            (std::vector<double>{1, 9, 9, 1}));
  EXPECT_EQ(n::slice(out.concatenated, {0, 0, 0, 0}, {1, 1, 1, 4}).to_vector<double>(),
            (std::vector<double>{1, 4, 9, 1}));
  EXPECT_EQ(n::models::GoogLeNetWidths()[0].KernelWeights(192), 163328);
  EXPECT_THROW(w.Divided(2), std::invalid_argument);
  EXPECT_THROW(w.KernelWeights(0), std::invalid_argument);
}
TEST(NnClassicArchitectures, AlexNetGeometryAllLeavesAndDropoutModes) {
  n::models::AlexNetConfig cfg;
  cfg.image_size = 67;
  cfg.width_divisor = 8;
  cfg.classes = 3;
  cfg.dense_width = 16;
  EXPECT_EQ(n::models::AlexNetConfig{}.FinalSpatialSize(), 6);
  n::ParameterStore p(17);
  n::Generator rng(19);
  const auto x = n::normal({2, 67, 67, 3}, 0, .2, rng);
  const auto out = n::models::AlexNet(p, x, cfg);
  EXPECT_EQ(out.pooling[2].shape(), (n::Shape{2, 1, 1, 32}));
  for (const auto &g : n::grad(n::sum(out.logits), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  n::NoGradGuard guard;
  EXPECT_EQ(n::max(n::abs(n::models::AlexNet(p, x, cfg).logits -
                          n::models::AlexNet(p, x, cfg).logits))
                .item(),
            0);
  EXPECT_GT(n::max(n::abs(n::models::AlexNet(p, x, cfg, true).logits -
                          n::models::AlexNet(p, x, cfg, true).logits))
                .item(),
            0);
  EXPECT_THROW(n::models::AlexNet(p, n::zeros({1, 68, 68, 3}), cfg), std::invalid_argument);
  EXPECT_THROW(n::models::KeepDropout(p, x, 2), std::invalid_argument);
}
TEST(NnClassicArchitectures, GoogLeNetAuxiliariesReceiveGradientsAndLeaveInferenceUnchanged) {
  n::models::GoogLeNetConfig cfg;
  cfg.image_size = 65;
  cfg.width_divisor = 8;
  cfg.classes = 3;
  cfg.auxiliary_hidden = 16;
  EXPECT_EQ(n::models::GoogLeNetConfig{}.AuxiliarySpatialSize(), 4);
  n::ParameterStore p(17);
  n::Generator rng(23);
  const auto x = n::normal({2, 65, 65, 3}, 0, .2, rng);
  const auto out = n::models::GoogLeNet(p, x, cfg, false, true);
  EXPECT_EQ(out.auxiliary.size(), 2);
  EXPECT_EQ(out.inception[8].shape(), (n::Shape{2, 3, 3, 128}));
  auto loss =
      n::sparse_softmax_cross_entropy(out.logits, n::tensor({0, 2}).to(n::DType::Int64));
  for (const auto &a : out.auxiliary)
    loss =
        loss + .3 * n::sparse_softmax_cross_entropy(a, n::tensor({0, 2}).to(n::DType::Int64));
  for (const auto &g : n::grad(loss, p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  n::NoGradGuard guard;
  const auto before = n::models::GoogLeNet(p, x, cfg).logits;
  p.set("aux1/output/b", p.at("aux1/output/b") + 2);
  EXPECT_EQ(n::max(n::abs(before - n::models::GoogLeNet(p, x, cfg).logits)).item(), 0);
  EXPECT_TRUE(n::models::GoogLeNet(p, x, cfg).auxiliary.empty());
}
TEST(NnClassicArchitectures, UnpoolUsesForwardSwitchesAndDeconvUsesBackwardSign) {
  const auto x = n::tensor({1., 9., 2., 3., 4., 5., 6., 10., 7., 8., 12., 0., 11., 2., 1., 4.},
                           {1, 4, 4, 1});
  const auto signal = n::tensor({1., 2., 3., 4.}, {1, 2, 2, 1});
  const auto unpooled = n::models::UnpoolAtSwitches(x, signal, 2, 2, "VALID");
  std::vector<float> expected(16, 0);
  expected[1] = 1;
  expected[7] = 2;
  expected[12] = 3;
  expected[10] = 4;
  EXPECT_EQ(unpooled.to_vector<float>(), expected);
  auto signed_x = n::tensor({-1., 2.}, {1, 1, 1, 2}).requires_grad_();
  const auto backward = n::tensor({3., -4.}, {1, 1, 1, 2});
  const auto kernel = n::tensor({1., 0., 0., 1.}, {1, 1, 2, 2});
  EXPECT_EQ(n::grad(n::sum(n::relu(signed_x) * backward), {signed_x})[0].to_vector<float>(),
            (std::vector<float>{0, -4}));
  EXPECT_EQ(n::models::DeconvLayer(signed_x, kernel, backward, 1, "VALID").to_vector<float>(),
            (std::vector<float>{3, 0}));
  EXPECT_THROW(n::models::ParsePadding("same"), std::invalid_argument);
}
TEST(NnClassicArchitectures, ZfNetRetainsGeometryAndProjectsBackToInputGrid) {
  n::models::ZFNetConfig cfg;
  cfg.image_size = 67;
  cfg.width_divisor = 8;
  cfg.classes = 3;
  cfg.dense_width = 16;
  EXPECT_EQ(n::models::ZFNetConfig{}.FinalSpatialSize(), 6);
  n::ParameterStore p(17);
  n::Generator rng(23);
  const auto x = n::normal({1, 67, 67, 3}, 0, .2, rng);
  const auto out = n::models::ZFNet(p, x, cfg);
  EXPECT_EQ(out.convolution[0].shape(), (n::Shape{1, 32, 32, 12}));
  EXPECT_EQ(out.pooling[2].shape(), (n::Shape{1, 1, 1, 32}));
  const auto reconstruction =
      n::models::ZFNetDeconvolution(p, out, n::ones_like(out.convolution[4]));
  EXPECT_EQ(reconstruction.shape(), x.shape());
  EXPECT_TRUE(std::isfinite(n::sum(reconstruction).item()));
  EXPECT_THROW(n::models::ZFNet(p, n::zeros({1, 68, 68, 3}), cfg), std::invalid_argument);
}
TEST(NnClassicArchitectures, XceptionSeparableIsLinearBetweenItsTwoFactors) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  n::models::ConvNormConfig cfg;
  auto x = n::tensor({-2., 1., 3., -4., 2., -1., 4., -3., 1.}, {1, 3, 3, 1})
               .to(type)
               .detach()
               .requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::XceptionSeparable(p, state, "sep", x, 1, cfg);
  }
  p.set("sep/depthwise/W",
        n::tensor({0., 0., 0., 0., 1., 0., 0., 0., 0.}, {3, 3, 1, 1}).to(type));
  p.set("sep/pointwise/W", n::full({1, 1, 1, 1}, -2., type));
  const auto out = n::models::XceptionSeparable(p, state, "sep", x, 1, cfg);
  EXPECT_LT(n::max(n::abs(out + 2 * x / std::sqrt(1.001))).item(), 1e-12);
  EXPECT_LT(n::max(n::abs(n::grad(n::sum(out), {x})[0] + 2 / std::sqrt(1.001))).item(), 1e-12);
  const auto wrong = n::conv2d(n::relu(n::depthwise_conv2d(x, p.at("sep/depthwise/W"))),
                               p.at("sep/pointwise/W")) /
                     std::sqrt(1.001);
  EXPECT_GT(n::sum(n::abs(out - wrong)).item(), 10);
  EXPECT_EQ(p.count(), 12);
  EXPECT_EQ(state.count(), 2);
}
TEST(NnClassicArchitectures, XceptionKeepsEightMiddleBlocksAndSeparateMovingState) {
  n::models::XceptionConfig cfg;
  cfg.image_size = 71;
  cfg.classes = 3;
  cfg.width_divisor = 8;
  cfg.momentum = .9;
  n::ParameterStore p(17), state(17);
  n::Generator rng(29);
  const auto x = n::normal({2, 71, 71, 3}, 0, .2, rng);
  const auto out = n::models::Xception(p, state, x, cfg, true);
  ASSERT_EQ(out.stages.size(), 16);
  EXPECT_EQ(out.stages[12].first, "middle8");
  EXPECT_EQ(out.stages[12].second.shape(), (n::Shape{2, 5, 5, 91}));
  for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  const auto saved = state.flatten();
  n::NoGradGuard guard;
  n::models::Xception(p, state, x, cfg, false);
  EXPECT_EQ(state.flatten(), saved);
  EXPECT_GT(state.count(), 0);
}
TEST(NnClassicArchitectures, RectangularBetaOnlyBnUsesSeparateMovingState) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  n::models::ConvNormConfig cfg;
  cfg.momentum = .5;
  const auto x = n::linspace(1, 6, 6, type).reshape({1, 2, 3, 1});
  {
    n::NoGradGuard guard;
    n::models::NormalizedConv(p, state, "rect", x, 2, 1, 3, cfg, 1, "SAME", false);
  }
  p.set("rect/W", n::tensor({1., -1., 1., -1., 1., -1.}, {1, 3, 1, 2}).to(type));
  const auto raw = n::conv2d(x, p.at("rect/W"));
  const auto mean = n::mean(raw, {0, 1, 2}, true);
  const auto variance = n::mean(n::square(raw - mean), {0, 1, 2}, true);
  cfg.training = true;
  const auto out =
      n::models::NormalizedConv(p, state, "rect", x, 2, 1, 3, cfg, 1, "SAME", false);
  EXPECT_LT(n::max(n::abs(out - (raw - mean) / n::sqrt(variance + .001))).item(), 1e-12);
  EXPECT_EQ(p.count(), 8);
  EXPECT_EQ(state.count(), 4);
  EXPECT_LT(n::max(n::abs(state.at("rect/bn/moving_mean") - .5 * mean.reshape({2}))).item(),
            1e-12);
  EXPECT_LT(n::max(n::abs(state.at("rect/bn/moving_var") -
                          (.5 + .5 * variance.reshape({2}) * 6. / 5.)))
                .item(),
            1e-12);
}
TEST(NnClassicArchitectures, InceptionResidualScalesValuesAndGradients) {
  const auto type = n::DType::Float64;
  for (char kind : {'A', 'B', 'C'}) {
    n::ParameterStore p(17, type), state(17, type);
    n::models::Inception23Config cfg;
    cfg.width_divisor = 8;
    n::models::Inception23Outputs out;
    n::models::InceptionBuilder b(p, state, out, cfg, false);
    const auto x =
        n::linspace(.1, 1., 200, type).reshape({1, 5, 5, 8}).detach().requires_grad_();
    const n::models::ConvFeature feature{x, 8};
    const auto zero = n::models::InceptionResidual(b, "r", feature, kind, 0, false).value;
    const auto one = n::models::InceptionResidual(b, "r", feature, kind, .17, false).value;
    const auto two = n::models::InceptionResidual(b, "r", feature, kind, .34, false).value;
    EXPECT_EQ(n::max(n::abs(zero - x)).item(), 0);
    EXPECT_LT(n::max(n::abs(two - x - 2 * (one - x))).item(), 1e-12);
    auto variables = p.parameters();
    variables.push_back(x);
    const auto g1 = n::grad(n::sum(one), variables), g2 = n::grad(n::sum(two), variables);
    for (size_t i = 0; i + 1 < variables.size(); ++i)
      EXPECT_LT(n::max(n::abs(g2[i] - 2 * g1[i])).item(), 1e-10);
    EXPECT_LT(n::max(n::abs(g2.back() - 1 - 2 * (g1.back() - 1))).item(), 1e-10);
    EXPECT_TRUE(p.contains("r/project/b"));
    EXPECT_FALSE(p.contains("r/project/bn/beta"));
  }
}
TEST(NnClassicArchitectures, AllInceptionVariantsKeepTheirDepthAndAuxiliaryHeads) {
  n::set_num_threads(2);
  n::Generator rng(29);
  for (int variant : {2, 3, 4, 5}) {
    n::ParameterStore p(17), state(17);
    n::models::Inception23Outputs out;
    const int image = variant == 2 ? 64 : 107;
    const auto x = n::normal({2, image, image, 3}, 0, .2, rng);
    if (variant < 4) {
      n::models::Inception23Config cfg;
      cfg.version = variant;
      cfg.image_size = image;
      cfg.classes = 3;
      cfg.width_divisor = 8;
      out = n::models::Inception23(p, state, x, cfg, true, true);
    } else {
      n::models::Inception4Config cfg;
      cfg.residual = variant == 5;
      cfg.image_size = 107;
      cfg.classes = 3;
      cfg.width_divisor = 8;
      out = n::models::Inception4(p, state, x, cfg, true, true);
    }
    EXPECT_EQ(out.stages.size(), variant == 2   ? 15
                                 : variant == 3 ? 18
                                 : variant == 4 ? 22
                                                : 51);
    EXPECT_EQ(out.auxiliary.size(), variant == 2 ? 0 : 1);
    auto objective = n::mean(n::square(out.logits));
    for (const auto &a : out.auxiliary)
      objective = objective + .4 * n::mean(n::square(a));
    for (const auto &g : n::grad(objective, p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  }
}
