
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnMBConvModels, BlockStringTablesAndInvalidFields) {
  for (const auto &variant : {"a1", "b1"}) {
    int total = 0;
    for (const auto &b : n::models::MnasNetBlocks(variant)) {
      const auto text = n::models::EncodeBlock(b);
      EXPECT_EQ(n::models::EncodeBlock(n::models::DecodeBlock(text)), text);
      total += b.repeats;
    }
    EXPECT_EQ(total, std::string(variant) == "a1" ? 16 : 17);
  }
  for (int v = 0; v < 8; ++v) {
    int total = 0;
    for (const auto &b : n::models::EfficientNetBlocks(v))
      total += b.repeats;
    EXPECT_EQ(total, (std::vector<int>{16, 23, 23, 26, 32, 39, 45, 55})[v]);
  }
  for (const auto &variant : {"s", "m", "l", "b0", "b1", "b2", "b3"})
    for (const auto &b :
         n::models::EfficientNetV2Blocks(n::models::EfficientNetV2Spec(variant))) {
      EXPECT_EQ(n::models::EncodeBlock(n::models::DecodeBlock(n::models::EncodeBlock(b))),
                n::models::EncodeBlock(b));
    }
  for (const auto &text :
       {"r1_k0_s1_e1_i8_o8", "r1_k3_s0_e1_i8_o8", "r1_k3_s1_e0_i8_o8", "r1_k3_s12_e1_i8_o8",
        "r1_k3_s1_e1_i8_o8_c2", "r1_k3_s1_e1_i8_o8_se0.2.3"})
    EXPECT_THROW(n::models::DecodeBlock(text), std::invalid_argument);
  EXPECT_THROW(n::models::RoundFilters(8, 0), std::invalid_argument);
  EXPECT_EQ(n::models::RoundFilters(24, 1.3), 32);
}
TEST(NnMBConvModels, DropConnectMasksRowsAndPreservesDerivativesAtEndpoints) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    n::Generator gen(77);
    const auto x = n::ones({512, 2, 2, 3}, type).requires_grad_();
    const auto out = n::models::DropConnect(x, .8, gen), dx = n::grad(n::sum(out), {x})[0];
    EXPECT_EQ(out.dtype(), type);
    EXPECT_LT(n::max(n::abs(out - dx)).item(), 1e-6);
    const auto live = n::greater(n::sum(out, {1, 2, 3}), 0);
    EXPECT_NEAR(n::mean(live.to(type)).item(), .8, .06);
    const auto ratio = out.reshape({512, 12});
    EXPECT_EQ(n::max(n::abs(ratio - n::slice_axis(ratio, 1, 0, 1))).item(), 0);
    for (double survival : {0., 1.}) {
      const auto endpoint = n::models::DropConnect(x, survival, gen);
      EXPECT_EQ(n::max(n::abs(endpoint - survival)).item(), 0);
      EXPECT_EQ(n::max(n::abs(n::grad(n::sum(endpoint), {x})[0] - survival)).item(), 0);
    }
  }
  n::Generator gen(17);
  EXPECT_THROW(n::models::DropConnect(n::ones({2}), .8, gen), std::invalid_argument);
}
TEST(NnMBConvModels, ExpandedDepthwiseSeAndShortcutGradientFiniteDifference) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  std::vector<n::models::MobileLayer> layers;
  n::models::MobileContext c{p, state, layers, false};
  const auto x = n::linspace(-.7, .9, 2 * 5 * 5 * 4, type)
                     .reshape({2, 5, 5, 4})
                     .detach()
                     .requires_grad_();
  const auto b = n::models::DecodeBlock("r1_k3_s1_e2_i4_o4_se0.25");
  const auto out = n::models::MBConv(c, "b", x, 4, b, 1, true);
  EXPECT_EQ(out.expanded.shape(), n::Shape({2, 5, 5, 8}));
  EXPECT_EQ(out.gate.shape(), n::Shape({2, 1, 1, 8}));
  EXPECT_LT(n::max(n::abs(out.output - x - out.projected)).item(), 1e-15);
  auto vars = p.parameters();
  vars.insert(vars.begin(), x);
  for (const auto &g : n::grad(n::mean(n::square(out.output)), vars))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  const auto fresh = n::models::MBConv(c, "b", x, 4, b, 1, true);
  const auto dx = n::grad(n::mean(n::square(fresh.output)), {x})[0];
  const auto direction =
      n::one_hot(n::tensor({48}).to(n::DType::Int64), 200, 1, 0, type).reshape(x.shape());
  const double h = 1e-6;
  n::NoGradGuard guard;
  const double
      plus =
          n::mean(
              n::square(
                  n::models::MBConv(c, "b", x.detach() + h * direction, 4, b, 1, true).output))
              .item(),
      minus =
          n::mean(
              n::square(
                  n::models::MBConv(c, "b", x.detach() - h * direction, 4, b, 1, true).output))
              .item();
  EXPECT_NEAR(dx.reshape({200}).at({48}), (plus - minus) / (2 * h), 1e-9);
}
TEST(NnMBConvModels, FusedAndDepthwiseHaveDifferentExecutedBudgets) {
  for (bool fused : {true, false}) {
    n::ParameterStore p(17), state(17);
    std::vector<n::models::MobileLayer> layers;
    n::models::MobileContext c{p, state, layers, false};
    auto b = n::models::DecodeBlock("r1_k3_s2_e4_i24_o48_c1");
    b.fused = fused;
    n::NoGradGuard guard;
    const auto out = n::models::MBConv(c, "b", n::zeros({1, 96, 96, 24}), 48, b, 2, true);
    EXPECT_EQ(out.output.shape(), n::Shape({1, 48, 48, 48}));
    n::models::MobileNetOutputs net;
    net.layers = layers;
    EXPECT_EQ(net.Macs(), fused ? 58392576 : 33841152);
    EXPECT_EQ(p.count(), fused ? 25632 : 8256);
  }
  n::ParameterStore p(17), state(17);
  std::vector<n::models::MobileLayer> layers;
  n::models::MobileContext c{p, state, layers, false};
  const auto b = n::models::DecodeBlock("r1_k3_s2_e1_i8_o16_c1");
  const auto out = n::models::MBConv(c, "f", n::ones({1, 5, 5, 8}), 16, b, 2, true);
  EXPECT_EQ(out.output.shape(), n::Shape({1, 3, 3, 16}));
  EXPECT_EQ(layers.size(), 1);
  EXPECT_EQ(n::max(n::abs(out.output - out.projected)).item(), 0);
  EXPECT_TRUE(std::isfinite(n::sum(out.output).item()));
}
TEST(NnMBConvModels, CompleteCifarCountsGradientsAndFrozenInferenceState) {
  n::set_num_threads(2);
  for (int model = 0; model < 3; ++model) {
    n::ParameterStore p(17), state(17);
    auto forward = [&](const auto &x, bool training) {
      if (model == 0) {
        n::models::MnasNetConfig cfg;
        cfg.classes = 10;
        cfg.small_stem = true;
        cfg.update_stats = false;
        return n::models::MnasNet(p, state, x, cfg, training);
      }
      if (model == 1) {
        n::models::EfficientNetConfig cfg;
        cfg.classes = 10;
        cfg.small_stem = true;
        cfg.update_stats = false;
        return n::models::EfficientNet(p, state, x, cfg, training);
      }
      n::models::EfficientNetV2Config cfg;
      cfg.variant = "b0";
      cfg.classes = 10;
      cfg.small_stem = true;
      cfg.update_stats = false;
      return n::models::EfficientNetV2(p, state, x, cfg, training);
    };
    const auto x = n::sin(.37 * n::arange(0, 2 * 32 * 32 * 3).to(n::DType::Float32))
                       .reshape({2, 32, 32, 3});
    {
      n::NoGradGuard guard;
      forward(x, false);
    }
    EXPECT_EQ(p.count(), (std::vector<int64_t>{2618848, 4020358, 5871514})[model]);
    const auto before = state.flatten();
    const auto out = forward(x, true);
    for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    EXPECT_EQ(before, state.flatten());
    {
      n::NoGradGuard guard;
      forward(x, false);
    }
    EXPECT_EQ(before, state.flatten());
  }
}
