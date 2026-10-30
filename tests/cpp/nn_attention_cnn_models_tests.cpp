
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnAttentionModels, SeFullGateDerivativeMatchesIndependentDenseComposition) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const auto x = n::linspace(-.8, 1.2, 2 * 5 * 5 * 8, type)
                     .reshape({2, 5, 5, 8})
                     .detach()
                     .requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::SqueezeExcitationModule(p, "se", x, 2);
    p.set("se/reduce/b", n::full({4}, .3, type));
  }
  const auto actual = n::models::SqueezeExcitationModule(p, "se", x, 2);
  auto vars = p.parameters();
  vars.insert(vars.begin(), x);
  const auto derivatives = n::grad(n::sum(n::square(actual.output)), vars);
  const auto reduced =
      n::relu(n::matmul(n::mean(x, {1, 2}), p.at("se/reduce/W").reshape({8, 4})) +
              p.at("se/reduce/b"));
  const auto gate =
      n::sigmoid(n::matmul(reduced, p.at("se/expand/W").reshape({4, 8})) + p.at("se/expand/b"))
          .reshape({2, 1, 1, 8});
  const auto ref = x * gate;
  EXPECT_LT(n::max(n::abs(actual.output - ref)).item(), 1e-12);
  const auto expected = n::grad(n::sum(n::square(ref)), vars);
  for (size_t i = 0; i < vars.size(); ++i)
    EXPECT_LT(n::max(n::abs(derivatives[i] - expected[i])).item(), 1e-10);
  EXPECT_EQ(p.count(), 76);
}
TEST(NnAttentionModels, CbamSharedMlpAndNontrivialSpatialBnMatchAllDerivatives) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  const auto x = n::sin(.37 * n::arange(0, 2 * 5 * 5 * 8).to(type))
                     .reshape({2, 5, 5, 8})
                     .detach()
                     .requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::Cbam(p, state, "cb", x, 2, false, .9);
    state.set("cb/spatial/bn/moving_mean", n::full({1}, .2, type));
    state.set("cb/spatial/bn/moving_var", n::full({1}, 1.3, type));
    p.set("cb/spatial/bn/gamma", n::full({1}, .7, type));
    p.set("cb/spatial/bn/beta", n::full({1}, -.1, type));
  }
  const auto actual = n::models::Cbam(p, state, "cb", x, 2, false, .9);
  auto vars = p.parameters();
  vars.insert(vars.begin(), x);
  const auto a = n::grad(n::mean(n::square(actual.output)), vars);
  auto mlp = [&](const auto &h) {
    return n::matmul(n::relu(n::matmul(h, p.at("cb/mlp1/W")) + p.at("cb/mlp1/b")),
                     p.at("cb/mlp2/W")) +
           p.at("cb/mlp2/b");
  };
  const auto channel = n::sigmoid(mlp(n::mean(x, {1, 2})) + mlp(n::max(x, {1, 2})))
                           .reshape({2, 1, 1, 8}),
             refined = x * channel;
  const auto maps = n::concat({n::max(refined, {3}, true), n::mean(refined, {3}, true)}, 3);
  auto opts = n::conv_options(1, n::Padding::Explicit);
  opts.explicit_paddings = {3, 3, 3, 3};
  const auto filtered = n::conv2d(maps, p.at("cb/spatial/W"), opts);
  const auto gate = n::sigmoid((filtered - state.at("cb/spatial/bn/moving_mean")) *
                                   n::rsqrt(state.at("cb/spatial/bn/moving_var") + 1e-5) *
                                   p.at("cb/spatial/bn/gamma") +
                               p.at("cb/spatial/bn/beta"));
  const auto ref = refined * gate;
  EXPECT_LT(n::max(n::abs(actual.output - ref)).item(), 1e-12);
  const auto b = n::grad(n::mean(n::square(ref)), vars);
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_LT(n::max(n::abs(a[i] - b[i])).item(), 1e-10);
  EXPECT_EQ(p.count(), 176);
}
TEST(NnAttentionModels, EcaChannelConvolutionHasExactlyKernelWeights) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const auto x = n::linspace(-1, 2, 2 * 4 * 4 * 8, type).reshape({2, 4, 4, 8});
  const auto y = n::models::Eca(p, "e", x, 5);
  const auto pooled = n::mean(x, {1, 2}).to_vector<double>(),
             w = p.at("e/W").to_vector<double>();
  for (int batch = 0; batch < 2; ++batch)
    for (int ch = 0; ch < 8; ++ch) {
      double logit = 0;
      for (int j = 0; j < 5; ++j) {
        const int source = ch + j - 2;
        if (source >= 0 && source < 8)
          logit += pooled[batch * 8 + source] * w[j];
      }
      EXPECT_NEAR(y.gate.at({batch, 0, 0, ch}), 1 / (1 + std::exp(-logit)), 1e-12);
    }
  EXPECT_EQ(p.count(), 5);
  std::vector<int> sizes;
  for (int c : {64, 128, 256, 512, 256, 512, 1024, 2048})
    sizes.push_back(n::models::EcaKernel(c));
  EXPECT_EQ(sizes, (std::vector<int>{3, 5, 5, 5, 5, 5, 5, 7}));
}
TEST(NnAttentionModels, SelectiveKernelSoftmaxAndDilatedReceptiveSupport) {
  for (int stride : {1, 2}) {
    n::ParameterStore p(17), state(17);
    const auto x = n::ones({1, 9, 9, 8}).requires_grad_();
    {
      n::NoGradGuard guard;
      n::models::SelectiveKernel(p, state, "sk", x, 8, 2, stride, 2, 4, false, .9);
      p.set("sk/branch1/W", n::full({3, 3, 4, 8}, .01));
      p.set("sk/branch2/W", n::full({3, 3, 4, 8}, .01));
    }
    const auto out =
        n::models::SelectiveKernel(p, state, "sk", x, 8, 2, stride, 2, 4, false, .9);
    EXPECT_LT(n::max(n::abs(out.weights[0] + out.weights[1] - 1)).item(), 1e-6);
    EXPECT_GT(n::min(out.weights[0]).item(), 0);
    EXPECT_LT(n::max(n::abs(out.output - out.branches[0] * out.weights[0] -
                            out.branches[1] * out.weights[1]))
                  .item(),
              1e-5);
    const int centre = int(out.output.size(1)) / 2;
    for (int branch = 0; branch < 2; ++branch) {
      const auto fresh =
          n::models::SelectiveKernel(p, state, "sk", x, 8, 2, stride, 2, 4, false, .9);
      const auto picked =
          n::slice(fresh.branches[branch], {0, centre, centre, 0}, {1, 1, 1, 8});
      const auto support =
          n::greater(n::sum(n::abs(n::grad(n::sum(picked), {x})[0]), {0, 3}), 0);
      EXPECT_EQ(n::count_nonzero(support).item(), 9);
      const auto coords = n::arange(0, 9).to(n::DType::Float32);
      const auto rows =
          n::greater(n::sum(support.to(n::DType::Float32), {1}), 0).to(n::DType::Float32);
      const double low =
                       n::min(n::where(n::greater(rows, 0), coords, n::full({9}, 100))).item(),
                   high = n::max(coords * rows).item();
      EXPECT_EQ(high - low + 1, branch ? 5 : 3);
      EXPECT_EQ((low + high) / 2, centre * stride);
    }
  }
}
TEST(NnAttentionModels, GroupedAndHookStridesRespectTheSelectedPlacement) {
  n::set_num_threads(2);
  for (bool on_spatial : {true, false})
    for (bool hook : {true, false}) {
      n::ParameterStore p(17), state(17);
      n::models::AttentionResNetConfig cfg;
      cfg.width_divisor = 4;
      cfg.classes = 10;
      cfg.small_stem = true;
      cfg.cardinality = 32;
      cfg.base_width = 4;
      cfg.stride_on_3x3 = on_spatial;
      cfg.update_stats = false;
      if (hook)
        cfg.conv2 = n::models::SelectiveKernelHook(32, 16, .9, false);
      const auto x = n::sin(.17 * n::arange(0, 2 * 32 * 32 * 3).to(n::DType::Float32))
                         .reshape({2, 32, 32, 3});
      const auto out = n::models::AttentionResNet(p, state, x, cfg, true);
      EXPECT_EQ(out.features.shape(), n::Shape({2, 4, 4, 512}));
      EXPECT_EQ(out.blocks.size(), 16);
      for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
        EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    }
}
TEST(NnAttentionModels, CompleteAttentionCountsAndFrozenMovingState) {
  n::set_num_threads(2);
  for (auto attention :
       {n::models::ChannelAttention::kSqueezeExcitation, n::models::ChannelAttention::kCbam,
        n::models::ChannelAttention::kEca}) {
    n::ParameterStore p(17), state(17);
    n::models::AttentionResNetConfig cfg;
    cfg.classes = 10;
    cfg.width_divisor = 4;
    cfg.small_stem = true;
    cfg.stride_on_3x3 = false;
    cfg.attention = attention;
    cfg.adaptive_eca = true;
    cfg.update_stats = false;
    const auto x = n::ones({2, 32, 32, 3});
    {
      n::NoGradGuard guard;
      n::models::AttentionResNet(p, state, x, cfg);
    }
    EXPECT_EQ(p.count(), attention == n::models::ChannelAttention::kSqueezeExcitation ? 1645382
                         : attention == n::models::ChannelAttention::kCbam            ? 1646982
                                                                           : 1484260);
    const auto before = state.flatten();
    const auto out = n::models::AttentionResNet(p, state, x, cfg, true);
    for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    EXPECT_EQ(state.flatten(), before);
  }
}
TEST(NnAttentionModels, InvalidGeometryRejectedBeforeDivisionOrAllocation) {
  n::ParameterStore p(17), state(17);
  const auto x = n::ones({2, 5, 5, 8});
  EXPECT_THROW(n::models::SqueezeExcitationModule(p, "s", x, 0), std::invalid_argument);
  EXPECT_THROW(n::models::Cbam(p, state, "c", x, 16, false, .9), std::invalid_argument);
  EXPECT_THROW(n::models::EcaKernel(0), std::invalid_argument);
  EXPECT_THROW(n::models::SelectiveKernel(p, state, "sk", x, 8, 0, 1, 2, 4, false, .9),
               std::invalid_argument);
  n::models::AttentionResNetConfig cfg;
  cfg.width_divisor = 65;
  EXPECT_THROW(n::models::AttentionResNet(p, state, x, cfg), std::invalid_argument);
}
