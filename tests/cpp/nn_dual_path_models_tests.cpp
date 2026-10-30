#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
#include <numeric>
namespace n = datamunge::nn;
TEST(NnDualPathModels, ZeroOrdinaryUpdatesPreserveSignedPathsAndIdentityDerivatives) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  const auto r = n::linspace(-1, 1, 100, type).reshape({1, 5, 5, 4}).detach().requires_grad_();
  const auto d =
      n::linspace(-.7, .8, 50, type).reshape({1, 5, 5, 2}).detach().requires_grad_();
  const n::models::DPNPaths x{r, d, 4, 2};
  {
    n::NoGradGuard guard;
    n::models::DPNBlock(p, state, "u", x, 4, 4, 1, 2, false);
  }
  p.set("u/conv3/W", n::zeros_like(p.at("u/conv3/W")));
  const auto out = n::models::DPNBlock(p, state, "u", x, 4, 4, 1, 2, false);
  EXPECT_EQ(out.output.residual_channels, 4);
  EXPECT_EQ(out.output.dense_channels, 3);
  EXPECT_EQ(n::max(n::abs(out.output.residual - r)).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::slice_axis(out.output.dense, 3, 0, 2) - d)).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::slice_axis(out.output.dense, 3, 2, 3))).item(), 0);
  for (const auto &dx : n::grad(n::sum(n::models::DPNJoin(out.output)), {r, d}))
    EXPECT_EQ(n::max(n::abs(dx - 1)).item(), 0);
}
TEST(NnDualPathModels, ProjectionInitializesPathsBeforeAddingAndAppending) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  const auto r = n::linspace(-1, 1, 100, type).reshape({1, 5, 5, 4}).detach().requires_grad_();
  const auto d =
      n::linspace(-.7, .8, 50, type).reshape({1, 5, 5, 2}).detach().requires_grad_();
  const n::models::DPNPaths x{r, d, 4, 2};
  {
    n::NoGradGuard guard;
    n::models::DPNBlock(p, state, "u", x, 6, 4, 1, 2, true, 2);
  }
  const auto identity = n::eye(6, type);
  p.set("u/projection/W",
        n::concat({identity, n::slice_axis(identity, 1, 0, 2)}, 1).reshape({1, 1, 6, 8}));
  p.set("u/projection/bn/gamma", n::full({6}, std::sqrt(1.001), type));
  p.set("u/conv3/W", n::zeros_like(p.at("u/conv3/W")));
  const auto out = n::models::DPNBlock(p, state, "u", x, 6, 4, 1, 2, true, 2);
  const auto sampled =
      n::slice_axis(n::slice_axis(n::relu(n::models::DPNJoin(x)), 1, 0, 5, 2), 2, 0, 5, 2);
  const auto extra = n::slice_axis(sampled, 3, 0, 2);
  const auto expected = n::concat({sampled, extra, n::zeros({1, 3, 3, 1}, type)}, 3);
  EXPECT_LT(n::max(n::abs(n::models::DPNJoin(out.output) - expected)).item(), 1e-12);
  EXPECT_EQ(out.input.residual_channels, 6);
  EXPECT_EQ(out.input.dense_channels, 2);
  EXPECT_EQ(out.output.dense_channels, 3);
  const auto actual = n::grad(n::sum(n::models::DPNJoin(out.output)), {r, d});
  const auto reference = n::grad(n::sum(expected), {r, d});
  for (size_t i = 0; i < 2; ++i)
    EXPECT_LT(n::max(n::abs(actual[i] - reference[i])).item(), 1e-12);
}
TEST(NnDualPathModels, CompactNinetyTwoLayerStageAlgebraCountAndGradients) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::Generator rng(29);
  n::models::DPNConfig cfg;
  cfg.classes = 10;
  cfg.width_divisor = 4;
  cfg.groups = 8;
  cfg.small_stem = true;
  cfg.update_stats = false;
  const auto x = n::normal({2, 16, 16, 3}, 0, .2, rng);
  const auto out = n::models::DPN(p, state, x, cfg, true);
  EXPECT_EQ(p.count(), 2474330);
  EXPECT_EQ(state.count(), 31304);
  const std::array<int, 4> widths{64, 128, 256, 512}, dense{20, 48, 132, 160};
  {
    n::NoGradGuard guard;
    for (size_t stage = 0; stage < 4; ++stage) {
      auto expected_residual = out.blocks[stage][0].input.residual;
      std::vector<n::Tensor> parts{out.blocks[stage][0].input.dense};
      for (const auto &block : out.blocks[stage]) {
        expected_residual = expected_residual + block.residual_update;
        parts.push_back(block.dense_update);
      }
      EXPECT_EQ(n::max(n::abs(out.stages[stage].residual - expected_residual)).item(), 0);
      EXPECT_EQ(n::max(n::abs(out.stages[stage].dense - n::concat(parts, 3))).item(), 0);
      EXPECT_EQ(out.stages[stage].residual_channels, widths[stage]);
      EXPECT_EQ(out.stages[stage].dense_channels, dense[stage]);
    }
  }
  const auto saved = state.flatten();
  for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  n::NoGradGuard guard;
  n::models::DPN(p, state, x, cfg, false);
  EXPECT_EQ(state.flatten(), saved);
}
TEST(NnDualPathModels, LayoutsAndCardinalityMetadataAreValidated) {
  for (int depth : {68, 92, 98, 107, 131}) {
    const auto layout = n::models::DPNLayoutFor(depth);
    EXPECT_EQ(3 * std::accumulate(layout.blocks.begin(), layout.blocks.end(), 0) + 2, depth);
  }
  n::models::DPNConfig cfg;
  cfg.width_divisor = 4;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  cfg.groups = 8;
  EXPECT_NO_THROW(cfg.Validate());
  cfg.depth = 68;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  EXPECT_THROW(n::models::DPNJoin({n::ones({1, 3, 3, 3}), {}, 4, 0}), std::invalid_argument);
  EXPECT_THROW(n::models::DPNChannelSlice(n::ones({1, 3, 3, 4}), 3, 2), std::invalid_argument);
  EXPECT_EQ(n::models::DPNChannelSlice(n::ones({1, 3, 3, 4}), 1, 0).size(3), 0);
  n::ParameterStore p(17), state(17);
  const n::models::DPNPaths x{n::ones({1, 3, 3, 4}), n::ones({1, 3, 3, 2}), 4, 2};
  EXPECT_THROW(n::models::DPNBlock(p, state, "bad", x, 4, 3, 1, 2, false),
               std::invalid_argument);
}
