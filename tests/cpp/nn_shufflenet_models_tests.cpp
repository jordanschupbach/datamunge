
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnShuffleNetModels, ReleasePaperPermutationsAndWeightedInverseDerivatives) {
  const auto type = n::DType::Float64;
  const auto x = n::arange(0, 12).to(type).reshape({1, 1, 1, 12}).detach().requires_grad_();
  const auto release = n::models::ChannelShuffle(x, 12, 3),
             paper = n::models::PaperChannelShuffle(x, 12, 3);
  EXPECT_EQ(n::max(n::abs(n::models::ChannelShuffle(paper, 12, 3) - x)).item(), 0);
  const auto weights = n::arange(1, 13).to(type).reshape(x.shape()),
             dx = n::grad(n::sum(release * weights), {x})[0];
  for (int c = 0; c < 12; ++c) {
    EXPECT_EQ(release.at({0, 0, 0, (c % 3) * 4 + c / 3}), c);
    EXPECT_EQ(paper.at({0, 0, 0, (c % 4) * 3 + c / 4}), c);
    EXPECT_EQ(dx.at({0, 0, 0, c}), weights.at({0, 0, 0, (c % 3) * 4 + c / 3}));
  }
}
TEST(NnShuffleNetModels, ShuffleConnectsEveryGroupInControlledLinearGraph) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const auto x = n::arange(0, 12).to(type).reshape({1, 1, 1, 12}).detach().requires_grad_();
  n::ConvLayerOptions opts;
  opts.bias = false;
  auto build = [&](bool shuffle) {
    auto h = n::models::GroupedConv2D(p, "a", x, 12, 1, 3, opts);
    if (shuffle)
      h = n::models::ChannelShuffle(h, 12, 3);
    return n::models::GroupedConv2D(p, "b", h, 12, 1, 3, opts);
  };
  {
    n::NoGradGuard guard;
    build(false);
  }
  for (const auto &name : p.names())
    p.set(name, n::ones_like(p.at(name)));
  for (bool shuffle : {false, true}) {
    const auto dx = n::grad(n::sum(n::slice_axis(build(shuffle), 3, 0, 1)), {x})[0];
    for (int ch = 0; ch < 12; ++ch)
      EXPECT_EQ(dx.at({0, 0, 0, ch}), shuffle ? (ch < 4 ? 2 : 1) : (ch < 4 ? 4 : 0));
  }
}
TEST(NnShuffleNetModels, EvenChannelCarryAndStridedPaddedAverage) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  n::models::ShuffleContext c{p, state, false};
  const auto x = n::linspace(-1, 1, 288, type).reshape({1, 6, 6, 8}).detach().requires_grad_();
  const auto out = n::models::ShuffleV2Unit(c, "v2", x, 8, 1),
             carry = n::slice_axis(out, 3, 0, 4);
  EXPECT_EQ(n::max(n::abs(carry - n::slice_axis(x, 3, 0, 8, 2))).item(), 0);
  const auto dx = n::grad(n::sum(carry), {x})[0];
  for (int ch = 0; ch < 8; ++ch)
    EXPECT_EQ(n::max(n::abs(n::slice_axis(dx, 3, ch, ch + 1) - (ch % 2 == 0 ? 1 : 0))).item(),
              0);
  const auto pooled = n::models::AveragePoolWithPadding(n::ones({1, 6, 6, 8}, type));
  EXPECT_NEAR(pooled.at({0, 0, 0, 0}), 4. / 9, 1e-12);
  EXPECT_EQ(pooled.at({0, 1, 1, 0}), 1);
  const auto v1 =
      n::models::ShuffleV1Unit(c, "v1", n::ones({1, 6, 6, 8}, type), 32, 2, true, 2);
  EXPECT_EQ(v1.shape(), n::Shape({1, 3, 3, 32}));
  EXPECT_EQ(n::max(n::abs(n::slice_axis(v1, 3, 0, 8) - pooled)).item(), 0);
  EXPECT_EQ(n::models::ShuffleV2Unit(c, "down", x, 16, 2).shape(), n::Shape({1, 3, 3, 16}));
}
TEST(NnShuffleNetModels, ReleasedCountsStageShapesAndMeasuredMacs) {
  n::set_num_threads(2);
  n::NoGradGuard guard;
  for (auto [version, groups, size, count, moving, expected] :
       std::vector<std::tuple<int, int, std::string, int64_t, int64_t, int64_t>>{
           {1, 3, "1.0x", 1864728, 24480, 137460672},
           {1, 8, "0.5x", 1010448, 19584, 40469760},
           {2, 1, "0.5x", 1365792, 7952, 40476448},
           {2, 1, "1.0x", 2278760, 16180, 145814296}}) {
    n::ParameterStore p(17), state(17);
    n::models::ShuffleNetConfig cfg;
    cfg.version = version;
    cfg.groups = groups;
    cfg.size = size;
    const auto out = n::models::ShuffleNet(p, state, n::zeros({1, 224, 224, 3}), cfg);
    int64_t macs = out.classifier_macs;
    for (const auto &[shape, cost] : out.layers)
      macs += shape[1] * shape[2] * cost;
    EXPECT_EQ(p.count(), count);
    EXPECT_EQ(state.count(), moving);
    EXPECT_EQ(macs, expected);
    const auto widths = cfg.Widths();
    for (int i = 0; i < 3; ++i)
      EXPECT_EQ(out.stages[i].shape(),
                n::Shape({1, 28 / (1 << i), 28 / (1 << i), widths[i + 1]}));
  }
}
TEST(NnShuffleNetModels, CompleteCifarGradientsAndMovingMomentSemantics) {
  n::set_num_threads(2);
  for (int version : {1, 2}) {
    n::ParameterStore p(17), state(17);
    n::Generator gen(29);
    const auto x = n::normal({2, 32, 32, 3}, 0, .2, gen);
    n::models::ShuffleNetConfig cfg;
    cfg.version = version;
    cfg.classes = 10;
    cfg.small_stem = true;
    cfg.update_stats = false;
    const auto out = n::models::ShuffleNet(p, state, x, cfg, true);
    EXPECT_EQ(p.count(), version == 1 ? 914328 : 1265000);
    EXPECT_EQ(state.count(), version == 1 ? 24480 : 16180);
    for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    const auto saved = state.flatten();
    n::NoGradGuard guard;
    n::models::ShuffleNet(p, state, x, cfg);
    EXPECT_EQ(saved, state.flatten());
    cfg.update_stats = true;
    n::models::ShuffleNet(p, state, x, cfg, true);
    EXPECT_NE(saved, state.flatten());
  }
}
TEST(NnShuffleNetModels, RejectsInvalidChannelsGroupsUnitsAndLayouts) {
  EXPECT_THROW(n::models::ChannelShuffle(n::ones({1, 2, 2, 8}), 12, 3), std::invalid_argument);
  EXPECT_THROW(n::models::ChannelShuffle(n::ones({1, 2, 2, 8}), 8, 0), std::invalid_argument);
  EXPECT_THROW(n::models::ReshapeChannels(n::ones({1, 2, 3}), {3}), std::invalid_argument);
  n::ParameterStore p(17), state(17);
  n::models::ShuffleContext c{p, state, false};
  EXPECT_THROW(n::models::ShuffleV1Unit(c, "v1", n::ones({1, 4, 4, 8}), 32, 0, true, 2),
               std::invalid_argument);
  EXPECT_THROW(n::models::ShuffleV2Unit(c, "v2", n::ones({1, 4, 4, 8}), 16, 3),
               std::invalid_argument);
  n::models::ShuffleNetConfig cfg;
  cfg.version = 1;
  cfg.groups = 2;
  EXPECT_THROW(cfg.Widths(), std::invalid_argument);
  cfg = {};
  cfg.size = "invalid";
  EXPECT_THROW(cfg.Widths(), std::invalid_argument);
}
