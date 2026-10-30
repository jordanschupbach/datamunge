
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnCondenseNetModels, TensorRanksUseStableTiesAndCatchUpSkippedStages) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    n::ParameterStore p(17, type), state(17, type);
    std::vector<n::models::MobileLayer> layers;
    n::models::CondenseContext c{p, state, layers, false};
    {
      n::NoGradGuard guard;
      n::models::LearnedGroupConv(c, "g", n::ones({2, 5, 5, 16}, type), 16, 4, 4);
    }
    p.set("g/W", n::ones({1, 1, 16, 16}, type));
    n::models::UpdateCondensation(p, state, "g", 4, 4, .5);
    EXPECT_EQ(state.at("g/stage").item(), 3);
    EXPECT_EQ(state.at("g/mask").dtype(), type);
    for (int i = 0; i < 16; ++i)
      for (int o = 0; o < 16; ++o)
        EXPECT_EQ(state.at("g/mask").at({i, o}), i >= 12 ? 1 : 0);
    const auto saved = state.flatten();
    n::models::UpdateCondensation(p, state, "g", 4, 4, .1);
    EXPECT_EQ(state.flatten(), saved);
  }
}
TEST(NnCondenseNetModels, ForwardUsesOldMaskAndLassoWhileStateAdvancesForNextCall) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  std::vector<n::models::MobileLayer> layers;
  std::vector<n::Tensor> penalties;
  n::models::CondenseContext c{p, state, layers, false, .9, 0, &penalties, .01, true};
  const auto x = n::sin(.17 * n::arange(0, 2 * 5 * 5 * 16).to(type))
                     .reshape({2, 5, 5, 16})
                     .detach()
                     .requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::LearnedGroupConv(c, "g", x, 16, 4, 4);
  }
  p.set("g/W", n::ones({1, 1, 16, 16}, type));
  c.training = true;
  c.progress = .5;
  const auto old = n::models::LearnedGroupConv(c, "g", x, 16, 4, 4);
  ASSERT_EQ(penalties.size(), 1);
  EXPECT_EQ(n::sum(old.mask).item(), 256);
  EXPECT_EQ(n::sum(state.at("g/mask")).item(), 64);
  EXPECT_GT(penalties[0].item(), 0);
  c.update_stats = false;
  penalties.clear();
  const auto next = n::models::LearnedGroupConv(c, "g", x, 16, 4, 4);
  EXPECT_TRUE(penalties.empty());
  EXPECT_GT(n::sum(n::abs(old.output - next.output)).item(), 0);
  for (const auto &g : n::grad(n::sum(old.output), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
}
TEST(NnCondenseNetModels, PackedFloat64ConversionPreservesValuesAndAllDerivatives) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  std::vector<n::models::MobileLayer> layers;
  n::models::CondenseContext c{p, state, layers, false};
  const auto x = n::sin(.17 * n::arange(0, 2 * 5 * 5 * 16).to(type) + .3)
                     .reshape({2, 5, 5, 16})
                     .detach()
                     .requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::LearnedGroupConv(c, "g", x, 16, 4, 4);
  }
  for (double progress : {.17, .34, .5})
    n::models::UpdateCondensation(p, state, "g", 4, 4, progress);
  state.set("g/bn/moving_mean", n::linspace(-.2, .2, 16, type));
  state.set("g/bn/moving_var", n::linspace(.7, 1.2, 16, type));
  p.set("g/bn/gamma", n::linspace(.8, 1.1, 16, type));
  p.set("g/bn/beta", n::linspace(-.1, .1, 16, type));
  const auto masked = n::models::LearnedGroupConv(c, "g", x, 16, 4, 4).output;
  auto vars = p.parameters();
  vars.insert(vars.begin(), x);
  const auto expected = n::grad(n::mean(n::square(masked)), vars);
  const auto packed = n::models::Condense(p.at("g/W"), state.at("g/mask"), 4);
  EXPECT_EQ(packed.weight.dtype(), type);
  EXPECT_EQ(packed.weight.shape(), n::Shape({1, 1, 4, 16}));
  ASSERT_EQ(packed.index.size(), 16);
  const auto indices = n::from_vector(packed.index);
  auto pick = [&](const n::ParameterStore &store, const std::string &leaf) {
    return n::gather(store.at("g/bn/" + leaf), indices, 0);
  };
  const auto normalized =
      n::relu((n::gather(x, indices, 3) - pick(state, "moving_mean")) *
                  n::rsqrt(pick(state, "moving_var") + 1e-5) * pick(p, "gamma") +
              pick(p, "beta"));
  auto opts = n::conv_options(1, n::Padding::Valid);
  opts.groups = 4;
  const auto converted =
      n::models::PaperChannelShuffle(n::conv2d(normalized, packed.weight, opts), 16, 4);
  EXPECT_LT(n::max(n::abs(converted - masked)).item(), 1e-11);
  const auto actual = n::grad(n::mean(n::square(converted)), vars);
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_LT(n::max(n::abs(actual[i] - expected[i])).item(), 1e-10);
}
TEST(NnCondenseNetModels, FullCifarCountsGradientsAndExplicitConvertedCostAssumptions) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::models::CondenseNetConfig cfg;
  cfg.update_stats = false;
  n::Generator gen(29);
  const auto x = n::normal({2, 32, 32, 3}, 0, .2, gen);
  {
    n::NoGradGuard guard;
    const auto out = n::models::CondenseNet(p, state, x, cfg);
    EXPECT_EQ(out.channels, 800);
  }
  EXPECT_EQ(p.count(), 1451594);
  EXPECT_EQ(state.count(), 1273850);
  const auto out = n::models::CondenseNet(p, state, x, cfg, true, 100);
  EXPECT_EQ(out.penalties.size(), 42);
  EXPECT_EQ(state.at("condense/step").item(), 0);
  auto objective = n::mean(n::square(out.logits));
  for (const auto &penalty : out.penalties)
    objective = objective + penalty;
  for (const auto &g : n::grad(objective, p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  const auto half = n::models::ConvertedCondenseNetCost(cfg, 32),
             full = n::models::ConvertedCondenseNetCost(cfg, 32, false);
  EXPECT_EQ(half.parameters, 516202);
  EXPECT_EQ(half.macs, 62377888);
  EXPECT_EQ(half.features, 800);
  EXPECT_EQ(half.classifier_features, 400);
  EXPECT_EQ(full.parameters - half.parameters, 4000);
  for (int groups : {4, 8}) {
    n::models::CondenseNetConfig large;
    large.stages = {4, 6, 8, 10, 8};
    large.growth = {8, 16, 32, 64, 128};
    large.groups = large.condense = groups;
    large.classes = 1000;
    large.imagenet = true;
    const auto cost = n::models::ConvertedCondenseNetCost(large, 224);
    EXPECT_EQ(cost.parameters, groups == 4 ? 4773944 : 2935416);
    EXPECT_EQ(cost.macs, groups == 4 ? 516640576 : 261545792);
  }
}
TEST(NnCondenseNetModels, RejectsInconsistentMasksStagesGroupsAndLayouts) {
  n::ParameterStore p(17), state(17);
  std::vector<n::models::MobileLayer> layers;
  n::models::CondenseContext c{p, state, layers, false};
  EXPECT_THROW(n::models::LearnedGroupConv(c, "g", n::ones({2, 5, 5, 16}), 16, 0, 4),
               std::invalid_argument);
  {
    n::NoGradGuard guard;
    n::models::LearnedGroupConv(c, "g", n::ones({2, 5, 5, 16}), 16, 4, 4);
  }
  state.set("g/mask", n::full({16, 16}, .5));
  EXPECT_THROW(n::models::UpdateCondensation(p, state, "g", 4, 4, .5), std::invalid_argument);
  EXPECT_THROW(n::models::Condense(p.at("g/W"), state.at("g/mask"), 4), std::invalid_argument);
  n::models::CondenseNetConfig cfg;
  cfg.condense = 0;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  cfg = {};
  cfg.growth = {7, 16, 32};
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  EXPECT_THROW(n::models::CondenseNet(p, state, n::ones({1, 8, 8, 3}), {}, true, 0),
               std::invalid_argument);
}
