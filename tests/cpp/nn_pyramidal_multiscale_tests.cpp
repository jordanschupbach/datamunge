
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnPyramidalMultiscale, ZeroBranchesPreservePooledSignedShortcutAndWeightedDerivatives) {
  const auto type = n::DType::Float64;
  for (bool bottle : {false, true})
    for (int stride : {1, 2}) {
      n::ParameterStore p(17, type), state(17, type);
      const auto x =
          n::linspace(-1, 1, 108, type).reshape({1, 6, 6, 3}).detach().requires_grad_();
      const int planes = bottle ? 2 : 5, out = bottle ? 8 : 5, side = 6 / stride;
      {
        n::NoGradGuard guard;
        n::models::PyramidBlock(p, state, "b", x, planes, bottle, stride);
      }
      const std::string last = bottle ? "b/bn4/" : "b/bn3/";
      p.set(last + "gamma", n::zeros({out}, type));
      p.set(last + "beta", n::zeros({out}, type));
      const auto y = n::models::PyramidBlock(p, state, "b", x, planes, bottle, stride).output,
                 weights =
                     n::linspace(.2, 1, side * side * out, type).reshape({1, side, side, out});
      const auto dx = n::grad(n::sum(y * weights), {x})[0];
      auto expected = n::zeros({1, side, side, 3}, type);
      for (int r = 0; r < stride; ++r)
        for (int c = 0; c < stride; ++c)
          expected =
              expected + n::slice_axis(n::slice_axis(x, 1, r, 6, stride), 2, c, 6, stride) /
                             double(stride * stride);
      expected = n::concat({expected, n::zeros({1, side, side, out - 3}, type)}, 3);
      EXPECT_LT(n::max(n::abs(y - expected)).item(), 1e-12);
      EXPECT_LT(n::min(y).item(), 0);
      const auto reference = n::grad(n::sum(expected * weights), {x})[0];
      EXPECT_LT(n::max(n::abs(dx - reference)).item(), 1e-12);
    }
}
TEST(NnPyramidalMultiscale, PyramidSchedulesGeometryAndFrozenMovingState) {
  n::set_num_threads(2);
  n::models::PyramidNetConfig cfg;
  auto widths = cfg.UnitWidths();
  ASSERT_EQ(widths.size(), 54);
  EXPECT_EQ(widths[0], 17);
  EXPECT_EQ(widths[1], 18);
  EXPECT_EQ(widths.back(), 64);
  cfg.depth = 272;
  cfg.alpha = 200;
  cfg.bottleneck = true;
  EXPECT_EQ(cfg.UnitWidths().back(), 216);
  cfg = {};
  cfg.depth = 8;
  cfg.alpha = 3;
  cfg.update_stats = false;
  n::ParameterStore p(17), state(17);
  n::Generator rng(29);
  const auto x = n::normal({2, 8, 8, 3}, 0, .2, rng);
  const auto y = n::models::PyramidNet(p, state, x, cfg, true);
  EXPECT_EQ(y.channels, std::vector<int64_t>({17, 18, 19}));
  EXPECT_EQ(y.stages[0].shape(), n::Shape({2, 8, 8, 17}));
  EXPECT_EQ(y.stages[2].shape(), n::Shape({2, 2, 2, 19}));
  for (const auto &g : n::grad(n::mean(n::square(y.logits)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  const auto saved = state.flatten();
  n::NoGradGuard guard;
  n::models::PyramidNet(p, state, x, cfg);
  EXPECT_EQ(saved, state.flatten());
  cfg.update_stats = true;
  n::models::PyramidNet(p, state, x, cfg, true);
  EXPECT_NE(saved, state.flatten());
}
TEST(NnPyramidalMultiscale, PyramidRejectsInvalidSchedulesAndShortcutGeometry) {
  n::models::PyramidNetConfig cfg;
  cfg.alpha = std::numeric_limits<double>::infinity();
  EXPECT_THROW(cfg.UnitWidths(), std::invalid_argument);
  cfg = {};
  cfg.alpha = -1;
  EXPECT_THROW(cfg.UnitWidths(), std::invalid_argument);
  cfg = {};
  cfg.imagenet = true;
  EXPECT_THROW(cfg.StageCounts(), std::invalid_argument);
  EXPECT_THROW(n::models::PyramidShortcut(n::ones({1, 5, 5, 3}), 5, 2), std::invalid_argument);
  EXPECT_THROW(n::models::PyramidShortcut(n::ones({1, 4, 4, 3}), 2, 1), std::invalid_argument);
  EXPECT_THROW(n::models::PyramidShortcut(n::ones({1, 4}), 4, 1), std::invalid_argument);
}
TEST(NnPyramidalMultiscale, Res2HierarchyStageIndependenceAndBypassPooling) {
  const auto type = n::DType::Float64;
  for (bool stage : {false, true})
    for (int stride : {1, 2}) {
      if (!stage && stride == 2)
        continue;
      n::ParameterStore p(17, type), state(17, type);
      n::models::ResNetNormConfig cfg;
      cfg.scale = true;
      cfg.epsilon = 1e-5;
      const auto x =
          (n::ones({1, 11, 11, 4}, type) * n::arange(1, 5).to(type)).detach().requires_grad_();
      {
        n::NoGradGuard guard;
        n::models::Res2ScaleTransform(p, state, "s", x, 1, 4, cfg, stage, stride);
      }
      for (int i = 0; i < 3; ++i) {
        auto name = "s/group" + std::to_string(i);
        p.set(name + "/W", n::full({3, 3, 1, 1}, 1. / 9, type));
        p.set(name + "/bn/gamma", n::full({1}, std::sqrt(1.00001), type));
      }
      const auto out =
          n::models::Res2ScaleTransform(p, state, "s", x, 1, 4, cfg, stage, stride);
      const int side = (11 + stride - 1) / stride;
      EXPECT_EQ(out.mixed.shape(), n::Shape({1, side, side, 4}));
      EXPECT_NEAR(out.groups.back().at({0, 0, 0, 0}), stage ? 16. / 9 : 4, 1e-12);
      for (int g = 0; g < 4; ++g) {
        const auto centre = n::slice(out.groups[g], {0, side / 2, side / 2, 0}, {1, 1, 1, 1});
        const auto dx = n::grad(n::sum(centre), {x}, {.retain_graph = true})[0];
        for (int ch = 0; ch < 4; ++ch) {
          const auto map = n::slice_axis(dx, 3, ch, ch + 1);
          int diameter = 0;
          if (stage)
            diameter = ch == g ? 3 : 0;
          else if (g == 3)
            diameter = ch == 3 ? 1 : 0;
          else if (ch <= g)
            diameter = 2 * (g - ch + 1) + 1;
          EXPECT_EQ(n::sum(n::greater(n::abs(map), 1e-12).to(type)).item(),
                    diameter * diameter);
          EXPECT_NEAR(n::sum(map).item(), diameter ? 1 : 0, 1e-12);
        }
      }
    }
}
TEST(NnPyramidalMultiscale, CompleteCompactRes2NetCountsGradientsAndState) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::Generator rng(29);
  n::models::Res2NetConfig cfg;
  cfg.stage_width = 16;
  cfg.small_stem = true;
  cfg.classes = 10;
  cfg.update_stats = false;
  const auto x = n::normal({2, 16, 16, 3}, 0, .2, rng);
  const auto y = n::models::Res2Net(p, state, x, cfg, true);
  EXPECT_EQ(p.count(), 1493062);
  EXPECT_EQ(state.count(), 14852);
  EXPECT_EQ(y.blocks.size(), 16);
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(y.stages[i].shape(), n::Shape({2, 16 / (1 << i), 16 / (1 << i), 64 << i}));
  for (const auto &g : n::grad(n::mean(n::square(y.logits)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  const auto saved = state.flatten();
  n::NoGradGuard guard;
  n::models::Res2Net(p, state, x, cfg);
  EXPECT_EQ(saved, state.flatten());
  cfg.update_stats = true;
  n::models::Res2Net(p, state, x, cfg, true);
  EXPECT_NE(saved, state.flatten());
}
TEST(NnPyramidalMultiscale, Res2RejectsInvalidScaleStrideAndLayouts) {
  n::models::Res2NetConfig cfg;
  cfg.depth = 18;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  cfg = {};
  cfg.scale = 0;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  cfg = {};
  cfg.stage_width = 1;
  cfg.base_width = 1;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  n::ParameterStore p(17), state(17);
  n::models::ResNetNormConfig norm;
  EXPECT_THROW(n::models::Res2ScaleTransform(p, state, "s", n::ones({1, 5, 5, 4}), 1, 4, norm,
                                             false, 2),
               std::invalid_argument);
  EXPECT_THROW(n::models::Res2ScaleTransform(p, state, "s", n::ones({1, 5, 5, 3}), 1, 4, norm),
               std::invalid_argument);
  EXPECT_THROW(n::models::Res2ScaleTransform(p, state, "s", n::ones({1, 4}), 1, 4, norm),
               std::invalid_argument);
}
