
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnGhostNetModels, CeilingTrimOrderAndDiscardedChannelGradientsForTwoRatios) {
  const auto type = n::DType::Float64;
  for (int ratio : {2, 3}) {
    const int out = ratio == 2 ? 5 : 7;
    n::ParameterStore p(17, type), state(17, type);
    std::vector<n::models::MobileLayer> layers;
    n::models::MobileContext c{p, state, layers, false};
    const auto x =
        n::linspace(-1, 1, 75, type).reshape({1, 5, 5, 3}).detach().requires_grad_();
    {
      n::NoGradGuard guard;
      n::models::GhostModule(c, "g", x, out, false, ratio);
    }
    p.set("g/primary/W", n::eye(3, type).reshape({1, 1, 3, 3}));
    const auto delta = n::one_hot(n::from_vector(std::vector<int64_t>{4}), 9, 1, 0, type)
                           .reshape({3, 3, 1, 1});
    const auto gains =
        n::from_vector(ratio == 2 ? std::vector<double>{2} : std::vector<double>{2, 3})
            .reshape({1, 1, 1, ratio - 1});
    p.set("g/cheap/W", delta * n::ones({1, 1, 3, ratio - 1}, type) * gains);
    p.set("g/primary/bn/gamma", n::full({3}, std::sqrt(1.00001), type));
    p.set("g/cheap/bn/gamma", n::full({3 * (ratio - 1)}, std::sqrt(1.00001), type));
    const auto y = n::models::GhostModule(c, "g", x, out, false, ratio);
    const auto cheap = ratio == 2 ? 2 * x : n::stack({2 * x, 3 * x}, 4).reshape({1, 5, 5, 6});
    const auto expected = n::slice_axis(n::concat({x, cheap}, 3), 3, 0, out);
    EXPECT_LT(n::max(n::abs(y.intrinsic - x)).item(), 1e-12);
    EXPECT_LT(n::max(n::abs(y.ghost - cheap)).item(), 1e-12);
    EXPECT_LT(n::max(n::abs(y.output - expected)).item(), 1e-12);
    const auto gradients = n::grad(n::sum(y.output), {x, p.at("g/cheap/W")});
    const auto target = n::from_vector(ratio == 2 ? std::vector<double>{3, 3, 1}
                                                  : std::vector<double>{6, 6, 1});
    EXPECT_LT(n::max(n::abs(gradients[0] - target)).item(), 1e-12);
    EXPECT_EQ(n::max(n::abs(n::slice_axis(gradients[1], 2, 2, 3))).item(), 0);
  }
}
TEST(NnGhostNetModels, ZeroLinearGhostBranchRetainsSignedShortcutAndUnitDerivative) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  std::vector<n::models::MobileLayer> layers;
  n::models::MobileContext c{p, state, layers, false};
  const auto x = n::linspace(-1, 1, 200, type).reshape({1, 5, 5, 8}).detach().requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::GhostBottleneck(c, "b", x, 16, 8, 3, 1, .25);
  }
  for (const auto *part : {"primary", "cheap"})
    for (const auto *leaf : {"gamma", "beta"}) {
      const auto name = std::string("b/ghost2/") + part + "/bn/" + leaf;
      p.set(name, n::zeros_like(p.at(name)));
    }
  const auto y = n::models::GhostBottleneck(c, "b", x, 16, 8, 3, 1, .25);
  EXPECT_EQ(n::max(n::abs(y - x)).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::grad(n::sum(y), {x})[0] - 1)).item(), 0);
}
TEST(NnGhostNetModels, ReleasedWidthsCountsFeaturesAndMeasuredMacs) {
  n::set_num_threads(2);
  n::NoGradGuard guard;
  for (auto [width, count, moving, macs] :
       std::vector<std::tuple<double, int64_t, int64_t, int64_t>>{
           {.5, 2587292, 10984, 42328152},
           {1, 5182508, 21968, 141150512},
           {1.3, 7358820, 28536, 227003256}}) {
    n::ParameterStore p(17), state(17);
    n::models::GhostNetConfig cfg;
    cfg.width = width;
    const auto y = n::models::GhostNet(p, state, n::zeros({1, 224, 224, 3}), cfg);
    EXPECT_EQ(p.count(), count);
    EXPECT_EQ(state.count(), moving);
    EXPECT_EQ(y.Macs(), macs);
    EXPECT_EQ(y.features.shape(),
              n::Shape({1, 7, 7, n::models::MakeDivisible(960 * width, 4)}));
    EXPECT_EQ(y.logits.shape(), n::Shape({1, 1000}));
  }
  EXPECT_EQ(n::models::GhostNetBlocks().size(), 16);
}
TEST(NnGhostNetModels, CompleteCifarParameterGradientsAndMovingStateSemantics) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::Generator gen(29);
  n::models::GhostNetConfig cfg;
  cfg.small_stem = true;
  cfg.classes = 10;
  cfg.keep_probability = 1;
  cfg.update_stats = false;
  const auto x = n::normal({2, 32, 32, 3}, 0, .2, gen),
             y = n::models::GhostNet(p, state, x, cfg, true).logits;
  EXPECT_EQ(p.count(), 3914318);
  EXPECT_EQ(state.count(), 21968);
  for (const auto &g : n::grad(n::mean(n::square(y)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  const auto saved = state.flatten();
  n::NoGradGuard guard;
  n::models::GhostNet(p, state, x, cfg);
  EXPECT_EQ(saved, state.flatten());
  cfg.update_stats = true;
  n::models::GhostNet(p, state, x, cfg, true);
  EXPECT_NE(saved, state.flatten());
}
TEST(NnGhostNetModels, RejectsInvalidRatiosDepthMultipliersShapesAndSettings) {
  n::ParameterStore p(17), state(17);
  std::vector<n::models::MobileLayer> layers;
  n::models::MobileContext c{p, state, layers, false};
  EXPECT_THROW(n::models::GhostModule(c, "g", n::ones({1, 5, 5, 3}), 5, false, 0),
               std::invalid_argument);
  EXPECT_THROW(n::models::GhostModule(c, "g", n::ones({1, 5, 5, 3}), 5, false, 1),
               std::invalid_argument);
  EXPECT_THROW(n::models::GhostConvBN(c, "d", n::ones({1, 5, 5, 3}), 4, 3, 1, true, false),
               std::invalid_argument);
  EXPECT_THROW(n::models::GhostBottleneck(c, "b", n::ones({1, 5, 5, 3}), 8, 8, 3, 1, 2),
               std::invalid_argument);
  n::models::GhostNetConfig cfg;
  cfg.width = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(n::models::GhostNet(p, state, n::ones({1, 32, 32, 3}), cfg),
               std::invalid_argument);
}
