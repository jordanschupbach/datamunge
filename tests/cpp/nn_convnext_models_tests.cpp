
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnConvNeXtModels, AnalyticalCountsMatchAllReleasedPresetsAndActualLeaves) {
  for (auto [name, v2, count] :
       std::vector<std::tuple<std::string, bool, int64_t>>{{"tiny", false, 28589128},
                                                           {"small", false, 50223688},
                                                           {"base", false, 88591464},
                                                           {"large", false, 197767336},
                                                           {"atto", true, 3708400},
                                                           {"femto", true, 5233240},
                                                           {"pico", true, 9066280},
                                                           {"nano", true, 15623800},
                                                           {"tiny", true, 28635496},
                                                           {"base", true, 88717800},
                                                           {"large", true, 197956840},
                                                           {"huge", true, 660289640}})
    EXPECT_EQ(n::models::ConvNeXtParameters(n::models::ConvNeXtLayout(name), 1000, 4, v2),
              count);
  for (bool v2 : {false, true}) {
    n::NoGradGuard guard;
    n::ParameterStore p(17);
    n::models::ConvNeXtConfig cfg;
    cfg.size = "atto";
    cfg.v2 = v2;
    const auto out = n::models::ConvNeXt(p, n::zeros({1, 32, 32, 3}), cfg);
    EXPECT_EQ(p.count(),
              n::models::ConvNeXtParameters(n::models::ConvNeXtLayout("atto"), 1000, 4, v2));
    EXPECT_EQ(out.stages.size(), 4);
    EXPECT_EQ(out.features.shape(), n::Shape({1, 1, 1, 320}));
  }
}
TEST(NnConvNeXtModels, GrnInitializationAndDeadChannelsHaveFiniteIdentityDerivatives) {
  const auto type = n::DType::Float64;
  for (bool zero : {false, true}) {
    n::ParameterStore p(17, type);
    const auto x =
        (zero ? n::zeros({2, 3, 4, 8}, type)
              : n::concat({n::zeros({2, 3, 4, 1}, type), n::ones({2, 3, 4, 7}, type)}, 3))
            .detach()
            .requires_grad_();
    const auto out = n::models::GlobalResponseNorm(p, "g", x);
    EXPECT_EQ(out.output.dtype(), type);
    EXPECT_EQ(n::max(n::abs(out.output - x)).item(), 0);
    const auto dx = n::grad(n::sum(out.output), {x})[0];
    EXPECT_EQ(n::max(n::abs(dx - 1)).item(), 0);
    EXPECT_TRUE(std::isfinite(n::sum(out.response).item()));
  }
}
TEST(NnConvNeXtModels, ActiveGrnMatchesFormulaAndFiniteDifferences) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const auto x = n::linspace(-.7, .9, 2 * 3 * 4 * 8, type)
                     .reshape({2, 3, 4, 8})
                     .detach()
                     .requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::GlobalResponseNorm(p, "g", x);
    p.set("g/gamma", n::linspace(.1, .3, 8, type));
    p.set("g/beta", n::linspace(-.1, .1, 8, type));
  }
  const auto out = n::models::GlobalResponseNorm(p, "g", x);
  const auto norm = n::sqrt(n::sum(n::square(x), {1, 2}, true));
  const auto ref =
      p.at("g/gamma") * x * norm / (n::mean(norm, {3}, true) + 1e-6) + p.at("g/beta") + x;
  EXPECT_LT(n::max(n::abs(ref - out.output)).item(), 1e-12);
  const auto dx = n::grad(n::sum(n::square(out.output)), {x})[0];
  const auto direction =
      n::one_hot(n::tensor({48}).to(n::DType::Int64), 192, 1, 0, type).reshape(x.shape());
  const double h = 1e-6;
  n::NoGradGuard guard;
  const double
      a = n::sum(n::square(
                     n::models::GlobalResponseNorm(p, "g", x.detach() + h * direction).output))
              .item(),
      b = n::sum(n::square(
                     n::models::GlobalResponseNorm(p, "g", x.detach() - h * direction).output))
              .item();
  EXPECT_NEAR(dx.reshape({192}).at({48}), (a - b) / (2 * h), 1e-7);
}
TEST(NnConvNeXtModels, LayerScaleAndDepthwiseSupportRetainSignedShortcuts) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const auto x = n::sin(.17 * n::arange(0, 2 * 9 * 9 * 8).to(type))
                     .reshape({2, 9, 9, 8})
                     .detach()
                     .requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::ConvNeXtBlock(p, "b", x, false, false);
    p.set("b/layer_scale", n::zeros({8}, type));
  }
  const auto out = n::models::ConvNeXtBlock(p, "b", x, false, false);
  EXPECT_EQ(n::max(n::abs(out.output - x)).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::grad(n::sum(out.output), {x})[0] - 1)).item(), 0);
  p.set("b/layer_scale", n::ones({8}, type));
  const auto fresh = n::models::ConvNeXtBlock(p, "b", x, false, false);
  const auto picked = n::slice(fresh.output, {0, 4, 4, 0}, {1, 1, 1, 8});
  const auto derivative = n::grad(n::sum(picked), {x})[0];
  const auto support =
      n::greater(n::sum(n::abs(n::slice_axis(derivative, 0, 0, 1)), {0, 3}), 1e-15);
  EXPECT_EQ(n::count_nonzero(support).item(), 49);
}
TEST(NnConvNeXtModels, CompleteCifarBackwardAndRectangularMacGeometry) {
  n::set_num_threads(2);
  for (bool v2 : {false, true}) {
    n::ParameterStore p(17);
    n::models::ConvNeXtConfig cfg;
    cfg.size = "atto";
    cfg.cifar_stem = true;
    cfg.classes = 10;
    cfg.v2 = v2;
    cfg.drop_path = .1;
    const auto x = n::sin(.37 * n::arange(0, 2 * 32 * 32 * 3).to(n::DType::Float32))
                       .reshape({2, 32, 32, 3});
    const auto out = n::models::ConvNeXt(p, x, cfg, true);
    EXPECT_EQ(p.count(),
              n::models::ConvNeXtParameters(n::models::ConvNeXtLayout("atto"), 10, 2, v2));
    EXPECT_EQ(out.features.shape(), n::Shape({2, 2, 2, 320}));
    for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    n::NoGradGuard guard;
    const auto rect = n::models::ConvNeXt(p, n::zeros({1, 32, 48, 3}), cfg),
               square = n::models::ConvNeXt(p, n::zeros({1, 32, 32, 3}), cfg);
    const auto layout = n::models::ConvNeXtLayout("atto");
    const int64_t classifier = 3200;
    EXPECT_EQ((n::models::ConvNeXtMacs(rect, layout, 10, 2) - classifier) * 2,
              (n::models::ConvNeXtMacs(square, layout, 10, 2) - classifier) * 3);
  }
}
TEST(NnConvNeXtModels, InvalidLayoutsAndCountOverflowAreRejected) {
  EXPECT_THROW(n::models::ConvNeXtLayout("unknown"), std::invalid_argument);
  EXPECT_THROW(n::models::ConvNeXtParameters(
                   {{1, 1, 1, 1}, {2147483647, 2147483647, 2147483647, 2147483647}}),
               std::invalid_argument);
  n::ParameterStore p(17);
  EXPECT_THROW(n::models::ConvNeXtBlock(p, "b", n::ones({1, 3, 3, 4}), false, true, 1),
               std::invalid_argument);
  EXPECT_THROW(n::models::ConvNeXt(p, n::ones({1, 16, 16, 3})), std::invalid_argument);
}
