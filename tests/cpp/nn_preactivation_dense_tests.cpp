#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnPreactivationDense, FiftyZeroResidualUnitsPreserveSignedIdentityAndDerivative) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  const auto x = n::linspace(-1, 1, 100, type).reshape({1, 5, 5, 4}).detach().requires_grad_();
  auto build = [&](const n::Tensor &input) {
    auto h = input;
    for (int i = 0; i < 50; ++i)
      h = n::models::PreActBottleneck(p, state, "b" + std::to_string(i), h, 1).output;
    return h;
  };
  {
    n::NoGradGuard guard;
    build(x);
  }
  for (int i = 0; i < 50; ++i)
    p.set("b" + std::to_string(i) + "/conv3/W", n::zeros({1, 1, 1, 4}, type));
  const auto output = build(x);
  EXPECT_EQ(n::max(n::abs(output - x)).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::grad(n::sum(output), {x})[0] - 1)).item(), 0);
}
TEST(NnPreactivationDense,
     ProjectionConsumesPreactivationAndNonzeroBranchMatchesFiniteDifference) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  const auto x = n::linspace(-1, 1, 50, type).reshape({1, 5, 5, 2}).detach().requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::PreActBottleneck(p, state, "b", x, 1, 2);
  }
  p.set("b/conv3/W", n::zeros({1, 1, 1, 4}, type));
  p.set("b/shortcut/W", n::tensor({1., 0., 1., 0., 0., 1., 0., 1.}, {1, 1, 2, 4}).to(type));
  p.set("b/bn1/gamma", n::full({2}, std::sqrt(1.00001), type));
  const auto actual = n::models::PreActBottleneck(p, state, "b", x, 1, 2).output;
  const auto sampled = n::slice_axis(n::slice_axis(n::relu(x), 1, 0, 5, 2), 2, 0, 5, 2);
  const auto expected = n::concat({sampled, sampled}, 3);
  EXPECT_LT(n::max(n::abs(actual - expected)).item(), 1e-12);
  n::ParameterStore q(19, type), moments(19, type);
  const auto positive = n::full({1, 3, 3, 4}, .5, type).requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::PreActBottleneck(q, moments, "b", positive, 1);
  }
  for (const auto &name : q.names())
    if (name.ends_with("/W"))
      q.set(name, n::full(q.at(name).shape(), .2, type));
  const auto y = n::models::PreActBottleneck(q, moments, "b", positive, 1).output;
  const auto dx = n::grad(n::sum(y), {positive})[0];
  const auto direction = n::one_hot(n::tensor({16}).to(n::DType::Int64), 36, 1, 0, type)
                             .reshape(positive.shape());
  n::NoGradGuard guard;
  const double h = 1e-6;
  const double plus =
      n::sum(n::models::PreActBottleneck(q, moments, "b", positive.detach() + h * direction, 1)
                 .output)
          .item();
  const double minus =
      n::sum(n::models::PreActBottleneck(q, moments, "b", positive.detach() - h * direction, 1)
                 .output)
          .item();
  EXPECT_NEAR((plus - minus) / (2 * h), dx.at({0, 1, 1, 0}), 1e-7);
}
TEST(NnPreactivationDense, DenseLayersRetainOrderingAndDirectIdentityGradients) {
  const auto type = n::DType::Float64;
  for (bool bottleneck : {false, true}) {
    n::ParameterStore p(17, type), state(17, type);
    const auto x =
        n::linspace(-1, 1, 75, type).reshape({1, 5, 5, 3}).detach().requires_grad_();
    auto build = [&]() {
      auto h = x;
      for (int layer = 0; layer < 4; ++layer) {
        const auto old = h;
        const auto out = n::models::DenseGrowthLayer(p, state, "g" + std::to_string(layer), h,
                                                     2, bottleneck);
        EXPECT_EQ(out.output.size(3), 3 + 2 * (layer + 1));
        EXPECT_EQ(n::max(n::abs(n::slice_axis(out.output, 3, 0, old.size(3)) - old)).item(),
                  0);
        EXPECT_EQ(n::max(n::abs(n::slice_axis(out.output, 3, old.size(3), old.size(3) + 2) -
                                out.added))
                      .item(),
                  0);
        h = out.output;
      }
      return h;
    };
    const auto full = build();
    EXPECT_EQ(n::max(n::abs(n::grad(n::sum(n::slice_axis(full, 3, 0, 3)), {x})[0] - 1)).item(),
              0);
    for (const auto &name : p.names())
      if (name.ends_with("/conv2/W"))
        p.set(name, n::zeros_like(p.at(name)));
    const auto zero_added = build();
    EXPECT_EQ(n::max(n::abs(n::slice_axis(zero_added, 3, 3, 11))).item(), 0);
    EXPECT_EQ(n::max(n::abs(n::grad(n::sum(zero_added), {x})[0] - 1)).item(), 0);
  }
}
TEST(NnPreactivationDense, TransitionHasKnownAveragesAndChannelDerivatives) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  const auto a = n::linspace(1, 16, 16, type).reshape({1, 4, 4, 1});
  const auto x = n::concat({a, 2 * a}, 3).detach().requires_grad_();
  {
    n::NoGradGuard guard;
    n::models::DenseTransition(p, state, "t", x, 1);
  }
  p.set("t/bn/gamma", n::full({2}, std::sqrt(1.00001), type));
  p.set("t/conv/W", n::tensor({.25, .75}, {1, 1, 2, 1}).to(type));
  const auto y = n::models::DenseTransition(p, state, "t", x, 1);
  EXPECT_LT(
      n::max(n::abs(y - n::tensor({6.125, 9.625, 20.125, 23.625}, {1, 2, 2, 1}).to(type)))
          .item(),
      1e-12);
  const auto dx = n::grad(n::sum(y), {x})[0];
  EXPECT_LT(n::max(n::abs(dx - n::tensor({.0625, .1875}, {1, 1, 1, 2}).to(type))).item(),
            1e-12);
}
TEST(NnPreactivationDense, ModelCountsCompactGradientsAndInvalidConfigurations) {
  n::set_num_threads(2);
  {
    n::NoGradGuard guard;
    n::ParameterStore p(17), state(17);
    n::models::DenseNetConfig cfg;
    cfg.cifar_layout = true;
    cfg.depth = 40;
    cfg.growth_rate = 12;
    cfg.classes = 10;
    n::models::DenseNet(p, state, n::zeros({1, 32, 32, 3}), cfg);
    EXPECT_EQ(p.count(), 176122);
  }
  {
    n::ParameterStore p(17), state(17);
    n::models::DenseNetConfig cfg;
    cfg.cifar_layout = true;
    cfg.depth = 10;
    cfg.growth_rate = 2;
    cfg.classes = 3;
    cfg.update_stats = false;
    const auto out = n::models::DenseNet(p, state, n::ones({2, 8, 8, 3}), cfg, true);
    for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  }
  {
    n::ParameterStore p(17), state(17);
    n::models::PreActResNetConfig cfg;
    cfg.depth = 11;
    cfg.base_width = 2;
    cfg.update_stats = false;
    const auto out = n::models::PreActResNet(p, state, n::ones({2, 8, 8, 3}), cfg, true);
    for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  }
  n::models::DenseNetConfig dense;
  dense.cifar_layout = true;
  dense.depth = 10;
  dense.growth_rate = 1;
  dense.compression = .01;
  EXPECT_THROW(dense.Validate(), std::invalid_argument);
  n::models::PreActResNetConfig pre;
  pre.depth = 12;
  EXPECT_THROW(pre.Validate(), std::invalid_argument);
  n::ParameterStore p(17);
  EXPECT_THROW(n::models::RawResidualConv(p, "bad", n::ones({1, 3, 3, 2}), 2, 3, 1, 0),
               std::invalid_argument);
}
