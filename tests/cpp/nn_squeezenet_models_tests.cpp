
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnSqueezeNetModels, FireExpandValuesAndAllParameterInputDerivatives) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const auto x = n::sin(.37 * n::arange(0, 360).to(type) + .1)
                     .reshape({2, 6, 6, 5})
                     .detach()
                     .requires_grad_();
  const auto fire = n::models::Fire(p, "f", x, 3, 4, 6);
  const auto kernel =
                 n::concat({n::pad(p.at("f/expand1x1/W"), {{1, 1}, {1, 1}, {0, 0}, {0, 0}}),
                            p.at("f/expand3x3/W")},
                           3),
             bias = n::concat({p.at("f/expand1x1/b"), p.at("f/expand3x3/b")}, 0);
  const auto merged =
      n::relu(n::conv2d(fire.squeeze, kernel, n::conv_options(1, n::Padding::Same)) + bias);
  EXPECT_LT(n::max(n::abs(fire.output - merged)).item(), 1e-12);
  auto vars = p.parameters();
  vars.insert(vars.begin(), x);
  const auto a = n::grad(n::sum(n::square(fire.output)), vars, {.retain_graph = true}),
             b = n::grad(n::sum(n::square(merged)), vars);
  for (size_t i = 0; i < a.size(); ++i)
    EXPECT_LT(n::max(n::abs(a[i] - b[i])).item(), 1e-10);
  EXPECT_EQ(p.count(), 202);
  EXPECT_EQ(fire.output.shape(), n::Shape({2, 6, 6, 10}));
}
TEST(NnSqueezeNetModels, CeilPoolingPreservesNegativeEdgeWindowsAndSwitchGradients) {
  const auto type = n::DType::Float64;
  const auto x = n::arange(-16, 0).to(type).reshape({1, 4, 4, 1}).detach().requires_grad_();
  int size = 4;
  const auto y = n::models::CeilMaxPool(x, size);
  EXPECT_EQ(size, 2);
  EXPECT_EQ(y.to_vector<double>(), std::vector<double>({-6, -5, -2, -1}));
  const auto dx = n::grad(n::sum(y), {x})[0];
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      EXPECT_EQ(dx.at({0, r, c, 0}), double(r >= 2 && c >= 2));
  int odd = 5;
  const auto z = n::models::CeilMaxPool(n::ones({1, 5, 5, 2}), odd);
  EXPECT_EQ(odd, 2);
  EXPECT_EQ(z.shape(), n::Shape({1, 2, 2, 2}));
}
TEST(NnSqueezeNetModels, ReleasedCountsAndAllFireSizesIncludingBypasses) {
  n::set_num_threads(2);
  n::NoGradGuard guard;
  for (auto [version, bypass] :
       std::vector<std::pair<int, bool>>{{10, false}, {10, true}, {11, false}}) {
    n::ParameterStore p(17);
    n::models::SqueezeNetConfig cfg;
    cfg.version = version;
    cfg.simple_bypass = bypass;
    const auto out = n::models::SqueezeNet(p, n::zeros({1, 224, 224, 3}), cfg);
    EXPECT_EQ(p.count(), version == 10 ? 1248424 : 1235496);
    EXPECT_EQ(out.sizes, version == 10 ? std::vector<int>({54, 54, 54, 27, 27, 27, 27, 13})
                                       : std::vector<int>({55, 55, 27, 27, 13, 13, 13, 13}));
    EXPECT_EQ(out.fires.size(), 8);
    EXPECT_EQ(out.features.shape(), n::Shape({1, 13, 13, 1000}));
  }
}
TEST(NnSqueezeNetModels, CompleteCifarGradientsDropoutAndInferenceGeneratorSemantics) {
  n::set_num_threads(2);
  n::ParameterStore p(17);
  n::models::SqueezeNetConfig cfg;
  cfg.classes = 10;
  cfg.image_size = 32;
  cfg.cifar_stem = true;
  n::Generator gen(101), reference(101), data(29);
  const auto x = n::normal({2, 32, 32, 3}, 0, .2, data);
  const auto infer = n::models::SqueezeNet(p, x, cfg, false, &gen);
  EXPECT_EQ(p.count(), 727626);
  EXPECT_EQ(n::uniform({8}, 0, 1, gen).to_vector<float>(),
            n::uniform({8}, 0, 1, reference).to_vector<float>());
  const auto train = n::models::SqueezeNet(p, x, cfg, true, &gen);
  for (const auto &g : n::grad(n::sparse_softmax_cross_entropy(
                                   train.logits, n::from_vector(std::vector<int64_t>{0, 1})),
                               p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  n::NoGradGuard guard;
  EXPECT_EQ(n::max(n::abs(infer.logits - n::models::SqueezeNet(p, x, cfg).logits)).item(), 0);
  EXPECT_THROW(n::models::SqueezeNet(p, x, cfg, true), std::invalid_argument);
}
TEST(NnSqueezeNetModels, RejectsInvalidFirePoolingAndModelGeometry) {
  n::ParameterStore p(17);
  EXPECT_THROW(n::models::Fire(p, "f", n::ones({2, 3}), 2, 3, 3), std::invalid_argument);
  EXPECT_THROW(n::models::Fire(p, "f", n::ones({1, 5, 5, 3}), 0, 3, 3), std::invalid_argument);
  int side = 4;
  EXPECT_THROW(n::models::CeilMaxPool(n::ones({1, 5, 5, 2}), side), std::invalid_argument);
  n::models::SqueezeNetConfig cfg;
  cfg.version = 12;
  EXPECT_THROW(n::models::SqueezeNet(p, n::ones({1, 224, 224, 3}), cfg),
               std::invalid_argument);
  cfg = {};
  EXPECT_THROW(n::models::SqueezeNet(p, n::ones({1, 32, 32, 3}), cfg), std::invalid_argument);
}
