#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
#include <numeric>
namespace n = datamunge::nn;
TEST(NnResidualModels, ZeroResidualRetainsShortcutAndPostAddGate) {
  const auto type = n::DType::Float64;
  for (bool bottleneck : {false, true})
    for (bool projection : {false, true}) {
      n::ParameterStore p(17, type), state(17, type);
      n::models::ResNetNormConfig cfg;
      cfg.scale = true;
      cfg.epsilon = 1e-5;
      const int64_t in = projection   ? 2
                         : bottleneck ? 16
                                      : 4,
                    width = projection && bottleneck ? 1 : 4;
      const int64_t spatial = projection ? 5 : 3, out = bottleneck ? 4 * width : width;
      const int stride = projection ? 2 : 1;
      const auto x = n::linspace(-1, 1, spatial * spatial * in, type)
                         .reshape({1, spatial, spatial, in})
                         .detach()
                         .requires_grad_();
      {
        n::NoGradGuard guard;
        n::models::ResNetBlock(p, state, "b", x, width, cfg, stride, bottleneck);
      }
      p.set(std::string("b/") + (bottleneck ? "conv3" : "conv2") + "/bn/gamma",
            n::zeros({out}, type));
      if (projection) {
        std::vector<double> values(in * out, 0);
        for (int64_t c = 0; c < out; ++c)
          values[(c % in) * out + c] = 1;
        p.set("b/shortcut/W", n::from_vector(values, {1, 1, in, out}));
        p.set("b/shortcut/bn/gamma", n::full({out}, std::sqrt(1 + cfg.epsilon), type));
      }
      const auto block =
          n::models::ResNetBlock(p, state, "b", x, width, cfg, stride, bottleneck);
      const auto dx = n::grad(n::sum(block.output), {x})[0];
      EXPECT_EQ(n::max(n::abs(block.residual)).item(), 0);
      const auto values = x.to_vector<double>(), actual = block.output.to_vector<double>(),
                 gradient = dx.to_vector<double>();
      const int side = projection ? 3 : spatial;
      for (int r = 0; r < side; ++r)
        for (int c = 0; c < side; ++c)
          for (int ch = 0; ch < out; ++ch) {
            const auto source = ((stride * r) * spatial + stride * c) * in + ch % in;
            EXPECT_NEAR(actual[(r * side + c) * out + ch], std::max(0., values[source]),
                        1e-12);
          }
      for (int r = 0; r < spatial; ++r)
        for (int c = 0; c < spatial; ++c)
          for (int ch = 0; ch < in; ++ch) {
            const auto source = (r * spatial + c) * in + ch;
            const double expected = r % stride == 0 && c % stride == 0 && values[source] > 0
                                        ? double(out) / in
                                        : 0;
            EXPECT_NEAR(gradient[source], expected, 1e-12);
          }
    }
}
TEST(NnResidualModels, BottleneckStridePlacementChangesSampling) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type), state(17, type);
  n::models::ResNetNormConfig cfg;
  cfg.scale = true;
  cfg.epsilon = 1e-5;
  const auto x =
      n::one_hot(n::tensor({48}).to(n::DType::Int64), 200, 1, 0, type).reshape({1, 5, 5, 8});
  {
    n::NoGradGuard guard;
    n::models::ResNetBlock(p, state, "b", x, 2, cfg, 2, true);
  }
  for (const auto &name : p.names()) {
    const auto shape = p.at(name).shape();
    if (name.ends_with("/W"))
      p.set(name,
            name.starts_with("b/shortcut") ? n::zeros(shape, type) : n::ones(shape, type));
    if (name.ends_with("/gamma"))
      p.set(name, n::full(shape, std::sqrt(1 + cfg.epsilon), type));
  }
  const auto original = n::models::ResNetBlock(p, state, "b", x, 2, cfg, 2, true, false);
  const auto later = n::models::ResNetBlock(p, state, "b", x, 2, cfg, 2, true, true);
  EXPECT_EQ(n::max(original.output).item(), 0);
  EXPECT_NEAR(n::max(later.output).item(), 4, 1e-12);
  EXPECT_NEAR(n::sum(later.output).item(), 128, 1e-10);
  EXPECT_EQ(later.output.shape(), (n::Shape{1, 3, 3, 8}));
}
TEST(NnResidualModels, AllDepthSchedulesRetainStagesAndAllowFeatureOnlyExtraction) {
  n::set_num_threads(2);
  n::Generator rng(29);
  for (int depth : {18, 34, 50, 101, 152}) {
    n::ParameterStore p(17), state(17);
    n::models::ResNetConfig cfg;
    cfg.depth = depth;
    cfg.classes = 3;
    cfg.width_divisor = 8;
    cfg.small_stem = true;
    cfg.include_classifier = false;
    cfg.update_stats = false;
    const auto x = n::normal({2, 8, 8, 3}, 0, .2, rng);
    const auto out = n::models::ResNet(p, state, x, cfg, true);
    EXPECT_FALSE(out.logits.defined());
    EXPECT_FALSE(p.contains("output/W"));
    const auto counts = n::models::ResNetBlockCounts(depth);
    EXPECT_EQ(out.blocks.size(), std::accumulate(counts.begin(), counts.end(), 0));
    EXPECT_EQ(out.stages[3].shape(), (n::Shape{2, 1, 1, depth >= 50 ? 256 : 64}));
    const auto before = state.flatten();
    for (const auto &g : n::grad(n::mean(n::square(out.stages[3])), p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
    n::NoGradGuard guard;
    n::models::ResNet(p, state, x, cfg, false);
    EXPECT_EQ(state.flatten(), before);
  }
}
TEST(NnResidualModels, InvalidShapesAndConfigurationFailBeforeReadingAxes) {
  n::ParameterStore p(17), state(17);
  n::models::ResNetNormConfig norm;
  EXPECT_THROW(n::models::ResNetBlock(p, state, "bad", n::scalar(1), 4, norm),
               std::invalid_argument);
  EXPECT_THROW(
      n::models::ResNetNormalizedConv(p, state, "bad", n::scalar(1), 4, 3, norm, 1, 1, true),
      std::invalid_argument);
  EXPECT_THROW(n::models::ResNetConv(p, state, "bad", n::ones({1, 5, 5, 3}), 4, 2, norm),
               std::invalid_argument);
  n::models::ResNetConfig cfg;
  cfg.depth = 20;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  cfg.depth = 18;
  cfg.width_divisor = 3;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
}
