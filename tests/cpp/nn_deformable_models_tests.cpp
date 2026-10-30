
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
TEST(NnDeformableModels, ZeroOffsetsAndHalfMasksMatchOrdinaryFilters) {
  const auto type = n::DType::Float64;
  const auto x = n::linspace(-1, 2, 1 * 7 * 6 * 3, type).reshape({1, 7, 6, 3});
  for (int stride : {1, 2})
    for (int dilation : {1, 2})
      for (bool modulated : {false, true}) {
        n::ParameterStore p(17, type);
        const auto out =
            n::models::DeformableConv2D(p, "d", x, 4, 3, stride, modulated, true, dilation);
        auto options = n::conv_options(stride, n::Padding::Explicit, dilation);
        options.explicit_paddings = {dilation, dilation, dilation, dilation};
        const auto ref = n::conv2d(x, p.at("d/W"), options) * (modulated ? .5 : 1.);
        EXPECT_LT(n::max(n::abs(ref - out.output)).item(), 1e-12);
        EXPECT_EQ(n::sum(n::abs(out.offsets)).item(), 0);
        if (modulated)
          EXPECT_EQ(n::max(n::abs(out.masks - .5)).item(), 0);
        EXPECT_EQ(out.output.size(1), (7 + stride - 1) / stride);
      }
}
TEST(NnDeformableModels, FractionalSamplingAndAllOffsetBiasDerivativesMatchOracle) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const auto x = n::sin(.37 * n::arange(0, 7 * 7 * 2).to(type)).reshape({1, 7, 7, 2});
  {
    n::NoGradGuard guard;
    n::models::DeformableConv2D(p, "d", x, 3, 3, 1, false);
    const auto pattern = .37 + .9 * n::sin(1.7 * n::arange(0, 18).to(type));
    p.set("d/offset/b", n::floor(pattern) + .2 + .6 * (pattern - n::floor(pattern)));
  }
  const auto out = n::models::DeformableConv2D(p, "d", x, 3, 3, 1, false);
  const auto values = x.to_vector<double>(), weights = p.at("d/W").to_vector<double>(),
             offsets = p.at("d/offset/b").to_vector<double>();
  auto sample = [&](double y, double z, int c) {
    const int y0 = int(std::floor(y)), x0 = int(std::floor(z));
    double result = 0;
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx) {
        const int yy = y0 + dy, xx = x0 + dx;
        if (yy >= 0 && yy < 7 && xx >= 0 && xx < 7)
          result += (dy ? y - y0 : 1 - y + y0) * (dx ? z - x0 : 1 - z + x0) *
                    values[(yy * 7 + xx) * 2 + c];
      }
    return result;
  };
  for (int y = 0; y < 7; ++y)
    for (int z = 0; z < 7; ++z)
      for (int o = 0; o < 3; ++o) {
        double ref = 0;
        for (int k = 0; k < 9; ++k)
          for (int c = 0; c < 2; ++c)
            ref +=
                sample(y - 1 + k / 3 + offsets[2 * k], z - 1 + k % 3 + offsets[2 * k + 1], c) *
                weights[(k * 2 + c) * 3 + o];
        EXPECT_NEAR(out.output.at({0, y, z, o}), ref, 1e-12);
      }
  const auto derivative =
      n::grad(n::sum(n::square(out.output)), {p.at("d/offset/b")})[0].to_vector<double>();
  n::NoGradGuard guard;
  const double h = 1e-5;
  for (int k = 0; k < 18; ++k) {
    auto plus = offsets, minus = offsets;
    plus[k] += h;
    minus[k] -= h;
    p.set("d/offset/b", n::from_vector(plus));
    const double a =
        n::sum(n::square(n::models::DeformableConv2D(p, "d", x, 3, 3, 1, false).output))
            .item();
    p.set("d/offset/b", n::from_vector(minus));
    const double b =
        n::sum(n::square(n::models::DeformableConv2D(p, "d", x, 3, 3, 1, false).output))
            .item();
    EXPECT_NEAR(derivative[k], (a - b) / (2 * h), 1e-7);
  }
}
TEST(NnDeformableModels, CompleteLastTwoStagesRetainCountsAndFiniteGradients) {
  n::set_num_threads(2);
  for (bool modulated : {false, true}) {
    n::ParameterStore p(17), state(17);
    n::models::AttentionResNetConfig cfg;
    cfg.width_divisor = 4;
    cfg.classes = 10;
    cfg.small_stem = true;
    cfg.stride_on_3x3 = false;
    cfg.deformable_from = 3;
    cfg.modulated = modulated;
    cfg.update_stats = false;
    const auto x = n::sin(.37 * n::arange(0, 2 * 32 * 32 * 3).to(n::DType::Float32))
                       .reshape({2, 32, 32, 3});
    const auto out = n::models::AttentionResNet(p, state, x, cfg, true);
    const int64_t maps = modulated ? 27 : 18;
    EXPECT_EQ(p.count(), 1484186 + 6 * (9 * 64 * maps + maps) + 3 * (9 * 128 * maps + maps));
    EXPECT_EQ(out.features.shape(), n::Shape({2, 4, 4, 512}));
    for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
      EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  }
}
TEST(NnDeformableModels, InvalidGeometryFailsBeforeCreatingPredictors) {
  n::ParameterStore p(17);
  const auto x = n::ones({1, 5, 5, 3});
  EXPECT_THROW(n::models::DeformableConv2D(p, "d", x, 4, 0, 1, false), std::invalid_argument);
  EXPECT_THROW(n::models::DeformableConv2D(p, "d", x, 4, 3, 0, false), std::invalid_argument);
  EXPECT_THROW(n::models::DeformableConv2D(p, "d", x, 4, 2, 1, false), std::invalid_argument);
  EXPECT_EQ(p.count(), 0);
}
