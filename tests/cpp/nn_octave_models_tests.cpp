
#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
namespace n = datamunge::nn;
namespace {
using Image = std::vector<std::vector<std::vector<double>>>;
Image image(const n::Tensor &x) {
  Image out(x.size(1),
            std::vector<std::vector<double>>(x.size(2), std::vector<double>(x.size(3))));
  for (int r = 0; r < x.size(1); ++r)
    for (int c = 0; c < x.size(2); ++c)
      for (int k = 0; k < x.size(3); ++k)
        out[r][c][k] = x.at({0, r, c, k});
  return out;
}
Image pool(const Image &x) {
  const int side = int(x.size()) / 2, channels = int(x[0][0].size());
  Image out(side, std::vector<std::vector<double>>(side, std::vector<double>(channels)));
  for (int r = 0; r < side; ++r)
    for (int c = 0; c < side; ++c)
      for (int k = 0; k < channels; ++k)
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            out[r][c][k] += x[2 * r + dy][2 * c + dx][k] / 4;
  return out;
}
Image conv(const Image &x, const n::Tensor &kernel) {
  const int side = int(x.size()), in = int(kernel.size(2)), out = int(kernel.size(3));
  const auto w = kernel.to_vector<double>();
  Image result(side, std::vector<std::vector<double>>(side, std::vector<double>(out)));
  for (int r = 0; r < side; ++r)
    for (int c = 0; c < side; ++c)
      for (int ky = 0; ky < 3; ++ky)
        for (int kx = 0; kx < 3; ++kx) {
          const int y = r + ky - 1, z = c + kx - 1;
          if (y < 0 || z < 0 || y >= side || z >= side)
            continue;
          for (int i = 0; i < in; ++i)
            for (int o = 0; o < out; ++o)
              result[r][c][o] += x[y][z][i] * w[((ky * 3 + kx) * in + i) * out + o];
        }
  return result;
}
} // namespace
TEST(NnOctaveModels, FourPathsMatchIndependentHostConvolutionAndPoolForBothStrides) {
  const auto type = n::DType::Float64;
  for (int stride : {1, 2}) {
    n::ParameterStore p(17, type);
    const auto high =
                   n::sin(.37 * n::arange(0, 8 * 8 * 12).to(type) + .2).reshape({1, 8, 8, 12}),
               low = n::sin(.37 * n::arange(0, 4 * 4 * 4).to(type) + .9).reshape({1, 4, 4, 4});
    const n::models::OctaveMaps x{high, low, 12, 4};
    n::models::OctaveCost cost;
    const auto out = n::models::OctaveConv(p, "o", x, 16, .25, 3, stride, 8, &cost);
    auto h = image(high), l = image(low);
    if (stride == 2) {
      h = pool(h);
      l = pool(l);
    }
    const auto hh = conv(h, p.at("o/hh/W")), lh = conv(l, p.at("o/lh/W")),
               hl = conv(pool(h), p.at("o/hl/W")), ll = conv(l, p.at("o/ll/W"));
    for (int r = 0; r < out.high.size(1); ++r)
      for (int c = 0; c < out.high.size(2); ++c)
        for (int o = 0; o < 12; ++o)
          EXPECT_NEAR(out.high.at({0, r, c, o}), hh[r][c][o] + lh[r / 2][c / 2][o], 1e-12);
    for (int r = 0; r < out.low.size(1); ++r)
      for (int c = 0; c < out.low.size(2); ++c)
        for (int o = 0; o < 4; ++o)
          EXPECT_NEAR(out.low.at({0, r, c, o}), hl[r][c][o] + ll[r][c][o], 1e-12);
    EXPECT_EQ(p.count(), 2304);
    EXPECT_DOUBLE_EQ(double(cost.macs) / (int64_t(8 / stride) * (8 / stride) * 9 * 16 * 16),
                     .671875);
  }
}
TEST(NnOctaveModels, AllFourKernelDerivativesMatchFiniteDifferences) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  const n::models::OctaveMaps x{
      n::sin(.37 * n::arange(0, 8 * 8 * 12).to(type)).reshape({1, 8, 8, 12}),
      n::sin(.27 * n::arange(0, 4 * 4 * 4).to(type)).reshape({1, 4, 4, 4}), 12, 4};
  auto objective = [&]() {
    const auto y = n::models::OctaveConv(p, "o", x, 16, .25, 3, 1, 8);
    return n::mean(n::square(y.high)) + n::mean(n::square(y.low));
  };
  const auto loss = objective();
  const auto vars = p.parameters();
  const auto gradients = n::grad(loss, vars);
  const auto names = p.names();
  n::NoGradGuard guard;
  const double h = 1e-6;
  for (size_t i = 0; i < vars.size(); ++i) {
    const auto original = p.at(names[i]).clone();
    const auto direction =
        n::one_hot(n::tensor({0}).to(n::DType::Int64), original.numel(), 1, 0, type)
            .reshape(original.shape());
    p.set(names[i], original + h * direction);
    const double plus = objective().item();
    p.set(names[i], original - h * direction);
    const double minus = objective().item();
    p.set(names[i], original);
    EXPECT_NEAR(gradients[i].reshape({gradients[i].numel()}).at({0}), (plus - minus) / (2 * h),
                1e-7);
  }
}
TEST(NnOctaveModels, AlphaZeroIsDenseAtStrideOneAndMergeRestoresHighOnly) {
  n::ParameterStore p(17);
  const auto h = n::sin(.1 * n::arange(0, 8 * 8 * 8)).reshape({1, 8, 8, 8});
  const n::models::OctaveMaps input{h, n::Tensor(), 8, 0};
  const auto ordinary = n::models::OctaveConv(p, "plain", input, 8, 0, 3, 1, 8);
  auto options = n::conv_options(1, n::Padding::Explicit);
  options.explicit_paddings = {1, 1, 1, 1};
  EXPECT_EQ(n::max(n::abs(ordinary.high - n::conv2d(h, p.at("plain/hh/W"), options))).item(),
            0);
  EXPECT_FALSE(ordinary.low.defined());
  const auto split = n::models::OctaveConv(p, "split", input, 8, .25, 3, 1, 8),
             merge = n::models::OctaveConv(p, "merge", split, 8, 0, 3, 1, 8);
  EXPECT_EQ(merge.high.shape(), h.shape());
  EXPECT_EQ(merge.low_channels, 0);
  EXPECT_FALSE(merge.low.defined());
}
TEST(NnOctaveModels, FiveFullWidthSplitsKeepCountsAndNominalBudgetIsExplicit) {
  n::set_num_threads(2);
  int64_t previous = std::numeric_limits<int64_t>::max();
  for (double alpha : {0., .125, .25, .5, .75}) {
    n::NoGradGuard guard;
    n::ParameterStore p(17), state(17);
    n::models::OctResNetConfig cfg;
    cfg.alpha = alpha;
    const auto out = n::models::OctResNet(p, state, n::zeros({1, 32, 32, 3}), cfg);
    EXPECT_EQ(p.count(), 25557032);
    EXPECT_LE(out.macs, previous);
    previous = out.macs;
    if (alpha == 0)
      EXPECT_EQ(out.macs, 4089184256);
    EXPECT_EQ(out.features.shape(), n::Shape({1, 1, 1, 2048}));
  }
}
TEST(NnOctaveModels, CompleteCompactBackwardAndFrozenStreams) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::models::OctResNetConfig cfg;
  cfg.width_divisor = 4;
  cfg.classes = 10;
  cfg.small_stem = true;
  cfg.image_size = 32;
  cfg.update_stats = false;
  const auto x = n::sin(.37 * n::arange(0, 2 * 32 * 32 * 3).to(n::DType::Float32))
                     .reshape({2, 32, 32, 3});
  {
    n::NoGradGuard guard;
    n::models::OctResNet(p, state, x, cfg);
  }
  const auto before = state.flatten();
  const auto out = n::models::OctResNet(p, state, x, cfg, true);
  EXPECT_EQ(p.count(), 1484186);
  EXPECT_EQ(out.features.shape(), n::Shape({2, 4, 4, 512}));
  for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  EXPECT_EQ(state.flatten(), before);
}
TEST(NnOctaveModels, OddLowMapCanMergeDirectlyAtStrideTwo) {
  const auto type = n::DType::Float64;
  n::ParameterStore p(17, type);
  n::models::OctaveMaps x{
      n::sin(.1 * n::arange(0, 14 * 14 * 6).to(type)).reshape({1, 14, 14, 6}),
      n::sin(.2 * n::arange(0, 7 * 7 * 2).to(type)).reshape({1, 7, 7, 2}), 6, 2};
  n::models::OctaveCost cost;
  const auto y = n::models::OctaveConv(p, "merge", x, 8, 0, 1, 2, 14, &cost);
  const auto expected =
      n::conv2d(n::avg_pool2d(x.high, 2, 2, n::Padding::Valid), p.at("merge/hh/W")) +
      n::conv2d(x.low, p.at("merge/lh/W"));
  EXPECT_EQ(y.high.shape(), n::Shape({1, 7, 7, 8}));
  EXPECT_FALSE(y.low.defined());
  EXPECT_LT(n::max(n::abs(y.high - expected)).item(), 1e-12);
  EXPECT_EQ(cost.macs, 7 * 7 * 8 * 8);
  n::set_num_threads(2);
  n::NoGradGuard guard;
  n::ParameterStore full(19), state(19);
  const auto out = n::models::OctResNet(full, state, n::zeros({1, 224, 224, 3}));
  EXPECT_EQ(out.features.shape(), n::Shape({1, 7, 7, 2048}));
  EXPECT_EQ(full.count(), 25557032);
}
TEST(NnOctaveModels, InvalidSplitsAndOddHalfResolutionPoolingAreRejected) {
  EXPECT_THROW(n::models::LowChannels(8, 1), std::invalid_argument);
  n::ParameterStore p(17);
  n::models::OctaveMaps x{n::ones({1, 6, 6, 6}), n::ones({1, 3, 3, 2}), 6, 2};
  EXPECT_THROW(n::models::OctaveConv(p, "o", x, 8, .25, 3, 2, 6), std::invalid_argument);
  x.high_channels = 7;
  EXPECT_THROW(n::models::OctaveConv(p, "o", x, 8, .25, 3, 1, 6), std::invalid_argument);
  n::ParameterStore state(17);
  n::models::OctResNetConfig cfg;
  cfg.width_divisor = 0;
  EXPECT_THROW(n::models::OctResNet(p, state, n::ones({1, 32, 32, 3}), cfg),
               std::invalid_argument);
}
