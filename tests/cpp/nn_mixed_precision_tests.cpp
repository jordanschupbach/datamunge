#include <gtest/gtest.h>

#include <bit>
#include <cfenv>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <limits>
namespace nn = datamunge::nn;
TEST(MixedPrecision, HalfEveryFiniteEncodingAndTieBoundaries) {
  for (unsigned bits = 0; bits < 65536; ++bits) {
    unsigned e = (bits >> 10) & 31, m = bits & 1023;
    if (e == 31)
      continue;
    float x = e ? std::ldexp(1.f + m / 1024.f, int(e) - 15) : std::ldexp(float(m), -24);
    if (bits & 32768)
      x = -x;
    EXPECT_EQ(std::bit_cast<unsigned>(nn::round_float16(x)), std::bit_cast<unsigned>(x));
  }
  EXPECT_EQ(nn::round_float16(1.f + std::ldexp(1.f, -11)), 1.f);
  EXPECT_EQ(nn::round_float16(1.f + 3 * std::ldexp(1.f, -11)), 1.f + std::ldexp(1.f, -9));
  EXPECT_EQ(nn::round_float16(std::ldexp(1.f, -25)), 0);
  EXPECT_EQ(nn::round_float16(3 * std::ldexp(1.f, -25)), std::ldexp(1.f, -23));
  EXPECT_EQ(nn::round_float16(65519.f), 65504.f);
  EXPECT_TRUE(std::isinf(nn::round_float16(65520.f)));
  EXPECT_TRUE(std::signbit(nn::round_float16(-1e-12f)));
  EXPECT_TRUE(std::isnan(nn::round_float16(std::numeric_limits<float>::quiet_NaN())));
  const int original = std::fegetround();
  std::fesetround(FE_UPWARD);
  const float tied = nn::round_float16(std::ldexp(1.f, -25));
  std::fesetround(original);
  EXPECT_EQ(tied, 0);
}
TEST(MixedPrecision, BfloatExponentAndTies) {
  EXPECT_EQ(nn::round_bfloat16(1.f + std::ldexp(1.f, -8)), 1.f);
  EXPECT_EQ(nn::round_bfloat16(1.f + 3 * std::ldexp(1.f, -8)), 1.f + std::ldexp(1.f, -6));
  EXPECT_NE(nn::round_bfloat16(1e-8f), 0);
  EXPECT_TRUE(std::isinf(nn::round_bfloat16(std::numeric_limits<float>::infinity())));
  EXPECT_TRUE(std::isnan(nn::round_bfloat16(std::numeric_limits<float>::quiet_NaN())));
  EXPECT_TRUE(std::signbit(nn::round_bfloat16(-0.f)));
}
TEST(MixedPrecision, LossScalingRecoversSurrogateGradientAndUnscaleOwnsBuffers) {
  const auto w        = nn::scalar(1).requires_grad_();
  auto       gradient = [&](float scale) {
    return nn::grad(nn::quantize_precision(w, nn::PrecisionEmulation::Float16) * 1e-8 * scale, {w});
  };
  EXPECT_EQ(gradient(1)[0].item(), 0);
  auto                    scaled = gradient(65536);
  std::vector<nn::Tensor> g;
  ASSERT_TRUE(nn::unscale_gradients(scaled, 65536, &g));
  EXPECT_NEAR(g[0].item(), 1e-8, 1e-11);
  scaled[0].fill_(0);
  EXPECT_NE(g[0].item(), 0);
  EXPECT_FALSE(nn::unscale_gradients({nn::scalar(std::numeric_limits<float>::infinity())}, 1, &g));
  EXPECT_TRUE(g.empty());
  g = {nn::scalar(3)};
  EXPECT_THROW(nn::unscale_gradients({}, 0, &g), std::invalid_argument);
  EXPECT_EQ(g[0].item(), 3);
  EXPECT_THROW(
      nn::unscale_gradients({nn::zeros({1}, nn::DType::Float64)}, 1, &g), std::invalid_argument);
  EXPECT_EQ(g[0].item(), 3);
}
TEST(MixedPrecision, HostOverflowPolicyPreservesNontrivialAdamState) {
  auto                w = nn::scalar(1).requires_grad_();
  nn::OptimizerConfig c;
  c.kind = nn::OptimizerKind::Adam;
  nn::Optimizer opt({w}, .1, c);
  opt.minimize(nn::square(w));
  auto snapshot = [&] {
    std::vector<std::vector<double>> values{w.to_vector<double>()};
    for (const auto& [name, t] : opt.state())
      values.push_back(t.to_vector<double>());
    return values;
  };
  const auto before = snapshot();
  const auto scaled =
      nn::grad(nn::quantize_precision(w, nn::PrecisionEmulation::Float16) * 1e12, {w});
  std::vector<nn::Tensor> g;
  ASSERT_FALSE(nn::unscale_gradients(scaled, 1e12, &g));
  EXPECT_EQ(snapshot(), before);
  EXPECT_EQ(opt.iterations(), 1);
  nn::DynamicLossScale s(128, 2, 1, 256);
  s.applied();
  EXPECT_EQ(s.value(), 128);
  s.applied();
  EXPECT_EQ(s.value(), 256);
  s.overflow();
  EXPECT_EQ(s.value(), 128);
  for (int i = 0; i < 20; ++i)
    s.overflow();
  EXPECT_EQ(s.value(), 1);
  EXPECT_THROW((nn::DynamicLossScale(0)), std::invalid_argument);
}
TEST(MixedPrecision, DenseCreatesMasterParametersAndChecksSchema) {
  nn::ParameterStore p(11);
  const auto         x = nn::zeros({4, 2});
  const auto         y = nn::mixed_dense(p, "d", x, 2, 1, nn::PrecisionEmulation::Float16);
  EXPECT_EQ(y.shape(), (std::vector<std::int64_t>{4, 1}));
  EXPECT_EQ(p.parameters().size(), 2);
  EXPECT_EQ(p.at("d/W").dtype(), nn::DType::Float32);
  EXPECT_THROW(
      (void)nn::mixed_dense(p, "d", x, 3, 1, nn::PrecisionEmulation::Float16),
      std::invalid_argument);
  EXPECT_THROW(
      (void)nn::quantize_precision(
          nn::zeros({1}, nn::DType::Float64), nn::PrecisionEmulation::Float16),
      std::invalid_argument);
  EXPECT_THROW(
      (void)nn::precision_name(static_cast<nn::PrecisionEmulation>(99)), std::invalid_argument);
}
