#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cmath>

namespace nn = datamunge::nn;
namespace init = datamunge::nn::init;

namespace {
double sample_std(const nn::Tensor& t) {
  return std::sqrt(nn::mean(nn::square(t - nn::mean(t))).item());
}
}  // namespace

TEST(NnVarianceScaling, TruncatedNormalStd) {
  EXPECT_NEAR(init::truncated_normal_std(2.0), 0.87962566103423978, 1e-14);
  EXPECT_NEAR(init::truncated_normal_std(50.0), 1.0, 1e-12);
  EXPECT_THROW((void)init::truncated_normal_std(0.0), std::invalid_argument);
  nn::Generator g(5);
  const auto z = nn::truncated_normal({200000}, 0.0, 1.0, g, nn::DType::Float64);
  EXPECT_NEAR(sample_std(z), init::truncated_normal_std(), 0.005);
}

TEST(NnVarianceScaling, VarianceMatchesScaleOverFan) {
  nn::Generator g(1);
  const nn::Shape shape{400, 100};  // fan_in 400, fan_out 100, mean 250
  struct Case { double scale; init::FanMode mode; init::Distribution dist; double n; };
  for (const Case c : {Case{1, init::FanMode::FanIn, init::Distribution::TruncatedNormal, 400},
                       Case{2, init::FanMode::FanOut, init::Distribution::Normal, 100},
                       Case{1, init::FanMode::FanAvg, init::Distribution::Uniform, 250}}) {
    const auto w = init::variance_scaling(c.scale, c.mode, c.dist)(shape, nn::DType::Float64, g);
    EXPECT_NEAR(sample_std(w) / std::sqrt(c.scale / c.n), 1.0, 0.01);
  }
  EXPECT_THROW((void)init::variance_scaling(-1.0), std::invalid_argument);
}

TEST(NnVarianceScaling, Gains) {
  EXPECT_DOUBLE_EQ(init::gain("linear"), 1.0);
  EXPECT_DOUBLE_EQ(init::gain("sigmoid"), 1.0);
  EXPECT_DOUBLE_EQ(init::gain("tanh"), 5.0 / 3.0);
  EXPECT_DOUBLE_EQ(init::gain("relu"), std::sqrt(2.0));
  EXPECT_DOUBLE_EQ(init::gain("leaky_relu", 0.2), std::sqrt(2.0 / 1.04));
  EXPECT_DOUBLE_EQ(init::gain("selu"), 0.75);
  EXPECT_THROW((void)init::gain("swish"), std::invalid_argument);
}

TEST(NnVarianceScaling, SignalPropagationLinearIsExact) {
  // Orthogonal square weights preserve every row's norm, so the forward RMS stays at the input's.
  nn::Generator g(2);
  const auto sp = init::signal_propagation({64, 64, 64, 64, 64}, init::orthogonal(), [](const nn::Tensor& x) { return x; }, 32, g);
  ASSERT_EQ(sp.forward_rms.size(), 4u);
  for (std::size_t l = 1; l < 4; ++l) EXPECT_NEAR(sp.forward_rms[l], sp.forward_rms[0], 1e-10);
  for (std::size_t l = 1; l < 4; ++l) EXPECT_NEAR(sp.backward_rms[l], sp.backward_rms[3], 1e-10);
  // He-initialised ReLU layers keep the RMS roughly constant; a 0.5 scale halves it at every layer.
  const auto relu = [](const nn::Tensor& x) { return nn::relu(x); };
  const auto he   = init::signal_propagation(std::vector<std::int64_t>(11, 512), init::he_normal(), relu, 256, g);
  EXPECT_NEAR(he.forward_rms.back() / he.forward_rms.front(), 1.0, 0.3);
  const auto small = init::signal_propagation(std::vector<std::int64_t>(11, 512), init::variance_scaling(0.5, init::FanMode::FanIn, init::Distribution::Normal), relu, 256, g);
  // Variance factor n Var(w) / 2 = 0.25 per layer: the RMS halves, 2^-9 over the 9 steps from layer 1 to 10.
  EXPECT_NEAR(std::log2(small.forward_rms.back() / small.forward_rms.front()), -9.0, 0.3);
  EXPECT_THROW((void)init::signal_propagation({4}, init::he_normal(), relu, 2, g), std::invalid_argument);
}
