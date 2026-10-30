
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_factorized_video_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.10_3d_and_video_convolutions/factorized_video.cpp"
#undef main
TEST(FactorVideo, Every530HostFactorKernelInputDerivativeAndAllP3dVariants) {
  FactorChecks();
}
TEST(FactorVideo, ActualBothFullReferenceGraphsAllEndpointsAndEveryLearnedGradient) {
  nn::set_num_threads(2);
  GraphChecks(true);
}
TEST(FactorVideo, ExplicitBatchStatsMutationAndImmutableInference) {
  namespace m = dm::nn::models;
  nn::ParameterStore p(727), s(729);
  const m::FactorVideoConfig c{false, 4, 3};
  const auto pools = m::LoadVideo("build/cnn-datasets/action_video");
  const auto x = nn::slice_axis(pools[0].rgb, 0, 0, 2);
  m::FactorVideo(p, s, x, c);
  std::vector<nn::Tensor> before;
  for (const auto &v : s.parameters())
    before.push_back(v.clone());
  m::FactorVideo(p, s, x, c, true, false);
  for (std::size_t i = 0; i < before.size(); ++i)
    EXPECT_EQ(nn::max(nn::abs(before[i] - s.parameters()[i])).item(), 0);
  m::FactorVideo(p, s, x, c, true, true);
  auto change = nn::zeros({});
  for (std::size_t i = 0; i < before.size(); ++i)
    change = change + nn::sum(nn::abs(before[i] - s.parameters()[i]));
  EXPECT_GT(change.item(), 0);
  const auto state = s.flatten(), learned = p.flatten();
  const auto z = m::FactorVideo(p, s, x, c).logits;
  EXPECT_EQ(nn::max(nn::abs(m::FactorVideo(p, s, x, c).logits - z)).item(), 0);
  EXPECT_EQ(s.flatten(), state);
  EXPECT_EQ(p.flatten(), learned);
}
TEST(FactorVideo, ExplicitSymmetricStrideBordersAndInvalidContracts) {
  namespace m = dm::nn::models;
  const auto x = nn::arange(0, 64).reshape({1, 4, 4, 4, 1}), w = nn::ones({3, 3, 3, 1, 1});
  EXPECT_EQ(nn::max(nn::abs(m::FactorConv(x, w, 2, 2, 2) -
                            nn::conv3d(nn::pad(x, {{0, 0}, {1, 1}, {1, 1}, {1, 1}, {0, 0}}), w,
                                       {.strides = {2, 2, 2}, .padding = nn::Padding::Valid})))
                .item(),
            0);
  EXPECT_THROW(m::FactorMidplanes(0, 4), std::invalid_argument);
  EXPECT_THROW(m::FactorMidplanes(2147483647, 4), std::invalid_argument);
  EXPECT_THROW(m::P3dMix(
                   x,
                   [](const auto &t) {
                     return t;
                   },
                   [](const auto &t) {
                     return t;
                   },
                   static_cast<m::P3dKind>(7)),
               std::invalid_argument);
  nn::ParameterStore p, s;
  EXPECT_THROW(
      m::FactorVideo(p, s, nn::zeros({1, 16, 32, 32, 3}), {true, 4, 3}, true, false, nullptr),
      std::invalid_argument);
}

TEST(FactorVideo, FusedMovingVarianceUnbiasedCorrectionMatchesIndependentValues) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::ParameterStore p(1), s(2);
  const auto x = n::arange(1, 5).reshape({2, 1, 1, 2, 1});
  m::FactorUnit(p, s, "u", x, 1, 1, 1, 1, 1, 1, 1, false, false, false);
  {
    n::NoGradGuard guard;
    p.set("u/W", n::ones({1, 1, 1, 1, 1}));
  }
  const auto y = m::FactorUnit(p, s, "u", x, 1, 1, 1, 1, 1, 1, 1, false, true, true);
  EXPECT_NEAR(s.at("u/bn/moving_mean").item(), .25, 1e-7);
  EXPECT_NEAR(s.at("u/bn/moving_var").item(), .9 + .1 * 5. / 3., 2e-7);
  const auto values = y.to_vector<double>();
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(values[i], (i + 1 - 2.5) / std::sqrt(1.25 + 1e-5), 2e-7);
}
