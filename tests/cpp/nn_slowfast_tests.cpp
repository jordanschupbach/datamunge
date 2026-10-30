
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_slowfast_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.10_3d_and_video_convolutions/slowfast.cpp"
#undef main
TEST(SlowFast, EverySelectedFrameAndInputAdjoint) {
  SamplingChecks();
}
TEST(SlowFast, All56LateralValuesAnd192IndependentHostFiniteDifferences) {
  FusionChecks();
}
TEST(SlowFast, ZeroResidualIdentityBlockedKernelAndHostGammaBetaAdjoints) {
  ZeroGateChecks();
}
TEST(SlowFast, All32CompactBlocksFourFusionsAndAnalyticalFullCensus) {
  nn::set_num_threads(2);
  GraphChecks();
}
TEST(SlowFast, EveryMovingArrayMutationPolicyAndZeroGateInventory) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  n::ParameterStore p(751), s(753);
  n::Generator rng(743);
  const auto pools = m::LoadVideo("build/cnn-datasets/action_video");
  const auto x = n::slice_axis(pools[0].rgb, 0, 0, 2);
  const m::SlowFastConfig c{8, 3, 16};
  m::SlowFast(p, s, x, c);
  int gates = 0;
  for (const auto &name : p.names())
    if (name.ends_with("/expand/bn/gamma")) {
      EXPECT_EQ(n::max(n::abs(p.at(name))).item(), 0);
      ++gates;
    }
  EXPECT_EQ(gates, 32);
  auto before = s.flatten();
  m::SlowFast(p, s, x, c, true, false, &rng);
  EXPECT_EQ(s.flatten(), before);
  m::SlowFast(p, s, x, c, true, true, &rng);
  EXPECT_NE(s.flatten(), before);
  before = s.flatten();
  const auto learned = p.flatten();
  const auto y = m::SlowFast(p, s, x, c).logits;
  EXPECT_EQ(n::max(n::abs(y - m::SlowFast(p, s, x, c).logits)).item(), 0);
  EXPECT_EQ(s.flatten(), before);
  EXPECT_EQ(p.flatten(), learned);
}
TEST(SlowFast, InvalidFrameScaleGeometryAndTrainingRng) {
  namespace n = dm::nn;
  namespace m = n::models;
  EXPECT_THROW(m::SlowFastIndices(8), std::invalid_argument);
  EXPECT_THROW(m::SlowFastIndices(17), std::invalid_argument);
  EXPECT_THROW(m::SlowFastSample(n::ones({1, 15, 1, 1, 1}), 16), std::invalid_argument);
  n::ParameterStore p, s;
  EXPECT_THROW(m::SlowFast(p, s, n::zeros({1, 16, 32, 32, 3}), {8, 3, 16}, true),
               std::invalid_argument);
  EXPECT_THROW(m::SlowFast(p, s, n::zeros({1, 16, 32, 32, 3}), {8, 3, 16, NAN}),
               std::invalid_argument);
}
