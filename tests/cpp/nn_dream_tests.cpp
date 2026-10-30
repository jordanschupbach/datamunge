
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_dream_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.11_interpretability/deepdream.cpp"
#undef main
#include "../fixtures/nn_dream_reference.hpp"
TEST(DeepDream, AnalyticRmsPixelsFourBacktracksZeroRateStationaryAndNineHalfPixelValues) {
  Checks();
}
TEST(DeepDream, EveryDoubleTensorOctavePixelMatchesIndependentHostConvention) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto x = (n::sin(n::arange(0, 90) * .29) * .4 + .5).reshape({2, 3, 5, 3});
  for (const auto &size : std::vector<std::pair<int, int>>{{3, 5}, {4, 7}, {20, 24}, {2, 2}}) {
    const auto a = m::DreamResize(x, size.first, size.second),
               b = dream_reference::DreamResize(x, size.first, size.second);
    EXPECT_EQ(a.to_vector<float>(), b.to_vector<float>());
  }
}
TEST(DeepDream, EveryTensorSgdBoundedProposalMatchesRetainedHostAscent) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto x = (n::sin(n::arange(0, 12) * .4) * .4 + .5).reshape({2, 2, 3, 1});
  auto f = [](const n::Tensor &point) {
    auto x = point.detach().clone().requires_grad_();
    const auto s = -n::mean(n::square(x - .7));
    return m::DreamEvaluation{s.item(), n::grad(s, {x})[0].detach()};
  };
  auto host = [&](const n::Tensor &x) {
    const auto e = f(x);
    return dream_reference::DreamEvaluation{e.score, e.gradient};
  };
  for (double rate : {0., .02, 2.}) {
    const auto a = m::DreamAscent(x, rate, f);
    const auto b = dream_reference::DreamAscent(x, rate, host);
    EXPECT_EQ(a.backtracks, b.backtracks);
    EXPECT_EQ(a.step, b.step);
    EXPECT_LE(n::max(n::abs(a.image - b.image)).item(), 1e-7);
    EXPECT_NEAR(a.rms, b.rms, 1e-14);
  }
}
