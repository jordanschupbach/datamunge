
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_feature_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.11_interpretability/feature_visualization.cpp"
#undef main
TEST(FeatureVisualization, All784SelectedChannelL2And1512EdgeTvAnalyticDerivatives) {
  Checks();
}
TEST(FeatureVisualization, BatchMeanScaleAndUniformRawDrawOracle) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto noise = m::FeatureNoise(2, 1423);
  std::mt19937 rng(1423);
  auto expected = noise.to_vector<float>();
  for (auto &v : expected)
    v = float((double(rng()) + .5) / 4294967296.);
  EXPECT_EQ(noise.to_vector<float>(), expected);
  auto x = noise.clone().requires_grad_();
  const auto channels = n::ones({2}, n::DType::Int32);
  const auto out = m::VisualizeFeature(x, n::concat({x, x * 2.}, 3), channels, 2, .01f, .2f);
  const auto g = n::grad(out.mean, {x})[0];
  auto first = n::slice_axis(x, 0, 0, 1).detach().clone().requires_grad_();
  const auto o = m::VisualizeFeature(first, n::concat({first, first * 2.}, 3),
                                     n::ones({1}, n::DType::Int32), 2, .01f, .2f);
  EXPECT_LE(
      n::max(n::abs(n::slice_axis(g, 0, 0, 1) * 2. - n::grad(o.mean, {first})[0])).item(),
      2e-9);
  EXPECT_THROW(m::VisualizeFeature(x, n::concat({x, x}, 3), channels, 2, -.1f, .2f),
               std::invalid_argument);
  EXPECT_THROW(m::FeatureNoise(0, 1), std::invalid_argument);
}
