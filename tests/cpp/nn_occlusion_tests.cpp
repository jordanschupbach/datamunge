
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_occlusion_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.11_interpretability/occlusion_sensitivity.cpp"
#undef main
#include "../fixtures/nn_occlusion_reference.hpp"
TEST(Occlusion, Every240CounterfactualPixel12Drops20Coverage40OverlapAndEdgeNullFullControls) {
  Checks();
}
TEST(Occlusion, EveryTensorInterventionAndProjectedValueMatchesIndependentHostOracle) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto x = (n::sin(n::arange(0, 60) * .3) * .4 + .5).reshape({3, 4, 5, 1}),
             replacement = n::full(x.shape(), .2),
             targets = n::from_vector(std::vector<int>{0, 1, 0});
  auto predict = [](const n::Tensor &p) {
    const auto score = n::sum(p * p, {1, 2, 3});
    return n::stack({score, -score}, 1);
  };
  for (int batch : {1, 7, 32}) {
    const auto a =
        m::OcclusionSensitivity(x, replacement, targets, {2, 2, 2, 2}, batch, predict);
    const auto b = occlusion_reference::OcclusionSensitivity(x, replacement, targets,
                                                             {2, 2, 2, 2}, batch, predict);
    EXPECT_EQ(a.rectangles.to_vector<int>(), b.rectangles);
    EXPECT_EQ(a.coverage.to_vector<int>(), b.coverage);
    EXPECT_EQ(a.occluded_inputs.to_vector<float>(), b.occluded_inputs);
    EXPECT_EQ(a.occluded_logits.to_vector<float>(), b.occluded_logits);
    EXPECT_EQ(a.drops.to_vector<double>(), b.drops);
    const auto mean = a.pixel_mean.to_vector<double>();
    for (int i = 0; i < mean.size(); ++i)
      EXPECT_NEAR(mean[i], b.pixel_mean[i], 1e-12);
  }
}
TEST(Occlusion, InvalidCoverageTargetsAndPredictorContracts) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto x = n::ones({2, 4, 5, 1}), target = n::zeros({2}, n::DType::Int32);
  auto predict = [](const n::Tensor &p) {
    return n::zeros({p.size(0), 2});
  };
  EXPECT_THROW(m::OcclusionAnchors(5, 2, 3), std::invalid_argument);
  EXPECT_THROW(m::OcclusionSensitivity(x, x, n::full({2}, 2, n::DType::Int32), {2, 2, 2, 2}, 4,
                                       predict),
               std::invalid_argument);
  EXPECT_THROW(m::OcclusionSensitivity(x, x, target, {2, 2, 2, 2}, 4,
                                       [](const n::Tensor &p) {
                                         return n::full({p.size(0), 2}, NAN);
                                       }),
               std::invalid_argument);
}
