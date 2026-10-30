
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_cam_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.11_interpretability/grad_cam.cpp"
#undef main
TEST(GradCam, EveryHostWeightAlphaMapAnd24TrueExpAffineHigherDerivatives) {
  FormulaChecks();
}
TEST(GradCam, AllThirteenSharedModelOutputsMatchOriginalWorkflowAndImmutableWeights) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  sal::SaliencyModel source(false);
  source.Assign(sal::LoadArrays("output/nn/quick/models/vanilla_saliency/model_arrays.npz"));
  CamModel original(source.Export());
  const auto pools = m::LoadSaliency(std::filesystem::path(std::getenv("HOME")) /
                                     ".cache/datamunge/datasets/saliency");
  const auto d = m::SaliencyRows(pools[2], sal::FirstPerDigit(pools[2]));
  const auto targets = n::argmax(source.Predict(d), 1).to(n::DType::Int32);
  const auto before = source.parameters.flatten();
  const auto a = m::SaliencyCam(source.parameters, d.images, targets);
  const auto b = original.Maps(d, targets);
  std::vector<n::Tensor> values = {a.activation,
                                   a.gradient,
                                   a.maps.grad_weights,
                                   a.maps.grad_signed,
                                   a.maps.grad_map,
                                   a.maps.positive_alpha,
                                   a.maps.plus_weights,
                                   a.maps.plus_map,
                                   a.normalized,
                                   a.plus_normalized,
                                   a.resized,
                                   a.plus_resized,
                                   a.logits};
  ASSERT_EQ(values.size(), b.size());
  for (int i = 0; i < 13; ++i)
    EXPECT_EQ(n::max(n::abs(values[i] - b[i])).item(), 0);
  EXPECT_EQ(before, source.parameters.flatten());
  EXPECT_LE(
      n::max(n::abs(m::SaliencyCamTail(source.parameters, a.activation) - a.logits)).item(),
      2e-6);
}
TEST(GradCam, InvalidNegativeFeatureNonfiniteAndNormalizationContracts) {
  namespace n = dm::nn;
  namespace m = n::models;
  EXPECT_THROW(m::ClassActivationMaps(n::full({1, 2, 2, 1}, -1), n::ones({1, 2, 2, 1})),
               std::invalid_argument);
  EXPECT_THROW(m::ClassActivationMaps(n::ones({1, 2, 2, 1}), n::full({1, 2, 2, 1}, NAN)),
               std::invalid_argument);
  EXPECT_EQ(n::sum(m::NormalizeCam(n::zeros({2, 3, 3}))).item(), 0);
  EXPECT_THROW(m::NormalizeCam(n::full({1, 2, 2}, -1)), std::invalid_argument);
}
