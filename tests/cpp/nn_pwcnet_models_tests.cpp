#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main cnn_pwc_original_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/pwcnet.cpp"
#undef main
TEST(PwcTensor, Radius4CostEveryValueBothJacobians) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(CorrelationChecks());
}
TEST(PwcTensor, WarpValuesCoverageBothJacobians) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(WarpChecks());
}
TEST(PwcTensor, LearnedTransposeEveryValueInputKernelDerivative) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(TransposeChecks());
}
TEST(PwcTensor, UnitConversionsPyramidLossStrictMetrics) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(UnitsLossMetricChecks());
}
TEST(PwcTensor, IndependentDilatedContextValuesAndJacobians) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(ContextChecks());
}
TEST(PwcTensor, FullCompactCensusEvery126GradientAll67Stages) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(ModelChecks());
}

TEST(PwcTensor, SharedPyramidReusesAll18ConvolutionsWithoutHiddenState) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::ParameterStore p(163);
  const auto image = n::full({1, 64, 64, 3}, .4);
  const auto out = m::PWCNet(p, image, image, 64, 64, true);
  int feature_leaves = 0;
  for (const auto &name : p.names())
    feature_leaves += name.starts_with("pyramid/");
  EXPECT_EQ(feature_leaves, 36);
  EXPECT_EQ(p.size(), 126);
  for (int level = 0; level < 6; ++level)
    EXPECT_TRUE(n::all(n::equal(out.stages[level], out.stages[level + 6])).item<bool>());
  const auto count = p.size();
  m::PWCNet(p, image, image, 64, 64, true);
  EXPECT_EQ(p.size(), count);
  EXPECT_THROW(m::PwcHeNormal(0), std::invalid_argument);
  EXPECT_THROW(m::PWCNet(p, n::zeros({1, 64, 64, 1}), image, 64, 64, true),
               std::invalid_argument);
}
