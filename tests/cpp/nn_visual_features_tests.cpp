
#include <gtest/gtest.h>

#include <dm/dm.hpp>

#include "../fixtures/nn_visual_features_checks.hpp"
TEST(VisualFeatures, OriginalZeroConstantPositiveAndSharedHalfRenderingChecks) {
  RenderingChecks();
}
TEST(VisualFeatures, TensorChannelOrderSharedAndPerChannelScalesAndInvalidInput) {
  namespace n       = dm::nn;
  namespace m       = n::models;
  const auto x      = n::from_vector(std::vector<float>{0, 2, 0, 2, 0, 4, 0, 4}, {2, 1, 2, 2});
  const auto shared = m::VisualChannelImages(x, false), local = m::VisualChannelImages(x, true);
  EXPECT_EQ(shared.shape(), (n::Shape{4, 1, 2, 1}));
  EXPECT_EQ(shared.to_vector<float>(), (std::vector<float>{0, 0, .5, .5, 0, 0, 1, 1}));
  EXPECT_EQ(local.to_vector<float>(), (std::vector<float>{0, 0, 1, 1, 0, 0, 1, 1}));
  EXPECT_THROW(m::VisualChannelImages(n::full({1, 2, 2, 1}, -1)), std::invalid_argument);
  EXPECT_THROW(m::VisualChannelImages(n::full({1, 2, 2, 1}, NAN)), std::invalid_argument);
}
TEST(VisualFeatures, ActualFullFrozenVgg19FiveTapGeometryDeterminismAndAll32ImmutableConstants) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  dm::datasets::prepare_example_data(
      "style", (std::filesystem::path(dm::datasets::data_home()) / "style_transfer").string());
  const auto tensors = m::LoadTensorMap(
      (std::filesystem::path(dm::datasets::data_home()) / "style_transfer/vgg19.npz").string());
  const m::FrozenVgg19 model(tensors);
  const auto           assets = std::filesystem::path(dm::datasets::data_home()) / "style_transfer";
  const auto           rgb    = n::concat(
      {m::LoadStyleImage(assets / "content.jpg", 64), m::LoadStyleImage(assets / "style.jpg", 64)},
      0);
  std::map<std::string, std::vector<float>> before;
  for (const auto& [key, value] : model.constants)
    before.emplace(key, value.to_vector<float>());
  n::NoGradGuard guard;
  const auto     a = m::VggVisualFeatures(model, rgb), b = m::VggVisualFeatures(model, rgb);
  ASSERT_EQ(a.activation.size(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(a.activation[i].shape(), (n::Shape{2, 64 >> i, 64 >> i, m::FrozenVgg19::Widths[i]}));
    EXPECT_EQ(a.activation[i].to_vector<float>(), b.activation[i].to_vector<float>());
  }
  for (const auto& [key, value] : model.constants)
    EXPECT_EQ(value.to_vector<float>(), before.at(key));
}
