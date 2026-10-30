
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_filter_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.11_interpretability/filter_activations.cpp"
#undef main
TEST(VisualFeatures, OriginalZeroConstantPositiveAndSharedHalfRenderingChecks) {
  RenderingChecks();
}
TEST(VisualFeatures, TensorChannelOrderSharedAndPerChannelScalesAndInvalidInput) {
  namespace n = dm::nn;
  namespace m = n::models;
  const auto x = n::from_vector(std::vector<float>{0, 2, 0, 2, 0, 4, 0, 4}, {2, 1, 2, 2});
  const auto shared = m::VisualChannelImages(x, false),
             local = m::VisualChannelImages(x, true);
  EXPECT_EQ(shared.shape(), (n::Shape{4, 1, 2, 1}));
  EXPECT_EQ(shared.to_vector<float>(), (std::vector<float>{0, 0, .5, .5, 0, 0, 1, 1}));
  EXPECT_EQ(local.to_vector<float>(), (std::vector<float>{0, 0, 1, 1, 0, 0, 1, 1}));
  EXPECT_THROW(m::VisualChannelImages(n::full({1, 2, 2, 1}, -1)), std::invalid_argument);
  EXPECT_THROW(m::VisualChannelImages(n::full({1, 2, 2, 1}, NAN)), std::invalid_argument);
}
TEST(VisualFeatures,
     ActualFullFrozenVgg19FiveTapGeometryDeterminismAndAll32ImmutableConstants) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  const auto tensors = m::LoadTensorMap("build/cnn-datasets/style_transfer/vgg19.npz");
  const m::FrozenVgg19 model(tensors);
  n::Tensor rgb;
  for (const auto &[key, value] :
       n::load_npz("build/cnn-datasets/filter_activations/inputs.npz"))
    if (key == "rgb")
      rgb = value;
  std::map<std::string, std::vector<float>> before;
  for (const auto &[key, value] : model.constants)
    before.emplace(key, value.to_vector<float>());
  n::NoGradGuard guard;
  const auto a = m::VggVisualFeatures(model, rgb), b = m::VggVisualFeatures(model, rgb);
  ASSERT_EQ(a.activation.size(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(a.activation[i].shape(),
              (n::Shape{2, 64 >> i, 64 >> i, m::FrozenVgg19::Widths[i]}));
    EXPECT_EQ(a.activation[i].to_vector<float>(), b.activation[i].to_vector<float>());
  }
  for (const auto &[key, value] : model.constants)
    EXPECT_EQ(value.to_vector<float>(), before.at(key));
}
