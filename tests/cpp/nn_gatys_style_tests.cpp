
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_gatys_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/gatys_style_transfer.cpp"
#undef main
TEST(GatysStyleTransfer, IndependentPreprocessAndAdjoints) {
  PreprocessChecks();
}
TEST(GatysStyleTransfer, IndependentGramContentEveryAdjoint) {
  GramAndContentChecks();
}
TEST(GatysStyleTransfer, GenuineFullWidthKerasEveryPixelParity) {
  const std::filesystem::path root = "build/cnn-datasets/style_transfer";
  if (!std::filesystem::exists(root / "reference.npz"))
    GTEST_SKIP() << "prepare official VGG/Keras reference assets";
  dm::nn::set_num_threads(2);
  const auto w = LoadTensorMap((root / "vgg19.npz").string()),
             r = LoadTensorMap((root / "reference.npz").string());
  KerasParity(w, r);
}
TEST(GatysStyleTransfer, FrozenOwnershipValidationAndMutationDetection) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  const std::filesystem::path root = "build/cnn-datasets/style_transfer";
  if (!std::filesystem::exists(root / "vgg19.npz"))
    GTEST_SKIP() << "prepare official VGG weights";
  const auto construct = [&] {
    auto weights = m::LoadTensorMap((root / "vgg19.npz").string());
    return m::FrozenVgg19(weights);
  };
  auto vgg = construct();
  vgg.CheckFrozen();
  const auto x = n::full({1, 16, 32, 3}, .4).requires_grad_();
  const auto out = vgg.Extract(x);
  const auto g = n::grad(n::mean(out.content), {x})[0];
  EXPECT_TRUE(n::all(n::is_finite(g)).item<bool>());
  EXPECT_GT(n::sum(n::abs(g)).item(), 0.);
  for (const auto &[name, w] : vgg.constants)
    EXPECT_FALSE(w.requires_grad());
  vgg.CheckFrozen();
  {
    n::NoGradGuard guard;
    auto &w = vgg.constants.at("block1_conv1_b");
    w.copy_(w + 1.);
  }
  EXPECT_THROW(vgg.CheckFrozen(), std::invalid_argument);
  EXPECT_THROW(m::FrozenVgg19(m::TensorMap{}), std::invalid_argument);
  EXPECT_THROW(vgg.Extract(n::ones({1, 15, 32, 3})), std::invalid_argument);
}
TEST(GatysStyleTransfer, EveryHalfBytePngAndTensorImageIngress) {
  namespace n = dm::nn;
  namespace m = n::models;
  std::vector<double> pixels;
  for (int i = 0; i < 256; ++i) {
    const double x = std::min(1., (i + .5) / 255.);
    pixels.insert(pixels.end(), {x, x, x});
  }
  const auto path = std::filesystem::temp_directory_path() / "dm-gatys-every-half-byte.png";
  m::WriteStylePng(n::from_vector(pixels, {1, 1, 256, 3}), path);
  const auto decoded = n::from_image(dm::image::read_png(path.string())).to_vector<double>();
  ASSERT_EQ(decoded.size(), pixels.size());
  for (std::size_t i = 0; i < pixels.size(); ++i)
    EXPECT_EQ(decoded[i], std::lround(255 * pixels[i]));
  std::filesystem::remove(path);
  const std::filesystem::path source = "build/cnn-datasets/style_transfer/content.jpg";
  if (!std::filesystem::exists(source))
    GTEST_SKIP() << "prepare original source JPEG";
  const auto image = m::LoadStyleImage(source, 64);
  EXPECT_EQ(image.shape(), n::Shape({1, 64, 64, 3}));
  EXPECT_TRUE(n::all(n::is_finite(image)).item<bool>());
}
