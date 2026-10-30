
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_johnson_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/johnson_fast_style.cpp"
#undef main
TEST(JohnsonFastStyle, IndependentReflectionResidualJacobian) {
  ReflectResidualChecks();
}
TEST(JohnsonFastStyle, IndependentStrideEveryValueAndAdjoint) {
  StrideChecks();
}
TEST(JohnsonFastStyle, IndependentContentStyleTvEveryAdjoint) {
  LossChecks();
}
TEST(JohnsonFastStyle, FullAndCompactStageParameterBnCensus) {
  ModelChecks();
}
TEST(JohnsonFastStyle, GenuineFullWidthKerasEveryBatchedPixelParity) {
  const std::filesystem::path root = "build/cnn-datasets/style_transfer";
  if (!std::filesystem::exists(root / "johnson_reference.npz"))
    GTEST_SKIP() << "prepare genuine Keras fixtures";
  dm::nn::set_num_threads(2);
  KerasParity(LoadTensorMap((root / "vgg16.npz").string()),
              LoadTensorMap((root / "johnson_reference.npz").string()));
}
TEST(JohnsonFastStyle, FrozenOwnershipAndGeometry) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::set_num_threads(2);
  const std::filesystem::path root = "build/cnn-datasets/style_transfer";
  if (!std::filesystem::exists(root / "vgg16.npz"))
    GTEST_SKIP() << "prepare official weights";
  auto make = [&] {
    const auto weights = m::LoadTensorMap((root / "vgg16.npz").string());
    return m::FrozenVgg16(weights);
  };
  auto vgg = make();
  vgg.CheckFrozen();
  const auto x = n::full({1, 16, 24, 3}, .4).requires_grad_();
  const auto f = vgg.Extract(x);
  const auto g = n::grad(n::mean(f.content), {x})[0];
  EXPECT_TRUE(n::all(n::is_finite(g)).item<bool>());
  EXPECT_GT(n::sum(n::abs(g)).item(), 0.);
  EXPECT_THROW(vgg.Extract(x, 5), std::invalid_argument);
  EXPECT_THROW(vgg.Extract(x, 0), std::invalid_argument);
  EXPECT_THROW(m::FrozenVgg16(m::TensorMap{}), std::invalid_argument);
  {
    n::NoGradGuard guard;
    auto &b = vgg.constants.at("block1_conv1_b");
    b.copy_(b + 1.);
  }
  EXPECT_THROW(vgg.CheckFrozen(), std::invalid_argument);
}
TEST(JohnsonFastStyle, EveryPreparedPixelAndPreviewHalfByte) {
  namespace n = dm::nn;
  namespace m = n::models;
  std::vector<std::int32_t> values(3 * 3072);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = (i * 31 + i / 3072 * 17) % 256;
  const auto raw = n::from_vector(values, {3, 32, 32, 3});
  const auto data = m::FastStyleImages(raw, {2, 0}), paint = n::full({1, 64, 64, 3}, .37),
             pred = data * .8;
  const auto tiles = m::JohnsonPreviewTiles(data, paint, pred).to_vector<float>(),
             x = data.to_vector<float>(), p = pred.to_vector<float>();
  const float style = .37f;
  for (int b = 0; b < 2; ++b)
    for (int y = 0; y < 64; ++y)
      for (int col = 0; col < 64; ++col)
        for (int c = 0; c < 3; ++c) {
          const int source = b ? 0 : 2, top = y / 2, left = col / 2,
                    bottom = std::min(top + 1, 31), right = std::min(left + 1, 31);
          const float dy = (y % 2) * .5f, dx = (col % 2) * .5f;
          auto pixel = [&](int row, int column) {
            return float(values[((source * 32 + row) * 32 + column) * 3 + c]) / 255.f;
          };
          const float a = pixel(top, left) * (1 - dx) + pixel(top, right) * dx,
                      bb = pixel(bottom, left) * (1 - dx) + pixel(bottom, right) * dx;
          const int index = ((b * 64 + y) * 64 + col) * 3 + c;
          EXPECT_NEAR(x[index], a * (1 - dy) + bb * dy, 1e-7);
          const double channel[] = {x[index], style, p[index]};
          for (int panel = 0; panel < 3; ++panel)
            EXPECT_NEAR(tiles[((b * 64 + y) * 192 + panel * 64 + col) * 3 + c],
                        std::lround(255 * channel[panel]) / 255., 4e-8);
        }
  EXPECT_THROW(m::FastStyleImages(raw, {3}), std::invalid_argument);
}
TEST(JohnsonFastStyle, PairedCheckpointValidationBeforeMutation) {
  namespace n = dm::nn;
  namespace m = n::models;
  n::ParameterStore p(17), state(18);
  p.get("w", {2}, n::init::ones());
  state.get("mean", {2}, n::init::zeros());
  const auto path =
      (std::filesystem::temp_directory_path() / "dm-johnson-native-checkpoint.npz").string();
  m::SaveJohnson(p, state, path);
  const auto original = p.at("w").clone(), before = state.at("mean").clone();
  {
    n::NoGradGuard guard;
    p.at("w").copy_(n::zeros({2}));
  }
  m::RestoreJohnson(p, state, path);
  EXPECT_TRUE(n::all(n::equal(original, p.at("w"))).item<bool>());
  n::save_npz(path, {{"transform/w", n::ones({2})}, {"statistics/mean", n::ones({3})}});
  EXPECT_THROW(m::RestoreJohnson(p, state, path), std::invalid_argument);
  EXPECT_TRUE(n::all(n::equal(original, p.at("w"))).item<bool>());
  EXPECT_TRUE(n::all(n::equal(before, state.at("mean"))).item<bool>());
  std::filesystem::remove(path);
}
