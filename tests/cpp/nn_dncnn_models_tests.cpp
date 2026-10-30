#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main cnn_dncnn_original_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/dncnn.cpp"
#undef main
TEST(DnTensor, CorruptionMomentsResidualPairingAndMetrics) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(NoiseAndMetricChecks());
}
TEST(DnTensor, EverySignedResidualLossJacobian) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(ResidualChecks());
}
TEST(DnTensor, EveryBnInputAffineDerivativeAndMovingState) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(BatchNormChecks());
}
TEST(DnTensor, FullCompactOrthogonalGraphEveryGradient) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(ModelChecks());
}

TEST(DnTensor, IndependentBoxMullerEveryNoisyPixelAndOddLength) {
  namespace n = dm::nn;
  namespace m = n::models;
  std::vector<float> clean(19);
  for (int i = 0; i < 19; ++i)
    clean[i] = float(i % 7) / 6;
  const auto actual = m::DnCorrupt(n::from_vector(clean, {1, 1, 19, 1}), 25. / 255, 41);
  const auto noisy = actual.noisy.to_vector<float>(), noise = actual.noise.to_vector<float>();
  std::mt19937_64 rng(41);
  for (int i = 0; i < 19; i += 2) {
    const double u = ((rng() >> 11) + 1.) / 9007199254740992.,
                 v = ((rng() >> 11) + 1.) / 9007199254740992.;
    const double radius = std::sqrt(-2 * std::log(u)), angle = 6.2831853071795864769 * v;
    const double z[] = {radius * std::cos(angle), radius * std::sin(angle)};
    for (int k = 0; k < 2 && i + k < 19; ++k) {
      const float expected = float(clean[i + k] + (25. / 255) * z[k]);
      EXPECT_EQ(noisy[i + k], expected);
      EXPECT_EQ(noise[i + k], expected - clean[i + k]);
    }
  }
  std::vector<int32_t> rgb(3 * 3072);
  for (size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = (i * 37 + i / 3) % 256;
  const auto gray =
      m::DnImages(n::from_vector(rgb, {3, 32, 32, 3}), {2, 0, 2}).to_vector<float>();
  const int ids[] = {2, 0, 2};
  for (int b = 0; b < 3; ++b)
    for (int i = 0; i < 1024; ++i) {
      const int j = (ids[b] * 1024 + i) * 3;
      const float expected = float(.299 * (rgb[j] / 255.) + .587 * (rgb[j + 1] / 255.) +
                                   .114 * (rgb[j + 2] / 255.));
      EXPECT_EQ(gray[b * 1024 + i], expected);
    }
}
TEST(DnTensor, IndependentReflectedValuesAndEveryInputDerivative) {
  namespace n = dm::nn;
  namespace m = n::models;
  for (int h : {1, 2, 4})
    for (int w : {1, 3, 5}) {
      std::vector<float> values(2 * h * w), up(values.size());
      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = float(int(i % 13) - 4) * .17f;
        up[i] = float(int(i % 7) - 3) * .11f;
      }
      const auto x = n::from_vector(values, {2, h, w, 1}).requires_grad_();
      const auto y = m::DnBoxFilter(x);
      const auto g =
          n::grad(n::sum(y * n::from_vector(up, {2, h, w, 1})), {x})[0].to_vector<double>();
      const auto out = y.to_vector<double>();
      std::vector<double> expected_gradient(values.size());
      auto reflect = [](int i, int length) {
        if (length == 1)
          return 0;
        while (i < 0 || i >= length)
          i = i < 0 ? -i : 2 * length - 2 - i;
        return i;
      };
      for (int b = 0; b < 2; ++b)
        for (int row = 0; row < h; ++row)
          for (int col = 0; col < w; ++col) {
            double sum = 0;
            const int q = (b * h + row) * w + col;
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx) {
                const int i = (b * h + reflect(row + dy, h)) * w + reflect(col + dx, w);
                sum += values[i];
                expected_gradient[i] += double(up[q]) / 9;
              }
            EXPECT_NEAR(out[q], sum / 9, 1e-7);
          }
      for (size_t i = 0; i < g.size(); ++i)
        EXPECT_NEAR(g[i], expected_gradient[i], 1e-7);
    }
}
TEST(DnTensor, EveryFivePanelPixelIncludingClippingAndHalfByte) {
  namespace n = dm::nn;
  namespace m = n::models;
  std::vector<float> values(2 * 3 * 4);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = float(int(i % 17) - 2) / 13;
  values[0] = std::nextafter(.5f / 255.f, 0.f);
  const auto clean = n::from_vector(values, {2, 3, 4, 1}), noisy = clean + .2,
             restored = clean - .1, box = clean * .8;
  const m::DnData data{clean, noisy, noisy - clean};
  const auto tiles = m::DnPreviewTiles(data, restored, box).to_vector<float>();
  const auto nv = noisy.to_vector<float>(), rv = restored.to_vector<float>(),
             bv = box.to_vector<float>();
  for (int b = 0; b < 2; ++b)
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 4; ++x) {
        const int i = (b * 3 + y) * 4 + x;
        const double cols[] = {nv[i], bv[i], rv[i], values[i],
                               4 * std::abs(double(rv[i]) - values[i])};
        for (int j = 0; j < 5; ++j)
          for (int c = 0; c < 3; ++c) {
            const double v = j == 4 && c != 0 ? 0. : cols[j];
            const float expected = float(std::lround(255 * std::clamp(v, 0., 1.))) / 255.f;
            EXPECT_EQ(tiles[((b * 3 + y) * 20 + j * 4 + x) * 3 + c], expected);
          }
      }
}
