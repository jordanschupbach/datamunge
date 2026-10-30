#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main cnn_color_original_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/image_colorization.cpp"
#undef main
namespace reference {
using ColorTriple = std::array<double, 3>;
inline double ColorLinear(double value) {
  return value <= .04045 ? value / 12.92 : std::pow((value + .055) / 1.055, 2.4);
}
inline double ColorGamma(double value) {
  return value <= .0031308 ? 12.92 * value : 1.055 * std::pow(value, 1 / 2.4) - .055;
}
inline double ColorLabF(double value) {
  constexpr double delta = 6. / 29;
  return value > delta * delta * delta ? std::cbrt(value)
                                       : value / (3 * delta * delta) + 4. / 29;
}
inline double ColorLabInverse(double value) {
  constexpr double delta = 6. / 29;
  return value > delta ? value * value * value : 3 * delta * delta * (value - 4. / 29);
}
inline ColorTriple ColorRGBToLab(ColorTriple rgb) {
  for (auto &value : rgb) {
    check::Require(std::isfinite(value) && value >= 0 && value <= 1, "sRGB range");
    value = ColorLinear(value);
  }
  const double x =
      ColorLabF((.412453 * rgb[0] + .357580 * rgb[1] + .180423 * rgb[2]) / .95047);
  const double y = ColorLabF(.212671 * rgb[0] + .715160 * rgb[1] + .072169 * rgb[2]);
  const double z =
      ColorLabF((.019334 * rgb[0] + .119193 * rgb[1] + .950227 * rgb[2]) / 1.08883);
  return {116 * y - 16, 500 * (x - y), 200 * (y - z)};
}
/// Unclipped sRGB. Out-of-gamut fractions are reported before display clipping.
inline ColorTriple ColorLabToRGB(const ColorTriple &lab) {
  for (double value : lab)
    check::Require(std::isfinite(value), "finite Lab reconstruction");
  const double y = (lab[0] + 16) / 116;
  const double x = .95047 * ColorLabInverse(y + lab[1] / 500);
  const double yy = ColorLabInverse(y), z = 1.08883 * ColorLabInverse(y - lab[2] / 200);
  return {ColorGamma(3.2404813432005266 * x - 1.5371515162713185 * yy - .4985363261688878 * z),
          ColorGamma(-.9692549499965684 * x + 1.8759900014898907 * yy + .0415559265582928 * z),
          ColorGamma(.0556466391351772 * x - .2040413383665112 * yy + 1.0573110696453443 * z)};
}

} // namespace reference
TEST(ColorTensor, KnownColorsAnd2197RoundTrips) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(ColorScienceChecks());
}
TEST(ColorTensor, DataAndIndependentMetricNormalizations) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(DataMetricChecks());
}
TEST(ColorTensor, IndependentTanhRegressionAndAmbiguousMean) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(LossAndResizeChecks());
}
TEST(ColorTensor, EveryHalfPixelValueAndJacobian) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(SpatialResizeChecks());
}
TEST(ColorTensor, All20ParameterGradientsOddRectangularStages) {
  dm::nn::set_num_threads(2);
  EXPECT_NO_THROW(ModelChecks());
}

TEST(ColorTensor, EverySelectedByteAndPhysicalColorMatchesIndependentOracle) {
  namespace n = dm::nn;
  namespace m = n::models;
  std::vector<int32_t> bytes(3 * 3072);
  for (size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = (i * 31 + i / 7) % 256;
  const auto data = m::ColorPairs(n::from_vector(bytes, {3, 32, 32, 3}), {2, 0, 2});
  const auto l = data.lightness.to_vector<float>(), ab = data.chroma.to_vector<float>(),
             rgb = data.rgb.to_vector<float>();
  const int ids[] = {2, 0, 2};
  for (int b = 0; b < 3; ++b)
    for (int i = 0; i < 1024; ++i) {
      reference::ColorTriple color;
      for (int c = 0; c < 3; ++c) {
        color[c] = bytes[(ids[b] * 1024 + i) * 3 + c] / 255.;
        EXPECT_EQ(rgb[(b * 1024 + i) * 3 + c], float(color[c]));
      }
      const auto lab = reference::ColorRGBToLab(color);
      EXPECT_NEAR(l[b * 1024 + i], float(lab[0] / 50 - 1), 1e-7);
      for (int c = 0; c < 2; ++c)
        EXPECT_NEAR(ab[(b * 1024 + i) * 2 + c], float(lab[c + 1] / 128), 1e-7);
    }
}
TEST(ColorTensor, IndependentColorTransformJacobiansAcrossPiecewiseBranches) {
  namespace n = dm::nn;
  namespace m = n::models;
  const std::vector<double> values = {.001, .02, .03, .2, .4, .8, .9, .7, .3};
  auto x = n::from_vector(values, {3, 3}).requires_grad_();
  const auto up =
      n::from_vector(std::vector<double>{.1, -.2, .3, -.4, .5, -.6, .7, -.8, .9}, {3, 3});
  const auto g = n::grad(n::sum(m::ColorRGBToLabTensor(x) * up), {x})[0].to_vector<double>();
  const auto uv = up.to_vector<double>();
  for (int i = 0; i < 9; ++i) {
    auto hi = values, lo = values;
    hi[i] += 1e-6;
    lo[i] -= 1e-6;
    double diff = 0;
    const int b = i / 3;
    reference::ColorTriple h{hi[3 * b], hi[3 * b + 1], hi[3 * b + 2]},
        l{lo[3 * b], lo[3 * b + 1], lo[3 * b + 2]};
    const auto hl = reference::ColorRGBToLab(h), ll = reference::ColorRGBToLab(l);
    for (int c = 0; c < 3; ++c)
      diff += (hl[c] - ll[c]) * uv[3 * b + c] / 2e-6;
    EXPECT_NEAR(g[i], diff, 2e-7);
  }
  const auto raw = m::ColorLabToRGBTensor(n::tensor({50., 128., -128.}));
  EXPECT_TRUE(n::any(n::logical_or(n::less(raw, 0.), n::greater(raw, 1.))).item<bool>());
}
TEST(ColorTensor, EveryPreviewPixelAndHalfByteRounding) {
  namespace n = dm::nn;
  namespace m = n::models;
  std::vector<float> values(2 * 3 * 4 * 3);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = (int(i % 31) - 3) / 26.f;
  values[0] = std::nextafter(.5f / 255.f, 0.f);
  const auto original = n::from_vector(values, {2, 3, 4, 3});
  const m::ColorData data{n::zeros({2, 3, 4, 1}), n::zeros({2, 3, 4, 2}), original};
  const auto gray = original * .5, mean = original * .7, learned = original * .9;
  const auto tiles = m::ColorPreviewTiles(data, learned, gray, mean).to_vector<float>();
  const std::vector<float> cols[] = {gray.to_vector<float>(), mean.to_vector<float>(),
                                     learned.to_vector<float>(), values};
  for (int b = 0; b < 2; ++b)
    for (int y = 0; y < 3; ++y)
      for (int x = 0; x < 4; ++x)
        for (int k = 0; k < 4; ++k)
          for (int c = 0; c < 3; ++c) {
            const int i = ((b * 3 + y) * 4 + x) * 3 + c;
            const float expected =
                float(std::lround(255 * std::clamp(cols[k][i], 0.f, 1.f))) / 255.f;
            EXPECT_EQ(tiles[((b * 3 + y) * 16 + k * 4 + x) * 3 + c], expected);
          }
}
