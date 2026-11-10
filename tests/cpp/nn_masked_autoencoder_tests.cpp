#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <limits>

namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnMaskedAutoencoder, PatchMappingInverseStridesAndAllPixelGradients) {
  std::vector<double> values(784);
  for (int i = 0; i < 784; ++i)
    values[i] = double(i) / 784;
  const auto image   = n::from_vector(values, {1, 28, 28, 1}).requires_grad_();
  const auto patches = m::MaePatches(image);
  for (int patch = 0; patch < 16; ++patch)
    for (int pixel = 0; pixel < 49; ++pixel)
      EXPECT_EQ(patches.at({0, patch, pixel}), values[m::MaePixel(patch, pixel)]);
  const auto flat = m::MaeImage(patches);
  EXPECT_EQ(flat.to_vector<double>(), values);
  EXPECT_EQ(n::grad(n::sum(flat), {image})[0].to_vector<double>(), (std::vector<double>(784, 1)));
  const auto strided = n::transpose(n::transpose(image, {0, 2, 1, 3}).contiguous(), {0, 2, 1, 3});
  EXPECT_FALSE(strided.is_contiguous());
  EXPECT_EQ(m::MaePatches(strided).to_vector<double>(), patches.to_vector<double>());
  EXPECT_EQ(m::MaeImage(n::zeros({0, 16, 49})).shape(), (n::Shape{0, 784}));
  EXPECT_THROW(m::MaePixel(16, 0), std::invalid_argument);
  EXPECT_THROW(m::MaePixel(0, 49), std::invalid_argument);
  EXPECT_THROW(m::MaePatches(n::zeros({1, 28, 28, 2})), std::invalid_argument);
}

TEST(NnMaskedAutoencoder, SelectionIsDeterministicUniqueAndPositionsUseGroupedOrder) {
  const auto ids       = n::tensor({0., 1., 60000., 66921.}, {4}, n::DType::Int64);
  const auto selection = m::MaeSelection(ids);
  EXPECT_EQ(selection.shape(), (n::Shape{4, 4, 16}));
  EXPECT_FALSE(selection.requires_grad());
  for (int b = 0; b < 4; ++b) {
    int columns[16]{};
    for (int row = 0; row < 4; ++row) {
      int count = 0;
      for (int p = 0; p < 16; ++p) {
        const auto v = selection.at({b, row, p});
        EXPECT_TRUE(v == 0 || v == 1);
        count += int(v);
        columns[p] += int(v);
      }
      EXPECT_EQ(count, 1);
    }
    for (int count : columns)
      EXPECT_LE(count, 1);
  }
  EXPECT_EQ(selection.to_vector<float>(), m::MaeSelection(ids).to_vector<float>());
  EXPECT_NE(selection.to_vector<float>(), m::MaeSelection(ids, 1).to_vector<float>());
  const auto shuffled = n::take_rows(ids, {3, 1, 0, 2});
  EXPECT_EQ(
      m::MaeSelection(shuffled).to_vector<float>(),
      n::take_rows(selection, {3, 1, 0, 2}).to_vector<float>());
  EXPECT_EQ(
      m::MaeSelection(ids, 0, 1907, n::DType::Float64).to_vector<double>(),
      n::cast(selection, n::DType::Float64).to_vector<double>());
  EXPECT_EQ(m::MaeSelection(n::zeros({0}, n::DType::Int32)).shape(), (n::Shape{0, 4, 16}));
  const auto position = m::MaePosition(4, n::DType::Float64);
  EXPECT_EQ(n::select(position, 0, 0).to_vector<double>(), (std::vector<double>{0, 1, 0, 1}));
  EXPECT_NEAR(position.at({5, 0}), std::sin(1.), 1e-15);
  EXPECT_NEAR(position.at({5, 2}), std::sin(1.), 1e-15);
}

TEST(NnMaskedAutoencoder, HiddenOnlyLossHasExactDenominatorAndBothContinuousGradients) {
  const auto type       = n::DType::Float64;
  const auto prediction = n::full({1, 2, 2}, .3, type).requires_grad_();
  const auto target     = n::full({1, 2, 2}, .7, type).requires_grad_();
  const auto mask       = n::tensor({0., 1.}, {1, 2}, type);
  const auto loss       = m::MaeLoss(prediction, target, mask);
  const auto g          = n::grad(loss, {prediction, target});
  EXPECT_NEAR(loss.item(), .16, 1e-14);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(g[0].to_vector<double>()[i], i < 2 ? 0 : -.4, 1e-14);
    EXPECT_NEAR(g[1].to_vector<double>()[i], i < 2 ? 0 : .4, 1e-14);
  }
  const auto changed = n::tensor({.99, .99, .3, .3}, {1, 2, 2}, type);
  EXPECT_EQ(m::MaeLoss(changed, target, mask).item(), loss.item());
  EXPECT_THROW(m::MaeLoss(prediction, target, n::zeros({1, 2}, type)), std::invalid_argument);
  EXPECT_THROW(m::MaeLoss(prediction, target, n::full({1, 2}, .5, type)), std::invalid_argument);
}

TEST(NnMaskedAutoencoder, OriginalModelCountStandaloneDecodeAndHiddenInputIndependence) {
  n::ParameterStore  p(1877);
  std::vector<float> pixels(2 * 784);
  for (int i = 0; i < 2 * 784; ++i)
    pixels[i] = float((i * 13) % 31) / 30;
  const auto image    = n::from_vector(pixels, {2, 28, 28, 1});
  const auto selector = m::MaeSelection(n::tensor({0., 1.}, {2}, n::DType::Int32));
  const auto net      = m::MaskedAutoencoder(p, image, selector);
  EXPECT_EQ(p.count(), 13649);
  EXPECT_EQ(p.size(), 35u);
  EXPECT_EQ(net.code.shape(), (n::Shape{2, 4, 32}));
  EXPECT_EQ(net.encoder_attention.shape(), (n::Shape{2, 2, 4, 4}));
  EXPECT_EQ(net.decoder_attention.shape(), (n::Shape{2, 2, 16, 16}));
  const auto gradients = n::grad(m::MaeLoss(net.prediction, net.patches, net.mask), p.parameters());
  ASSERT_EQ(gradients.size(), 35u);
  for (const auto& g : gradients) {
    double squared = 0;
    for (float v : g.to_vector<float>()) {
      EXPECT_TRUE(std::isfinite(v));
      squared += double(v) * v;
    }
    EXPECT_GT(squared, 0);
  }
  const auto     before = p.flatten();
  n::NoGradGuard guard;
  EXPECT_EQ(
      m::MaeDecode(p, net.code, selector).prediction.to_vector<float>(),
      net.prediction.to_vector<float>());
  const auto changed = net.patches * (1. - n::expand_dims(net.mask, 2)) +
                       (1. - net.patches) * n::expand_dims(net.mask, 2);
  const auto hidden =
      m::MaskedAutoencoder(p, m::MaeImage(changed).reshape({2, 28, 28, 1}), selector);
  EXPECT_EQ(hidden.code.to_vector<float>(), net.code.to_vector<float>());
  EXPECT_EQ(hidden.prediction.to_vector<float>(), net.prediction.to_vector<float>());
  EXPECT_LT(n::max(n::abs(n::sum(net.encoder_attention, {-1}) - 1.)).item(), 2e-6);
  EXPECT_LT(n::max(n::abs(n::sum(net.decoder_attention, {-1}) - 1.)).item(), 2e-6);
  EXPECT_EQ(before, p.flatten());
}

TEST(NnMaskedAutoencoder, EncoderInputGradientsExcludeEveryHiddenPixelAndBlockFiniteDifferences) {
  const auto          type = n::DType::Float64;
  n::ParameterStore   p(1877, type);
  auto                image   = n::full({1, 28, 28, 1}, .25, type).requires_grad_();
  const auto          s       = m::MaeSelection(n::zeros({1}, n::DType::Int32), 0, 1907, type);
  const auto          encoded = m::MaeEncode(p, image, s);
  std::vector<double> weights(32);
  for (int i = 0; i < 32; ++i)
    weights[i] = i + 1;
  const auto g =
      n::grad(n::sum(encoded.code * n::from_vector(weights, {32})), {image})[0].to_vector<double>();
  double visible = 0;
  for (int patch = 0; patch < 16; ++patch)
    for (int within = 0; within < 49; ++within) {
      const auto v = g[m::MaePixel(patch, within)];
      if (encoded.mask.at({0, patch}) == 1)
        EXPECT_EQ(v, 0);
      else
        visible += std::abs(v);
    }
  EXPECT_GT(visible, 0);
  n::ParameterStore block(17, type);
  auto x = n::tensor({.1, .2, .3, .4, .2, .4, .3, .1}, {1, 2, 4}, type).requires_grad_();
  const n::TensorFunction f = [&](const auto& inputs) {
    return n::mean(n::square(m::MaeBlock(block, "block", inputs[0], 4).output));
  };
  EXPECT_TRUE(n::gradcheck(f, {x}, {.atol = 1e-7, .rtol = 1e-5}).ok);
}

TEST(NnMaskedAutoencoder, InvalidSelectorsAndShapesFailBeforeParameterCreation) {
  n::ParameterStore  p(1);
  const auto         image = n::zeros({1, 28, 28, 1});
  std::vector<float> duplicate(64);
  duplicate[0] = duplicate[16] = duplicate[34] = duplicate[51] = 1;
  EXPECT_THROW(
      m::MaskedAutoencoder(p, image, n::from_vector(duplicate, {1, 4, 16})), std::invalid_argument);
  EXPECT_THROW(m::MaskedAutoencoder(p, image, n::zeros({1, 4, 16})), std::invalid_argument);
  auto selection = m::MaeSelection(n::zeros({1}, n::DType::Int32));
  EXPECT_THROW(
      m::MaskedAutoencoder(p, image, selection.clone().requires_grad_()), std::invalid_argument);
  EXPECT_THROW(
      m::MaskedAutoencoder(p, n::zeros({1, 28, 28, 1}, n::DType::Float64), selection),
      std::invalid_argument);
  EXPECT_THROW(m::MaeDecode(p, n::zeros({1, 4, 31}), selection), std::invalid_argument);
  EXPECT_EQ(p.size(), 0u);
  EXPECT_THROW(m::MaeSelection(n::zeros({1})), std::invalid_argument);
  EXPECT_THROW(m::MaeSelection(n::tensor({-1.}, {1}, n::DType::Int64)), std::invalid_argument);
  EXPECT_THROW(m::MaePosition(3), std::invalid_argument);
}
