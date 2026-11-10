#include <gtest/gtest.h>

#include <datamunge/nn/convolutional_variational_autoencoder.hpp>
#include <datamunge/nn/nn.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
TEST(NnConvolutionalVae, DefaultStageShapesParameterNamesAndCodeOnlyDecoder) {
  n::ParameterStore p(2011);
  const auto        x = n::full({2, 28, 28, 1}, .3), e = n::full({2, 2}, .7);
  const auto        o = m::ConvolutionalVariationalAutoencoder(p, x, e);
  EXPECT_EQ(p.size(), 14);
  EXPECT_EQ(p.count(), 2117);
  EXPECT_EQ(
      p.names(),
      (std::vector<std::string>{
          "encoder1/W",
          "encoder1/b",
          "encoder2/W",
          "encoder2/b",
          "mu/W",
          "mu/b",
          "log_variance/W",
          "log_variance/b",
          "decoder_projection/W",
          "decoder_projection/b",
          "decoder1/W",
          "decoder1/b",
          "decoder2/W",
          "decoder2/b"}));
  EXPECT_EQ(o.conv1.shape(), (n::Shape{2, 28, 28, 8}));
  EXPECT_EQ(o.pool1.shape(), (n::Shape{2, 14, 14, 8}));
  EXPECT_EQ(o.conv2.shape(), (n::Shape{2, 14, 14, 4}));
  EXPECT_EQ(o.pool2.shape(), (n::Shape{2, 7, 7, 4}));
  EXPECT_EQ(o.decoder_base.shape(), o.pool2.shape());
  EXPECT_EQ(o.decoder1.shape(), o.pool1.shape());
  EXPECT_EQ(o.spatial_logits.shape(), x.shape());
  EXPECT_EQ(o.probability.shape(), (n::Shape{2, 784}));
  EXPECT_EQ(o.loss.item(), o.unweighted_loss.item());
  const auto before = p.flatten();
  const auto d      = m::ConvolutionalVaeDecode(p, o.z);
  EXPECT_EQ(d.probability.to_vector<float>(), o.probability.to_vector<float>());
  EXPECT_EQ(d.decoder_base.to_vector<float>(), o.decoder_base.to_vector<float>());
  EXPECT_EQ(before, p.flatten());
  n::ParameterStore decoder(1);
  m::ConvolutionalVaeDecode(decoder, n::zeros({2, 2}));
  EXPECT_EQ(decoder.size(), 6);
  EXPECT_EQ(decoder.count(), 957);
}
TEST(NnConvolutionalVae, All84SmallModelParameterDerivativesAndInputFirstSecondChecks) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(2011, t);
  auto              x =
      n::tensor({.1, .2, .3, .4, .2, .4, .1, .3, .3, .1, .4, .2, .4, .3, .2, .1}, {1, 4, 4, 1}, t)
          .requires_grad_();
  auto e = n::full({1, 1}, .3, t).requires_grad_();
  m::ConvolutionalVariationalAutoencoder(p, x, e, 1, 2, 1);
  for (const auto& name : p.names())
    p.set(name, n::full(p.at(name).shape(), name.ends_with("/b") ? .1 : .03, t));
  auto value = [&]() { return m::ConvolutionalVariationalAutoencoder(p, x, e, 1, 2, 1).loss; };
  const auto          loss    = value();
  const auto          gs      = n::grad(loss, p.parameters());
  const auto          initial = p.flatten();
  std::vector<double> analytic;
  for (const auto& g : gs) {
    const auto a = g.to_vector<double>();
    analytic.insert(analytic.end(), a.begin(), a.end());
  }
  ASSERT_EQ(initial.size(), 84);
  ASSERT_EQ(analytic.size(), 84);
  for (std::size_t i = 0; i < initial.size(); ++i) {
    auto changed = initial;
    changed[i] += 1e-6;
    p.unflatten(changed);
    const double plus = value().item();
    changed[i] -= 2e-6;
    p.unflatten(changed);
    const double minus = value().item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-8);
  }
  p.unflatten(initial);
  n::TensorFunction f = [&](const auto& a) {
    return m::ConvolutionalVariationalAutoencoder(p, a[0], a[1], 1, 2, 1).loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {x, e}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {x, e}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnConvolutionalVae, StridedImagesNoiseControlsAndNonSquareGeometry) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(2011, t);
  const auto        x = n::full({2, 4, 8, 1}, .3, t), e = n::ones({2, 1}, t);
  const auto        o = m::ConvolutionalVariationalAutoencoder(p, x, e, 1, 2, 1);
  const auto strided  = n::transpose(n::transpose(x, {0, 2, 1, 3}).contiguous(), {0, 2, 1, 3});
  EXPECT_FALSE(strided.is_contiguous());
  const auto s = m::ConvolutionalVariationalAutoencoder(p, strided, e, 1, 2, 1);
  EXPECT_EQ(o.probability.to_vector<double>(), s.probability.to_vector<double>());
  const auto zero = m::ConvolutionalVariationalAutoencoder(p, x, n::zeros({2, 1}, t), 1, 2, 1);
  EXPECT_EQ(o.mu.to_vector<double>(), zero.mu.to_vector<double>());
  EXPECT_EQ(o.log_variance.to_vector<double>(), zero.log_variance.to_vector<double>());
  EXPECT_EQ(zero.z.to_vector<double>(), zero.mu.to_vector<double>());
  EXPECT_NE(o.z.to_vector<double>(), zero.z.to_vector<double>());
  const auto d = m::ConvolutionalVaeDecode(p, o.z, 4, 8, 1, 2, 1);
  EXPECT_EQ(d.probability.to_vector<double>(), o.probability.to_vector<double>());
  EXPECT_EQ(d.spatial_logits.shape(), x.shape());
}
TEST(NnConvolutionalVae, InvalidInputOptionsBeforeAnyLeavesAndGeometryOverflow) {
  n::ParameterStore p(1);
  const auto        e = n::zeros({1, 2});
  for (const auto& shape : std::vector<n::Shape>{
           {0, 4, 4, 1}, {1, 0, 4, 1}, {1, 5, 4, 1}, {1, 4, 6, 1}, {1, 4, 4, 2}, {4, 4, 1}})
    EXPECT_THROW(
        m::ConvolutionalVariationalAutoencoder(p, n::zeros(shape), e), std::invalid_argument);
  EXPECT_THROW(
      m::ConvolutionalVariationalAutoencoder(p, n::full({1, 4, 4, 1}, 1.1), e),
      std::invalid_argument);
  EXPECT_THROW(
      m::ConvolutionalVariationalAutoencoder(p, n::zeros({1, 4, 4, 1}), n::zeros({2, 2})),
      std::invalid_argument);
  EXPECT_THROW(
      m::ConvolutionalVariationalAutoencoder(p, n::zeros({1, 4, 4, 1}), e, 0),
      std::invalid_argument);
  EXPECT_THROW(m::ConvolutionalVaeDecode(p, e, 6, 4), std::invalid_argument);
  EXPECT_THROW(
      m::ConvolutionalVaeDecode(p, e, std::numeric_limits<std::int64_t>::max() - 3, 8),
      std::invalid_argument);
  EXPECT_THROW(
      m::ConvolutionalVaeDecode(p, n::zeros({1, 2}, n::DType::Int32)), std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
}
