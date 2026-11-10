#include <gtest/gtest.h>

#include <datamunge/nn/conditional_autoencoder.hpp>
#include <datamunge/nn/nn.hpp>
namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnConditionalAutoencoder, HandConditionEntersBothPosteriorAndDecoder) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(2111, t);
  const auto        x = n::zeros({2, 1}, t), labels = n::from_vector(std::vector<int>{0, 1}, {2}),
             e = n::full({2, 1}, .5, t);
  m::ConditionalVariationalAutoencoder(p, x, labels, e, 1, 2, 1, 1);
  EXPECT_EQ(p.size(), 10);
  EXPECT_EQ(p.count(), 14);
  for (const auto& name : p.names())
    p.set(name, n::zeros(p.at(name).shape(), t));
  p.set("encoder/W", n::tensor({0., 1., 2.}, {3, 1}, t));
  p.set("mu/W", n::ones({1, 1}, t));
  p.set("decoder/W", n::tensor({1., 3., 5.}, {3, 1}, t));
  p.set("output/W", n::ones({1, 1}, t));
  const auto o = m::ConditionalVariationalAutoencoder(p, x, labels, e, 1, 2, 1, 1);
  EXPECT_EQ(o.mu.to_vector<double>(), (std::vector<double>{1, 2}));
  EXPECT_EQ(o.log_variance.to_vector<double>(), (std::vector<double>{0, 0}));
  EXPECT_EQ(o.z.to_vector<double>(), (std::vector<double>{1.5, 2.5}));
  EXPECT_EQ(o.logits.to_vector<double>(), (std::vector<double>{4.5, 7.5}));
  EXPECT_EQ(o.kl.to_vector<double>(), (std::vector<double>{.5, 2}));
  EXPECT_EQ(o.condition.to_vector<double>(), (std::vector<double>{1, 0, 0, 1}));
  EXPECT_EQ(o.encoder_input.to_vector<double>(), (std::vector<double>{0, 1, 0, 0, 0, 1}));
  EXPECT_EQ(o.decoder_input.to_vector<double>(), (std::vector<double>{1.5, 1, 0, 2.5, 0, 1}));
  EXPECT_EQ(o.loss.item(), o.unweighted_loss.item());
  const auto fixed = m::ConditionalVaeDecode(p, n::ones({2, 1}, t), labels, 1, 2, 1, 1);
  EXPECT_EQ(fixed.to_vector<double>(), (std::vector<double>{4, 6}));
}
TEST(NnConditionalAutoencoder, All47LeafDerivativesAndInputFirstSecondChecks) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(2111, t);
  auto              x      = n::tensor({.1, .4, .8, .2, .6, .3}, {2, 3}, t).requires_grad_();
  auto              e      = n::tensor({.3, -.6, -.2, .8}, {2, 2}, t).requires_grad_();
  const auto        labels = n::from_vector(std::vector<std::int64_t>{0, 2}, {2});
  auto              value  = [&]() {
    return m::ConditionalVariationalAutoencoder(p, x, labels, e, 3, 3, 2, 2).loss;
  };
  const auto          loss      = value();
  const auto          initial   = p.flatten();
  const auto          gradients = n::grad(loss, p.parameters());
  std::vector<double> analytic;
  for (const auto& v : gradients) {
    const auto a = v.to_vector<double>();
    analytic.insert(analytic.end(), a.begin(), a.end());
  }
  ASSERT_EQ(initial.size(), 47);
  ASSERT_EQ(analytic.size(), 47);
  for (std::size_t i = 0; i < initial.size(); ++i) {
    auto v = initial;
    v[i] += 1e-6;
    p.unflatten(v);
    const double plus = value().item();
    v[i] -= 2e-6;
    p.unflatten(v);
    const double minus = value().item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-8);
  }
  p.unflatten(initial);
  n::TensorFunction f = [&](const auto& a) {
    return m::ConditionalVariationalAutoencoder(p, a[0], labels, a[1], 3, 3, 2, 2).loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {x, e}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {x, e}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnConditionalAutoencoder, NoiseAndStandaloneDecodeControlsAndImmutableInference) {
  n::ParameterStore p(2111);
  const auto        x = n::full({2, 3}, .3), e = n::full({2, 2}, .7),
             labels = n::from_vector(std::vector<int>{0, 2}, {2});
  const auto o      = m::ConditionalVariationalAutoencoder(p, x, labels, e, 3, 3, 2, 2);
  const auto before = p.flatten();
  const auto repeat = m::ConditionalVariationalAutoencoder(p, x, labels, e, 3, 3, 2, 2);
  const auto zero =
      m::ConditionalVariationalAutoencoder(p, x, labels, n::zeros({2, 2}), 3, 3, 2, 2);
  EXPECT_EQ(o.probability.to_vector<float>(), repeat.probability.to_vector<float>());
  EXPECT_EQ(o.mu.to_vector<float>(), zero.mu.to_vector<float>());
  EXPECT_EQ(zero.z.to_vector<float>(), zero.mu.to_vector<float>());
  EXPECT_NE(o.z.to_vector<float>(), zero.z.to_vector<float>());
  EXPECT_EQ(
      n::sigmoid(m::ConditionalVaeDecode(p, o.z, labels, 3, 3, 2, 2)).to_vector<float>(),
      o.probability.to_vector<float>());
  EXPECT_EQ(before, p.flatten());
  n::ParameterStore decoder(99);
  m::ConditionalVaeDecode(decoder, n::zeros({2, 2}), labels, 3, 3, 2, 2);
  EXPECT_EQ(decoder.size(), 4);
  EXPECT_EQ(decoder.count(), 21);
}
TEST(NnConditionalAutoencoder, RejectInvalidDiscreteConditionsBeforeAnyLeaves) {
  n::ParameterStore p(1);
  const auto        x = n::zeros({2, 3}), e = n::zeros({2, 2});
  for (const auto& labels :
       {n::from_vector(std::vector<int>{-1, 0}, {2}),
        n::from_vector(std::vector<int>{0, 3}, {2}),
        n::zeros({2}),
        n::zeros({1}, n::DType::Int32),
        n::zeros({2, 1}, n::DType::Int32)}) {
    EXPECT_THROW(
        m::ConditionalVariationalAutoencoder(p, x, labels, e, 3, 3, 2, 2), std::invalid_argument);
    EXPECT_THROW(m::ConditionalVaeDecode(p, e, labels, 3, 3, 2, 2), std::invalid_argument);
  }
  const auto labels = n::zeros({2}, n::DType::Int32);
  EXPECT_THROW(
      m::ConditionalVariationalAutoencoder(p, x, labels, n::zeros({1, 2}), 3, 3, 2, 2),
      std::invalid_argument);
  EXPECT_THROW(
      m::ConditionalVariationalAutoencoder(p, n::full({2, 3}, -1), labels, e, 3, 3, 2, 2),
      std::invalid_argument);
  EXPECT_THROW(
      m::ConditionalVariationalAutoencoder(p, x, labels, e, 3, 0, 2, 2), std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
}
