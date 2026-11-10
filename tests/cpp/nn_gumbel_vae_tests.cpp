#include <gtest/gtest.h>

#include <datamunge/nn/gumbel_categorical_autoencoder.hpp>
#include <datamunge/nn/nn.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
TEST(NnGumbelVae, HandProbabilitiesKlAndEveryLogitAndNoisePartial) {
  const auto t        = n::DType::Float64;
  auto       l        = n::tensor({.1, -.4, .7, .2, .6, -.3}, {2, 3}, t).requires_grad_(),
       g              = n::tensor({.3, -.2, .6, -.1, .4, .2}, {2, 3}, t).requires_grad_();
  const auto a        = m::GumbelCategorical(l, g, .7);
  const auto kl       = m::CategoricalUniformKl(a.probability, a.log_probability);
  auto       gradient = n::grad(.5 * n::sum(n::square(a.relaxed)) + .3 * n::sum(kl), {l, g});
  for (int b = 0; b < 2; ++b) {
    double q[3], y[3], sq = 0, sy = 0, expected = 0, yy = 0;
    for (int j = 0; j < 3; ++j) {
      q[j] = std::exp(l.at({b, j}));
      y[j] = std::exp((l.at({b, j}) + g.at({b, j})) / .7);
      sq += q[j];
      sy += y[j];
    }
    for (int j = 0; j < 3; ++j) {
      q[j] /= sq;
      y[j] /= sy;
      expected += q[j] * std::log(3 * q[j]);
      yy += y[j] * y[j];
      EXPECT_NEAR(a.probability.at({b, j}), q[j], 1e-15);
      EXPECT_NEAR(a.relaxed.at({b, j}), y[j], 1e-15);
    }
    EXPECT_NEAR(kl.at({b}), expected, 1e-15);
    for (int j = 0; j < 3; ++j) {
      const double rec   = y[j] * (y[j] - yy) / .7,
                   prior = .3 * q[j] * (std::log(3 * q[j]) - expected);
      EXPECT_NEAR(gradient[0].at({b, j}), rec + prior, 1e-15);
      EXPECT_NEAR(gradient[1].at({b, j}), rec, 1e-15);
    }
  }
  const auto cold = m::GumbelCategorical(l, g, .25), warm = m::GumbelCategorical(l, g, 2);
  EXPECT_EQ(a.indices.to_vector<std::int64_t>(), cold.indices.to_vector<std::int64_t>());
  EXPECT_EQ(a.indices.to_vector<std::int64_t>(), warm.indices.to_vector<std::int64_t>());
  EXPECT_NE(cold.relaxed.to_vector<double>(), warm.relaxed.to_vector<double>());
  EXPECT_FALSE(a.indices.requires_grad());
  const auto ties = m::GumbelCategorical(n::zeros({1, 3}, t), n::zeros({1, 3}, t));
  EXPECT_EQ(ties.indices.item<std::int64_t>(), 0);
}
TEST(NnGumbelVae, SamplerFirstSecondDerivativesBroadcastRowsStridesAndExtremeValues) {
  const auto t        = n::DType::Float64;
  auto       l        = n::tensor({.1, -.4, .7}, {1, 3}, t).requires_grad_(),
       g              = n::tensor({.3, -.2, .6, -.1, .4, .2}, {2, 3}, t).requires_grad_();
  n::TensorFunction f = [](const auto& a) {
    auto s = m::GumbelCategorical(a[0], a[1], .7);
    return n::mean(n::square(s.relaxed)) +
           .3 * n::sum(m::CategoricalUniformKl(s.probability, s.log_probability));
  };
  EXPECT_TRUE(n::gradcheck(f, {l, g}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {l, g}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  const auto draw = m::GumbelCategorical(l, g);
  EXPECT_EQ(draw.probability.shape(), (n::Shape{1, 3}));
  EXPECT_EQ(draw.relaxed.shape(), (n::Shape{2, 3}));
  const auto strided = n::transpose(n::transpose(g).contiguous());
  EXPECT_FALSE(strided.is_contiguous());
  EXPECT_EQ(
      draw.relaxed.to_vector<double>(),
      m::GumbelCategorical(l, strided).relaxed.to_vector<double>());
  const auto extreme =
      m::GumbelCategorical(n::tensor({100., -100., 0.}, {1, 3}, t), n::zeros({1, 3}, t));
  EXPECT_NEAR(
      m::CategoricalUniformKl(extreme.probability, extreme.log_probability).item(),
      std::log(3.),
      1e-14);
}
TEST(NnGumbelVae, ExplicitNoiseMatchesPrescribedMidpointEngineAndRejectsOverflow) {
  const auto bank  = m::GumbelNoise(3, 2, 4294967295U, 2, 3),
             again = m::GumbelNoise(3, 2, 4294967295U, 2, 3);
  EXPECT_EQ(bank.uniform.shape(), (n::Shape{2, 3, 2, 3}));
  EXPECT_FALSE(bank.noise.requires_grad());
  EXPECT_EQ(bank.uniform.dtype(), n::DType::Float64);
  EXPECT_EQ(bank.noise.dtype(), n::DType::Float32);
  EXPECT_EQ(bank.noise.to_vector<float>(), again.noise.to_vector<float>());
  std::mt19937 random(4294967295U);
  const auto   u = bank.uniform.to_vector<double>(), g = bank.gumbel.to_vector<double>();
  const auto   noise = bank.noise.to_vector<float>();
  for (std::size_t i = 0; i < u.size(); ++i) {
    const double expected = (double(random()) + .5) / 4294967296.,
                 value    = -std::log(-std::log(expected));
    EXPECT_DOUBLE_EQ(u[i], expected);
    EXPECT_DOUBLE_EQ(g[i], value);
    EXPECT_EQ(noise[i], float(value));
    EXPECT_GT(u[i], 0);
    EXPECT_LT(u[i], 1);
  }
  EXPECT_THROW(m::GumbelNoise(0, 1, 1), std::invalid_argument);
  EXPECT_THROW(m::GumbelNoise(1, 1, 1, 1, 1), std::invalid_argument);
  EXPECT_THROW(
      m::GumbelNoise(std::numeric_limits<std::int64_t>::max(), 2, 1), std::invalid_argument);
  EXPECT_THROW(
      m::GumbelNoise(std::numeric_limits<std::int64_t>::max() / 8, 1, 1, 1, 2),
      std::invalid_argument);
}
TEST(NnGumbelVae, AllThirtyNineTinyModelDerivativesIncludePathwiseAndCategoricalKl) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(2903, t);
  const auto        x = n::tensor({.1, .4, .8, .2, .6, .3}, {2, 3}, t),
             g        = n::tensor({.3, -.6, .1, .4, -.2, .8, .5, -.1}, {2, 2, 2}, t);
  auto value = [&]() { return m::GumbelCategoricalAutoencoder(p, x, g, .7, 3, 2, 2, 2).loss; };
  value();
  for (const auto& name : p.names()) {
    std::vector<double> v(p.at(name).numel());
    for (std::size_t i = 0; i < v.size(); ++i)
      v[i] = name.ends_with("/b") ? .12 + .003 * i : .015 + .002 * (i % 11);
    p.set(name, n::from_vector(v, p.at(name).shape()));
  }
  const auto          gs = n::grad(value(), p.parameters());
  std::vector<double> analytic;
  for (const auto& grad : gs) {
    auto v = grad.to_vector<double>();
    analytic.insert(analytic.end(), v.begin(), v.end());
  }
  EXPECT_EQ(p.size(), 8);
  EXPECT_EQ(p.count(), 39);
  EXPECT_EQ(analytic.size(), 39);
  const auto original = p.flatten();
  for (std::size_t i = 0; i < original.size(); ++i) {
    auto v = original;
    v[i] += 1e-6;
    p.unflatten(v);
    double plus = value().item();
    v[i] -= 2e-6;
    p.unflatten(v);
    double minus = value().item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-8);
  }
  p.unflatten(original);
  auto              live = x.clone().requires_grad_(), noise = g.clone().requires_grad_();
  n::TensorFunction f = [&](const auto& a) {
    return m::GumbelCategoricalAutoencoder(p, a[0], a[1], .7, 3, 2, 2, 2).loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {live, noise}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {live, noise}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  auto o = m::GumbelCategoricalAutoencoder(p, x, g, .7, 3, 2, 2, 2);
  EXPECT_EQ(o.hard.dtype(), t);
  EXPECT_FALSE(o.hard.requires_grad());
  EXPECT_EQ(
      o.hard_prediction.to_vector<double>(),
      m::GumbelDecodeIds(p, o.indices, 3, 2, 2, 2).to_vector<double>());
  EXPECT_EQ(
      o.soft_prediction.to_vector<double>(),
      m::GumbelDecodeSoft(p, o.relaxed, 3, 2, 2, 2).to_vector<double>());
}
TEST(NnGumbelVae, DefaultCensusOwningEndpointsTargetFreeDecodeAndZeroModel) {
  n::ParameterStore p(2903);
  n::Tensor         probability, ids, loss;
  {
    auto o      = m::GumbelCategoricalAutoencoder(p, n::full({2, 784}, .3), n::zeros({2, 4, 8}));
    probability = o.soft_prediction;
    ids         = o.indices;
    loss        = o.loss;
  }
  EXPECT_EQ(p.count(), 53104);
  EXPECT_EQ(p.size(), 8);
  EXPECT_EQ(probability.shape(), (n::Shape{2, 784}));
  EXPECT_EQ(ids.dtype(), n::DType::Int32);
  n::ParameterStore decoder(1);
  m::GumbelDecodeIds(decoder, n::zeros({2, 4}, n::DType::Int64));
  EXPECT_EQ(decoder.size(), 4);
  EXPECT_EQ(decoder.count(), 26928);
  const auto gradients = n::grad(loss, p.parameters());
  EXPECT_EQ(gradients.size(), 8);
  n::OptimizerConfig c;
  c.kind = n::OptimizerKind::Adam;
  n::Optimizer opt(p, .003, c);
  const auto   before = p.flatten();
  opt.step(gradients);
  EXPECT_NE(before, p.flatten());
  EXPECT_EQ(opt.iterations(), 1);
  for (const auto& name : p.names())
    p.set(name, n::zeros(p.at(name).shape()));
  const auto o = m::GumbelCategoricalAutoencoder(p, n::zeros({2, 784}), n::zeros({2, 4, 8}));
  EXPECT_NEAR(n::max(n::abs(o.kl)).item(), 0, 1e-6);
  EXPECT_NEAR(o.loss.item(), 784 * std::log(2.), 1e-3);
  EXPECT_EQ(n::min(o.soft_prediction).item(), .5);
}
TEST(NnGumbelVae, MalformedInputsFailBeforeLeavesAndFloat32TemperatureRoundingIsExplicit) {
  n::ParameterStore p(1);
  const auto        x = n::zeros({2, 3}), g = n::zeros({2, 2, 2});
  for (double temp :
       {0.,
        -1.,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::min()})
    EXPECT_THROW(m::GumbelCategoricalAutoencoder(p, x, g, temp, 3, 2, 2, 2), std::invalid_argument);
  EXPECT_THROW(
      m::GumbelCategoricalAutoencoder(p, x, n::zeros({1, 2, 2}), .7, 3, 2, 2, 2),
      std::invalid_argument);
  EXPECT_THROW(m::GumbelCategoricalAutoencoder(p, x, g, .7, 3, 2, 1, 2), std::invalid_argument);
  EXPECT_THROW(m::GumbelDecodeSoft(p, n::zeros({1, 2, 2}), 3, 2, 2, 2), std::invalid_argument);
  EXPECT_THROW(
      m::GumbelDecodeIds(p, n::from_vector(std::vector<int>{-1, 0}, {1, 2}), 3, 2, 2, 2),
      std::invalid_argument);
  EXPECT_THROW(
      m::GumbelDecodeIds(p, n::from_vector(std::vector<int>{2, 0}, {1, 2}), 3, 2, 2, 2),
      std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  const auto logits = n::tensor({.1f, .4f, -.2f}, {1, 3}),
             noise  = n::tensor({.2f, -.1f, .3f}, {1, 3});
  EXPECT_EQ(
      m::GumbelCategorical(logits, noise, .7).relaxed.to_vector<float>(),
      n::softmax((logits + noise) / double(.7f), -1).to_vector<float>());
}
