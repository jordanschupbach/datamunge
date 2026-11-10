#include <gtest/gtest.h>

#include <datamunge/nn/importance_weighted_autoencoder.hpp>
#include <datamunge/nn/nn.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
TEST(NnIwae, HandExtremeWeightsAllNormalizedPartialsAndShiftInvariance) {
  const auto t = n::DType::Float64;
  auto       w = n::tensor({1000., -1000., 1001., -1001., 999., -999.}, {3, 2}, t).requires_grad_();
  const auto bound = -m::IwaeLogMeanExp(w), gradient = n::grad(n::mean(bound), {w})[0];
  for (int b = 0; b < 2; ++b) {
    double maximum = -1e300, sum = 0;
    for (int k = 0; k < 3; ++k)
      maximum = std::max(maximum, w.at({k, b}));
    for (int k = 0; k < 3; ++k)
      sum += std::exp(w.at({k, b}) - maximum);
    EXPECT_NEAR(bound.at({b}), -maximum - std::log(sum / 3), 1e-12);
    for (int k = 0; k < 3; ++k)
      EXPECT_NEAR(gradient.at({k, b}), -.5 * std::exp(w.at({k, b}) - maximum) / sum, 1e-15);
  }
  EXPECT_NEAR(
      n::max(n::abs(m::IwaeLogMeanExp(w + 17.) - m::IwaeLogMeanExp(w) - 17.)).item(), 0, 1e-12);
  auto              a = n::tensor({.1, .2, -.3, .4, .5, -.6}, {3, 2}, t).requires_grad_();
  n::TensorFunction f = [](const auto& x) { return n::sum(m::IwaeLogMeanExp(x[0])); };
  EXPECT_TRUE(n::gradcheck(f, {a}, {.atol = 1e-9, .rtol = 1e-6}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {a}, {.atol = 1e-9, .rtol = 1e-6}).ok);
}
TEST(NnIwae, SingleDuplicateEqualAndDominantWeightsEssAndNonmonotoneFiniteDraw) {
  const auto t      = n::DType::Float64;
  auto       single = n::tensor({-700., -800.}, {1, 2}, t), repeated = n::tile(single, {3, 1});
  EXPECT_EQ(m::IwaeLogMeanExp(single).to_vector<double>(), single.reshape({2}).to_vector<double>());
  EXPECT_NEAR(n::max(n::abs(m::IwaeLogMeanExp(repeated) - single)).item(), 0, 1e-12);
  auto out = m::ImportanceWeightedBound(
      n::zeros({6, 2}, t), n::zeros({6, 2}, t), n::ones({6, 2}, t), n::full({6}, 3., t), 3);
  EXPECT_NEAR(out.loss.item(), 3., 1e-15);
  for (auto v : out.effective_samples.to_vector<double>())
    EXPECT_NEAR(v, 3., 1e-15);
  for (auto v : out.normalized_weights.to_vector<double>())
    EXPECT_NEAR(v, 1. / 3, 1e-15);
  auto dominant = m::ImportanceWeightedBound(
      n::zeros({3, 1}, t),
      n::zeros({3, 1}, t),
      n::zeros({3, 1}, t),
      n::tensor({0., 1000., 1000.}, {3}, t),
      3);
  EXPECT_EQ(dominant.effective_samples.item(), 1.);
  EXPECT_EQ(dominant.normalized_weights.to_vector<double>(), (std::vector<double>{1, 0, 0}));
  const auto one = m::IwaeLogMeanExp(n::tensor({0.}, {1, 1}, t)).item(),
             two = m::IwaeLogMeanExp(n::tensor({0., -10.}, {2, 1}, t)).item();
  EXPECT_LT(two, one); // A particular nested Monte Carlo estimate need not increase.
}
TEST(NnIwae, AllFourBoundInputPathsFirstSecondDerivativesAndStrides) {
  const auto t = n::DType::Float64;
  auto mu = n::full({6, 2}, .3, t).requires_grad_(), lv = n::full({6, 2}, -.2, t).requires_grad_(),
       z = n::tensor({.1, .5, -.4, .3, .7, .8, .2, -.3, .4, -.7, .9, .3}, {6, 2}, t)
               .requires_grad_(),
       nll            = n::tensor({1., 2., 3., 4., 5., 6.}, {6}, t).requires_grad_();
  n::TensorFunction f = [](const auto& a) {
    return m::ImportanceWeightedBound(a[0], a[1], a[2], a[3], 3).loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {mu, lv, z, nll}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {mu, lv, z, nll}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  const auto a       = m::ImportanceWeightedBound(mu, lv, z, nll, 3);
  const auto strided = n::transpose(n::transpose(z).contiguous());
  EXPECT_FALSE(strided.is_contiguous());
  EXPECT_EQ(
      a.log_weights.to_vector<double>(),
      m::ImportanceWeightedBound(mu, lv, strided, nll, 3).log_weights.to_vector<double>());
  const auto g = n::grad(a.loss, {nll})[0].reshape({3, 2});
  EXPECT_NEAR(n::max(n::abs(g - .5 * a.normalized_weights)).item(), 0, 1e-15);
}
TEST(NnIwae, FullTinyParameterAndImageNoiseGradientsAtK1AndK3) {
  const auto t = n::DType::Float64;
  for (int k : {1, 3}) {
    n::ParameterStore p(2237, t);
    auto              x = n::tensor({.2, .4, .3, .7, .5, .6, .3, .6, .2, .4, .7, .5}, {2, 6}, t),
         e              = n::full({k, 2, 2}, .27, t);
    auto value          = [&]() { return m::ImportanceWeightedAutoencoder(p, x, e, 6, 2, 3).loss; };
    value();
    // Positive biases keep this derivative fixture away from ReLU kinks.
    p.set("encoder/b", n::full({3}, .8, t));
    p.set("decoder/b", n::full({3}, .8, t));
    const auto gradients=n::grad(value(),p.parameters());
    std::vector<double> analytic;
    for(const auto& g:gradients){auto v=g.to_vector<double>();analytic.insert(analytic.end(),v.begin(),v.end());}
    const auto original=p.flatten();
    EXPECT_EQ(p.size(), 10);
    EXPECT_EQ(p.count(), 70);
    for (std::size_t j = 0; j < original.size(); ++j) {
      auto a = original;
      a[j] += 1e-6;
      p.unflatten(a);
      double plus = value().item();
      a[j] -= 2e-6;
      p.unflatten(a);
      double minus = value().item();
      EXPECT_NEAR((plus - minus) / 2e-6, analytic[j], 2e-8);
    }
    p.unflatten(original);
    auto              image = x.clone().requires_grad_(), epsilon = e.clone().requires_grad_();
    n::TensorFunction f = [&](const auto& a) {
      return m::ImportanceWeightedAutoencoder(p, a[0], a[1], 6, 2, 3).loss;
    };
    EXPECT_TRUE(n::gradcheck(f, {image, epsilon}, {.atol = 1e-8, .rtol = 1e-5}).ok);
    EXPECT_TRUE(n::gradgradcheck(f, {image, epsilon}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  }
}
TEST(NnIwae, SampledK1DensityRatioDiffersFromAnalyticKlAndCommonGaussianLeaves) {
  n::ParameterStore p(2237);
  const auto        x = n::full({2, 784}, .3), e = n::full({1, 2, 4}, .2);
  const auto        out   = m::ImportanceWeightedAutoencoder(p, x, e);
  const auto        plain = m::BetaVariationalAutoencoder(p, x, e.reshape({2, 4}));
  EXPECT_EQ(p.count(), 51416);
  EXPECT_EQ(p.size(), 10);
  EXPECT_EQ(out.probability.to_vector<float>(), plain.probability.to_vector<float>());
  EXPECT_NE(out.loss.item(), plain.loss.item());
  EXPECT_EQ(out.loss.item(), n::mean(-out.log_weights).item());
  for (const auto& name : p.names())
    p.set(name, n::zeros(p.at(name).shape()));
  const auto zero = m::ImportanceWeightedAutoencoder(p, x, n::ones({3, 2, 4}));
  EXPECT_EQ(zero.kl.to_vector<float>(), (std::vector<float>(6, 0)));
  EXPECT_NEAR(zero.loss.item(), 784 * std::log(2.), 1e-3);
}
TEST(NnIwae, OwningEndpointsCodeOnlyInferenceAndExplicitAdamSaveRestore) {
  n::ParameterStore p(2237);
  const auto        x = n::full({2, 784}, .3), e = n::ones({3, 2, 4});
  n::Tensor         probability, z, weights, ess;
  {
    const auto out = m::ImportanceWeightedAutoencoder(p, x, e);
    probability    = out.probability;
    z              = out.z;
    weights        = out.normalized_weights;
    ess            = out.effective_samples;
  }
  const auto before = p.flatten();
  EXPECT_EQ(
      probability.to_vector<float>(), n::sigmoid(m::GaussianVaeDecode(p, z)).to_vector<float>());
  EXPECT_EQ(p.flatten(), before);
  EXPECT_EQ(weights.shape(), (n::Shape{3, 2}));
  EXPECT_EQ(ess.shape(), (n::Shape{2}));
  n::OptimizerConfig config;
  config.kind=n::OptimizerKind::Adam;
  n::Optimizer adam(p,.003,config);
  adam.step(n::grad(m::ImportanceWeightedAutoencoder(p, x, e).loss, p.parameters()));
  EXPECT_NE(p.flatten(), before);
  n::ParameterStore q(9);
  m::ImportanceWeightedAutoencoder(q, x, e);
  q.unflatten(p.flatten());
  EXPECT_EQ(
      m::ImportanceWeightedAutoencoder(p, x, e).probability.to_vector<float>(),
      m::ImportanceWeightedAutoencoder(q, x, e).probability.to_vector<float>());
}
TEST(NnIwae, InvalidInputsBeforeLeavesAndNoncontiguousNoiseImages) {
  n::ParameterStore p(1);
  auto              x = n::full({2, 6}, .3), e = n::ones({3, 2, 2});
  EXPECT_THROW(m::ImportanceWeightedAutoencoder(p, x, e, 6, 2, 0), std::invalid_argument);
  EXPECT_THROW(
      m::ImportanceWeightedAutoencoder(p, n::full({2, 6}, 1.1), e, 6, 2, 3), std::invalid_argument);
  EXPECT_THROW(
      m::ImportanceWeightedAutoencoder(p, x, n::ones({3, 1, 2}), 6, 2, 3), std::invalid_argument);
  EXPECT_THROW(
      m::ImportanceWeightedAutoencoder(p, x, n::ones({3, 2, 2}, n::DType::Float64), 6, 2, 3),
      std::invalid_argument);
  EXPECT_THROW(
      m::ImportanceWeightedAutoencoder(p, x, n::full({3, 2, 2}, NAN), 6, 2, 3),
      std::invalid_argument);
  EXPECT_THROW(
      m::ImportanceWeightedAutoencoder(p, x, e, 6, 2, std::numeric_limits<std::int64_t>::max()),
      std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  auto sx = n::transpose(n::transpose(x).contiguous()),
       se = n::transpose(n::transpose(e, {2, 1, 0}).contiguous(), {2, 1, 0});
  EXPECT_FALSE(sx.is_contiguous());
  EXPECT_FALSE(se.is_contiguous());
  EXPECT_EQ(
      m::ImportanceWeightedAutoencoder(p, sx, se, 6, 2, 3).probability.to_vector<float>(),
      m::ImportanceWeightedAutoencoder(p, x, e, 6, 2, 3).probability.to_vector<float>());
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 2}),
        n::zeros({2, 0}),
        n::ones({2, 2}, n::DType::Int32),
        n::full({2, 2}, INFINITY)})
    EXPECT_THROW(m::IwaeLogMeanExp(bad), std::invalid_argument);
  EXPECT_THROW(
      m::ImportanceWeightedBound(
          n::zeros({6, 2}), n::zeros({6, 2}), n::zeros({6, 2}), n::ones({6}), 4),
      std::invalid_argument);
  EXPECT_THROW(
      m::ImportanceWeightedBound(
          n::zeros({6, 2}), n::zeros({6, 2}), n::zeros({6, 2}), n::full({6}, NAN), 3),
      std::invalid_argument);
}
