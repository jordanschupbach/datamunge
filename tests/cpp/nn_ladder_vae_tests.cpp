#include <gtest/gtest.h>

#include <datamunge/nn/ladder_variational_autoencoder.hpp>
#include <datamunge/nn/nn.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
TEST(NnLadder, HandPrecisionFusionFourLivePathsAndConditionalKl) {
  const auto t  = n::DType::Float64;
  auto       em = n::full({2, 2}, 2., t).requires_grad_(),
       el       = n::full({2, 2}, std::log(4.), t).requires_grad_(),
       pm       = n::full({2, 2}, -1., t).requires_grad_(),
       pl       = n::full({2, 2}, std::log(9.), t).requires_grad_();
  const auto q  = m::LadderPrecisionFusion(em, el, pm, pl);
  for (double v : q.mu.to_vector<double>())
    EXPECT_NEAR(v, 14. / 13, 1e-14);
  for (double v : q.log_variance.to_vector<double>())
    EXPECT_NEAR(std::exp(v), 36. / 13, 1e-14);
  n::TensorFunction f = [](const auto& a) {
    auto q = m::LadderPrecisionFusion(a[0], a[1], a[2], a[3]);
    return n::sum(n::square(q.mu) + n::square(q.log_variance)) +
           n::sum(m::ConditionalGaussianKl(q.mu, q.log_variance, a[2], a[3]));
  };
  EXPECT_TRUE(n::gradcheck(f, {em, el, pm, pl}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {em, el, pm, pl}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  auto g = n::grad(f({em, el, pm, pl}), {em, el, pm, pl});
  for (const auto& v : g)
    EXPECT_GT(n::max(n::abs(v)).item(), 0);
}
TEST(NnLadder, EqualSourcesExtremePrecisionsSymmetryStridesAndOwningFields) {
  const auto t = n::DType::Float64;
  auto em = n::tensor({2., -1., .3, .8}, {2, 2}, t), pm = n::tensor({-1., 1., .1, -.4}, {2, 2}, t),
       el = n::tensor({-1000., 1000., 1000., -1000.}, {2, 2}, t), pl = -el;
  n::Tensor mu, lv;
  {
    auto q = m::LadderPrecisionFusion(em, el, pm, pl);
    mu     = q.mu;
    lv     = q.log_variance;
  }
  EXPECT_EQ(mu.to_vector<double>(), (std::vector<double>{2., 1., .1, .8}));
  for (double v : lv.to_vector<double>())
    EXPECT_EQ(v, -1000.);
  const auto same = m::LadderPrecisionFusion(em, n::zeros({2, 2}, t), em, n::zeros({2, 2}, t));
  EXPECT_NEAR(n::max(n::abs(same.mu - em)).item(), 0, 1e-15);
  for (double v : same.log_variance.to_vector<double>())
    EXPECT_EQ(v, -std::log(2.));
  auto a = m::LadderPrecisionFusion(em, n::full({2, 2}, .3, t), pm, n::full({2, 2}, -.1, t)),
       b = m::LadderPrecisionFusion(pm, n::full({2, 2}, -.1, t), em, n::full({2, 2}, .3, t));
  EXPECT_EQ(a.mu.to_vector<double>(), b.mu.to_vector<double>());
  const auto strided = n::transpose(n::transpose(em).contiguous());
  EXPECT_FALSE(strided.is_contiguous());
  EXPECT_EQ(
      a.mu.to_vector<double>(),
      m::LadderPrecisionFusion(strided, n::full({2, 2}, .3, t), pm, n::full({2, 2}, -.1, t))
          .mu.to_vector<double>());
}
TEST(NnLadder, EverySmallParameterAndImageNoiseFirstSecondDerivatives) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(2503, t);
  auto image = n::full({2, 6}, .3, t), epsilon = n::tensor({.2, -.1, .3, -.2, .1, .4}, {2, 3}, t);
  auto value = [&]() {
    return m::LadderVariationalAutoencoder(p, image, epsilon, 6, 2, 1, 3, 2).loss;
  };
  value();
  for (const auto& name : p.names()) {
    auto a = std::vector<double>(p.at(name).numel());
    for (std::size_t i = 0; i < a.size(); ++i)
      a[i] = name.ends_with("/b") ? .2 + .001 * i : .03 + .002 * (i % 7);
    p.set(name, n::from_vector(a, p.at(name).shape()));
  }
  auto                gs = n::grad(value(), p.parameters());
  std::vector<double> analytic;
  for (const auto& g : gs) {
    auto v = g.to_vector<double>();
    analytic.insert(analytic.end(), v.begin(), v.end());
  }
  EXPECT_EQ(p.size(), 22);
  EXPECT_EQ(p.count(), 100);
  auto original = p.flatten();
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
  auto              x = image.clone().requires_grad_(), e = epsilon.clone().requires_grad_();
  n::TensorFunction f = [&](const auto& a) {
    return m::LadderVariationalAutoencoder(p, a[0], a[1], 6, 2, 1, 3, 2).loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {x, e}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {x, e}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnLadder, DefaultCensusExplicitNoiseOrderingAndLowerOnlyDecode) {
  n::ParameterStore p(2503);
  auto              image = n::full({2, 784}, .3), zero = n::zeros({2, 6});
  auto              base = m::LadderVariationalAutoencoder(p, image, zero);
  EXPECT_EQ(p.size(), 22);
  EXPECT_EQ(p.count(), 52196);
  EXPECT_EQ(base.lower_z.to_vector<float>(), base.lower_mu.to_vector<float>());
  EXPECT_EQ(base.top_z.to_vector<float>(), base.top_mu.to_vector<float>());
  auto lower = n::concat({n::full({2, 4}, .5), n::zeros({2, 2})}, 1),
       upper = n::concat({n::zeros({2, 4}), n::full({2, 2}, .5)}, 1);
  auto low   = m::LadderVariationalAutoencoder(p, image, lower),
       top   = m::LadderVariationalAutoencoder(p, image, upper);
  EXPECT_EQ(base.evidence_mu.to_vector<float>(), top.evidence_mu.to_vector<float>());
  EXPECT_EQ(base.prior_mu.to_vector<float>(), low.prior_mu.to_vector<float>());
  EXPECT_EQ(base.lower_mu.to_vector<float>(), low.lower_mu.to_vector<float>());
  EXPECT_NE(base.lower_z.to_vector<float>(), low.lower_z.to_vector<float>());
  EXPECT_GT(n::max(n::abs(base.prior_mu - top.prior_mu)).item(), 1e-6);
  EXPECT_GT(n::max(n::abs(base.lower_mu - top.lower_mu)).item(), 1e-6);
  EXPECT_EQ(
      top.probability.to_vector<float>(),
      n::sigmoid(m::GaussianVaeDecode(p, top.lower_z)).to_vector<float>());
  EXPECT_EQ(top.loss.item(), n::mean(top.nll + (top.kl_lower + top.kl_top)).item());
}
TEST(NnLadder, ImageFreeAncestralFieldsBothInputPathsAndPriorOnlyCensus) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(2503, t);
  auto top = n::full({2, 1}, .2, t).requires_grad_(), e = n::full({2, 2}, .3, t).requires_grad_();
  auto prior = m::LadderPrior(p, top, e, 6, 2, 1, 3, 2);
  EXPECT_EQ(p.size(), 10);
  EXPECT_EQ(p.count(), 49);
  n::TensorFunction f = [&](const auto& a) {
    return n::mean(n::square(m::LadderPrior(p, a[0], a[1], 6, 2, 1, 3, 2).probability));
  };
  EXPECT_TRUE(n::gradcheck(f, {top, e}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {top, e}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  auto means = m::LadderPrior(p, top, n::zeros({2, 2}, t), 6, 2, 1, 3, 2);
  EXPECT_EQ(means.lower_z.to_vector<double>(), means.prior_mu.to_vector<double>());
  n::ParameterStore standard(1);
  m::LadderPrior(standard, n::zeros({1, 2}), n::zeros({1, 4}));
  EXPECT_EQ(standard.size(), 10);
  EXPECT_EQ(standard.count(), 26216);
}
TEST(NnLadder, OwnedForwardAndPriorSurviveParentsVisibleAdamAndRestoredWeights) {
  n::ParameterStore p(2503);
  auto              image = n::full({2, 784}, .3), noise = n::ones({2, 6});
  n::Tensor         z, probability, loss;
  {
    auto out    = m::LadderVariationalAutoencoder(p, image, noise);
    z           = out.lower_z;
    probability = out.probability;
    loss        = out.loss;
  }
  auto before = p.flatten();
  EXPECT_EQ(
      probability.to_vector<float>(), n::sigmoid(m::GaussianVaeDecode(p, z)).to_vector<float>());
  EXPECT_EQ(p.flatten(), before);
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Adam;
  n::Optimizer optimizer(p, .003, config);
  optimizer.step(n::grad(loss, p.parameters()));
  EXPECT_EQ(optimizer.iterations(), 1);
  EXPECT_NE(p.flatten(), before);
  n::ParameterStore restored(7);
  m::LadderVariationalAutoencoder(restored, image, noise);
  restored.unflatten(p.flatten());
  EXPECT_EQ(
      m::LadderVariationalAutoencoder(p, image, noise).probability.to_vector<float>(),
      m::LadderVariationalAutoencoder(restored, image, noise).probability.to_vector<float>());
  n::Tensor top, pm, lv, code, pixels, epsilon;
  {
    auto prior = m::LadderPrior(p, n::ones({2, 2}), n::zeros({2, 4}));
    top        = prior.top_z;
    pm         = prior.prior_mu;
    lv         = prior.prior_lv;
    code       = prior.lower_z;
    pixels     = prior.probability;
    epsilon    = prior.lower_noise;
  }
  EXPECT_EQ(code.to_vector<float>(), pm.to_vector<float>());
  EXPECT_EQ(lv.shape(), (n::Shape{2, 4}));
  EXPECT_EQ(epsilon.shape(), lv.shape());
  EXPECT_EQ(
      pixels.to_vector<float>(), n::sigmoid(m::GaussianVaeDecode(p, code)).to_vector<float>());
}
TEST(NnLadder, InvalidInputsOverflowAndStridesBeforeLeafCreation) {
  n::ParameterStore p(1);
  auto              image = n::full({2, 6}, .3), e = n::zeros({2, 3});
  EXPECT_THROW(m::LadderVariationalAutoencoder(p, image, e, 6, 2, 1, 0, 2), std::invalid_argument);
  EXPECT_THROW(m::LadderVariationalAutoencoder(p, image, e, 6, 2, 1, 3, 0), std::invalid_argument);
  EXPECT_THROW(
      m::LadderVariationalAutoencoder(p, n::full({2, 6}, 1.1), e, 6, 2, 1, 3, 2),
      std::invalid_argument);
  EXPECT_THROW(
      m::LadderVariationalAutoencoder(p, image, n::zeros({1, 3}), 6, 2, 1, 3, 2),
      std::invalid_argument);
  EXPECT_THROW(
      m::LadderVariationalAutoencoder(
          p, image, e, 6, std::numeric_limits<std::int64_t>::max(), 1, 3, 2),
      std::invalid_argument);
  EXPECT_THROW(
      m::LadderPrior(p, n::zeros({2, 1}), n::zeros({1, 2}), 6, 2, 1, 3, 2), std::invalid_argument);
  EXPECT_THROW(
      m::LadderPrior(p, n::zeros({2, 1}), n::zeros({2, 2}, n::DType::Float64), 6, 2, 1, 3, 2),
      std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  auto sx = n::transpose(n::transpose(image).contiguous()),
       se = n::transpose(n::transpose(e).contiguous());
  EXPECT_FALSE(sx.is_contiguous());
  EXPECT_EQ(
      m::LadderVariationalAutoencoder(p, sx, se, 6, 2, 1, 3, 2).probability.to_vector<float>(),
      m::LadderVariationalAutoencoder(p, image, e, 6, 2, 1, 3, 2).probability.to_vector<float>());
  EXPECT_THROW(
      m::LadderPrecisionFusion(
          n::zeros({2, 2}), n::zeros({1, 2}), n::zeros({2, 2}), n::zeros({2, 2})),
      std::invalid_argument);
  EXPECT_THROW(
      m::LadderPrecisionFusion(
          n::zeros({2, 2}), n::full({2, 2}, INFINITY), n::zeros({2, 2}), n::zeros({2, 2})),
      std::invalid_argument);
}
