#include <gtest/gtest.h>

#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/vae_gan.hpp>
#include <limits>
namespace n = datamunge::nn;
namespace m = n::models;

namespace {
std::vector<n::Tensor> Fields(const m::VaeGanOutput& o) {
  return {
      o.mu,
      o.log_variance,
      o.z,
      o.probability,
      o.prior_probability,
      o.real_feature,
      o.reconstruction_feature,
      o.prior_feature,
      o.real_logit,
      o.reconstruction_logit,
      o.prior_logit,
      o.feature_nll,
      o.kl,
      o.pixel_nll,
      o.mse,
      o.encoder_loss,
      o.generator_loss,
      o.discriminator_loss,
      o.generator_adversarial};
}
} // namespace

TEST(NnVaeGan, FeatureDistanceSumsCoordinatesAndStopsOnlyTheRealTarget) {
  const auto t    = n::DType::Float64;
  auto       real = n::ones({2, 3}, t).requires_grad_(), rec = n::zeros({2, 3}, t).requires_grad_();
  const auto loss = m::VaeGanFeatureNll(real, rec);
  EXPECT_EQ(loss.to_vector<double>(), (std::vector<double>{1.5, 1.5}));
  const auto gradients = n::grad(n::sum(loss), {real, rec}, {.allow_unused = true});
  EXPECT_FALSE(gradients[0].defined());
  EXPECT_EQ(gradients[1].to_vector<double>(), (std::vector<double>(6, -1)));
  n::TensorFunction f = [&](const auto& a) { return n::sum(m::VaeGanFeatureNll(real, a[0])); };
  EXPECT_TRUE(n::gradcheck(f, {rec}, {.atol = 1e-9, .rtol = 1e-6}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {rec}, {.atol = 1e-9, .rtol = 1e-6}).ok);
  const auto strided = n::transpose(n::transpose(rec).contiguous());
  EXPECT_FALSE(strided.is_contiguous());
  EXPECT_TRUE(n::equal_bits(loss, m::VaeGanFeatureNll(real, strided)));
  EXPECT_THROW(m::VaeGanFeatureNll(real, n::zeros({2, 2}, t)), std::invalid_argument);
  EXPECT_THROW(
      m::VaeGanFeatureNll(n::zeros({0, 3}, t), n::zeros({0, 3}, t)), std::invalid_argument);
  EXPECT_THROW(m::VaeGanFeatureNll(real, n::zeros({2, 3})), std::invalid_argument);
  EXPECT_THROW(
      m::VaeGanFeatureNll(real, n::full({2, 3}, std::numeric_limits<double>::infinity(), t)),
      std::invalid_argument);
}

TEST(NnVaeGan, EverySmallSelectedGroupDerivativeAndNoiseHigherPaths) {
  const auto        t = n::DType::Float64;
  n::ParameterStore e(3301, t), g(3307, t), d(3313, t);
  auto              x       = n::tensor({.1, .2, .7, .8, .3, .4}, {2, 3}, t);
  auto              epsilon = n::tensor({.2, -.1, .3, -.2}, {2, 2}, t).requires_grad_();
  auto              prior   = n::tensor({-.2, .4, .5, -.3, .1, .8}, {3, 2}, t).requires_grad_();
  auto              out     = m::VaeGan(e, g, d, x, epsilon, prior, 3, 2, 2, 2);
  EXPECT_EQ(e.count() + g.count() + d.count(), 46);
  std::vector<n::ParameterStore*> stores{&e, &g, &d};
  for (int group = 0; group < 3; ++group) {
    auto objective = [&]() {
      auto o = m::VaeGan(e, g, d, x, epsilon, prior, 3, 2, 2, 2);
      return group == 0 ? o.encoder_loss : group == 1 ? o.generator_loss : o.discriminator_loss;
    };
    auto&               p = *stores[group];
    std::vector<double> analytic;
    for (const auto& v : n::grad(objective(), p.parameters())) {
      const auto a = v.to_vector<double>();
      analytic.insert(analytic.end(), a.begin(), a.end());
    }
    const auto initial = p.flatten();
    for (std::size_t i = 0; i < initial.size(); ++i) {
      auto changed = initial;
      changed[i] += 1e-6;
      p.unflatten(changed);
      const auto plus = objective().item();
      changed[i] -= 2e-6;
      p.unflatten(changed);
      const auto minus = objective().item();
      EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-8);
    }
    p.unflatten(initial);
  }
  n::TensorFunction f = [&](const auto& a) {
    auto o = m::VaeGan(e, g, d, x, a[0], a[1], 3, 2, 2, 2);
    return o.encoder_loss + o.generator_loss + o.discriminator_loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {epsilon, prior}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {epsilon, prior}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}

TEST(NnVaeGan, ExplicitThreeAdamStatesAndRecomputedDThenEThenGIsolation) {
  const auto        t = n::DType::Float64;
  n::ParameterStore e(3301, t), g(3307, t), d(3313, t);
  const auto        x = n::full({2, 3}, .3, t), epsilon = n::full({2, 2}, .2, t),
             prior           = n::full({3, 2}, -.4, t);
  auto               forward = [&]() { return m::VaeGan(e, g, d, x, epsilon, prior, 3, 2, 2, 2); };
  auto               initial = forward();
  const auto         old_encoder = n::grad(initial.encoder_loss, e.parameters());
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Adam;
  n::Optimizer                    eo(e, .003, config), go(g, .003, config), disc(d, .003, config);
  std::vector<n::ParameterStore*> stores{&e, &g, &d};
  std::vector<n::Optimizer*>      optimizers{&eo, &go, &disc};
  for (int group : {2, 0, 1}) {
    const std::vector<std::vector<double>> before{e.flatten(), g.flatten(), d.flatten()};
    const auto                             o           = forward();
    const auto                             loss        = group == 0   ? o.encoder_loss
                                                         : group == 1 ? o.generator_loss
                                                                      : o.discriminator_loss;
    const auto                             derivatives = n::grad(loss, stores[group]->parameters());
    if (group == 0) {
      double change = 0;
      for (std::size_t k = 0; k < derivatives.size(); ++k)
        change += n::sum(n::abs(derivatives[k] - old_encoder[k])).item();
      EXPECT_GT(change, 1e-8);
    }
    optimizers[group]->step(derivatives);
    for (int k = 0; k < 3; ++k) {
      if (k == group)
        EXPECT_NE(stores[k]->flatten(), before[k]);
      else
        EXPECT_EQ(stores[k]->flatten(), before[k]);
    }
  }
  EXPECT_EQ(eo.iterations(), 1);
  EXPECT_EQ(go.iterations(), 1);
  EXPECT_EQ(disc.iterations(), 1);
}

TEST(NnVaeGan, ZeroModelAlgebraBalancedUnequalBatchesAndDiagnosticClipping) {
  const auto        t = n::DType::Float64;
  n::ParameterStore e(3301, t), g(3307, t), d(3313, t);
  const auto x = n::zeros({2, 3}, t), epsilon = n::zeros({2, 2}, t), prior = n::zeros({3, 2}, t);
  m::VaeGan(e, g, d, x, epsilon, prior, 3, 2, 2, 2);
  for (auto* p : {&e, &g, &d})
    p->unflatten(std::vector<double>(p->count(), 0));
  const auto out = m::VaeGan(e, g, d, x, epsilon, prior, 3, 2, 2, 2);
  EXPECT_EQ(out.encoder_loss.item(), 0);
  EXPECT_DOUBLE_EQ(out.generator_loss.item(), std::log(2.));
  EXPECT_DOUBLE_EQ(out.discriminator_loss.item(), 2 * std::log(2.));
  for (double v : out.pixel_nll.to_vector<double>())
    EXPECT_NEAR(v, 3 * std::log(2.), 1e-14);
  EXPECT_EQ(out.mse.to_vector<double>(), (std::vector<double>{.25, .25}));
  g.set("output/b", n::full({3}, 1000., t));
  const auto clipped = m::VaeGan(e, g, d, x, epsilon, prior, 3, 2, 2, 2);
  const auto upper   = double(1.f - 1e-6f);
  for (double v : clipped.pixel_nll.to_vector<double>())
    EXPECT_NEAR(v, -3 * std::log(1 - upper), 1e-12);
  EXPECT_DOUBLE_EQ(clipped.encoder_loss.item(), out.encoder_loss.item());
  EXPECT_DOUBLE_EQ(clipped.generator_loss.item(), out.generator_loss.item());
  EXPECT_DOUBLE_EQ(clipped.discriminator_loss.item(), out.discriminator_loss.item());
  EXPECT_EQ(clipped.prior_probability.shape(), (n::Shape{3, 3}));
}

TEST(NnVaeGan, IndependentNoisePriorInterventionsStridesAndOwnedNineteenFields) {
  const auto             t = n::DType::Float64;
  std::vector<n::Tensor> saved;
  {
    n::ParameterStore e(3301, t), g(3307, t), d(3313, t);
    const auto        x       = n::tensor({.1, .2, .7, .8, .3, .4}, {2, 3}, t);
    const auto        epsilon = n::full({2, 2}, .2, t), prior = n::full({3, 2}, -.4, t);
    const auto        a        = m::VaeGan(e, g, d, x, epsilon, prior, 3, 2, 2, 2);
    const auto        before_e = e.flatten(), before_g = g.flatten(), before_d = d.flatten();
    const auto        b = m::VaeGan(e, g, d, x, n::full({2, 2}, -.3, t), prior, 3, 2, 2, 2);
    EXPECT_TRUE(n::equal_bits(a.mu, b.mu));
    EXPECT_TRUE(n::equal_bits(a.log_variance, b.log_variance));
    EXPECT_TRUE(n::equal_bits(a.prior_probability, b.prior_probability));
    EXPECT_GT(n::max(n::abs(a.probability - b.probability)).item(), 1e-6);
    const auto c = m::VaeGan(e, g, d, x, epsilon, n::ones({4, 2}, t), 3, 2, 2, 2);
    EXPECT_TRUE(n::equal_bits(a.probability, c.probability));
    EXPECT_TRUE(n::equal_bits(a.encoder_loss, c.encoder_loss));
    EXPECT_TRUE(n::equal_bits(a.probability, m::VaeGanDecode(g, a.z, 3, 2, 2)));
    const auto strided = n::transpose(n::transpose(x).contiguous());
    EXPECT_FALSE(strided.is_contiguous());
    const auto fields = Fields(m::VaeGan(e, g, d, strided, epsilon, prior, 3, 2, 2, 2));
    saved             = Fields(a);
    for (std::size_t k = 0; k < saved.size(); ++k)
      EXPECT_TRUE(n::equal_bits(fields[k], saved[k]));
    EXPECT_EQ(e.flatten(), before_e);
    EXPECT_EQ(g.flatten(), before_g);
    EXPECT_EQ(d.flatten(), before_d);
  }
  ASSERT_EQ(saved.size(), 19u);
  for (const auto& field : saved) {
    EXPECT_TRUE(field.defined());
    for (double v : field.to_vector<double>())
      EXPECT_TRUE(std::isfinite(v));
  }
  n::Tensor feature, logit;
  {
    n::ParameterStore d(3313, t);
    auto              o = m::VaeGanDiscriminator(d, n::full({2, 3}, -.2, t), 3, 2);
    feature             = o.feature;
    logit               = o.logit;
  }
  EXPECT_EQ(feature.shape(), (n::Shape{2, 2}));
  EXPECT_EQ(logit.shape(), (n::Shape{2, 1}));
}

TEST(NnVaeGan, InvalidInputsAndOverflowFailBeforeAnyGroupCreatesLeaves) {
  const auto        t = n::DType::Float64;
  n::ParameterStore e(3301, t), g(3307, t), d(3313, t), wrong(1);
  const auto x = n::zeros({2, 3}, t), epsilon = n::zeros({2, 2}, t), prior = n::zeros({3, 2}, t);
  for (const auto& image :
       {n::Tensor{},
        n::zeros({2, 3}),
        n::zeros({0, 3}, t),
        n::zeros({2, 2}, t),
        n::full({2, 3}, 1.1, t),
        n::full({2, 3}, std::numeric_limits<double>::quiet_NaN(), t)})
    EXPECT_THROW(m::VaeGan(e, g, d, image, epsilon, prior, 3, 2, 2, 2), std::invalid_argument);
  for (const auto& noise :
       {n::Tensor{},
        n::zeros({1, 2}, t),
        n::zeros({2, 3}, t),
        n::zeros({2, 2}),
        n::full({2, 2}, std::numeric_limits<double>::infinity(), t)})
    EXPECT_THROW(m::VaeGan(e, g, d, x, noise, prior, 3, 2, 2, 2), std::invalid_argument);
  for (const auto& code :
       {n::Tensor{},
        n::zeros({0, 2}, t),
        n::zeros({3, 3}, t),
        n::zeros({3, 2}),
        n::full({3, 2}, std::numeric_limits<double>::infinity(), t)})
    EXPECT_THROW(m::VaeGan(e, g, d, x, epsilon, code, 3, 2, 2, 2), std::invalid_argument);
  EXPECT_THROW(m::VaeGan(e, e, d, x, epsilon, prior, 3, 2, 2, 2), std::invalid_argument);
  EXPECT_THROW(m::VaeGan(e, g, e, x, epsilon, prior, 3, 2, 2, 2), std::invalid_argument);
  EXPECT_THROW(m::VaeGan(e, g, g, x, epsilon, prior, 3, 2, 2, 2), std::invalid_argument);
  EXPECT_THROW(m::VaeGan(e, wrong, d, x, epsilon, prior, 3, 2, 2, 2), std::invalid_argument);
  for (auto widths : std::vector<std::vector<std::int64_t>>{
           {0, 2, 2, 2},
           {3, 0, 2, 2},
           {3, 2, 0, 2},
           {3, 2, 2, 0},
           {3, 2, std::numeric_limits<std::int64_t>::max(), 2}})
    EXPECT_THROW(
        m::VaeGan(e, g, d, x, epsilon, prior, widths[0], widths[1], widths[2], widths[3]),
        std::invalid_argument);
  EXPECT_THROW(
      m::VaeGanDecode(g, prior, 3, 2, std::numeric_limits<std::int64_t>::max()),
      std::invalid_argument);
  EXPECT_THROW(
      m::VaeGanDiscriminator(d, x, 3, std::numeric_limits<std::int64_t>::max()),
      std::invalid_argument);
  EXPECT_EQ(e.size() + g.size() + d.size() + wrong.size(), 0u);
}

TEST(NnVaeGan, DefaultCensusCreationOrderAndStandaloneGeneratorGradients) {
  n::ParameterStore e(3301), g(3307), d(3313);
  const auto out = m::VaeGan(e, g, d, n::zeros({2, 784}), n::zeros({2, 4}), n::zeros({3, 4}));
  EXPECT_EQ(e.count(), 12696);
  EXPECT_EQ(g.count(), 13408);
  EXPECT_EQ(d.count(), 6289);
  EXPECT_EQ(e.size() + g.size() + d.size(), 14u);
  EXPECT_EQ(
      e.names(),
      (std::vector<std::string>{
          "hidden/W", "hidden/b", "mu/W", "mu/b", "log_variance/W", "log_variance/b"}));
  EXPECT_EQ(g.names(), (std::vector<std::string>{"hidden/W", "hidden/b", "output/W", "output/b"}));
  EXPECT_EQ(d.names(), (std::vector<std::string>{"feature/W", "feature/b", "logit/W", "logit/b"}));
  EXPECT_EQ(out.probability.shape(), (n::Shape{2, 784}));
  EXPECT_EQ(out.prior_probability.shape(), (n::Shape{3, 784}));
  n::ParameterStore standalone(3307, n::DType::Float64);
  auto              z      = n::full({2, 2}, -.2, n::DType::Float64).requires_grad_();
  const auto        pixels = m::VaeGanDecode(standalone, z, 3, 2, 2);
  EXPECT_GT(n::max(n::abs(n::grad(n::sum(pixels), {z})[0])).item(), 0);
  EXPECT_EQ(standalone.size(), 4u);
}
