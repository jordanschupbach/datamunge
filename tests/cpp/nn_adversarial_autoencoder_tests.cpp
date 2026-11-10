#include <gtest/gtest.h>

#include <datamunge/nn/adversarial_autoencoder.hpp>
#include <datamunge/nn/nn.hpp>
#include <limits>
namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnAdversarialAutoencoder, StableSummedBernoulliNllAndBothArgumentDerivatives) {
  const auto t      = n::DType::Float64;
  auto       logits = n::tensor({1000., 1000., 0., -1000., -1000., 0.}, {2, 3}, t).requires_grad_();
  auto       targets = n::tensor({0., 1., .5, 1., 0., .5}, {2, 3}, t).requires_grad_();
  auto       values  = m::BernoulliPixelNll(logits, targets);
  for (double v : values.to_vector<double>())
    EXPECT_NEAR(v, 1000 + std::log(2.), 1e-12);
  auto g = n::grad(n::sum(values), {logits, targets});
  EXPECT_EQ(g[0].to_vector<double>(), (std::vector<double>{1, 0, 0, -1, 0, 0}));
  EXPECT_EQ(g[1].to_vector<double>(), (std::vector<double>{-1000, -1000, 0, 1000, 1000, 0}));
  EXPECT_THROW(m::BernoulliPixelNll(n::zeros({2, 0}), n::zeros({2, 0})), std::invalid_argument);
  EXPECT_THROW(m::BernoulliPixelNll(n::zeros({1, 2}), n::full({1, 2}, 1.1)), std::invalid_argument);
  EXPECT_THROW(
      m::BernoulliPixelNll(
          n::full({1, 2}, std::numeric_limits<double>::infinity()), n::zeros({1, 2})),
      std::invalid_argument);
}
TEST(NnAdversarialAutoencoder, AllSmallModelParameterAndInputFirstSecondDerivatives) {
  const auto        t = n::DType::Float64;
  n::ParameterStore e(3501, t), g(3511, t), d(3527, t);
  auto              x     = n::tensor({.1, .2, .7, .8, .3, .4}, {2, 3}, t).requires_grad_();
  auto              prior = n::tensor({-.2, .4, .5, -.3, .1, .8}, {3, 2}, t).requires_grad_();
  auto              out   = m::AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2);
  EXPECT_EQ(e.count() + g.count() + d.count(), 38);
  EXPECT_EQ(out.prior_logit.shape(), (n::Shape{3, 1}));
  std::vector<n::ParameterStore*> stores{&e, &g, &d};
  for (int objective = 0; objective < 3; ++objective) {
    auto value = [&]() {
      auto o = m::AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2);
      return objective == 0   ? o.reconstruction_loss
             : objective == 1 ? o.discriminator_loss
                              : o.adversarial_loss;
    };
    for (int group = 0; group < 3; ++group) {
      if (objective == 0 ? group == 2 : group == 1)
        continue;
      auto&               p    = *stores[group];
      auto                grad = n::grad(value(), p.parameters());
      std::vector<double> analytic;
      for (const auto& v : grad) {
        auto a = v.to_vector<double>();
        analytic.insert(analytic.end(), a.begin(), a.end());
      }
      const auto initial = p.flatten();
      for (std::size_t i = 0; i < initial.size(); ++i) {
        auto changed = initial;
        changed[i] += 1e-6;
        p.unflatten(changed);
        double plus = value().item();
        changed[i] -= 2e-6;
        p.unflatten(changed);
        double minus = value().item();
        EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 1e-8);
      }
      p.unflatten(initial);
    }
  }
  n::TensorFunction all = [&](const auto& args) {
    auto o = m::AdversarialAutoencoder(e, g, d, args[0], args[1], 3, 2, 2);
    return o.reconstruction_loss + o.discriminator_loss + o.adversarial_loss;
  };
  EXPECT_TRUE(n::gradcheck(all, {x, prior}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(all, {x, prior}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  const auto changed = m::AdversarialAutoencoder(e, g, d, x, n::full({4, 2}, 2., t), 3, 2, 2);
  EXPECT_EQ(out.code.to_vector<double>(), changed.code.to_vector<double>());
  EXPECT_EQ(out.probability.to_vector<double>(), changed.probability.to_vector<double>());
  EXPECT_EQ(
      n::sigmoid(m::AaeDecode(g, out.code, 3, 2, 2)).to_vector<double>(),
      out.probability.to_vector<double>());
  EXPECT_GT(
      n::max(n::abs(n::take_rows(changed.prior_logit, {0, 1, 2}) - out.prior_logit)).item(), 0);
}
TEST(NnAdversarialAutoencoder, ReconstructionGradientBarrierAndFourIndependentOptimizerStates) {
  const auto        t = n::DType::Float64;
  n::ParameterStore e(3501, t), g(3511, t), d(3527, t);
  auto              x = n::full({2, 3}, .3, t), prior = n::tensor({-.5, .4, .8, -.2}, {2, 2}, t);
  m::AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2);
  n::OptimizerConfig config;
  config.kind = n::OptimizerKind::Adam;
  n::Optimizer er(e, .003, config), gr(g, .003, config), disc(d, .003, config), ea(e, .003, config);
  auto         before_e = e.flatten(), before_g = g.flatten(), before_d = d.flatten();
  auto         out  = m::AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2);
  auto         vars = e.parameters();
  auto         gp   = g.parameters();
  vars.insert(vars.end(), gp.begin(), gp.end());
  const auto             gradients = n::grad(out.reconstruction_loss, vars);
  const auto             count     = e.size();
  std::vector<n::Tensor> de(gradients.begin(), gradients.begin() + count),
      dg(gradients.begin() + count, gradients.end());
  er.step(de);
  // Recomputing G after writing E would use a different gradient snapshot.
  const auto wrong = n::grad(
      m::AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2).reconstruction_loss, g.parameters());
  double difference = 0;
  for (std::size_t i = 0; i < dg.size(); ++i)
    difference += n::sum(n::abs(dg[i] - wrong[i])).item();
  EXPECT_GT(difference, 1e-6);
  gr.step(dg);
  EXPECT_NE(e.flatten(), before_e);
  EXPECT_NE(g.flatten(), before_g);
  EXPECT_EQ(d.flatten(), before_d);
  before_e = e.flatten();
  before_g = g.flatten();
  disc.minimize(m::AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2).discriminator_loss);
  EXPECT_EQ(e.flatten(), before_e);
  EXPECT_EQ(g.flatten(), before_g);
  EXPECT_NE(d.flatten(), before_d);
  before_d = d.flatten();
  ea.minimize(m::AdversarialAutoencoder(e, g, d, x, prior, 3, 2, 2).adversarial_loss);
  EXPECT_NE(e.flatten(), before_e);
  EXPECT_EQ(g.flatten(), before_g);
  EXPECT_EQ(d.flatten(), before_d);
  EXPECT_EQ(er.iterations(), 1);
  EXPECT_EQ(gr.iterations(), 1);
  EXPECT_EQ(disc.iterations(), 1);
  EXPECT_EQ(ea.iterations(), 1);
}
TEST(NnAdversarialAutoencoder, OriginalCensusStandaloneComponentsAndValidationBeforeCreation) {
  n::ParameterStore e(3501), g(3511), d(3527);
  EXPECT_THROW(
      m::AdversarialAutoencoder(e, g, d, n::full({1, 784}, -1), n::zeros({1, 4})),
      std::invalid_argument);
  EXPECT_THROW(
      m::AdversarialAutoencoder(e, g, d, n::zeros({1, 784}), n::zeros({1, 3})),
      std::invalid_argument);
  EXPECT_THROW(
      m::AdversarialAutoencoder(e, e, d, n::zeros({1, 784}), n::zeros({1, 4})),
      std::invalid_argument);
  EXPECT_EQ(e.size() + g.size() + d.size(), 0u);
  const auto out = m::AdversarialAutoencoder(e, g, d, n::full({2, 784}, .2), n::zeros({3, 4}));
  EXPECT_EQ(e.count(), 12628);
  EXPECT_EQ(g.count(), 13408);
  EXPECT_EQ(d.count(), 97);
  EXPECT_EQ(e.size() + g.size() + d.size(), 12u);
  EXPECT_EQ(out.nll.shape(), (n::Shape{2}));
  EXPECT_EQ(out.reconstruction_loss.dim(), 0);
  n::ParameterStore standalone(3511);
  auto              code   = n::full({2, 4}, -.2).requires_grad_();
  auto              logits = m::AaeDecode(standalone, code);
  EXPECT_EQ(standalone.count(), 13408);
  EXPECT_EQ(standalone.size(), 4u);
  EXPECT_GT(n::sum(n::abs(n::grad(n::sum(logits), {code})[0])).item(), 0);
}
