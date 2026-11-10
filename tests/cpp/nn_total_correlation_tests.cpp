#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/total_correlation_autoencoder.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
TEST(NnTotalCorrelation, ExactTwoComponentHandRatiosRetainNegativeSampleMean) {
  const auto t = n::DType::Float64;
  auto z = n::tensor({1., 1., 1., -1.}, {2, 2}, t), mu = n::tensor({-1., -1., 1., 1.}, {2, 2}, t),
       lv          = n::zeros({2, 2}, t);
  const auto   out = m::EmpiricalTotalCorrelation(z, mu, lv);
  const double v = std::exp(-2.), positive = std::log(2 * (1 + v * v) / std::pow(1 + v, 2)),
               negative = std::log(4 * v / std::pow(1 + v, 2));
  EXPECT_NEAR(out.per_case.at({0}), positive, 1e-14);
  EXPECT_NEAR(out.per_case.at({1}), negative, 1e-14);
  EXPECT_LT(out.mean.item(), 0);
  const auto single = m::EmpiricalTotalCorrelation(z, n::zeros({1, 2}, t), n::zeros({1, 2}, t));
  for (double x : single.per_case.to_vector<double>())
    EXPECT_NEAR(x, 0, 1e-14);
  const auto one =
      m::EmpiricalTotalCorrelation(n::zeros({3, 1}, t), n::ones({2, 1}, t), n::zeros({2, 1}, t));
  for (double x : one.per_case.to_vector<double>())
    EXPECT_NEAR(x, 0, 1e-14);
}
TEST(NnTotalCorrelation, AllSampleReferenceHeadFirstAndSecondDerivativesAndExtremeValues) {
  const auto        t  = n::DType::Float64;
  auto              z  = n::tensor({.3, -.2, 1., .4}, {2, 2}, t).requires_grad_();
  auto              mu = n::tensor({0., 0., 1., -.3, -.5, .8}, {3, 2}, t).requires_grad_();
  auto              lv = n::tensor({.2, -.3, .4, .1, -.6, .5}, {3, 2}, t).requires_grad_();
  n::TensorFunction f  = [](const auto& a) {
    return m::EmpiricalTotalCorrelation(a[0], a[1], a[2]).mean;
  };
  EXPECT_TRUE(n::gradcheck(f, {z, mu, lv}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {z, mu, lv}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  const auto g = n::grad(f({z, mu, lv}), {z, mu, lv});
  for (const auto& v : g)
    EXPECT_GT(n::sum(n::square(v)).item(), 0);
  const auto extreme = m::EmpiricalTotalCorrelation(n::tensor({1000., -1000.}, {1, 2}, t), mu, lv);
  EXPECT_TRUE(std::isfinite(extreme.mean.item()));
}
TEST(NnTotalCorrelation, PermutationMatchesOriginalShufflePreservesMarginalsAndDetaches) {
  const auto t = n::DType::Float64;
  auto       x = n::tensor({0., 10., 100., 1., 11., 101., 2., 12., 102., 3., 13., 103.}, {4, 3}, t)
               .requires_grad_();
  const auto out = m::PermuteLatentColumns(x, 2311), repeat = m::PermuteLatentColumns(x, 2311);
  EXPECT_FALSE(out.requires_grad());
  EXPECT_EQ(out.dtype(), t);
  EXPECT_EQ(out.to_vector<double>(), repeat.to_vector<double>());
  const auto          v = x.to_vector<double>();
  std::vector<double> expected(v.size());
  std::mt19937        random(2311);
  std::vector<int>    rows(4);
  for (int d = 0; d < 3; ++d) {
    std::iota(rows.begin(), rows.end(), 0);
    std::shuffle(rows.begin(), rows.end(), random);
    for (int i = 0; i < 4; ++i)
      expected[i * 3 + d] = v[rows[i] * 3 + d];
  }
  EXPECT_EQ(out.to_vector<double>(), expected);
  const auto strided = n::transpose(n::transpose(x).contiguous());
  EXPECT_FALSE(strided.is_contiguous());
  EXPECT_EQ(out.to_vector<double>(), m::PermuteLatentColumns(strided, 2311).to_vector<double>());
}
TEST(NnTotalCorrelation, AllSmallFactorAndBetaTcParameterDerivativesIncludeReferencePaths) {
  const auto t = n::DType::Float64;
  auto       x = n::tensor({.1, .4, .8, .2, .6, .3}, {2, 3}, t),
       e       = n::tensor({.3, -.6, -.2, .8}, {2, 2}, t),
       refs    = n::tensor({.2, .3, .7, .5, .4, .1, .8, .2, .6}, {3, 3}, t);
  for (bool factor : {true, false}) {
    n::ParameterStore p(2237, t), d(2309, t);
    auto              value = [&]() {
      return factor ? m::FactorVariationalAutoencoder(p, d, x, e, 3, 3, 2, 2, 2).loss
                                 : m::BetaTcVariationalAutoencoder(p, x, e, refs, 3, 3, 2, 2).loss;
    };
    value();
    for (auto* store : {&p, &d})
      for (const auto& name : store->names())
        store->set(name, n::full(store->at(name).shape(), name.ends_with("/b") ? .1 : .03, t));
    const auto loss = value();
    auto       vars = p.parameters();
    const auto ds   = d.parameters();
    vars.insert(vars.end(), ds.begin(), ds.end());
    const auto          gs = n::grad(loss, vars);
    std::vector<double> analytic;
    for (const auto& g : gs) {
      const auto a = g.to_vector<double>();
      analytic.insert(analytic.end(), a.begin(), a.end());
    }
    EXPECT_EQ(analytic.size(), factor ? 44 : 35);
    std::size_t offset = 0;
    for (auto* store : {&p, &d}) {
      const auto initial = store->flatten();
      for (std::size_t i = 0; i < initial.size(); ++i) {
        auto v = initial;
        v[i] += 1e-6;
        store->unflatten(v);
        const double plus = value().item();
        v[i] -= 2e-6;
        store->unflatten(v);
        const double minus = value().item();
        EXPECT_NEAR((plus - minus) / 2e-6, analytic[offset + i], 1e-8);
      }
      store->unflatten(initial);
      offset += initial.size();
    }
    if (!factor) {
      auto       live = refs.clone().requires_grad_();
      const auto o    = m::BetaTcVariationalAutoencoder(p, x, e, live, 3, 3, 2, 2);
      const auto rg   = n::grad(o.loss, {live})[0];
      EXPECT_GT(n::sum(n::square(rg)).item(), 0);
    }
  }
}
TEST(NnTotalCorrelation, OppositeStoresFrozenByExplicitTwoPhaseCallerAndBalancedLoss) {
  n::ParameterStore p(2237), d(2309);
  const auto        x = n::full({2, 3}, .3), e = n::full({2, 2}, .7);
  m::FactorVariationalAutoencoder(p, d, x, e, 3, 3, 2, 2, 2);
  n::OptimizerConfig c;
  c.kind = n::OptimizerKind::Adam;
  n::Optimizer po(p, .003, c), disc(d, .003, c);
  const auto   before = d.flatten();
  auto         out    = m::FactorVariationalAutoencoder(p, d, x, e, 3, 3, 2, 2, 2);
  po.step(n::grad(out.loss, p.parameters()));
  EXPECT_EQ(before, d.flatten());
  const auto vae_before = p.flatten();
  out                   = m::FactorVariationalAutoencoder(p, d, x, e, 3, 3, 2, 2, 2);
  const auto joint = out.net.z.detach(), product = m::PermuteLatentColumns(joint, 2311);
  disc.step(n::grad(m::FactorDiscriminatorLoss(d, joint, product, 2, 2), d.parameters()));
  EXPECT_EQ(vae_before, p.flatten());
  EXPECT_EQ(po.iterations(), 1);
  EXPECT_EQ(disc.iterations(), 1);
  for (const auto& name : d.names())
    d.set(name, n::zeros(d.at(name).shape()));
  EXPECT_NEAR(m::FactorDiscriminatorLoss(d, joint, product, 2, 2).item(), std::log(2.), 1e-7);
}
TEST(NnTotalCorrelation, InvalidInputsReferencesAndAliasedStoresBeforeAnyLeaves) {
  n::ParameterStore p(1), d(2);
  const auto        x = n::zeros({2, 3}), e = n::zeros({2, 2}), refs = n::zeros({3, 3});
  EXPECT_THROW(m::FactorVariationalAutoencoder(p, p, x, e, 3, 3, 2, 2, 2), std::invalid_argument);
  EXPECT_THROW(m::FactorVariationalAutoencoder(p, d, x, e, -1, 3, 2, 2, 2), std::invalid_argument);
  EXPECT_THROW(
      m::BetaTcVariationalAutoencoder(p, x, e, n::full({3, 3}, 1.1), 3, 3, 2, 2),
      std::invalid_argument);
  EXPECT_THROW(
      m::BetaTcVariationalAutoencoder(p, x, n::zeros({1, 2}), refs, 3, 3, 2, 2),
      std::invalid_argument);
  EXPECT_THROW(
      m::FactorDiscriminatorLoss(d, n::zeros({2, 2}), n::zeros({3, 3}), 2, 2),
      std::invalid_argument);
  EXPECT_THROW(
      m::EmpiricalTotalCorrelation(e, n::zeros({0, 2}), n::zeros({0, 2})), std::invalid_argument);
  EXPECT_THROW(
      m::EmpiricalTotalCorrelation(e, n::zeros({3, 2}), n::zeros({3, 2}, n::DType::Float64)),
      std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  EXPECT_EQ(d.size(), 0);
}
