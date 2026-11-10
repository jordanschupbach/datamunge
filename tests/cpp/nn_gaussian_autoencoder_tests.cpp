#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>
#include <limits>
namespace n = datamunge::nn;
namespace m = n::models;
TEST(NnGaussianAutoencoder, PathwiseValuesAndAllTwelveAnalyticDerivatives) {
  const auto                t = n::DType::Float64;
  const std::vector<double> a{.2, -.4, .1, .3}, b{0, std::log(4.), std::log(.25), .2},
      c{1, -.5, 2, -1};
  auto mu      = n::from_vector(a, {2, 2}).requires_grad_(),
       lv      = n::from_vector(b, {2, 2}).requires_grad_(),
       e       = n::from_vector(c, {2, 2}).requires_grad_();
  const auto z = m::GaussianReparameterize(mu, lv, e), kl = m::DiagonalGaussianKl(mu, lv);
  const auto g = n::grad(n::mean(.5 * n::sum(n::square(z), {1}) + 4 * kl), {mu, lv, e});
  for (int i = 0; i < 4; ++i) {
    const double sigma = std::exp(.5 * b[i]), v = a[i] + sigma * c[i];
    EXPECT_NEAR(z.to_vector<double>()[i], v, 1e-14);
    EXPECT_NEAR(g[0].to_vector<double>()[i], (v + 4 * a[i]) / 2, 1e-14);
    EXPECT_NEAR(
        g[1].to_vector<double>()[i], (.5 * v * sigma * c[i] + 2 * (std::exp(b[i]) - 1)) / 2, 1e-14);
    EXPECT_NEAR(g[2].to_vector<double>()[i], v * sigma / 2, 1e-14);
  }
  EXPECT_EQ(m::DiagonalGaussianKl(n::zeros({1, 2}, t), n::zeros({1, 2}, t)).item(), 0);
  EXPECT_EQ(m::DiagonalGaussianKl(n::ones({1, 3}, t), n::zeros({1, 3}, t)).item(), 1.5);
}
TEST(NnGaussianAutoencoder, PairedBetaChangesOnlyObjectiveAndExplicitNoiseOnlySamples) {
  n::ParameterStore p(2237);
  const auto        x = n::full({2, 3}, .3), e = n::full({2, 2}, .7);
  const auto        a = m::BetaVariationalAutoencoder(p, x, e, 1, 3, 2, 2),
             b        = m::BetaVariationalAutoencoder(p, x, e, 4, 3, 2, 2);
  EXPECT_EQ(p.size(), 10);
  EXPECT_EQ(p.count(), 35);
  EXPECT_EQ(a.mu.to_vector<float>(), b.mu.to_vector<float>());
  EXPECT_EQ(a.log_variance.to_vector<float>(), b.log_variance.to_vector<float>());
  EXPECT_EQ(a.probability.to_vector<float>(), b.probability.to_vector<float>());
  EXPECT_NEAR(b.loss.item() - a.loss.item(), 3 * n::mean(a.kl).item(), 2e-6);
  EXPECT_EQ(a.loss.item(), a.unweighted_loss.item());
  const auto before = p.flatten();
  const auto zero   = m::BetaVariationalAutoencoder(p, x, n::zeros({2, 2}), 1, 3, 2, 2);
  EXPECT_EQ(zero.z.to_vector<float>(), zero.mu.to_vector<float>());
  EXPECT_EQ(zero.mu.to_vector<float>(), a.mu.to_vector<float>());
  EXPECT_NE(zero.z.to_vector<float>(), a.z.to_vector<float>());
  EXPECT_EQ(
      n::sigmoid(m::GaussianVaeDecode(p, a.z, 3, 2, 2)).to_vector<float>(),
      a.probability.to_vector<float>());
  EXPECT_EQ(before, p.flatten());
  const auto g = n::grad(b.loss, p.parameters());
  ASSERT_EQ(g.size(), 10);
  for (const auto& v : g)
    for (double d : v.to_vector<double>())
      EXPECT_TRUE(std::isfinite(d));
}
TEST(NnGaussianAutoencoder, InvalidInputsBeforeParameterCreationAndStridedGradients) {
  n::ParameterStore p(1);
  auto              x = n::zeros({2, 3}), e = n::zeros({2, 2});
  for (double beta : {0., -1., std::numeric_limits<double>::infinity(), 1e-100, 1e100})
    EXPECT_THROW(m::BetaVariationalAutoencoder(p, x, e, beta, 3, 2, 2), std::invalid_argument);
  EXPECT_THROW(
      m::BetaVariationalAutoencoder(p, x, n::zeros({1, 2}), 1, 3, 2, 2), std::invalid_argument);
  EXPECT_THROW(
      m::BetaVariationalAutoencoder(p, n::full({2, 3}, 1.1), e, 1, 3, 2, 2), std::invalid_argument);
  EXPECT_THROW(m::BetaVariationalAutoencoder(p, x, e, 1, 0, 2, 2), std::invalid_argument);
  EXPECT_THROW(
      m::GaussianVaeDecode(p, n::zeros({2, 2}, n::DType::Int32), 3, 2, 2), std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
  auto base = n::ones({2, 3}, n::DType::Float64).requires_grad_();
  auto mu = n::transpose(base), lv = n::zeros({3, 2}, n::DType::Float64),
       eps     = n::ones({3, 2}, n::DType::Float64);
  const auto g = n::grad(n::sum(m::GaussianReparameterize(mu, lv, eps)), {base});
  EXPECT_EQ(g[0].to_vector<double>(), std::vector<double>(6, 1));
  EXPECT_THROW(m::DiagonalGaussianKl(mu, n::zeros({3, 2})), std::invalid_argument);
}
TEST(NnGaussianAutoencoder, GenericFactorCountsFitOnlySelectionAndNegativeR2) {
  // L=2,K=1; second coordinate duplicates first. Correlation ties choose axis0.
  const auto fit  = n::from_vector(std::vector<double>{-1, -1, 0, 0, 1, 1}, {3, 2});
  const auto test = n::from_vector(std::vector<double>{2, 2, 3, 3}, {2, 2});
  const auto y    = n::from_vector(std::vector<int>{1, 3, 5}, {3, 1});
  const auto yt   = n::from_vector(std::vector<int>{7, 9}, {2, 1});
  auto out = m::LatentFactorDiagnostics(fit, test, y, yt, n::zeros({3, 2}, n::DType::Float64));
  ASSERT_EQ(out.size(), 13);
  EXPECT_EQ(out[8].second.item(), 0);
  EXPECT_EQ(out[0].second.at({0}), 0);
  EXPECT_NEAR(out[1].second.at({0}), std::sqrt(2. / 3), 1e-14);
  EXPECT_EQ(out[2].second.item(), 3);
  EXPECT_NEAR(out[11].second.item(), 1, 1e-6);
  auto changed = m::LatentFactorDiagnostics(
      fit,
      test,
      y,
      n::from_vector(std::vector<int>{-100, -99}, {2, 1}),
      n::zeros({3, 2}, n::DType::Float64));
  for (int i : {0, 1, 2, 3, 4, 5, 7, 8, 9, 10})
    EXPECT_EQ(out[i].second.to_vector<double>(), changed[i].second.to_vector<double>());
  EXPECT_LT(changed[11].second.item(), 0);
  for (const auto& entry : out)
    EXPECT_FALSE(entry.second.requires_grad());
  EXPECT_THROW(
      m::LatentFactorDiagnostics(
          fit, test, y, n::ones({2, 1}, n::DType::Int32), n::zeros({3, 2}, n::DType::Float64)),
      std::invalid_argument);
  EXPECT_THROW(
      m::LatentFactorDiagnostics(fit, test, y, yt, n::zeros({3, 2}, n::DType::Float64), 0),
      std::invalid_argument);
}
TEST(NnGaussianAutoencoder, AllSmallModelLeafDerivativesAndSecondInputDerivatives) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(2237, t);
  auto              x = n::tensor({.1, .4, .8, .2, .6, .3}, {2, 3}, t).requires_grad_();
  auto              e = n::tensor({.3, -.6, -.2, .8}, {2, 2}, t).requires_grad_();
  auto       value    = [&]() { return m::BetaVariationalAutoencoder(p, x, e, 4, 3, 2, 2).loss; };
  const auto initial_loss = value();
  const auto gradients    = n::grad(initial_loss, p.parameters());
  std::vector<double> analytic;
  for (const auto& v : gradients) {
    const auto a = v.to_vector<double>();
    analytic.insert(analytic.end(), a.begin(), a.end());
  }
  const auto initial = p.flatten();
  ASSERT_EQ(initial.size(), 35);
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
  n::TensorFunction f = [&](const auto& args) {
    return m::BetaVariationalAutoencoder(p, args[0], args[1], 4, 3, 2, 2).loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {x, e}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {x, e}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
