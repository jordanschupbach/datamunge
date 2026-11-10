#include <gtest/gtest.h>

#include <datamunge/nn/wasserstein_autoencoder.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/training.hpp>

namespace n = datamunge::nn;
namespace m = n::models;

TEST(NnWae, DefaultCensusCreationOrderDecoderOnlyRestoreAndCallerAdam) {
  n::ParameterStore p(3701);
  const auto        image = n::full({2, 784}, .3), prior = n::zeros({3, 4});
  auto              out = m::WassersteinAutoencoder(p, image, prior);
  EXPECT_EQ(p.size(), 8);
  EXPECT_EQ(p.count(), 26036);
  EXPECT_EQ(out.code.shape(), (n::Shape{2, 4}));
  EXPECT_EQ(out.mmd.xy.shape(), (n::Shape{2, 3}));
  EXPECT_EQ(
      p.names(),
      (std::vector<std::string>{
          "encoder/W",
          "encoder/b",
          "code/W",
          "code/b",
          "decoder/W",
          "decoder/b",
          "output/W",
          "output/b"}));
  EXPECT_TRUE(n::equal_bits(out.probability, n::sigmoid(m::WaeDecode(p, out.code))));
  n::ParameterStore decoder(1);
  m::WaeDecode(decoder, n::zeros({20, 4}));
  EXPECT_EQ(decoder.count(), 13408);
  EXPECT_EQ(decoder.size(), 4);
  for (const auto& name : decoder.names())
    decoder.set(name, p.at(name));
  EXPECT_TRUE(n::equal_bits(out.probability, n::sigmoid(m::WaeDecode(decoder, out.code))));
  const auto         before = p.flatten();
  n::OptimizerConfig cfg;
  cfg.kind = n::OptimizerKind::Adam;
  n::Optimizer adam(p, .003, cfg);
  adam.step(n::grad(out.loss, p.parameters()));
  EXPECT_EQ(adam.iterations(), 1);
  EXPECT_NE(p.flatten(), before);
}

TEST(NnWae, HandPixelSumsSignedObjectiveBothPathsAndZeroWeight) {
  const auto t    = n::DType::Float64;
  auto       x    = n::tensor({.1, .2, .3, .4}, {2, 2}, t).requires_grad_();
  auto       y    = n::tensor({.4, .6, .8, .9}, {2, 2}, t).requires_grad_();
  auto       cost = m::WaeSquaredPixelCost(x, y);
  EXPECT_NEAR(cost.at({0}), .25, 1e-15);
  EXPECT_NEAR(cost.at({1}), .5, 1e-15);
  auto mm = n::scalar(-.1, t).requires_grad_();
  auto c  = n::tensor({2., 4.}, {2}, t).requires_grad_();
  auto g  = n::grad(m::WaeObjective(c, mm), {c, mm});
  EXPECT_EQ(g[0].to_vector<double>(), (std::vector<double>{.5, .5}));
  EXPECT_EQ(g[1].item(), 10);
  EXPECT_EQ(m::WaeObjective(c, mm).item(), 2);
  EXPECT_EQ(n::grad(m::WaeObjective(c, mm, 0), {mm})[0].item(), 0);
  n::TensorFunction f = [](const auto& a) { return n::mean(m::WaeSquaredPixelCost(a[0], a[1])); };
  EXPECT_TRUE(n::gradcheck(f, {x, y}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {x, y}).ok);
  n::TensorFunction objective = [](const auto& a) { return m::WaeObjective(a[0], a[1], .7); };
  EXPECT_TRUE(n::gradcheck(objective, {c, mm}).ok);
  EXPECT_TRUE(n::gradgradcheck(objective, {c, mm}).ok);
}

TEST(NnWae, EveryTinyParameterDerivativeMatchesIndependentCentralDifferences) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(3701, t);
  const auto        x     = n::tensor({.2, .4, .3, .7, .5, .6, .3, .6, .2, .4, .7, .5}, {2, 6}, t);
  const auto        prior = n::tensor({.5, -.2, .1, .7, -.3, .2}, {3, 2}, t);
  auto value = [&]() { return m::WassersteinAutoencoder(p, x, prior, 10, 6, 2, 3).loss; };
  value();
  EXPECT_EQ(p.count(), 62);
  EXPECT_EQ(p.size(), 8);
  const auto          gradient = n::grad(value(), p.parameters());
  std::vector<double> analytic;
  for (const auto& g : gradient) {
    const auto v = g.to_vector<double>();
    analytic.insert(analytic.end(), v.begin(), v.end());
  }
  const auto original = p.flatten();
  for (std::size_t i = 0; i < original.size(); ++i) {
    auto a = original;
    a[i] += 1e-6;
    p.unflatten(a);
    double plus = value().item();
    a[i] -= 2e-6;
    p.unflatten(a);
    double minus = value().item();
    EXPECT_NEAR((plus - minus) / 2e-6, analytic[i], 2e-8) << i;
  }
  p.unflatten(original);
}

TEST(NnWae, ImagePriorAndDecoderFirstSecondDerivatives) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(3701, t);
  auto x = n::tensor({.2, .4, .3, .7, .5, .6, .3, .6, .2, .4, .7, .5}, {2, 6}, t).requires_grad_();
  auto prior          = n::tensor({.5, -.2, .1, .7, -.3, .2}, {3, 2}, t).requires_grad_();
  n::TensorFunction f = [&](const auto& a) {
    return m::WassersteinAutoencoder(p, a[0], a[1], 10, 6, 2, 3).loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {x, prior}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {x, prior}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  auto              code    = n::tensor({.2, -.3, .5, .7}, {2, 2}, t).requires_grad_();
  n::TensorFunction decoder = [&](const auto& a) {
    return n::mean(n::square(m::WaeDecode(p, a[0], 6, 2, 3)));
  };
  EXPECT_TRUE(n::gradcheck(decoder, {code}).ok);
  EXPECT_TRUE(n::gradgradcheck(decoder, {code}).ok);
}

TEST(NnWae, PriorChangesOnlyDiscrepancyUnequalCountsAndFullBatchCost) {
  n::ParameterStore p(3701);
  const auto        x      = n::full({3, 784}, .3);
  const auto        a      = m::WassersteinAutoencoder(p, x, n::zeros({4, 4}));
  const auto        before = p.flatten();
  const auto        b      = m::WassersteinAutoencoder(p, x, n::ones({5, 4}));
  EXPECT_TRUE(n::equal_bits(a.code, b.code));
  EXPECT_TRUE(n::equal_bits(a.logits, b.logits));
  EXPECT_TRUE(n::equal_bits(a.probability, b.probability));
  EXPECT_NE(a.mmd.value.item(), b.mmd.value.item());
  const auto small = m::WassersteinAutoencoder(p, n::take_rows(x, {0, 1}), n::zeros({7, 4}));
  EXPECT_TRUE(n::equal_bits(small.probability, n::take_rows(a.probability, {0, 1})));
  EXPECT_EQ(before, p.flatten());
  const auto zero = m::WassersteinAutoencoder(p, x, n::zeros({4, 4}), 0);
  EXPECT_EQ(zero.loss.item(), zero.reconstruction_loss.item());
}

TEST(NnWae, Float64StridesAllOwnedFieldsAndSavedLeafRestore) {
  const auto        t = n::DType::Float64;
  n::ParameterStore p(3701, t);
  const auto        x     = n::transpose(n::full({6, 2}, .3, t));
  const auto        prior = n::transpose(n::full({2, 3}, -.2, t));
  EXPECT_FALSE(x.is_contiguous());
  EXPECT_FALSE(prior.is_contiguous());
  auto out   = m::WassersteinAutoencoder(p, x, prior, 10, 6, 2, 3);
  auto saved = out.mmd;
  auto code  = out.code;
  EXPECT_TRUE(
      n::equal_bits(
          out.probability,
          m::WassersteinAutoencoder(p, x.contiguous(), prior.contiguous(), 10, 6, 2, 3)
              .probability));
  n::ParameterStore restored(8, t);
  m::WassersteinAutoencoder(restored, x, prior, 10, 6, 2, 3);
  for (const auto& name : p.names())
    restored.set(name, p.at(name).clone());
  EXPECT_TRUE(
      n::equal_bits(
          out.probability, m::WassersteinAutoencoder(restored, x, prior, 10, 6, 2, 3).probability));
  out = {};
  EXPECT_EQ(code.shape(), (n::Shape{2, 2}));
  EXPECT_EQ(saved.xy.shape(), (n::Shape{2, 3}));
  EXPECT_TRUE(std::isfinite(saved.value.item()));
}

TEST(NnWae, InvalidInputsControlsCountsAndGeometryPrecedeLeaves) {
  n::ParameterStore p(1);
  const auto        x = n::full({2, 6}, .3), prior = n::zeros({3, 2});
  for (double weight : {-1., double(INFINITY), double(NAN), 1e100, 1e-100})
    EXPECT_THROW(m::WassersteinAutoencoder(p, x, prior, weight, 6, 2, 3), std::invalid_argument);
  EXPECT_THROW(
      m::WassersteinAutoencoder(p, n::full({2, 6}, 1.1), prior, 10, 6, 2, 3),
      std::invalid_argument);
  EXPECT_THROW(
      m::WassersteinAutoencoder(p, n::zeros({1, 6}), prior, 0, 6, 2, 3), std::invalid_argument);
  EXPECT_THROW(
      m::WassersteinAutoencoder(p, x, n::zeros({1, 2}), 0, 6, 2, 3), std::invalid_argument);
  EXPECT_THROW(
      m::WassersteinAutoencoder(p, x, n::zeros({3, 2}, n::DType::Float64), 10, 6, 2, 3),
      std::invalid_argument);
  EXPECT_THROW(m::WassersteinAutoencoder(p, x, prior, 10, 6, 2, 0), std::invalid_argument);
  EXPECT_THROW(
      m::WassersteinAutoencoder(p, x, prior, 10, 6, 2, std::numeric_limits<std::int64_t>::max()),
      std::invalid_argument);
  EXPECT_THROW(m::WaeDecode(p, n::zeros({2, 3}), 6, 2, 3), std::invalid_argument);
  EXPECT_EQ(p.size(), 0);
}

TEST(NnWae, SquaredCostAndObjectiveRejectMalformedValuesAndDoNotClampSignedMmd) {
  EXPECT_THROW(m::WaeSquaredPixelCost(n::zeros({2, 3}), n::zeros({1, 3})), std::invalid_argument);
  EXPECT_THROW(
      m::WaeSquaredPixelCost(n::zeros({2, 3}), n::full({2, 3}, -.1)), std::invalid_argument);
  EXPECT_THROW(
      m::WaeSquaredPixelCost(n::full({2, 3}, NAN), n::zeros({2, 3})), std::invalid_argument);
  EXPECT_THROW(m::WaeObjective(n::tensor({-1., 2.}), n::scalar(.1)), std::invalid_argument);
  EXPECT_THROW(m::WaeObjective(n::zeros({0}), n::scalar(.1)), std::invalid_argument);
  EXPECT_THROW(m::WaeObjective(n::ones({2}), n::ones({2})), std::invalid_argument);
  EXPECT_THROW(m::WaeObjective(n::ones({2}), n::scalar(NAN)), std::invalid_argument);
  EXPECT_LT(m::WaeObjective(n::zeros({2}), n::scalar(-.1)).item(), 0);
}
