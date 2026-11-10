#include <gtest/gtest.h>

#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/conditional_gan.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/optim.hpp>
#include <limits>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
std::vector<n::Tensor> Fields(const m::DenseGanOutput& o) {
  return {o.fake, o.real_score, o.fake_score, o.discriminator_loss, o.generator_loss};
}
std::vector<n::Tensor> Fields(const m::ConditionalGanOutput& o) {
  return {
      o.fake,
      o.real_logit,
      o.fake_logit,
      o.real_probability,
      o.fake_probability,
      o.discriminator_loss,
      o.generator_loss};
}
} // namespace
TEST(NnDenseGan, LeastSquaresUnequalPoolsAlgebraDerivativesAndSquaredOverflow) {
  const auto t = n::DType::Float64;
  auto       r = n::tensor({-2., 1., 3.}, {3, 1}, t).requires_grad_(),
       f       = n::tensor({-1., 2.}, {2, 1}, t).requires_grad_();
  auto o       = m::LeastSquaresGanObjectives(r, f);
  EXPECT_DOUBLE_EQ(o.discriminator.item(), .5 * (13. / 3 + 2.5));
  EXPECT_DOUBLE_EQ(o.generator.item(), 1.25);
  auto d = n::grad(o.discriminator, {r, f});
  auto g = n::grad(o.generator, {f});
  EXPECT_EQ(d[0].to_vector<double>(), (std::vector<double>{-1, 0, 2. / 3}));
  EXPECT_EQ(d[1].to_vector<double>(), (std::vector<double>{-.5, 1}));
  EXPECT_EQ(g[0].to_vector<double>(), (std::vector<double>{-1, .5}));
  n::TensorFunction fn = [](const auto& a) {
    auto o = m::LeastSquaresGanObjectives(a[0], a[1]);
    return o.discriminator + o.generator;
  };
  EXPECT_TRUE(n::gradcheck(fn, {r, f}, {.atol = 1e-9, .rtol = 1e-6}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {r, f}, {.atol = 1e-9, .rtol = 1e-6}).ok);
  EXPECT_THROW(
      m::LeastSquaresGanObjectives(n::full({1, 1}, 1e30), n::zeros({1, 1})), std::invalid_argument);
}
TEST(NnDenseGan, HingeExactMarginSubgradientAndUnhingedConstantGGradient) {
  for (auto t : {n::DType::Float32, n::DType::Float64}) {
    auto r = n::tensor({-80., 1., 2.}, {3, 1}, t).requires_grad_(),
         f = n::tensor({-80., -1., 2., 80.}, {4, 1}, t).requires_grad_();
    auto o = m::HingeGanObjectives(r, f);
    EXPECT_NEAR(o.discriminator.item(), 48., 2e-6);
    EXPECT_NEAR(o.generator.item(), -.25, 1e-12);
    auto d = n::grad(o.discriminator, {r, f}), g = n::grad(o.generator, {f});
    EXPECT_NEAR(d[0].at({0, 0}), -1. / 3, 1e-7);
    EXPECT_EQ(d[0].at({1, 0}), 0);
    EXPECT_EQ(d[0].at({2, 0}), 0);
    EXPECT_EQ(d[1].to_vector<double>(), (std::vector<double>{0, 0, .25, .25}));
    EXPECT_EQ(g[0].to_vector<double>(), std::vector<double>(4, -.25));
  }
  const auto t         = n::DType::Float64;
  auto       r         = n::tensor({-.3, 1.4}, {2, 1}, t).requires_grad_(),
       f               = n::tensor({-1.5, .2, 2.}, {3, 1}, t).requires_grad_();
  n::TensorFunction fn = [](const auto& a) {
    auto o = m::HingeGanObjectives(a[0], a[1]);
    return o.discriminator + o.generator;
  };
  EXPECT_TRUE(n::gradcheck(fn, {r, f}, {.atol = 1e-9, .rtol = 1e-6}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {r, f}, {.atol = 1e-9, .rtol = 1e-6}).ok);
}
TEST(NnDenseGan, InvalidScorePoolsAndStridesForBothObjectives) {
  const auto t = n::DType::Float64;
  auto       r = n::full({2, 1}, .3, t), f = n::full({3, 1}, -.2, t);
  auto       rs = n::select(n::full({2, 2}, .3, t), 1, 0).reshape({2, 1}),
       fs       = n::select(n::full({3, 2}, -.2, t), 1, 0).reshape({3, 1});
  for (bool hinge : {false, true}) {
    auto loss = [&](const n::Tensor& a, const n::Tensor& b) {
      return hinge ? m::HingeGanObjectives(a, b) : m::LeastSquaresGanObjectives(a, b);
    };
    auto a = loss(r, f), b = loss(rs, fs);
    EXPECT_TRUE(n::equal_bits(a.discriminator, b.discriminator));
    EXPECT_TRUE(n::equal_bits(a.generator, b.generator));
    for (auto bad :
         {n::Tensor{},
          n::zeros({0, 1}, t),
          n::zeros({3, 2}, t),
          n::zeros({3, 1}),
          n::full({3, 1}, NAN, t)})
      EXPECT_THROW(loss(r, bad), std::invalid_argument);
  }
}
TEST(NnDenseGan, DefaultCensusAndSameMapsAcrossAllObjectiveSelections) {
  n::ParameterStore g(5407), d(5413);
  auto              x = n::full({2, 784}, .3), z = n::full({3, 16}, -.2);
  auto              base = m::DenseGan(g, d, x, z, m::DenseGanObjective::LeastSquares);
  EXPECT_EQ(g.count(), 26416);
  EXPECT_EQ(d.count(), 25153);
  EXPECT_EQ(g.size() + d.size(), 8u);
  for (auto* p : {&g, &d})
    EXPECT_EQ(
        p->names(), (std::vector<std::string>{"hidden/W", "hidden/b", "output/W", "output/b"}));
  const auto beforeg = g.flatten(), befored = d.flatten();
  for (auto kind : {m::DenseGanObjective::Logistic, m::DenseGanObjective::Hinge}) {
    auto out = m::DenseGan(g, d, x, z, kind);
    EXPECT_TRUE(n::equal_bits(base.fake, out.fake));
    EXPECT_TRUE(n::equal_bits(base.real_score, out.real_score));
    EXPECT_TRUE(n::equal_bits(base.fake_score, out.fake_score));
  }
  EXPECT_EQ(g.flatten(), beforeg);
  EXPECT_EQ(d.flatten(), befored);
  EXPECT_EQ(base.fake.shape(), (n::Shape{3, 784}));
  EXPECT_EQ(base.real_score.shape(), (n::Shape{2, 1}));
}
TEST(NnDenseGan, StandaloneRawScoreIsNotSigmoidAndGUsesOnlySuppliedNoise) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(1, t), d(2, t);
  auto              z    = n::full({2, 2}, -.2, t).requires_grad_();
  auto              fake = m::MlpGanGenerate(g, z, 3, 2);
  EXPECT_GT(n::max(n::abs(n::grad(n::sum(fake), {z})[0])).item(), 1e-9);
  m::MlpGanDiscriminate(d, n::zeros({2, 3}, t), 3, 2);
  d.unflatten(std::vector<double>(d.count(), 0));
  d.set("output/b", n::full({1}, 4., t));
  EXPECT_EQ(
      m::MlpGanDiscriminate(d, n::ones({2, 3}, t), 3, 2).to_vector<double>(),
      std::vector<double>(2, 4));
}
TEST(NnDenseGan, EverySmallSelectedGroupAndNoiseHigherDerivativeForAllObjectives) {
  const auto t = n::DType::Float64;
  auto       x = n::tensor({.1, .4, -.2, .5, .2, .7}, {2, 3}, t).requires_grad_(),
       z       = n::tensor({-.3, .2, .6, -.1, .4, .8}, {3, 2}, t).requires_grad_();
  for (auto kind :
       {m::DenseGanObjective::Logistic,
        m::DenseGanObjective::LeastSquares,
        m::DenseGanObjective::Hinge}) {
    n::ParameterStore g(5407, t), d(5413, t);
    auto              forward = [&]() { return m::DenseGan(g, d, x, z, kind, 3, 2); };
    forward();
    for (int group = 0; group < 2; ++group) {
      auto& p         = group == 0 ? g : d;
      auto  objective = [&]() {
        auto o = forward();
        return group == 0 ? o.generator_loss : o.discriminator_loss;
      };
      std::vector<double> a;
      for (const auto& v : n::grad(objective(), p.parameters())) {
        auto w = v.to_vector<double>();
        a.insert(a.end(), w.begin(), w.end());
      }
      const auto initial = p.flatten();
      for (std::size_t k = 0; k < initial.size(); ++k) {
        auto v = initial;
        v[k] += 1e-6;
        p.unflatten(v);
        auto plus = objective().item();
        v[k] -= 2e-6;
        p.unflatten(v);
        auto minus = objective().item();
        EXPECT_NEAR((plus - minus) / 2e-6, a[k], 1e-8);
      }
      p.unflatten(initial);
    }
    n::TensorFunction f = [&](const auto& a) {
      auto o = m::DenseGan(g, d, a[0], a[1], kind, 3, 2);
      return o.discriminator_loss + o.generator_loss;
    };
    EXPECT_TRUE(n::gradcheck(f, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
    EXPECT_TRUE(n::gradgradcheck(f, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  }
}
TEST(NnDenseGan, BothNetworkConditionsAffectRealAndGeneratedPredictionsWithoutAuxiliaryHead) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(5101, t), d(5113, t);
  auto x = n::full({2, 3}, .3, t), z = n::full({2, 2}, -.2, t), a = n::zeros({2}, n::DType::Int32),
       b     = n::ones({2}, n::DType::Int64);
  auto first = m::ConditionalGan(g, d, x, z, a, 3, 2, 3),
       other = m::ConditionalGan(g, d, x, z, b, 3, 2, 3);
  EXPECT_GT(n::max(n::abs(first.real_logit - other.real_logit)).item(), 1e-8);
  EXPECT_GT(n::max(n::abs(first.fake - other.fake)).item(), 1e-8);
  EXPECT_TRUE(n::equal_bits(first.fake, m::ConditionalGanGenerate(g, z, a, 3, 2, 3)));
  EXPECT_TRUE(n::equal_bits(first.real_logit, m::ConditionalGanDiscriminate(d, x, a, 3, 2, 3)));
  EXPECT_EQ(d.names(), (std::vector<std::string>{"hidden/W", "hidden/b", "output/W", "output/b"}));
  n::ParameterStore dg(5101), dd(5113);
  m::ConditionalGan(dg, dd, n::zeros({2, 784}), n::zeros({2, 16}), n::zeros({2}, n::DType::Int32));
  EXPECT_EQ(dg.count(), 26736);
  EXPECT_EQ(dd.count(), 25473);
}
TEST(NnDenseGan, ConditionalEverySmallGroupAndContinuousHigherPaths) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(5101, t), d(5113, t);
  auto x = n::full({2, 3}, .3, t).requires_grad_(), z = n::full({2, 2}, -.2, t).requires_grad_(),
       labels  = n::tensor({0., 1.}, {2}, n::DType::Int32);
  auto forward = [&]() { return m::ConditionalGan(g, d, x, z, labels, 3, 2, 2); };
  forward();
  for (int group = 0; group < 2; ++group) {
    auto& p = group == 0 ? g : d;
    auto  f = [&]() {
      auto o = forward();
      return group == 0 ? o.generator_loss : o.discriminator_loss;
    };
    std::vector<double> a;
    for (const auto& v : n::grad(f(), p.parameters())) {
      auto w = v.to_vector<double>();
      a.insert(a.end(), w.begin(), w.end());
    }
    const auto initial = p.flatten();
    for (std::size_t k = 0; k < initial.size(); ++k) {
      auto v = initial;
      v[k] += 1e-6;
      p.unflatten(v);
      auto plus = f().item();
      v[k] -= 2e-6;
      p.unflatten(v);
      auto minus = f().item();
      EXPECT_NEAR((plus - minus) / 2e-6, a[k], 1e-8);
    }
    p.unflatten(initial);
  }
  n::TensorFunction f = [&](const auto& a) {
    auto o = m::ConditionalGan(g, d, a[0], a[1], labels, 3, 2, 2);
    return o.discriminator_loss + o.generator_loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnDenseGan, ExplicitDThenFreshGAdamWritesOnlyOwnGroupForAllThreeExperiments) {
  const auto t = n::DType::Float64;
  auto       x = n::full({2, 3}, .3, t), z = n::full({2, 2}, -.2, t),
       labels = n::zeros({2}, n::DType::Int32);
  for (int experiment = 0; experiment < 3; ++experiment) {
    n::ParameterStore g(5101, t), d(5113, t);
    auto              forward = [&]() {
      if (experiment == 0) {
        auto o = m::ConditionalGan(g, d, x, z, labels, 3, 2, 3);
        return std::pair{o.discriminator_loss, o.generator_loss};
      }
      auto o = m::DenseGan(
          g,
          d,
          x,
          z,
          experiment == 1 ? m::DenseGanObjective::LeastSquares : m::DenseGanObjective::Hinge,
          3,
          3);
      return std::pair{o.discriminator_loss, o.generator_loss};
    };
    forward();
    const auto         old = n::grad(forward().second, g.parameters());
    n::OptimizerConfig cfg;
    cfg.kind  = n::OptimizerKind::Adam;
    cfg.beta1 = .5;
    n::Optimizer go(g, .0002, cfg), disc(d, .0002, cfg);
    auto         g0 = g.flatten(), d0 = d.flatten();
    disc.step(n::grad(forward().first, d.parameters()));
    EXPECT_EQ(g.flatten(), g0);
    EXPECT_NE(d.flatten(), d0);
    d0            = d.flatten();
    auto   fresh  = n::grad(forward().second, g.parameters());
    double change = 0;
    for (std::size_t k = 0; k < fresh.size(); ++k)
      change += n::sum(n::abs(fresh[k] - old[k])).item();
    EXPECT_GT(change, 1e-8);
    go.step(fresh);
    EXPECT_NE(g.flatten(), g0);
    EXPECT_EQ(d.flatten(), d0);
    EXPECT_EQ(go.iterations(), 1);
    EXPECT_EQ(disc.iterations(), 1);
  }
}
TEST(NnDenseGan, StridesAndAllOwningFieldsSurviveStores) {
  const auto             t = n::DType::Float64;
  std::vector<n::Tensor> saved;
  {
    n::ParameterStore g(1, t), d(2, t);
    auto              x = n::full({2, 3}, .3, t), z = n::full({2, 2}, -.2, t),
         labels = n::tensor({0., 1.}, {2}, n::DType::Int32);
    auto xs     = n::transpose(n::transpose(x).contiguous()),
         zs     = n::transpose(n::transpose(z).contiguous());
    auto ls     = n::select(n::tensor({0., 8., 1., 8.}, {2, 2}, n::DType::Int32), 1, 0);
    ASSERT_FALSE(xs.is_contiguous());
    ASSERT_FALSE(ls.is_contiguous());
    auto a = Fields(m::ConditionalGan(g, d, x, z, labels, 3, 2, 3)),
         b = Fields(m::ConditionalGan(g, d, xs, zs, ls, 3, 2, 3));
    for (std::size_t k = 0; k < a.size(); ++k)
      EXPECT_TRUE(n::equal_bits(a[k], b[k]));
    saved = a;
  }
  {
    n::ParameterStore g(1, t), d(2, t);
    auto              x = n::full({2, 3}, .3, t), z = n::full({3, 2}, -.2, t);
    auto              xs = n::transpose(n::transpose(x).contiguous());
    auto              a = Fields(m::DenseGan(g, d, x, z, m::DenseGanObjective::LeastSquares, 3, 3)),
         b = Fields(m::DenseGan(g, d, xs, z, m::DenseGanObjective::LeastSquares, 3, 3));
    for (std::size_t k = 0; k < a.size(); ++k)
      EXPECT_TRUE(n::equal_bits(a[k], b[k]));
    saved.insert(saved.end(), a.begin(), a.end());
  }
  ASSERT_EQ(saved.size(), 12u);
  for (const auto& v : saved)
    for (double x : v.to_vector<double>())
      EXPECT_TRUE(std::isfinite(x));
}
TEST(NnDenseGan, AllMalformedInputsLabelsObjectivesAndProductsFailBeforeLeaves) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(1, t), d(2, t), wrong(3);
  auto x = n::zeros({2, 3}, t), z = n::zeros({2, 2}, t), labels = n::zeros({2}, n::DType::Int32);
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 3}, t),
        n::zeros({2, 0}, t),
        n::zeros({2, 3}),
        n::full({2, 3}, NAN, t)}) {
    EXPECT_THROW(
        m::DenseGan(g, d, bad, z, m::DenseGanObjective::LeastSquares, 3, 2), std::invalid_argument);
    EXPECT_THROW(m::ConditionalGan(g, d, bad, z, labels, 3, 2, 2), std::invalid_argument);
  }
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 2}, t),
        n::zeros({2, 0}, t),
        n::zeros({2, 2}),
        n::full({2, 2}, INFINITY, t)})
    EXPECT_THROW(
        m::DenseGan(g, d, x, bad, m::DenseGanObjective::Hinge, 3, 2), std::invalid_argument);
  for (auto bad :
       {n::Tensor{},
        n::zeros({2}),
        n::zeros({1}, n::DType::Int32),
        n::full({2}, -1, n::DType::Int64),
        n::full({2}, 2, n::DType::Int32)}) {
    EXPECT_THROW(m::ConditionalGan(g, d, x, z, bad, 3, 2, 2), std::invalid_argument);
    EXPECT_THROW(m::ConditionalGanDiscriminate(d, x, bad, 3, 2, 2), std::invalid_argument);
  }
  EXPECT_THROW(
      m::ConditionalGan(g, d, x, n::zeros({1, 2}, t), labels, 3, 2, 2), std::invalid_argument);
  EXPECT_THROW(m::DenseGan(g, g, x, z, m::DenseGanObjective::Hinge, 3, 2), std::invalid_argument);
  EXPECT_THROW(
      m::DenseGan(g, wrong, x, z, m::DenseGanObjective::Hinge, 3, 2), std::invalid_argument);
  EXPECT_THROW(
      m::DenseGan(g, d, x, z, static_cast<m::DenseGanObjective>(99), 3, 2), std::invalid_argument);
  EXPECT_THROW(
      m::DenseGan(g, d, x, z, m::DenseGanObjective::Hinge, 3, INT64_MAX), std::invalid_argument);
  EXPECT_THROW(m::ConditionalGan(g, d, x, z, labels, 3, INT64_MAX, 1), std::invalid_argument);
  EXPECT_THROW(m::ConditionalGanDiscriminate(d, x, labels, 3, INT64_MAX, 1), std::invalid_argument);
  EXPECT_EQ(g.size() + d.size() + wrong.size(), 0u);
}
