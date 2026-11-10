#include <gtest/gtest.h>

#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/bidirectional_gan.hpp>
#include <datamunge/nn/gan_data.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/optim.hpp>
#include <limits>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
std::vector<n::Tensor> Fields(const m::BiganOutput& o) {
  return {
      o.fake,
      o.encoded,
      o.real_score,
      o.fake_score,
      o.discriminator_loss,
      o.eg_loss,
      o.reconstruction,
      o.latent_cycle};
}
} // namespace
TEST(NnBigan, OrderedRingCentresSeedsFinalCastAndInvalidOptions) {
  const auto                t      = n::DType::Float64;
  const auto                centre = m::GaussianRingPoints(9, 5507, 4, 2, 0, t).to_vector<double>();
  const std::vector<double> expected{2, 0, 0, 2, -2, 0, 0, -2, 2, 0, 0, 2, -2, 0, 0, -2, 2, 0};
  for (std::size_t k = 0; k < expected.size(); ++k)
    EXPECT_NEAR(centre[k], expected[k], 1e-14);
  const auto first = m::GaussianRingPoints(19, 5507, 8, 2, .08, t);
  EXPECT_FALSE(first.requires_grad());
  EXPECT_TRUE(first.is_contiguous());
  EXPECT_TRUE(n::equal_bits(first, m::GaussianRingPoints(19, 5507, 8, 2, .08, t)));
  EXPECT_FALSE(n::equal_bits(first, m::GaussianRingPoints(19, 5519, 8, 2, .08, t)));
  EXPECT_TRUE(n::equal_bits(n::cast(first, n::DType::Float32), m::GaussianRingPoints(19, 5507)));
  EXPECT_TRUE(
      n::equal_bits(
          n::cast(m::GaussianNoise({5, 2}, 1, 7).reshape({5, 2}), n::DType::Float32),
          m::GaussianRingPoints(5, 7, 1, 0, 1)));
  EXPECT_EQ(m::GaussianRingPoints(3, 1, 1, 0, 0).to_vector<float>(), std::vector<float>(6, 0));
  EXPECT_THROW(m::GaussianRingPoints(0), std::invalid_argument);
  EXPECT_THROW(m::GaussianRingPoints(INT64_MAX), std::invalid_argument);
  EXPECT_THROW(m::GaussianRingPoints(3, 1, 0), std::invalid_argument);
  for (auto value : {-1., double(INFINITY), double(NAN)}) {
    EXPECT_THROW(m::GaussianRingPoints(3, 1, 8, value), std::invalid_argument);
    EXPECT_THROW(m::GaussianRingPoints(3, 1, 8, 2, value), std::invalid_argument);
  }
  EXPECT_THROW(m::GaussianRingPoints(3, 1, 8, 2, .08, n::DType::Int32), std::invalid_argument);
  EXPECT_THROW(m::GaussianRingPoints(3, 1, 1, 1e300, 0), std::invalid_argument);
}
TEST(NnBigan, DefaultCensusCreationOrderAndCanonicalMapReuse) {
  n::ParameterStore g(6007), e(6011), d(6029), reference(6007);
  const auto        z = n::full({3, 2}, -.2), x = n::full({2, 2}, .3);
  const auto        out = m::BidirectionalGan(g, e, d, x, z);
  EXPECT_EQ(g.count(), 82);
  EXPECT_EQ(e.count(), 82);
  EXPECT_EQ(d.count(), 97);
  for (auto p : {&g, &e, &d}) {
    EXPECT_EQ(p->size(), 4u);
    EXPECT_EQ(
        p->names(), (std::vector<std::string>{"hidden/W", "hidden/b", "output/W", "output/b"}));
  }
  EXPECT_TRUE(n::equal_bits(out.fake, m::AaeDecode(reference, z, 2, 2, 16)));
  EXPECT_EQ(g.flatten(), reference.flatten());
  EXPECT_EQ(out.fake.shape(), (n::Shape{3, 2}));
  EXPECT_EQ(out.encoded.shape(), (n::Shape{2, 2}));
  EXPECT_EQ(out.real_score.shape(), (n::Shape{2, 1}));
  EXPECT_EQ(out.fake_score.shape(), (n::Shape{3, 1}));
  EXPECT_TRUE(n::equal_bits(out.reconstruction, m::BiganMap(g, out.encoded)));
  EXPECT_TRUE(n::equal_bits(out.latent_cycle, m::BiganMap(e, out.fake)));
}
TEST(NnBigan, UnequalPoolZeroModelAlgebraAndUnfittedDiagnosticPaths) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(1, t), e(2, t), d(3, t);
  auto              x = n::full({2, 3}, .4, t), z = n::full({4, 2}, -.3, t);
  m::BidirectionalGan(g, e, d, x, z, 2);
  for (auto p : {&g, &e, &d})
    p->unflatten(std::vector<double>(p->count(), 0));
  auto out = m::BidirectionalGan(g, e, d, x, z, 2);
  EXPECT_DOUBLE_EQ(out.discriminator_loss.item(), 2 * std::log(2.));
  EXPECT_DOUBLE_EQ(out.eg_loss.item(), 2 * std::log(2.));
  EXPECT_EQ(out.reconstruction.shape(), (n::Shape{2, 3}));
  EXPECT_EQ(out.latent_cycle.shape(), (n::Shape{4, 2}));
  const auto old_loss = out.eg_loss.item(), old_d = out.discriminator_loss.item();
  g.set("output/b", n::full({3}, 2., t));
  e.set("output/b", n::full({2}, -3., t));
  auto changed = m::BidirectionalGan(g, e, d, x, z, 2);
  EXPECT_EQ(changed.reconstruction.to_vector<double>(), std::vector<double>(6, 2));
  EXPECT_EQ(changed.latent_cycle.to_vector<double>(), std::vector<double>(8, -3));
  EXPECT_EQ(changed.eg_loss.item(), old_loss);
  EXPECT_EQ(changed.discriminator_loss.item(), old_d);
  // Nonzero diagnostics cannot secretly contribute to either training objective.
  EXPECT_EQ(n::sum(n::abs(n::grad(changed.eg_loss, g.parameters())[3])).item(), 0);
}
TEST(NnBigan, ExplicitPairPartsBothMatterAndStandaloneContinuousPathsStayLive) {
  const auto        t = n::DType::Float64;
  n::ParameterStore d(6029, t);
  auto x = n::full({2, 3}, .3, t).requires_grad_(), z = n::full({2, 2}, -.2, t).requires_grad_();
  auto original = m::BiganJointScore(d, x, z, 3);
  EXPECT_GT(
      n::max(n::abs(original - m::BiganJointScore(d, x, n::zeros({2, 2}, t), 3))).item(), 1e-8);
  EXPECT_GT(
      n::max(n::abs(original - m::BiganJointScore(d, n::zeros({2, 3}, t), z, 3))).item(), 1e-8);
  const auto derivatives = n::grad(n::sum(original), {x, z});
  for (const auto& v : derivatives)
    EXPECT_GT(n::max(n::abs(v)).item(), 1e-8);
  n::TensorFunction pair = [&](const auto& a) {
    return n::sum(n::square(m::BiganJointScore(d, a[0], a[1], 3)));
  };
  EXPECT_TRUE(n::gradcheck(pair, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(pair, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  n::ParameterStore map(6007, t);
  map.get("hidden/W", {2, 2}, n::init::zeros());
  map.get("hidden/b", {2}, n::init::zeros());
  map.get("output/W", {2, 3}, n::init::zeros());
  map.get("output/b", {3}, n::init::constant(4.));
  EXPECT_EQ(
      m::BiganMap(map, z, 3, 2).to_vector<double>(), std::vector<double>(6, 4.)); // no sigmoid
}
TEST(NnBigan, EverySmallSelectedParameterAndDataPriorHigherDerivative) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(6007, t), e(6011, t), d(6029, t);
  auto              x = n::tensor({-.3, .2, .7, .4, .1, -.2}, {2, 3}, t).requires_grad_(),
       z              = n::tensor({.3, -.5, .6, .2, -.1, .8}, {3, 2}, t).requires_grad_();
  auto forward        = [&]() { return m::BidirectionalGan(g, e, d, x, z, 2); };
  forward();
  std::vector<n::ParameterStore*> stores{&g, &e, &d};
  for (int group = 0; group < 3; ++group) {
    auto& p         = *stores[group];
    auto  objective = [&]() {
      auto o = forward();
      return group == 2 ? o.discriminator_loss : o.eg_loss;
    };
    std::vector<double> analytic;
    for (const auto& derivative : n::grad(objective(), p.parameters())) {
      auto a = derivative.to_vector<double>();
      analytic.insert(analytic.end(), a.begin(), a.end());
    }
    const auto initial = p.flatten();
    for (std::size_t k = 0; k < initial.size(); ++k) {
      auto changed = initial;
      changed[k] += 1e-6;
      p.unflatten(changed);
      auto plus = objective().item();
      changed[k] -= 2e-6;
      p.unflatten(changed);
      auto minus = objective().item();
      EXPECT_NEAR((plus - minus) / 2e-6, analytic[k], 1e-8);
    }
    p.unflatten(initial);
  }
  n::TensorFunction fn = [&](const auto& a) {
    auto o = m::BidirectionalGan(g, e, d, a[0], a[1], 2);
    return o.eg_loss + o.discriminator_loss + .1 * n::mean(n::square(o.reconstruction)) +
           .2 * n::mean(n::square(o.latent_cycle));
  };
  EXPECT_TRUE(n::gradcheck(fn, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnBigan, DThenEThenGOwnWritesAndELeavesGPartialBitExactWithDFixed) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    n::ParameterStore g(6007, type), e(6011, type), d(6029, type);
    auto              x = n::full({2, 3}, .3, type), z = n::full({3, 2}, -.2, type);
    auto              forward = [&]() { return m::BidirectionalGan(g, e, d, x, z, 3); };
    forward();
    n::OptimizerConfig cfg;
    cfg.kind  = n::OptimizerKind::Adam;
    cfg.beta1 = .5;
    n::Optimizer                    go(g, .0002, cfg), eo(e, .0002, cfg), disc(d, .0002, cfg);
    std::vector<n::ParameterStore*> stores{&g, &e, &d};
    std::vector<n::Optimizer*>      opts{&go, &eo, &disc};
    const auto                      old_e = n::grad(forward().eg_loss, e.parameters());
    std::vector<n::Tensor>          before_e;
    for (int group : {2, 1, 0}) {
      std::vector<std::vector<double>> before{g.flatten(), e.flatten(), d.flatten()};
      if (group == 1)
        before_e = n::grad(forward().eg_loss, g.parameters());
      if (group == 0) {
        const auto after = n::grad(forward().eg_loss, g.parameters());
        for (std::size_t k = 0; k < after.size(); ++k)
          EXPECT_TRUE(n::equal_bits(after[k], before_e[k]));
      }
      auto       o = forward();
      const auto derivatives =
          n::grad(group == 2 ? o.discriminator_loss : o.eg_loss, stores[group]->parameters());
      if (group == 1) {
        double change = 0;
        for (std::size_t k = 0; k < derivatives.size(); ++k)
          change += n::sum(n::abs(derivatives[k] - old_e[k])).item();
        EXPECT_GT(change, 1e-8);
      }
      opts[group]->step(derivatives);
      for (int k = 0; k < 3; ++k) {
        if (k == group)
          EXPECT_NE(stores[k]->flatten(), before[k]);
        else
          EXPECT_EQ(stores[k]->flatten(), before[k]);
      }
    }
    for (auto opt : opts)
      EXPECT_EQ(opt->iterations(), 1);
  }
}
TEST(NnBigan, StridesAndEightFieldsRetainLifetimeAfterStoresAndParentDisappear) {
  const auto             t = n::DType::Float64;
  std::vector<n::Tensor> saved;
  {
    n::ParameterStore g(6007, t), e(6011, t), d(6029, t);
    auto              x = n::tensor({-.3, .2, .7, .4, .1, -.2}, {2, 3}, t),
         z              = n::tensor({.3, -.5, .6, .2, -.1, .8}, {3, 2}, t);
    auto xs             = n::transpose(n::transpose(x).contiguous()),
         zs             = n::transpose(n::transpose(z).contiguous());
    ASSERT_FALSE(xs.is_contiguous());
    ASSERT_FALSE(zs.is_contiguous());
    saved            = Fields(m::BidirectionalGan(g, e, d, x, z, 2));
    const auto other = Fields(m::BidirectionalGan(g, e, d, xs, zs, 2));
    for (std::size_t k = 0; k < saved.size(); ++k)
      EXPECT_TRUE(n::equal_bits(saved[k], other[k]));
  }
  ASSERT_EQ(saved.size(), 8u);
  for (const auto& a : saved)
    for (double v : a.to_vector<double>())
      EXPECT_TRUE(std::isfinite(v));
}
TEST(NnBigan, InvalidInputsGeometryAndAliasedStoresFailBeforeAnyLeaves) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(1, t), e(2, t), d(3, t), wrong(4);
  auto              x = n::zeros({2, 3}, t), z = n::zeros({3, 2}, t);
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 3}, t),
        n::zeros({2, 0}, t),
        n::zeros({2, 3}),
        n::zeros({2, 3}, n::DType::Int32),
        n::full({2, 3}, NAN, t)})
    EXPECT_THROW(m::BidirectionalGan(g, e, d, bad, z, 2), std::invalid_argument);
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 2}, t),
        n::zeros({3, 0}, t),
        n::zeros({3, 2}),
        n::full({3, 2}, INFINITY, t)})
    EXPECT_THROW(m::BidirectionalGan(g, e, d, x, bad, 2), std::invalid_argument);
  EXPECT_THROW(m::BidirectionalGan(g, g, d, x, z, 2), std::invalid_argument);
  EXPECT_THROW(m::BidirectionalGan(g, e, g, x, z, 2), std::invalid_argument);
  EXPECT_THROW(m::BidirectionalGan(g, e, e, x, z, 2), std::invalid_argument);
  EXPECT_THROW(m::BidirectionalGan(g, wrong, d, x, z, 2), std::invalid_argument);
  EXPECT_THROW(m::BidirectionalGan(g, e, d, x, z, 0), std::invalid_argument);
  EXPECT_THROW(m::BidirectionalGan(g, e, d, x, z, INT64_MAX), std::invalid_argument);
  EXPECT_THROW(m::BiganMap(g, z, 0, 2), std::invalid_argument);
  EXPECT_THROW(m::BiganMap(g, z, INT64_MAX, 2), std::invalid_argument);
  EXPECT_THROW(m::BiganJointScore(d, x, z, 2), std::invalid_argument);
  EXPECT_THROW(m::BiganJointScore(d, x, n::zeros({2, 2}, t), INT64_MAX), std::invalid_argument);
  EXPECT_EQ(g.size() + e.size() + d.size() + wrong.size(), 0u);
}
