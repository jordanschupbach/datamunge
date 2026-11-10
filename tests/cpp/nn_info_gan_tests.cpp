#include <gtest/gtest.h>

#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/info_gan.hpp>
#include <datamunge/nn/optim.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
std::vector<n::Tensor> Fields(const m::InfoGanOutput& o) {
  return {
      o.fake,
      o.real_logit,
      o.fake_logit,
      o.real_probability,
      o.fake_probability,
      o.discriminator_loss,
      o.generator_loss,
      o.source_generator_loss,
      o.recognition_loss,
      o.categorical_loss,
      o.continuous_loss,
      o.mi_lower_bound,
      o.q_logits,
      o.q_probability,
      o.q_mean};
}
} // namespace
TEST(NnInfoGan, PriorGoldenWordsRowOrderFinalCastAndRejection) {
  auto p = m::InfoPrior(4, 5382);
  EXPECT_EQ(p.category.dtype(), n::DType::Int32);
  EXPECT_EQ(p.raw.dtype(), n::DType::Float64);
  EXPECT_EQ(p.category.to_vector<std::int32_t>(), (std::vector<std::int32_t>{0, 9, 1, 1}));
  EXPECT_EQ(
      p.raw.to_vector<double>(),
      (std::vector<double>{
          2873073520.,
          2159082293.,
          4137076914.,
          1184340869.,
          3924616428.,
          325940591.,
          1372298291.,
          1298187572.,
          3078163632.,
          3244090601.,
          3189635439.,
          2886599427.}));
  auto wide = m::InfoPrior(4, 5382, 10, 2, n::DType::Float64), repeat = m::InfoPrior(4, 5382);
  EXPECT_TRUE(n::equal_bits(p.continuous, n::cast(wide.continuous, n::DType::Float32)));
  EXPECT_TRUE(n::equal_bits(p.continuous, repeat.continuous));
  EXPECT_TRUE(n::equal_bits(p.raw, wide.raw));
  EXPECT_FALSE(n::equal_bits(p.continuous, m::InfoPrior(4, 5383).continuous));
  EXPECT_FALSE(p.continuous.requires_grad());
  EXPECT_FALSE(p.raw.requires_grad());
  auto rejection = m::InfoPrior(2, 1, 1073741825, 2);
  EXPECT_EQ(
      rejection.raw.to_vector<double>(),
      (std::vector<double>{
          1791095845., 4282876139., 3093770124., 491263., 550290313., 1298508491.}));
  EXPECT_EQ(
      rejection.category.to_vector<std::int32_t>(), (std::vector<std::int32_t>{717354020, 491263}));
  auto single = m::InfoPrior(7, 3, 1, 3);
  EXPECT_EQ(single.category.to_vector<std::int32_t>(), std::vector<std::int32_t>(7, 0));
  EXPECT_EQ(single.raw.shape(), n::Shape({7, 4}));
}
TEST(NnInfoGan, PriorRejectsInvalidCountsDtypesAndProducts) {
  auto top = std::numeric_limits<std::int64_t>::max();
  EXPECT_THROW(m::InfoPrior(0, 1), std::invalid_argument);
  EXPECT_THROW(m::InfoPrior(-1, 1), std::invalid_argument);
  EXPECT_THROW(m::InfoPrior(2, 1, 0), std::invalid_argument);
  EXPECT_THROW(m::InfoPrior(2, 1, 2147483648LL), std::invalid_argument);
  EXPECT_THROW(m::InfoPrior(2, 1, 10, 0), std::invalid_argument);
  EXPECT_THROW(m::InfoPrior(2, 1, 10, top), std::invalid_argument);
  EXPECT_THROW(m::InfoPrior(top, 1, 10, 2), std::invalid_argument);
  EXPECT_THROW(m::InfoPrior(2, 1, 10, 2, n::DType::Int32), std::invalid_argument);
}
TEST(NnInfoGan, RecognitionAlgebraTargetDerivativesAndUnusedProbability) {
  auto t      = n::DType::Float64;
  auto logits = n::zeros({2, 2}, t).requires_grad_(), mean = n::zeros({2, 1}, t).requires_grad_(),
       u               = n::tensor({-1., 1.}, {2, 1}, t).requires_grad_();
  auto               c = n::tensor({0, 1}, {2}, n::DType::Int64);
  m::InfoRecognition q{logits, {}, mean};
  auto               r = m::InfoGanRecognitionLoss(q, c, u);
  EXPECT_NEAR(r.categorical.item(), std::log(2.), 1e-14);
  EXPECT_NEAR(r.continuous.item(), .5 * (1 + std::log(2 * std::acos(-1.))), 1e-14);
  EXPECT_NEAR(r.lower_bound.item(), 2 * std::log(2.) - r.total.item(), 1e-14);
  EXPECT_LT(r.lower_bound.item(), 0);
  auto g = n::grad(r.total, {logits, mean, u});
  EXPECT_EQ(g[0].to_vector<double>(), (std::vector<double>{-.25, .25, .25, -.25}));
  EXPECT_EQ(g[1].to_vector<double>(), (std::vector<double>{.5, -.5}));
  EXPECT_EQ(g[2].to_vector<double>(), (std::vector<double>{-.5, .5}));
  q.probability = n::full({3}, NAN);
  EXPECT_TRUE(n::equal_bits(r.total, m::InfoGanRecognitionLoss(q, c, u).total));
  // Use interior targets for centred finite differences under the bounded-prior contract.
  auto              interior = n::tensor({-.3, .4}, {2, 1}, t).requires_grad_();
  n::TensorFunction fn       = [&](const auto& a) {
    return m::InfoGanRecognitionLoss({a[0], {}, a[1]}, c, a[2]).total;
  };
  EXPECT_TRUE(n::gradcheck(fn, {logits, mean, interior}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {logits, mean, interior}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnInfoGan, RecognitionStableExtremeLogitsAndMalformedHeads) {
  auto c      = n::tensor({0, 1}, {2}, n::DType::Int64);
  auto logits = n::tensor({1000, -1000, -1000, 1000}, {2, 2}), mean = n::zeros({2, 2}),
       u    = n::zeros({2, 2});
  auto good = m::InfoGanRecognitionLoss({logits, {}, mean}, c, u);
  EXPECT_DOUBLE_EQ(good.categorical.item(), 0);
  EXPECT_THROW(
      m::InfoGanRecognitionLoss({logits, {}, n::full({2, 2}, 1e30)}, c, u), std::invalid_argument);
  for (auto bad :
       {n::Tensor{},
        n::zeros({1, 2}),
        n::zeros({2, 3}),
        n::zeros({2, 2}, n::DType::Float64),
        n::full({2, 2}, NAN),
        n::full({2, 2}, 2.)})
    EXPECT_THROW(m::InfoGanRecognitionLoss({logits, {}, mean}, c, bad), std::invalid_argument);
  EXPECT_THROW(
      m::InfoGanRecognitionLoss({logits, {}, n::zeros({1, 2})}, c, u), std::invalid_argument);
  EXPECT_THROW(
      m::InfoGanRecognitionLoss({n::zeros({0, 2}), {}, mean}, c, u), std::invalid_argument);
  EXPECT_THROW(
      m::InfoGanRecognitionLoss({logits, {}, mean}, n::full({2}, 2, n::DType::Int32), u),
      std::invalid_argument);
}
TEST(NnInfoGan, DefaultCensusUnequalPoolsAndStandaloneImageOnlyQ) {
  n::ParameterStore g(5303), d(5309), q(5317);
  auto              codes = m::InfoPrior(3, 5351);
  auto              x = n::full({2, 784}, .3), z = n::full({3, 16}, -.2);
  auto              out = m::InfoGan(g, d, q, x, z, codes.category, codes.continuous);
  EXPECT_EQ(g.count(), 26800);
  EXPECT_EQ(d.count(), 25153);
  EXPECT_EQ(q.count(), 25516);
  EXPECT_EQ(g.size() + d.size() + q.size(), 14);
  EXPECT_EQ(
      q.names(),
      (std::vector<std::string>{
          "hidden/W", "hidden/b", "category/W", "category/b", "continuous/W", "continuous/b"}));
  EXPECT_EQ(out.real_logit.shape(), n::Shape({2, 1}));
  EXPECT_EQ(out.fake.shape(), n::Shape({3, 784}));
  EXPECT_TRUE(n::equal_bits(out.fake, m::InfoGanGenerate(g, z, codes.category, codes.continuous)));
  EXPECT_TRUE(n::equal_bits(out.real_logit, m::MlpGanDiscriminate(d, x)));
  auto pred = m::InfoGanRecognize(q, out.fake);
  EXPECT_TRUE(n::equal_bits(pred.probability, out.q_probability));
  EXPECT_TRUE(n::equal_bits(pred.mean, out.q_mean));
  auto before = q.flatten();
  m::InfoGanGenerate(g, z, n::zeros({3}, n::DType::Int32), n::zeros({3, 2}));
  EXPECT_EQ(q.flatten(), before);
  q.set("continuous/W", n::zeros({32, 2}));
  q.set("continuous/b", n::full({2}, 4.));
  EXPECT_EQ(m::InfoGanRecognize(q, x).mean.to_vector<double>(), std::vector<double>(4, 4.));
}
TEST(NnInfoGan, EverySmallG_D_QLeafDerivativeMatchesFiniteDifferences) {
  auto                            t = n::DType::Float64;
  n::ParameterStore               g(5303, t), d(5309, t), q(5317, t);
  std::vector<n::ParameterStore*> stores{&g, &d, &q};
  auto                            x = n::tensor({.1, .3, -.2, .4, .2, -.1}, {2, 3}, t),
       z                            = n::tensor({-.2, .1, .3, -.1, .4, .2}, {3, 2}, t),
       u = n::tensor({-.3, .1, .4}, {3, 1}, t), c = n::tensor({0, 1, 0}, {3}, n::DType::Int32);
  auto forward = [&]() { return m::InfoGan(g, d, q, x, z, c, u, 3, 2, 2); };
  forward();
  for (int group = 0; group < 3; ++group) {
    auto o         = forward();
    auto gradients = n::grad(
        group == 0   ? o.generator_loss
        : group == 1 ? o.discriminator_loss
                     : o.recognition_loss,
        stores[group]->parameters());
    auto names = stores[group]->names();
    for (std::size_t k = 0; k < names.size(); ++k) {
      auto parameter = stores[group]->at(names[k]);
      auto ptr       = parameter.data_ptr<double>();
      for (std::int64_t j = 0; j < parameter.numel(); ++j) {
        double old = ptr[j], h = 1e-5;
        ptr[j]      = old + h;
        auto   plus = forward();
        double a    = (group == 0   ? plus.generator_loss
                       : group == 1 ? plus.discriminator_loss
                                    : plus.recognition_loss)
                       .item();
        ptr[j]       = old - h;
        auto   minus = forward();
        double b     = (group == 0   ? minus.generator_loss
                        : group == 1 ? minus.discriminator_loss
                                     : minus.recognition_loss)
                       .item();
        ptr[j] = old;
        EXPECT_NEAR((a - b) / (2 * h), gradients[k].to_vector<double>()[j], 2e-8)
            << group << "/" << names[k] << "[" << j << "]";
      }
    }
  }
}
TEST(NnInfoGan, DataNoiseContinuousFirstAndHigherPaths) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(5303, t), d(5309, t), q(5317, t);
  auto x = n::full({2, 3}, .3, t).requires_grad_(), z = n::full({3, 2}, -.2, t).requires_grad_(),
       u = n::full({3, 1}, .1, t).requires_grad_(), c = n::zeros({3}, n::DType::Int32);
  n::TensorFunction f = [&](const auto& a) {
    auto out = m::InfoGan(g, d, q, a[0], a[1], c, a[2], 3, 2, 2);
    return out.discriminator_loss + out.generator_loss;
  };
  EXPECT_TRUE(n::gradcheck(f, {x, z, u}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {x, z, u}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnInfoGan, D_Q_FreshGAdamOwnershipAndUnrelatedGradientIndependence) {
  auto                            t = n::DType::Float64;
  n::ParameterStore               g(5303, t), d(5309, t), q(5317, t);
  std::vector<n::ParameterStore*> stores{&g, &d, &q};
  auto x = n::full({2, 3}, .3, t), z = n::full({2, 2}, -.2, t), u = n::full({2, 1}, .1, t),
       c       = n::tensor({0, 1}, {2}, n::DType::Int32);
  auto forward = [&]() { return m::InfoGan(g, d, q, x, z, c, u, 3, 2, 3); };
  forward();
  n::OptimizerConfig cfg;
  cfg.kind  = n::OptimizerKind::Adam;
  cfg.beta1 = .5;
  n::Optimizer               go(g, .0002, cfg), disc(d, .0002, cfg), rec(q, .0002, cfg);
  std::vector<n::Optimizer*> opts{&go, &disc, &rec};
  auto                       q_old = n::grad(forward().recognition_loss, q.parameters());
  auto                       g_old = n::grad(forward().generator_loss, g.parameters());
  for (int group : {1, 2, 0}) {
    std::vector<std::vector<double>> before;
    for (auto* p : stores)
      before.push_back(p->flatten());
    if (group == 2) {
      auto after = n::grad(forward().recognition_loss, q.parameters());
      for (std::size_t k = 0; k < after.size(); ++k)
        EXPECT_TRUE(n::equal_bits(q_old[k], after[k]));
    }
    if (group == 0) {
      auto   fresh  = n::grad(forward().generator_loss, g.parameters());
      double change = 0;
      for (std::size_t k = 0; k < fresh.size(); ++k)
        change += n::sum(n::abs(fresh[k] - g_old[k])).item();
      EXPECT_GT(change, 1e-8);
    }
    auto out = forward();
    opts[group]->step(
        n::grad(
            group == 0   ? out.generator_loss
            : group == 1 ? out.discriminator_loss
                         : out.recognition_loss,
            stores[group]->parameters()));
    for (int k = 0; k < 3; ++k)
      EXPECT_EQ(stores[k]->flatten() != before[k], k == group);
  }
  for (auto* o : opts)
    EXPECT_EQ(o->iterations(), 1);
}
TEST(NnInfoGan, StridesAndFifteenOwningFieldsAfterStores) {
  auto                   t = n::DType::Float64;
  std::vector<n::Tensor> saved;
  {
    n::ParameterStore g(1, t), d(2, t), q(3, t);
    auto x = n::full({2, 3}, .3, t), z = n::full({3, 2}, -.2, t), u = n::full({3, 1}, .1, t),
         c  = n::tensor({0, 1, 0}, {3}, n::DType::Int64);
    auto xs = n::transpose(n::transpose(x).contiguous()),
         zs = n::transpose(n::transpose(z).contiguous()),
         us = n::select(n::full({3, 2}, .1, t), 1, 0).reshape({3, 1}),
         cs = n::select(n::tensor({0, 8, 1, 8, 0, 8}, {3, 2}, n::DType::Int64), 1, 0);
    ASSERT_FALSE(xs.is_contiguous());
    ASSERT_FALSE(cs.is_contiguous());
    auto a = Fields(m::InfoGan(g, d, q, x, z, c, u, 3, 2, 3)),
         b = Fields(m::InfoGan(g, d, q, xs, zs, cs, us, 3, 2, 3));
    for (std::size_t k = 0; k < a.size(); ++k)
      EXPECT_TRUE(n::equal_bits(a[k], b[k]));
    saved = a;
  }
  EXPECT_EQ(saved.size(), 15);
  for (const auto& a : saved)
    for (double v : a.to_vector<double>())
      EXPECT_TRUE(std::isfinite(v));
}
TEST(NnInfoGan, RejectsBadInputsGeometryStoresBeforeAnyLeaves) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(1, t), d(2, t), q(3, t);
  auto              x = n::zeros({2, 3}, t), z = n::zeros({3, 2}, t), u = n::zeros({3, 1}, t),
       c = n::zeros({3}, n::DType::Int32);
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 2}, t),
        n::zeros({3, 0}, t),
        n::zeros({3, 2}),
        n::full({3, 2}, NAN, t)})
    EXPECT_THROW(m::InfoGan(g, d, q, x, bad, c, u, 3, 2, 3), std::invalid_argument);
  for (auto bad :
       {n::zeros({2, 1}, t),
        n::zeros({3, 0}, t),
        n::zeros({3, 1}),
        n::full({3, 1}, 2., t),
        n::full({3, 1}, INFINITY, t)})
    EXPECT_THROW(m::InfoGan(g, d, q, x, z, c, bad, 3, 2, 3), std::invalid_argument);
  for (auto bad :
       {n::zeros({2}, n::DType::Int32),
        n::full({3}, -1., n::DType::Int64),
        n::full({3}, 2., n::DType::Int64),
        n::zeros({3}, t)})
    EXPECT_THROW(m::InfoGan(g, d, q, x, z, bad, u, 3, 2, 3), std::invalid_argument);
  EXPECT_THROW(m::InfoGan(g, g, q, x, z, c, u, 3, 2, 3), std::invalid_argument);
  EXPECT_THROW(m::InfoGan(g, d, g, x, z, c, u, 3, 2, 3), std::invalid_argument);
  EXPECT_THROW(m::InfoGan(g, d, d, x, z, c, u, 3, 2, 3), std::invalid_argument);
  EXPECT_THROW(m::InfoGan(g, d, q, x, z, c, u, 3, 0, 3), std::invalid_argument);
  EXPECT_THROW(m::InfoGan(g, d, q, x, z, c, u, 3, 2, 0), std::invalid_argument);
  EXPECT_THROW(
      m::InfoGan(g, d, q, x, z, c, u, 3, std::numeric_limits<std::int64_t>::max(), 3),
      std::invalid_argument);
  EXPECT_THROW(
      m::InfoGanRecognize(q, x, 3, 2, std::numeric_limits<std::int64_t>::max(), 3),
      std::invalid_argument);
  EXPECT_THROW(
      m::InfoGanGenerate(g, z, c, u, 3, std::numeric_limits<std::int64_t>::max(), 3),
      std::invalid_argument);
  EXPECT_EQ(g.size() + d.size() + q.size(), 0);
}
