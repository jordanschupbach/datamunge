#include <gtest/gtest.h>

#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/auxiliary_classifier_gan.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/optim.hpp>
#include <limits>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
std::vector<n::Tensor> Fields(const m::AuxiliaryClassifierGanOutput& o) {
  return {
      o.fake,
      o.real_logit,
      o.fake_logit,
      o.real_probability,
      o.fake_probability,
      o.discriminator_loss,
      o.generator_loss,
      o.real_class_logits,
      o.fake_class_logits,
      o.real_class_probability,
      o.fake_class_probability,
      o.real_class_loss,
      o.fake_class_loss,
      o.source_discriminator_loss,
      o.source_generator_loss};
}
} // namespace
TEST(NnAuxiliaryGan, StableExtremeLogisticLossesPreserveSmallAndStrongDerivatives) {
  for (auto type : {n::DType::Float32, n::DType::Float64}) {
    auto real = n::full({2, 1}, 80., type).requires_grad_(),
         fake = n::full({3, 1}, -80., type).requires_grad_();
    auto o    = m::LogisticGanObjectives(real, fake);
    EXPECT_NEAR(o.generator.item(), 80., 1e-6);
    auto d = n::grad(o.discriminator, {real, fake});
    for (double v : d[0].to_vector<double>()) {
      EXPECT_LT(v, 0);
      EXPECT_NEAR(v / (-.5 * std::exp(-80.)), 1., 1e-6);
    }
    for (double v : d[1].to_vector<double>()) {
      EXPECT_GT(v, 0);
      EXPECT_NEAR(v / (std::exp(-80.) / 3), 1., 1e-6);
    }
    auto g = n::grad(o.generator, {real, fake}, {.allow_unused = true});
    EXPECT_FALSE(g[0].defined());
    for (double v : g[1].to_vector<double>())
      EXPECT_NEAR(v, -1. / 3, 1e-7);
  }
  const auto t         = n::DType::Float64;
  auto       r         = n::tensor({-.8, .2}, {2, 1}, t).requires_grad_(),
       f               = n::tensor({.4, -.3, .1}, {3, 1}, t).requires_grad_();
  n::TensorFunction fn = [](const auto& a) {
    auto o = m::LogisticGanObjectives(a[0], a[1]);
    return o.discriminator + o.generator;
  };
  EXPECT_TRUE(n::gradcheck(fn, {r, f}, {.atol = 1e-9, .rtol = 1e-6}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {r, f}, {.atol = 1e-9, .rtol = 1e-6}).ok);
  EXPECT_THROW(m::LogisticGanObjectives(r, n::zeros({3, 2}, t)), std::invalid_argument);
  EXPECT_THROW(m::LogisticGanObjectives(r, n::zeros({3, 1})), std::invalid_argument);
  EXPECT_THROW(m::LogisticGanObjectives(n::zeros({0, 1}, t), f), std::invalid_argument);
  EXPECT_THROW(m::LogisticGanObjectives(r, n::full({3, 1}, INFINITY, t)), std::invalid_argument);
}
TEST(NnAuxiliaryGan, DefaultGeometryAndClassBeforeSourceInitialization) {
  n::ParameterStore g(5209), d(5219), reference(5219);
  const auto        o = m::AuxiliaryClassifierGan(
      g, d, n::zeros({2, 784}), n::zeros({2, 16}), n::zeros({2}, n::DType::Int32));
  EXPECT_EQ(g.count(), 26736);
  EXPECT_EQ(d.count(), 25483);
  EXPECT_EQ(g.size(), 4u);
  EXPECT_EQ(d.size(), 6u);
  EXPECT_EQ(g.names(), (std::vector<std::string>{"hidden/W", "hidden/b", "output/W", "output/b"}));
  EXPECT_EQ(
      d.names(),
      (std::vector<std::string>{
          "hidden/W", "hidden/b", "class/W", "class/b", "source/W", "source/b"}));
  auto h = n::dense(reference, "hidden", n::zeros({2, 784}), 32, n::Activation::Tanh);
  n::dense(reference, "class", h, 10);
  n::dense(reference, "source", h, 1);
  EXPECT_EQ(reference.flatten(), d.flatten());
  EXPECT_EQ(o.fake.shape(), (n::Shape{2, 784}));
  EXPECT_EQ(o.real_class_logits.shape(), (n::Shape{2, 10}));
  EXPECT_EQ(o.fake_logit.shape(), (n::Shape{2, 1}));
  for (double v : n::sum(o.fake_class_probability, {1}).to_vector<double>())
    EXPECT_NEAR(v, 1, 2e-7);
}
TEST(NnAuxiliaryGan, ZeroModelAlgebraAndClassTargetsUseSourceGroupedCrossEntropy) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(1, t), d(2, t);
  const auto        x = n::zeros({2, 3}, t), z = n::zeros({2, 2}, t),
             labels = n::tensor({0., 1.}, {2}, n::DType::Int64);
  m::AuxiliaryClassifierGan(g, d, x, z, labels, 3, 2, 2);
  for (auto p : {&g, &d})
    p->unflatten(std::vector<double>(p->count(), 0));
  const auto o = m::AuxiliaryClassifierGan(g, d, x, z, labels, 3, 2, 2);
  EXPECT_EQ(o.fake.to_vector<double>(), std::vector<double>(6, .5));
  EXPECT_NEAR(o.source_discriminator_loss.item(), 2 * std::log(2.), 1e-15);
  EXPECT_NEAR(o.source_generator_loss.item(), std::log(2.), 1e-15);
  EXPECT_NEAR(o.discriminator_loss.item(), 4 * std::log(2.), 1e-15);
  EXPECT_NEAR(o.generator_loss.item(), 2 * std::log(2.), 1e-15);
  const auto logits = n::tensor({80., -80., -80., 80.}, {2, 2}, t).requires_grad_();
  auto       ce     = m::AuxiliaryClassLoss(logits, labels);
  const auto direct =
      n::mean(-n::sum(n::cast(n::one_hot(labels, 2), t) * n::log_softmax(logits), {1}));
  EXPECT_TRUE(n::equal_bits(ce, direct));
  auto grads = n::grad(ce, {logits});
  EXPECT_EQ(grads[0].shape(), logits.shape());
}
TEST(NnAuxiliaryGan, ClassConditionsAffectGeneratorAndTargetsButNeverEnterRealDiscriminator) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(5209, t), d(5219, t);
  auto              x = n::tensor({-.1, .3, .7, .4, .2, -.3}, {2, 3}, t),
       z              = n::tensor({.4, -.2, .1, .8}, {2, 2}, t);
  auto labels         = n::tensor({0., 1.}, {2}, n::DType::Int32),
       changed        = n::tensor({1., 0.}, {2}, n::DType::Int64);
  const auto a        = m::AuxiliaryClassifierGan(g, d, x, z, labels, 3, 2, 3),
             b        = m::AuxiliaryClassifierGan(g, d, x, z, changed, 3, 2, 3);
  EXPECT_TRUE(n::equal_bits(a.real_logit, b.real_logit));
  EXPECT_TRUE(n::equal_bits(a.real_class_logits, b.real_class_logits));
  EXPECT_TRUE(n::equal_bits(a.real_class_probability, b.real_class_probability));
  EXPECT_GT(n::max(n::abs(a.fake - b.fake)).item(), 1e-6);
  EXPECT_NE(a.real_class_loss.item(), b.real_class_loss.item());
  EXPECT_TRUE(n::equal_bits(a.fake, m::ConditionalGanGenerate(g, z, labels, 3, 2, 3)));
  auto standalone = m::AuxiliaryGanDiscriminate(d, x, 3, 2, 3);
  EXPECT_TRUE(n::equal_bits(standalone.source, a.real_logit));
  EXPECT_TRUE(n::equal_bits(standalone.class_probability, a.real_class_probability));
  const auto total  = n::grad(a.generator_loss, g.parameters(), {.retain_graph = true}),
             source = n::grad(a.source_generator_loss, g.parameters(), {.retain_graph = true}),
             aux    = n::grad(a.fake_class_loss, g.parameters());
  for (std::size_t k = 0; k < total.size(); ++k)
    EXPECT_LT(n::max(n::abs(total[k] - source[k] - aux[k])).item(), 1e-14);
}
TEST(NnAuxiliaryGan, ExplicitDThenRecomputedGAdamStatesWriteOnlyTheirOwnStore) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(5209, t), d(5219, t);
  auto              x = n::full({2, 3}, .3, t), z = n::full({2, 2}, -.2, t),
       labels        = n::tensor({0., 1.}, {2}, n::DType::Int32);
  auto       forward = [&]() { return m::AuxiliaryClassifierGan(g, d, x, z, labels, 3, 2, 3); };
  auto       initial = forward();
  const auto old_g   = n::grad(initial.generator_loss, g.parameters(), {.retain_graph = true});
  n::OptimizerConfig cfg;
  cfg.kind  = n::OptimizerKind::Adam;
  cfg.beta1 = .5;
  n::Optimizer go(g, .0002, cfg), disc(d, .0002, cfg);
  auto         g0 = g.flatten(), d0 = d.flatten();
  disc.step(n::grad(initial.discriminator_loss, d.parameters()));
  EXPECT_EQ(g.flatten(), g0);
  EXPECT_NE(d.flatten(), d0);
  const auto fresh  = forward();
  const auto new_g  = n::grad(fresh.generator_loss, g.parameters());
  double     change = 0;
  for (std::size_t k = 0; k < new_g.size(); ++k)
    change += n::sum(n::abs(new_g[k] - old_g[k])).item();
  EXPECT_GT(change, 1e-8);
  d0 = d.flatten();
  go.step(new_g);
  EXPECT_EQ(d.flatten(), d0);
  EXPECT_NE(g.flatten(), g0);
  EXPECT_EQ(go.iterations(), 1);
  EXPECT_EQ(disc.iterations(), 1);
  // Forward itself also keeps the mathematically valid cross-group paths.
  auto o = forward();
  EXPECT_EQ(n::grad(o.generator_loss, d.parameters(), {.retain_graph = true}).size(), d.size());
  EXPECT_EQ(n::grad(o.discriminator_loss, g.parameters()).size(), g.size());
}
TEST(NnAuxiliaryGan, EverySmallParameterDerivativeAndContinuousNoiseHigherPaths) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(5209, t), d(5219, t);
  auto              x = n::tensor({-.1, .3, .7, .4, .2, -.3}, {2, 3}, t).requires_grad_(),
       z              = n::tensor({.4, -.2, .1, .8}, {2, 2}, t).requires_grad_();
  auto labels         = n::tensor({0., 1.}, {2}, n::DType::Int32);
  auto forward        = [&]() { return m::AuxiliaryClassifierGan(g, d, x, z, labels, 3, 2, 2); };
  forward();
  for (int group = 0; group < 2; ++group) {
    auto& p         = group == 0 ? g : d;
    auto  objective = [&]() {
      auto o = forward();
      return group == 0 ? o.generator_loss : o.discriminator_loss;
    };
    std::vector<double> analytic;
    for (const auto& v : n::grad(objective(), p.parameters())) {
      auto a = v.to_vector<double>();
      analytic.insert(analytic.end(), a.begin(), a.end());
    }
    const auto initial = p.flatten();
    for (std::size_t k = 0; k < initial.size(); ++k) {
      auto a = initial;
      a[k] += 1e-6;
      p.unflatten(a);
      auto plus = objective().item();
      a[k] -= 2e-6;
      p.unflatten(a);
      auto minus = objective().item();
      EXPECT_NEAR((plus - minus) / 2e-6, analytic[k], 1e-8);
    }
    p.unflatten(initial);
  }
  n::TensorFunction fn = [&](const auto& a) {
    auto o = m::AuxiliaryClassifierGan(g, d, a[0], a[1], labels, 3, 2, 2);
    return o.generator_loss + o.discriminator_loss;
  };
  EXPECT_TRUE(n::gradcheck(fn, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnAuxiliaryGan, StridesAndOwningFifteenFieldsOutliveBothStores) {
  const auto             t = n::DType::Float64;
  std::vector<n::Tensor> saved;
  {
    n::ParameterStore g(5209, t), d(5219, t);
    auto              x = n::tensor({-.1, .3, .7, .4, .2, -.3}, {2, 3}, t),
         z              = n::tensor({.4, -.2, .1, .8}, {2, 2}, t),
         labels         = n::tensor({0., 1.}, {2}, n::DType::Int32);
    const auto xs       = n::transpose(n::transpose(x).contiguous()),
               zs       = n::transpose(n::transpose(z).contiguous());
    const auto ls       = n::select(n::tensor({0., 8., 1., 8.}, {2, 2}, n::DType::Int32), 1, 0);
    ASSERT_FALSE(xs.is_contiguous());
    ASSERT_FALSE(zs.is_contiguous());
    ASSERT_FALSE(ls.is_contiguous());
    saved         = Fields(m::AuxiliaryClassifierGan(g, d, x, z, labels, 3, 2, 2));
    auto       g0 = g.flatten(), d0 = d.flatten();
    const auto other = Fields(m::AuxiliaryClassifierGan(g, d, xs, zs, ls, 3, 2, 2));
    for (std::size_t i = 0; i < saved.size(); ++i)
      EXPECT_TRUE(n::equal_bits(saved[i], other[i]));
    EXPECT_EQ(g.flatten(), g0);
    EXPECT_EQ(d.flatten(), d0);
  }
  ASSERT_EQ(saved.size(), 15u);
  for (const auto& v : saved)
    for (double x : v.to_vector<double>())
      EXPECT_TRUE(std::isfinite(x));
}
TEST(NnAuxiliaryGan, InvalidInputsLabelsProductsAndAliasedStoresFailBeforeLeaves) {
  const auto        t = n::DType::Float64;
  n::ParameterStore g(1, t), d(2, t), wrong(3);
  const auto        x = n::zeros({2, 3}, t), z = n::zeros({2, 2}, t),
             labels = n::zeros({2}, n::DType::Int32);
  for (auto invalid :
       {n::Tensor{},
        n::zeros({0, 3}, t),
        n::zeros({2, 2}, t),
        n::zeros({2, 3}),
        n::full({2, 3}, NAN, t)})
    EXPECT_THROW(
        m::AuxiliaryClassifierGan(g, d, invalid, z, labels, 3, 2, 2), std::invalid_argument);
  for (auto invalid :
       {n::Tensor{},
        n::zeros({1, 2}, t),
        n::zeros({2, 0}, t),
        n::zeros({2, 2}),
        n::full({2, 2}, INFINITY, t)})
    EXPECT_THROW(
        m::AuxiliaryClassifierGan(g, d, x, invalid, labels, 3, 2, 2), std::invalid_argument);
  for (auto invalid :
       {n::Tensor{},
        n::zeros({2}),
        n::zeros({1}, n::DType::Int32),
        n::full({2}, -1, n::DType::Int64),
        n::full({2}, 2, n::DType::Int32)}) {
    EXPECT_THROW(m::AuxiliaryClassifierGan(g, d, x, z, invalid, 3, 2, 2), std::invalid_argument);
    EXPECT_THROW(m::ConditionalGanGenerate(g, z, invalid, 3, 2, 2), std::invalid_argument);
    EXPECT_THROW(m::AuxiliaryClassLoss(n::zeros({2, 2}, t), invalid), std::invalid_argument);
  }
  EXPECT_THROW(m::AuxiliaryClassifierGan(g, g, x, z, labels, 3, 2, 2), std::invalid_argument);
  EXPECT_THROW(m::AuxiliaryClassifierGan(g, wrong, x, z, labels, 3, 2, 2), std::invalid_argument);
  for (auto a : std::vector<std::vector<std::int64_t>>{
           {0, 2, 2},
           {3, 0, 2},
           {3, 2, 0},
           {3, 2, INT64_MAX},
           {INT64_MAX, 2, 2},
           {3, INT64_MAX, 2}})
    EXPECT_THROW(
        m::AuxiliaryClassifierGan(g, d, x, z, labels, a[0], a[1], a[2]), std::invalid_argument);
  EXPECT_EQ(g.size() + d.size() + wrong.size(), 0u);
}
TEST(NnAuxiliaryGan, CoarsePixelPoolScaleSignedStridesAndHigherPaths) {
  const auto t      = n::DType::Float64;
  auto       x      = n::ones({2, 784}, t).requires_grad_();
  auto       coarse = m::GanCoarsePixels(x);
  EXPECT_EQ(coarse.shape(), (n::Shape{2, 16}));
  for (double v : coarse.to_vector<double>())
    EXPECT_NEAR(v, .25, 1e-15);
  auto derivative = n::grad(n::sum(coarse), {x})[0];
  for (double v : derivative.to_vector<double>())
    EXPECT_DOUBLE_EQ(v, 1. / 196);
  auto strided = n::transpose(n::transpose(x).contiguous());
  EXPECT_TRUE(n::equal_bits(coarse, m::GanCoarsePixels(strided)));
  for (double v : m::GanCoarsePixels(n::full({1, 28, 28, 1}, -4, t)).to_vector<double>())
    EXPECT_NEAR(v, -1, 1e-15);
  n::TensorFunction fn = [](const auto& a) {
    return n::sum(n::square(m::GanCoarsePixels(n::ones({2, 784}, n::DType::Float64) * a[0])));
  };
  const auto scales = n::full({2, 1}, .7, t).requires_grad_();
  EXPECT_TRUE(n::gradcheck(fn, {scales}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {scales}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 784}, t),
        n::zeros({2, 783}, t),
        n::zeros({784}, t),
        n::zeros({2, 784}, n::DType::Int32),
        n::full({2, 784}, NAN, t)})
    EXPECT_THROW(m::GanCoarsePixels(bad), std::invalid_argument);
}
