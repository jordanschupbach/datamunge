#include <gtest/gtest.h>

#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/spectral_gan.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
std::vector<n::Tensor> Snapshot(const n::ParameterStore& p) {
  std::vector<n::Tensor> result;
  for (const auto& v : p.parameters())
    result.push_back(v.detach().clone());
  return result;
}
void Same(const n::ParameterStore& p, const std::vector<n::Tensor>& before) {
  ASSERT_EQ(p.size(), before.size());
  const auto values = p.parameters();
  for (std::size_t i = 0; i < before.size(); ++i)
    EXPECT_TRUE(n::equal_bits(values[i], before[i]));
}
} // namespace
TEST(NnSpectralGan, OrderedColumnPowerIterationWithoutInitialNormalization) {
  for (auto dtype : {n::DType::Float32, n::DType::Float64}) {
    auto w        = n::tensor({3., 0., 0., 2.}, {2, 2}, dtype).requires_grad_();
    auto u        = n::tensor({2., 1.}, {2, 1}, dtype).requires_grad_();
    auto pair     = n::cached_spectral_power_iteration(w, u, 1, 1.);
    auto v        = n::matmul(w, u);
    v             = v / n::sqrt(n::sum(n::square(v)) + 1.);
    auto expected = n::matmul(w, v, true);
    expected      = expected / n::sqrt(n::sum(n::square(expected)) + 1.);
    EXPECT_TRUE(n::equal_bits(pair.u, expected));
    EXPECT_TRUE(n::equal_bits(pair.v, v));
    EXPECT_FALSE(pair.u.requires_grad());
    EXPECT_FALSE(pair.v.requires_grad());
    auto two        = n::cached_spectral_power_iteration(w, u, 2, 1.);
    auto sequential = n::cached_spectral_power_iteration(w, pair.u, 1, 1.);
    EXPECT_TRUE(n::equal_bits(two.u, sequential.u));
    EXPECT_TRUE(n::equal_bits(two.v, sequential.v));
    auto zero = n::cached_spectral_power_iteration(n::zeros({2, 3}, dtype), n::ones({3, 1}, dtype));
    EXPECT_EQ(n::sum(zero.u).item(), 0);
    EXPECT_EQ(n::sum(zero.v).item(), 0);
  }
}
TEST(NnSpectralGan, CachedSigmaKeepsWeightDerivativeAndStopsBothColumns) {
  auto t      = n::DType::Float64;
  auto w      = n::tensor({2., 1., .5, 3.}, {2, 2}, t).requires_grad_();
  auto u      = n::tensor({.6, .8}, {2, 1}, t).requires_grad_();
  auto v      = n::tensor({.8, .6}, {2, 1}, t).requires_grad_();
  auto result = n::cached_spectral_kernel(w, u, v);
  EXPECT_NEAR(result.sigma.item(), 3.22, 1e-14);
  auto derivatives = n::grad(n::sum(result.weight), {w, u, v}, {.allow_unused = true});
  EXPECT_FALSE(derivatives[1].defined());
  EXPECT_FALSE(derivatives[2].defined());
  auto expected = n::ones({2, 2}, t) / result.sigma.item() -
                  n::matmul(v, u, false, true) * (6.5 / std::pow(result.sigma.item(), 2));
  EXPECT_LT(n::max(n::abs(expected - derivatives[0])).item(), 1e-14);
  n::TensorFunction f = [&](const auto& a) {
    return n::sum(n::square(n::cached_spectral_kernel(a[0], u, v).weight));
  };
  EXPECT_TRUE(n::gradcheck(f, {w}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(f, {w}, {.atol = 1e-8, .rtol = 1e-5}).ok);
}
TEST(NnSpectralGan, SignedSigmaFloorAndZeroKernelHaveFiniteCorrectDerivatives) {
  auto t      = n::DType::Float64;
  auto one    = n::ones({1, 1}, t);
  auto w      = n::full({1, 1}, -2., t).requires_grad_();
  auto result = n::cached_spectral_kernel(w, one, one, .5);
  EXPECT_EQ(result.sigma.item(), -2);
  EXPECT_EQ(result.weight.item(), -4);
  EXPECT_EQ(n::grad(n::sum(result.weight), {w})[0].item(), 2);
  auto z    = n::zeros({1, 1}, t).requires_grad_();
  auto zero = n::cached_spectral_kernel(z, one, one, .5);
  EXPECT_EQ(zero.sigma.item(), 0);
  EXPECT_EQ(zero.weight.item(), 0);
  EXPECT_EQ(n::grad(n::sum(zero.weight), {z})[0].item(), 2);
}
TEST(NnSpectralGan, StridedAndRankFourKernelsFlattenLeadingAxes) {
  auto t = n::DType::Float64;
  auto w = n::transpose(n::tensor({1., 2., 3., 4., 5., 6.}, {3, 2}, t));
  auto u = n::ones({3, 1}, t), v = n::ones({2, 1}, t);
  auto a = n::cached_spectral_kernel(w, u, v), b = n::cached_spectral_kernel(w.contiguous(), u, v);
  EXPECT_TRUE(n::equal_bits(a.weight, b.weight));
  auto shaped = w.contiguous().reshape({1, 1, 2, 3});
  auto c      = n::cached_spectral_kernel(shaped, u, v);
  EXPECT_EQ(c.weight.shape(), shaped.shape());
  EXPECT_TRUE(n::equal_bits(c.weight.reshape({2, 3}), b.weight));
  auto p = n::cached_spectral_power_iteration(shaped, u),
       q = n::cached_spectral_power_iteration(w, u);
  EXPECT_TRUE(n::equal_bits(p.u, q.u));
  EXPECT_TRUE(n::equal_bits(p.v, q.v));
}
TEST(NnSpectralGan, RejectsBadColumnsGuardsAndOverflowingArithmetic) {
  auto w = n::ones({2, 3}), u = n::ones({3, 1}), v = n::ones({2, 1});
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 3}),
        n::ones({3}),
        n::ones({2, 3}, n::DType::Int32),
        n::full({2, 3}, NAN)})
    EXPECT_THROW(n::cached_spectral_kernel(bad, u, v), std::invalid_argument);
  for (auto bad :
       {n::Tensor{},
        n::ones({1, 3}),
        n::ones({2, 1}),
        n::ones({3, 1}, n::DType::Float64),
        n::full({3, 1}, INFINITY)}) {
    EXPECT_THROW(n::cached_spectral_kernel(w, bad, v), std::invalid_argument);
    EXPECT_THROW(n::cached_spectral_power_iteration(w, bad), std::invalid_argument);
  }
  for (double bad : {0., -1., double(NAN), double(INFINITY), 1e-300, 1e300}) {
    EXPECT_THROW(n::cached_spectral_kernel(w, u, v, bad), std::invalid_argument);
    EXPECT_THROW(n::cached_spectral_power_iteration(w, u, 1, bad), std::invalid_argument);
  }
  EXPECT_THROW(n::cached_spectral_power_iteration(w, u, 0), std::invalid_argument);
  EXPECT_THROW(n::cached_spectral_kernel(w, u, n::ones({2})), std::invalid_argument);
  EXPECT_THROW(n::cached_spectral_power_iteration(n::full({2, 3}, 1e30), u), std::invalid_argument);
  EXPECT_THROW(
      n::cached_spectral_kernel(n::full({2, 3}, 1e30), n::full({3, 1}, 1e30), v),
      std::invalid_argument);
}
TEST(NnSpectralGan, StateAdvancePrevalidatesAllPairsBeforeAnyWrites) {
  n::ParameterStore d(2), s(0);
  d.get("a/W", {2, 3}, n::init::ones());
  d.get("b/W", {3, 1}, n::init::ones());
  s.get("a/u", {3, 1}, n::init::ones());
  s.get("a/v", {2, 1}, n::init::ones());
  s.get("b/u", {1, 1}, n::init::ones());
  auto initial = Snapshot(s), weights = Snapshot(d);
  EXPECT_THROW(n::advance_cached_spectral_state(d, s), std::out_of_range);
  Same(s, initial);
  Same(d, weights);
  s.get("b/v", {3, 1}, n::init::constant(NAN));
  initial = Snapshot(s);
  EXPECT_THROW(n::advance_cached_spectral_state(d, s), std::invalid_argument);
  Same(s, initial);
  Same(d, weights);
  s.set("b/v", n::ones({3, 1}));
  auto handle = s.at("a/u");
  n::advance_cached_spectral_state(d, s);
  Same(d, weights);
  EXPECT_TRUE(n::equal_bits(handle, s.at("a/u")));
  EXPECT_LT(handle.at({0, 0}), 1);
  n::ParameterStore empty(0), other(1), wide(0, n::DType::Float64);
  n::advance_cached_spectral_state(empty, other);
  EXPECT_EQ(other.size(), 0);
  EXPECT_THROW(n::advance_cached_spectral_state(d, d), std::invalid_argument);
  EXPECT_THROW(n::advance_cached_spectral_state(d, wide), std::invalid_argument);
  EXPECT_THROW(n::advance_cached_spectral_state(empty, other, 0), std::invalid_argument);
}
TEST(NnSpectralGan, DefaultCensusConstantsSeedRepeatAndImmutableInference) {
  n::ParameterStore g(5801), d(5807), s(0), g2(5801), d2(5807), s2(0);
  auto              x = n::zeros({2, 784}), z = n::zeros({3, 16});
  auto              out = m::SpectralGan(g, d, s, x, z), again = m::SpectralGan(g2, d2, s2, x, z);
  EXPECT_EQ(g.count(), 26416);
  EXPECT_EQ(d.count(), 25153);
  EXPECT_EQ(s.count(), 849);
  EXPECT_EQ(g.size() + d.size() + s.size(), 12);
  EXPECT_EQ(out.fake.shape(), n::Shape({3, 784}));
  EXPECT_EQ(out.real_score.shape(), n::Shape({2, 1}));
  EXPECT_TRUE(n::equal_bits(out.fake, again.fake));
  Same(d2, Snapshot(d));
  Same(s2, Snapshot(s));
  EXPECT_EQ(s.at("hidden/u").at({0, 0}), double(1.f / std::sqrt(32.f)));
  EXPECT_EQ(s.at("hidden/v").at({0, 0}), double(1.f / std::sqrt(784.f)));
  auto state = Snapshot(s), weights = Snapshot(d);
  m::SpectralGan(g, d, s, x, z);
  Same(s, state);
  Same(d, weights);
  n::advance_cached_spectral_state(d, s, 16);
  state = Snapshot(s);
  m::SpectralGanDiscriminate(d, s, x);
  m::SpectralGan(g, d, s, x, z);
  Same(s, state);
}
TEST(NnSpectralGan, BothInputAndWeightPathsSupportFirstAndHigherDerivatives) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(3, t), d(4, t), s(0, t);
  auto              x = n::tensor({.1, .4, -.2, .5, .2, .7}, {2, 3}, t).requires_grad_();
  auto              z = n::tensor({-.3, .2, .6, -.1, .4, .8}, {3, 2}, t).requires_grad_();
  m::SpectralGan(g, d, s, x, z, 3, 2);
  n::advance_cached_spectral_state(d, s, 16);
  n::TensorFunction fn = [&](const auto& a) {
    auto o = m::SpectralGan(g, d, s, a[0], a[1], 3, 2);
    return o.discriminator_loss + o.generator_loss;
  };
  EXPECT_TRUE(n::gradcheck(fn, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  EXPECT_TRUE(n::gradgradcheck(fn, {x, z}, {.atol = 1e-8, .rtol = 1e-5}).ok);
  auto out     = m::SpectralGan(g, d, s, x, z, 3, 2);
  auto stopped = n::grad(out.generator_loss, s.parameters(), {.allow_unused = true});
  for (const auto& v : stopped)
    EXPECT_FALSE(v.defined());
  for (int group : {0, 1}) {
    auto& p        = group ? d : g;
    auto  base     = p.flatten();
    auto  o        = m::SpectralGan(g, d, s, x, z, 3, 2);
    auto  partials = n::grad(group ? o.discriminator_loss : o.generator_loss, p.parameters());
    std::vector<double> analytic;
    for (const auto& v : partials) {
      auto a = v.to_vector<double>();
      analytic.insert(analytic.end(), a.begin(), a.end());
    }
    for (std::size_t i = 0; i < base.size(); ++i) {
      auto values = base;
      values[i] += 1e-6;
      p.unflatten(values);
      auto plus = m::SpectralGan(g, d, s, x, z, 3, 2);
      values[i] -= 2e-6;
      p.unflatten(values);
      auto   minus  = m::SpectralGan(g, d, s, x, z, 3, 2);
      double finite = ((group ? plus.discriminator_loss : plus.generator_loss).item() -
                       (group ? minus.discriminator_loss : minus.generator_loss).item()) /
                      2e-6;
      EXPECT_NEAR(finite, analytic[i], 1e-8);
    }
    p.unflatten(base);
  }
}
TEST(NnSpectralGan, ExplicitStateThenDThenFreshGOwnOnlyTheirStores) {
  n::ParameterStore g(3), d(4), s(0);
  auto              x = n::full({2, 3}, .3), z = n::full({3, 2}, -.2);
  m::SpectralGan(g, d, s, x, z, 3, 2);
  n::advance_cached_spectral_state(d, s, 16);
  n::OptimizerConfig config;
  config.kind  = n::OptimizerKind::Adam;
  config.beta1 = .5;
  n::Optimizer go(g, .0002, config), disc(d, .0002, config);
  auto         state = Snapshot(s), gb = Snapshot(g), db = Snapshot(d);
  n::advance_cached_spectral_state(d, s);
  Same(g, gb);
  Same(d, db);
  state       = Snapshot(s);
  auto before = m::SpectralGan(g, d, s, x, z, 3, 2);
  disc.step(n::grad(before.discriminator_loss, d.parameters()));
  Same(g, gb);
  Same(s, state);
  auto fresh = m::SpectralGan(g, d, s, x, z, 3, 2);
  EXPECT_FALSE(n::equal_bits(before.fake_score, fresh.fake_score));
  db = Snapshot(d);
  go.step(n::grad(fresh.generator_loss, g.parameters()));
  Same(d, db);
  Same(s, state);
  EXPECT_EQ(go.iterations(), 1);
  EXPECT_EQ(disc.iterations(), 1);
}
TEST(NnSpectralGan, InvalidInputsAndGeometryFailBeforeLeafCreation) {
  n::ParameterStore g(1), d(2), s(0), wide(0, n::DType::Float64);
  auto              x = n::zeros({2, 3}), z = n::zeros({3, 2});
  EXPECT_THROW(m::SpectralGan(g, g, s, x, z, 3, 2), std::invalid_argument);
  EXPECT_THROW(m::SpectralGan(g, d, g, x, z, 3, 2), std::invalid_argument);
  EXPECT_THROW(m::SpectralGan(g, d, wide, x, z, 3, 2), std::invalid_argument);
  for (auto bad :
       {n::Tensor{},
        n::zeros({0, 3}),
        n::zeros({2, 4}),
        n::zeros({2, 3}, n::DType::Float64),
        n::full({2, 3}, NAN)})
    EXPECT_THROW(m::SpectralGan(g, d, s, bad, z, 3, 2), std::invalid_argument);
  for (auto bad :
       {n::Tensor{}, n::zeros({0, 2}), n::zeros({2, 0}), n::zeros({2}), n::full({2, 2}, INFINITY)})
    EXPECT_THROW(m::SpectralGan(g, d, s, x, bad, 3, 2), std::invalid_argument);
  EXPECT_THROW(m::SpectralGan(g, d, s, x, z, 0, 2), std::invalid_argument);
  EXPECT_THROW(
      m::SpectralGan(g, d, s, x, z, 3, std::numeric_limits<std::int64_t>::max()),
      std::invalid_argument);
  EXPECT_THROW(m::SpectralGanDiscriminate(d, s, x, 3, 0), std::invalid_argument);
  EXPECT_EQ(g.size() + d.size() + s.size(), 0);
}
