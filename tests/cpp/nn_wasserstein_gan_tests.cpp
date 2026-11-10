#include <gtest/gtest.h>

#include <datamunge/nn/gan_priors.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/midpoint_uniform.hpp>
#include <datamunge/nn/optim.hpp>
#include <datamunge/nn/wasserstein_gan.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
std::vector<n::Tensor> Snapshot(const n::ParameterStore& p) {
  std::vector<n::Tensor> a;
  for (auto& v : p.parameters())
    a.push_back(v.detach().clone());
  return a;
}
void Same(const n::ParameterStore& p, const std::vector<n::Tensor>& a) {
  auto values = p.parameters();
  ASSERT_EQ(values.size(), a.size());
  for (std::size_t k = 0; k < a.size(); ++k)
    EXPECT_TRUE(n::equal_bits(values[k], a[k]));
}
} // namespace
TEST(NnWassersteinGan, MidpointUniformUsesLocalRawWordsOneFinalCastAndRejectsInvalidProducts) {
  auto wide   = n::MidpointUniform(3, 1, 6601, n::DType::Float64),
       single = n::MidpointUniform(3, 1, 6601);
  EXPECT_TRUE(n::equal_bits(single, n::cast(wide, n::DType::Float32)));
  EXPECT_EQ(wide.to_vector<double>(), (std::vector<double>{0.14499446202535182,0.4169559587026015,0.9257562280399725}));
  EXPECT_TRUE(n::equal_bits(wide, n::MidpointUniform(3, 1, 6601, n::DType::Float64)));
  EXPECT_FALSE(wide.requires_grad());
  EXPECT_GT(n::min(wide).item(), 0);
  EXPECT_LT(n::max(wide).item(), 1);
  EXPECT_THROW(n::MidpointUniform(0), std::invalid_argument);
  EXPECT_THROW(n::MidpointUniform(2, 0), std::invalid_argument);
  EXPECT_THROW(
      n::MidpointUniform(std::numeric_limits<std::int64_t>::max(), 2), std::invalid_argument);
  EXPECT_THROW(n::MidpointUniform(2, 1, 1, n::DType::Int64), std::invalid_argument);
}
TEST(NnWassersteinGan, MapsPreserveSourceCensusNamesAndIndependentOwningOutput) {
  for (auto t : {n::DType::Float32, n::DType::Float64}) {
    n::ParameterStore g(5501, t), c(5503, t);
    auto              x = n::tensor({.2, .7, -.3, .8}, {2, 2}, t),
         z = m::GaussianRejectionPrior(2, 2, 5601, 2, t).z, a = n::MidpointUniform(2, 1, 6601, t);
    auto out = m::WassersteinGan(g, c, x, z, a);
    EXPECT_EQ(g.count(), 82);
    EXPECT_EQ(c.count(), 65);
    EXPECT_EQ(g.size(), 4);
    EXPECT_EQ(c.size(), 4);
    EXPECT_EQ(g.names().front(), "hidden/W");
    EXPECT_EQ(c.names().back(), "output/b");
    EXPECT_EQ(out.fake.shape(), n::Shape({2, 2}));
    EXPECT_EQ(out.norm.shape(), n::Shape({2}));
    EXPECT_TRUE(n::equal_bits(out.fake, m::WassersteinGenerate(g, z)));
    EXPECT_TRUE(n::equal_bits(out.real_score, m::WassersteinCritic(c, x)));
  }
}
TEST(NnWassersteinGan, AnalyticAndSymbolicInputJacobiansAgreeAndPenaltyHasCriticHistory) {
  n::ParameterStore g(5501, n::DType::Float64), c(5503, n::DType::Float64);
  auto              x = n::tensor({.2, .7, -.3, .8}, {2, 2}, n::DType::Float64),
       z = n::ones({2, 2}, n::DType::Float64), a = n::full({2, 1}, .5, n::DType::Float64);
  auto out = m::WassersteinGan(g, c, x, z, a);
  EXPECT_LT(n::max(n::abs(out.input_gradient - out.symbolic_input_gradient)).item(), 1e-14);
  EXPECT_FALSE(out.interpolation.requires_grad());
  EXPECT_TRUE(out.symbolic_input_gradient.requires_grad());
  EXPECT_TRUE(out.penalty.requires_grad());
  auto gradient =
      n::grad(out.penalty, c.parameters(), {.create_graph = true, .allow_unused = true});
  EXPECT_TRUE(gradient[0].requires_grad());
  EXPECT_FALSE(gradient[3].defined());
  auto higher = n::grad(n::sum(n::square(gradient[0])), c.parameters(), {.allow_unused = true});
  EXPECT_TRUE(higher[0].defined());
  for (auto& v : higher)
    if (v.defined())
      for (double value : v.to_vector<double>())
        EXPECT_TRUE(std::isfinite(value));
}
TEST(NnWassersteinGan, PenaltyStopsExternalRealAlphaNoiseAndGeneratorWhileBaseLossKeepsG) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(5501, t), c(5503, t);
  auto              x = n::tensor({.2, .7, -.3, .8}, {2, 2}, t).requires_grad_(),
       z = n::ones({2, 2}, t).requires_grad_(), a = n::full({2, 1}, .5, t).requires_grad_();
  auto out     = m::WassersteinGan(g, c, x, z, a);
  auto stopped = n::grad(out.penalty, {x, z, a, g.at("hidden/W")}, {.allow_unused = true});
  for (auto& v : stopped)
    EXPECT_FALSE(v.defined());
  out       = m::WassersteinGan(g, c, x, z, a);
  auto base = n::grad(out.critic_loss, {g.at("hidden/W")});
  EXPECT_GT(n::max(n::abs(base[0])).item(), 1e-8);
}
TEST(NnWassersteinGan, NoGradEvaluationComputesActualInputGradientWithoutRestoringTrainingMode) {
  n::ParameterStore g(5501), c(5503);
  {
    n::NoGradGuard guard;
    auto           out =
        m::WassersteinGan(g, c, n::ones({2, 2}), n::ones({2, 2}), n::full({2, 1}, .5), 10, false);
    EXPECT_FALSE(n::is_grad_enabled());
    for (auto& v :
         {out.fake,
          out.real_score,
          out.fake_score,
          out.input_gradient,
          out.symbolic_input_gradient,
          out.norm,
          out.penalty,
          out.critic_loss})
      EXPECT_FALSE(v.requires_grad());
    EXPECT_TRUE(n::equal_bits(out.input_gradient, out.symbolic_input_gradient));
  }
  EXPECT_TRUE(n::is_grad_enabled());
}
TEST(NnWassersteinGan, ZeroPenaltyWeightSkipsAdditionButStillReportsFullDiagnostics) {
  n::ParameterStore g(5501), c(5503);
  auto              x = n::ones({2, 2}), z = n::zeros({2, 2}), a = n::full({2, 1}, .5);
  auto              base = m::WassersteinGan(g, c, x, z, a, 0);
  auto              gp   = m::WassersteinGan(g, c, x, z, a, 10);
  EXPECT_TRUE(n::equal_bits(base.critic_loss, -base.critic_gap));
  EXPECT_TRUE(n::equal_bits(base.penalty, gp.penalty));
  EXPECT_TRUE(n::equal_bits(base.generator_loss, gp.generator_loss));
  EXPECT_TRUE(n::equal_bits(gp.critic_loss, -gp.critic_gap + 10. * gp.penalty));
}
TEST(NnWassersteinGan, ClipIncludesAllBiasesPreservesFlagsAndValidatesBeforeAnyWrite) {
  n::ParameterStore c(5503);
  m::WassersteinCritic(c, n::zeros({1, 2}));
  for (auto& name : c.names())
    c.set(name, n::full(c.at(name).shape(), .2));
  auto names = c.names();
  m::ClipWassersteinCritic(c);
  EXPECT_EQ(c.names(), names);
  for (auto& v : c.parameters()) {
    EXPECT_EQ(n::max(n::abs(v)).item(), double(.01f));
    EXPECT_TRUE(v.requires_grad());
  }
  auto before = Snapshot(c);
  EXPECT_THROW(m::ClipWassersteinCritic(c, -1), std::invalid_argument);
  Same(c, before);
  c.set(c.names().back(), n::full({1}, std::numeric_limits<double>::quiet_NaN()));
  auto first = c.at(c.names().front()).detach().clone();
  EXPECT_THROW(m::ClipWassersteinCritic(c), std::invalid_argument);
  EXPECT_TRUE(n::equal_bits(first, c.at(c.names().front())));
  n::ParameterStore empty;
  EXPECT_THROW(m::ClipWassersteinCritic(empty), std::invalid_argument);
}
TEST(NnWassersteinGan, ZeroCriticHasFiniteGuardedNormPenaltyAndParameterPartials) {
  n::ParameterStore g(5501), c(5503);
  m::WassersteinGan(g, c, n::ones({2, 2}), n::ones({2, 2}), n::full({2, 1}, .5));
  for (auto& name : c.names())
    c.set(name, n::zeros(c.at(name).shape()));
  auto out = m::WassersteinGan(g, c, n::ones({2, 2}), n::ones({2, 2}), n::full({2, 1}, .5));
  EXPECT_NEAR(out.norm.at({0}), 1e-6, 1e-13);
  auto gradient = n::grad(out.critic_loss, c.parameters());
  for (auto& v : gradient)
    for (double x : v.to_vector<double>())
      EXPECT_TRUE(std::isfinite(x));
}
TEST(NnWassersteinGan, NoiseAndCriticInputPathsHaveCheckedFirstAndSecondDerivatives) {
  auto              t = n::DType::Float64;
  n::ParameterStore g(5501, t), c(5503, t);
  auto              x      = n::tensor({.2, -.3}, {1, 2}, t).requires_grad_();
  n::TensorFunction critic = [&](const auto& a) {
    return n::sum(n::square(m::WassersteinCritic(c, a[0])));
  };
  EXPECT_TRUE(n::gradcheck(critic, {x}).ok);
  EXPECT_TRUE(n::gradgradcheck(critic, {x}).ok);
  n::TensorFunction gen = [&](const auto& a) {
    return -n::mean(m::WassersteinCritic(c, m::WassersteinGenerate(g, a[0])));
  };
  EXPECT_TRUE(n::gradcheck(gen, {x}).ok);
  EXPECT_TRUE(n::gradgradcheck(gen, {x}).ok);
}
TEST(NnWassersteinGan, AllBadInputChecksPrecedeLeafCreationAndReadOnlyCallsPreserveStores) {
  n::ParameterStore g(5501), c(5503), other(0, n::DType::Float64);
  auto              x = n::ones({2, 2}), z = n::ones({2, 2}), a = n::full({2, 1}, .5);
  EXPECT_THROW(m::WassersteinGan(g, g, x, z, a), std::invalid_argument);
  EXPECT_THROW(m::WassersteinGan(g, other, x, z, a), std::invalid_argument);
  EXPECT_THROW(m::WassersteinGan(g, c, x, n::ones({1, 2}), a), std::invalid_argument);
  EXPECT_THROW(m::WassersteinGan(g, c, x, z, n::full({2, 1}, 1.1)), std::invalid_argument);
  EXPECT_THROW(m::WassersteinGan(g, c, x, z, a, -1), std::invalid_argument);
  EXPECT_THROW(m::WassersteinGenerate(g, n::ones({2, 3})), std::invalid_argument);
  EXPECT_EQ(g.size() + c.size(), 0);
  m::WassersteinGan(g, c, x, z, a);
  auto before_g = Snapshot(g), before_c = Snapshot(c);
  m::WassersteinGan(g, c, x, z, a, 10, false);
  Same(g, before_g);
  Same(c, before_c);
}
