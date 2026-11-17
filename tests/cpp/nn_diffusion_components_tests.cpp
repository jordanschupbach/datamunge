#include <gtest/gtest.h>

#include <datamunge/nn/diffusion_components.hpp>
#include <datamunge/nn/optim.hpp>
namespace n = datamunge::nn;
namespace m = n::models;
namespace {
n::Tensor I(std::initializer_list<std::int32_t> values) {
  return n::from_vector(std::vector<std::int32_t>(values), {std::int64_t(values.size())});
}
} // namespace
TEST(NnDiffusionComponents, ExplicitInitializationPreservesSourceLeavesAndWarmRandomStream) {
  n::ParameterStore p(11003), q(11003);
  m::InitializeDiffusionMlp(p, 72, 64);
  m::DdpmEpsilon(q, n::zeros({1, 8, 8, 1}), I({3}));
  EXPECT_EQ(p.count(), 5504);
  for (auto name : p.names())
    EXPECT_TRUE(n::equal_bits(p.at(name), q.at(name))) << name;
  auto before = p.flatten();
  m::InitializeDiffusionMlp(p, 72, 64);
  m::DiffusionMlp(p, n::zeros({2, 72}));
  EXPECT_EQ(before, p.flatten());
  EXPECT_TRUE(
      n::equal_bits(
          p.get("probe", {5}, n::init::normal(.1)), q.get("probe", {5}, n::init::normal(.1))));
}
TEST(NnDiffusionComponents, ContinuousTimeKeepsSourceRoundingAndLiveDerivatives) {
  auto   t          = n::from_vector(std::vector<double>{.5, 7.25}, {2}).requires_grad_();
  auto   output     = m::DiffusionContinuousTimeEmbedding(t, 15, n::DType::Float64);
  auto   normalized = n::reshape(n::cast(t, n::DType::Float32), {-1, 1}) / 15.;
  double pi         = std::acos(-1.);
  auto   phase =
      normalized *
      n::from_vector(
          std::vector<float>{float(pi), float(2 * pi), float(4 * pi), float(8 * pi)}, {1, 4});
  EXPECT_TRUE(
      n::equal_bits(
          output, n::cast(n::concat({n::sin(phase), n::cos(phase)}, 1), n::DType::Float64)));
  auto g = n::grad(n::sum(output), {t})[0];
  EXPECT_TRUE(g.defined());
  EXPECT_GT(n::max(n::abs(g)).item(), .01);
}
TEST(NnDiffusionComponents, DdimAdjacentEtaOneMatchesDdpmAndEtaZeroIgnoresInnovation) {
  auto schedule = m::CosineDiffusionSchedule();
  auto x        = n::from_vector(std::vector<float>{.1f, -.2f, .3f, -.4f}, {2, 2});
  auto t = I({4, 19}), previous = I({3, 18}), eps = n::full({2, 2}, .15), z = n::ones({2, 2});
  auto ddim = m::DdimReverse(x, t, previous, eps, z, n::scalar(1), schedule);
  auto ddpm = m::DdpmReverse(x, t, eps, z, schedule);
  EXPECT_LT(n::max(n::abs(ddim.value - ddpm.value)).item(), 2e-6);
  auto a = m::DdimReverse(x, t, previous, eps, z, n::scalar(0), schedule),
       b = m::DdimReverse(x, t, previous, eps, z * 99, n::scalar(0), schedule);
  EXPECT_TRUE(n::equal_bits(a.value, b.value));
  auto final = m::DdimReverse(x, I({0, 0}), I({-1, -1}), eps, z, n::scalar(1), schedule);
  EXPECT_TRUE(n::equal_bits(final.value, final.x0_clipped));
}
TEST(NnDiffusionComponents, NoiseConditionalScalesAndLangevinHaveHandComputedValues) {
  auto levels = I({0, 15});
  auto sigma  = m::NoiseConditionalSigma(levels).to_vector<float>();
  EXPECT_EQ(sigma[0], 1.f);
  EXPECT_EQ(sigma[1], .05f);
  auto x = n::zeros({2, 8, 8, 1}), score = n::ones(x.shape()), z = n::zeros(x.shape());
  auto out   = m::AnnealedLangevin(x, levels, score, z);
  auto alpha = out.alpha.to_vector<float>();
  EXPECT_NEAR(alpha[0], .008, 1e-9);
  EXPECT_NEAR(alpha[1], .00002, 1e-11);
  EXPECT_TRUE(n::equal_bits(out.value, out.mean));
  n::ParameterStore p(12003);
  m::InitializeDiffusionMlp(p, 72, 64);
  auto y = m::NoiseConditionalScore(p, x, levels);
  EXPECT_TRUE(n::equal_bits(y.score, -y.epsilon / m::NoiseConditionalSigma(levels)));
}
TEST(NnDiffusionComponents, GaussianScoreFieldAndZeroIntegratorStepMatchClosedForms) {
  auto x = n::full({2, 8, 8, 1}, .3), t = n::full({2}, .5);
  auto f = m::ScoreSdeField(nullptr, x, t);
  EXPECT_EQ(n::max(n::abs(f.score)).item(), 0);
  auto step = m::ScoreSdeStep(nullptr, x, t, n::scalar(0), n::ones(x.shape()));
  EXPECT_TRUE(n::equal_bits(step.em_value, x));
  EXPECT_TRUE(n::equal_bits(step.ode_value, x));
  EXPECT_EQ(step.ode_stage_states.size(), 4);
}
TEST(NnDiffusionComponents, ConditionalAndClassifierPredictorsHaveSourceParameterCensus) {
  n::ParameterStore conditional(14007), classifier(14003);
  m::CreateGuidance(conditional, false);
  m::CreateGuidance(classifier, true);
  EXPECT_EQ(conditional.count(), 5856);
  EXPECT_EQ(classifier.count(), 3722);
  auto x = n::zeros({2, 8, 8, 1}), time = I({0, 31});
  auto a = m::GuidancePredictor(conditional, x, time, I({0, 10}), false),
       b = m::GuidancePredictor(classifier, x, time, I({0, 9}), true);
  EXPECT_EQ(a.prediction.shape(), x.shape());
  EXPECT_EQ(b.prediction.shape(), (n::Shape{2, 10}));
  auto logits = n::zeros({2, 10});
  EXPECT_NEAR(
      m::ClassifierLogProbability(logits, I({0, 9})).to_vector<float>()[0], -std::log(10.), 1e-6);
  EXPECT_TRUE(n::equal_bits(m::ClassifierFreeEpsilon(a.prediction, x, n::scalar(1)), a.prediction));
  EXPECT_TRUE(n::equal_bits(m::ClassifierFreeEpsilon(a.prediction, x, n::scalar(0)), x));
}
TEST(NnDiffusionComponents, LatentPosteriorLeavesCleanEstimateUnclipped) {
  auto s   = m::CosineDiffusionSchedule();
  auto x   = n::full({1, 2}, 2.);
  auto out = m::LatentDiffusionReverse(x, I({0}), n::zeros({1, 2}), n::ones({1, 2}), s);
  EXPECT_GT(n::min(out.x0).item(), 2);
  EXPECT_EQ(out.variance.item(), 0);
  auto other = m::LatentDiffusionReverse(x, I({0}), n::zeros({1, 2}), n::full({1, 2}, 99), s);
  EXPECT_TRUE(n::equal_bits(out.value, other.value));
}
TEST(NnDiffusionComponents, ConsistencyBoundaryIsIdentityAndTeacherUsesEarlierTime) {
  n::ParameterStore p(19003);
  m::InitializeDiffusionMlp(p, 10, 2);
  auto x        = n::from_vector(std::vector<float>{.2f, -.3f, .7f, .9f}, {2, 2});
  auto boundary = n::full({2}, double(.01f));
  auto out      = m::ConsistencyMap(p, x, boundary);
  EXPECT_TRUE(n::equal_bits(out.value, x));
  EXPECT_EQ(n::max(n::abs(out.out)).item(), 0);
  auto time = n::ones({2}), previous = n::full({2}, .5);
  auto teacher  = m::ConsistencyEulerTeacher(x, time, previous);
  auto expected = x + n::reshape(previous - time, {-1, 1}) * m::ConsistencyGaussianDrift(x, time);
  EXPECT_TRUE(n::equal_bits(teacher, expected));
}
TEST(NnDiffusionComponents, FlowStagesExposeLiveInputJacobiansAndExplicitOptimizer) {
  n::ParameterStore p(20003);
  m::InitializeDiffusionMlp(p, 3, 2);
  EXPECT_EQ(p.count(), 1250);
  auto x = n::full({2, 2}, .2).requires_grad_(), t = n::full({2}, .5);
  auto path = m::FlowMatchingStep(&p, x, t, n::scalar(.1), true);
  EXPECT_EQ(path.states.size(), 4);
  auto g = n::grad(n::sum(path.value), {x})[0];
  EXPECT_TRUE(g.defined());
  n::OptimizerConfig c;
  c.kind = n::OptimizerKind::Adam;
  n::Optimizer optimizer(p, .003, c);
  auto         before = p.flatten();
  auto         loss   = n::mean(n::square(m::FlowMatchingVelocity(&p, x, t) - .4));
  optimizer.step(n::grad(loss, p.parameters()));
  EXPECT_NE(before, p.flatten());
  EXPECT_EQ(optimizer.iterations(), 1);
  auto oracle = m::FlowMatchingStep(nullptr, x, t, n::scalar(0), false);
  EXPECT_TRUE(n::equal_bits(oracle.value, x));
}
TEST(NnDiffusionComponents, PeriodicFixtureReconstructsEverySampleFromGeneratingParameters) {
  auto   data    = m::PeriodicDiffusionSignals(5, 18001);
  auto   params  = data.parameters.to_vector<double>();
  auto   signals = data.signals.to_vector<float>();
  auto   ids     = data.source_ids.to_vector<std::int32_t>();
  double pi      = std::acos(-1.);
  for (int b = 0; b < 5; ++b) {
    EXPECT_EQ(ids[b], 18001000 + b);
    for (int j = 0; j < 64; ++j) {
      double f = params[3 * b], a = params[3 * b + 1], phase = params[3 * b + 2];
      float  expected = float(
          .9 * a / 1.25 *
          (std::sin(2 * pi * f * j / 64 + phase) +
           .25 * std::sin(2 * pi * 3 * f * j / 64 + 2 * phase)));
      EXPECT_EQ(signals[64 * b + j], expected);
    }
  }
  EXPECT_TRUE(n::equal_bits(data.signals, m::PeriodicDiffusionSignals(5, 18001).signals));
}
TEST(NnDiffusionComponents, InvalidConditionsRejectWithoutChangingInitializedLeaves) {
  n::ParameterStore p(19003);
  m::InitializeDiffusionMlp(p, 10, 2);
  auto before = p.flatten();
  auto x      = n::zeros({2, 2});
  EXPECT_THROW(m::ConsistencyMap(p, x, n::zeros({2})), std::invalid_argument);
  EXPECT_EQ(before, p.flatten());
  EXPECT_THROW(m::PeriodicDiffusionSignals(0), std::invalid_argument);
  EXPECT_THROW(m::NoiseConditionalSigma(I({16})), std::invalid_argument);
  EXPECT_THROW(m::DiffusionContinuousTimeEmbedding(n::full({1}, -1)), std::invalid_argument);
  EXPECT_THROW(
      m::FlowMatchingStep(nullptr, x, n::full({2}, .9), n::scalar(-.2), true),
      std::invalid_argument);
  EXPECT_THROW(
      m::DdimReverse(x, I({3, 3}), I({3, 2}), x, x, n::scalar(0), m::CosineDiffusionSchedule()),
      std::invalid_argument);
}
TEST(NnDiffusionComponents, SignalRasterRetainsSourcePixelsAndDoesNotMutateObservations) {
  auto x    = n::zeros({1, 8, 8, 1});
  auto grid = m::PeriodicSignalGrid(x, 1);
  EXPECT_EQ(grid.shape(), (n::Shape{112, 256, 3}));
  EXPECT_FALSE(grid.requires_grad());
  auto pixels = grid.to_vector<float>();
  auto value  = [&](int y, int x, int c) { return pixels[(y * 256 + x) * 3 + c]; };
  EXPECT_EQ(value(0, 0, 0), 250);
  EXPECT_EQ(value(56, 0, 0), 30);
  EXPECT_EQ(value(56, 0, 1), 90);
  EXPECT_EQ(value(56, 0, 2), 190);
  EXPECT_EQ(value(56, 255, 0), 210);
  EXPECT_EQ(n::max(n::abs(x)).item(), 0);
  EXPECT_THROW(m::PeriodicSignalGrid(x, 0), std::invalid_argument);
  EXPECT_THROW(m::PeriodicSignalGrid(x, 65), std::invalid_argument);
}

TEST(NnDiffusionComponents, UndefinedTimesAreRejectedBeforeScalarOrRowAccess) {
  auto x     = n::zeros({2, 2});
  auto image = n::zeros({2, 8, 8, 1});
  EXPECT_THROW(
      m::FlowMatchingStep(nullptr, x, n::Tensor{}, n::scalar(.1), true), std::invalid_argument);
  EXPECT_THROW(
      m::ScoreSdeStep(nullptr, image, n::Tensor{}, n::scalar(.1), image), std::invalid_argument);
  EXPECT_THROW(m::AnnealedLangevin(image, n::Tensor{}, image, image), std::invalid_argument);
}
