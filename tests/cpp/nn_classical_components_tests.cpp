#include <gtest/gtest.h>

#include <array>
#include <datamunge/nn/classical_components.hpp>
#include <datamunge/nn/classical_inference.hpp>
#include <limits>
namespace n = datamunge::nn;
namespace m = datamunge::nn::ml;
TEST(ClassicalComponents, PairwiseDistanceAndKernelInputDerivatives) {
  auto x = n::from_vector(std::vector<double>{0, 2}, {2, 1}).requires_grad_();
  auto y = n::from_vector(std::vector<double>{1, 3}, {2, 1});
  EXPECT_EQ(m::SquaredDistances(x, y).to_vector<double>(), (std::vector<double>{1, 9, 1, 1}));
  auto k = m::RbfKernel(x, y, .5f);
  auto g = n::grad(n::sum(k), {x})[0].to_vector<double>();
  EXPECT_NEAR(g[0], std::exp(-.5) + 3 * std::exp(-4.5), 1e-12);
  EXPECT_NEAR(g[1], 0, 1e-12);
  EXPECT_THROW(m::SquaredDistances(n::zeros({0, 1}), n::zeros({1, 1})), std::invalid_argument);
  EXPECT_THROW(m::RbfKernel(x, y, 0), std::invalid_argument);
  EXPECT_THROW(m::SquaredDistances(x, n::zeros({1, 1})), std::invalid_argument);
}
TEST(ClassicalComponents, FourierSeedAndGradientMatchIndependentHostCalculation) {
  m::RandomFourierFeatures a(2, 4, .5f, 73), b(2, 4, .5f, 73);
  EXPECT_EQ(a.weights.to_vector<float>(), b.weights.to_vector<float>());
  std::mt19937_64                       rng(73);
  std::normal_distribution<float>       normal(0, 1);
  std::uniform_real_distribution<float> uniform(0, 2 * std::acos(-1.f));
  auto weights = a.weights.to_vector<float>(), phase = a.phase.to_vector<float>();
  for (auto w : weights)
    EXPECT_EQ(w, normal(rng));
  for (auto p : phase)
    EXPECT_EQ(p, uniform(rng));
  auto                  x = n::from_vector(std::vector<float>{.2f, -.3f}, {1, 2}).requires_grad_();
  auto                  z = a.Transform(x).to_vector<float>();
  auto                  g = n::grad(n::sum(a.Transform(x)), {x})[0].to_vector<float>();
  std::array<double, 2> expected{};
  for (int j = 0; j < 4; ++j) {
    double angle = .2f * weights[j] - .3f * weights[4 + j] + phase[j];
    EXPECT_NEAR(z[j], std::sqrt(.5) * std::cos(angle), 2e-7);
    for (int i = 0; i < 2; ++i)
      expected[i] -= std::sqrt(.5) * std::sin(angle) * weights[i * 4 + j];
  }
  EXPECT_NEAR(g[0], expected[0], 5e-7);
  EXPECT_NEAR(g[1], expected[1], 5e-7);
  EXPECT_THROW(a.Transform(n::zeros({1, 3})), std::invalid_argument);
  a.phase = n::zeros({3});
  EXPECT_THROW(a.Transform(x), std::invalid_argument);
}
TEST(ClassicalComponents, LloydTieAndEmptyClusterAndCallerIterations) {
  m::KMeans model(1);
  auto      x    = n::tensor({0, 2, 9, 11}, {4, 1});
  auto      c    = n::tensor({1, 10, 100}, {3, 1});
  auto      step = model.Step(x, c);
  EXPECT_EQ(step.centers.to_vector<float>(), (std::vector<float>{1, 10, 100}));
  EXPECT_EQ(step.counts.to_vector<float>(), (std::vector<float>{2, 2, 0}));
  EXPECT_EQ(step.inertia, 4);
  auto ties = model.Step(n::tensor({1}, {1, 1}), n::tensor({0, 2}, {2, 1}));
  EXPECT_EQ(ties.labels.to_vector<std::int64_t>()[0], 0);
  auto previous = model.Step(x, n::tensor({0, 9}, {2, 1}));
  auto next     = model.Step(x, previous.centers);
  EXPECT_LE(next.inertia, previous.inertia);
  EXPECT_EQ(c.to_vector<float>(), (std::vector<float>{1, 10, 100}));
  EXPECT_THROW(
      model.Step(x, n::tensor({std::numeric_limits<float>::infinity()}, {1, 1})),
      std::invalid_argument);
}
TEST(ClassicalComponents, SeededInitializationHandlesRepeatedAndDistinctRows) {
  m::KMeans model(1);
  auto      x = n::tensor({0, 1, 2, 3, 4}, {5, 1});
  for (bool plus : {false, true}) {
    auto a = model.Initialize(x, 5, 19, plus).to_vector<float>();
    auto b = model.Initialize(x, 5, 19, plus).to_vector<float>();
    EXPECT_EQ(a, b);
    std::sort(a.begin(), a.end());
    EXPECT_EQ(a, x.to_vector<float>());
  }
  auto same = model.Initialize(n::zeros({5, 1}), 5, 19, true);
  EXPECT_EQ(same.to_vector<float>(), (std::vector<float>(5, 0)));
  EXPECT_THROW(model.Initialize(x, 6, 1, true), std::invalid_argument);
}
namespace {
m::HmmState State() {
  return {
      n::from_vector(std::vector<double>{.6, .4}),
      n::from_vector(std::vector<double>{.7, .3, .4, .6}, {2, 2}),
      n::from_vector(std::vector<double>{.5, .5, .1, .9}, {2, 2})};
}
} // namespace
TEST(ClassicalComponents, HmmMatchesEnumerationPosteriorViterbiAndOneEmUpdate) {
  auto                  state        = State();
  auto                  observations = n::from_vector(std::vector<std::int32_t>{0, 1, 0}, {1, 3});
  m::HiddenMarkovModel  model(2, 2, 3);
  auto                  out   = model.Run(observations, state);
  double                total = 0, best = -1;
  std::array<double, 6> posterior{};
  std::array<int, 3>    path{};
  double                initial[] = {.6, .4}, transition[2][2] = {{.7, .3}, {.4, .6}},
         emission[2][2] = {{.5, .5}, {.1, .9}};
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 2; ++c) {
        double p = initial[a] * emission[a][0] * transition[a][b] * emission[b][1] *
                   transition[b][c] * emission[c][0];
        total += p;
        posterior[a] += p;
        posterior[2 + b] += p;
        posterior[4 + c] += p;
        if (p > best) {
          best = p;
          path = {a, b, c};
        }
      }
  EXPECT_NEAR(out.log_likelihood.item<double>(), std::log(total), 1e-12);
  EXPECT_NEAR(out.viterbi_score.item<double>(), std::log(best), 1e-12);
  EXPECT_EQ(
      out.path.to_vector<std::int32_t>(), (std::vector<std::int32_t>{path[0], path[1], path[2]}));
  auto got = out.posterior.to_vector<double>();
  for (int i = 0; i < 6; ++i)
    EXPECT_NEAR(got[i], posterior[i] / total, 1e-12);
  EXPECT_GE(
      model.Run(observations, out.next).log_likelihood.item<double>(),
      out.log_likelihood.item<double>() - 1e-12);
  auto evidence = n::gather(n::transpose(state.emission), observations);
  EXPECT_NEAR(
      n::ml::hmm_log_likelihood(n::log(state.initial), n::log(state.transition), n::log(evidence))
          .item<double>(),
      out.log_likelihood.item<double>(),
      1e-12);
  EXPECT_FALSE(out.posterior.requires_grad());
}
TEST(ClassicalComponents, HmmSingleStepKeepsTransitionsAndUnvisitedRows) {
  auto state       = State();
  state.initial    = n::from_vector(std::vector<double>{1, 0});
  state.transition = n::from_vector(std::vector<double>{1, 0, 0, 1}, {2, 2});
  m::HiddenMarkovModel model(2, 2, 1);
  auto                 out = model.Run(n::zeros({1, 1}, n::DType::Int32), state);
  EXPECT_EQ(out.next.transition.to_vector<double>(), state.transition.to_vector<double>());
  EXPECT_EQ(out.next.emission.to_vector<double>(), (std::vector<double>{1, 0, .1, .9}));
}
TEST(ClassicalComponents, HmmRejectsInvalidAndImpossibleObservations) {
  m::HiddenMarkovModel model(2, 2, 3);
  auto                 state = State();
  EXPECT_THROW(model.Run(n::full({1, 3}, 2, n::DType::Int32), state), std::invalid_argument);
  EXPECT_THROW(model.Run(n::zeros({1, 3}), state), std::invalid_argument);
  state.emission = n::from_vector(std::vector<double>{0, 1, 0, 1}, {2, 2});
  EXPECT_THROW(model.Run(n::zeros({1, 3}, n::DType::Int32), state), std::runtime_error);
  state         = State();
  state.initial = n::from_vector(std::vector<double>{.1, .1});
  EXPECT_THROW(model.Run(n::zeros({1, 3}, n::DType::Int32), state), std::invalid_argument);
}
