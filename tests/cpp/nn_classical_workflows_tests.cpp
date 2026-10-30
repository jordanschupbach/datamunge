#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>
#include <numeric>
namespace n = datamunge::nn;
namespace m = n::ml;
TEST(NnClassicalWorkflows, QrAndCholeskySatisfyExactNormalEquations) {
  const auto x = n::tensor({1, -2, 1, -1, 1, 0, 1, 1, 1, 2}, {5, 2}, n::DType::Float64);
  const auto y = n::tensor({0, 0, 1, 2, 2}, {5, 1}, n::DType::Float64);
  for (const auto w : {m::LeastSquaresQr(x, y), m::LeastSquaresCholesky(x, y)}) {
    EXPECT_NEAR(w.at({0, 0}), 1, 1e-12);
    EXPECT_NEAR(w.at({1, 0}), .6, 1e-12);
    EXPECT_LT(n::max(n::abs(n::matmul(x, n::matmul(x, w) - y, true, false))).item(), 1e-12);
  }
}
TEST(NnClassicalWorkflows, LegendreValuesAndPolynomialRecovery) {
  const auto x = n::tensor({-1, 0, 1}, {3, 1}, n::DType::Float64), X = m::PolynomialFeatures(x, 2);
  EXPECT_EQ(X.to_vector<double>(), (std::vector<double>{1, -1, 1, 1, 0, -.5, 1, 1, 1}));
  const auto w = m::LeastSquaresQr(X, n::square(x));
  EXPECT_NEAR(w.at({0, 0}), 1. / 3, 1e-12);
  EXPECT_NEAR(w.at({2, 0}), 2. / 3, 1e-12);
  EXPECT_THROW(m::PolynomialFeatures(x, -1), std::invalid_argument);
}
TEST(NnClassicalWorkflows, ProximalPathMatchesOrthogonalSolutionAndKeepsIntercept) {
  const auto x = n::tensor({-1, -1, 1, 1}, {4, 1}, n::DType::Float64),
             X = n::concat({n::ones_like(x), x}, 1), y = 1 + .8 * x,
             mask = n::tensor({0, 1}, {2, 1}, n::DType::Float64);
  for (int i = 0; i <= 20; ++i) {
    const float l1 = i / 20.f;
    auto        w  = n::zeros({2, 1}, n::DType::Float64);
    for (int k = 0; k < 100; ++k)
      w = m::ElasticNetStep(X, y, w, mask, l1, .5f, .4f);
    EXPECT_NEAR(w.at({0, 0}), 1, 1e-12);
    EXPECT_NEAR(w.at({1, 0}), std::max(.8 - double(l1), 0.) / 1.5, 1e-12);
  }
  EXPECT_THROW(m::ElasticNetStep(X, y, n::zeros({2, 1}), mask, -1, 0, .1), std::invalid_argument);
}
TEST(NnClassicalWorkflows, FoldCoverageBalanceAndWeightedMean) {
  const auto       folds = m::KFold(67, 5, 127), repeat = m::KFold(67, 5, 127);
  std::vector<int> held(67);
  for (size_t i = 0; i < folds.size(); ++i) {
    EXPECT_EQ(folds[i].validation, repeat[i].validation);
    EXPECT_EQ(folds[i].validation.size(), i < 2 ? 14 : 13);
    for (auto row : folds[i].validation) {
      ++held[row];
      EXPECT_EQ(std::count(folds[i].training.begin(), folds[i].training.end(), row), 0);
    }
  }
  for (auto count : held)
    EXPECT_EQ(count, 1);
  size_t     i     = 0;
  const auto score = m::CrossValidate(
      m::KFold(7, 3, 1, false), [&](const auto&, const auto&) { return double(++i); });
  EXPECT_DOUBLE_EQ(score.mean, 13. / 7);
  EXPECT_THROW(m::CompleteFolds(3, {{0, 1}, {1, 2}}), std::invalid_argument);
  EXPECT_THROW(
      m::CrossValidate(
          folds, [](const auto&, const auto&) { return std::numeric_limits<double>::infinity(); }),
      std::runtime_error);
}
TEST(NnClassicalWorkflows, StratifiedCountsAndInsufficientClass) {
  const std::vector<int> labels{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  const auto             folds = m::StratifiedKFold(labels, 3, 17);
  for (const auto& fold : folds) {
    int c = 0;
    for (auto i : fold.validation)
      c += labels[i] == 0;
    EXPECT_GE(c, 1);
    EXPECT_LE(c, 2);
  }
  EXPECT_THROW(m::StratifiedKFold({0, 0, 1}, 2, 17), std::invalid_argument);
}
TEST(NnClassicalWorkflows, GlmLikelihoodAndScoresMatchScalarDistributions) {
  for (const auto family :
       {m::GlmFamily::kPoisson, m::GlmFamily::kGamma, m::GlmFamily::kNegativeBinomial})
    for (double value : {-2., -1., 0., 1., 2.}) {
      auto         eta = n::scalar(value, n::DType::Float64).requires_grad_();
      const double mu = std::exp(value), y = 2, shape = 3;
      const auto   loss  = m::GlmNll(eta, n::scalar(y, n::DType::Float64), family, shape);
      const auto   g     = n::grad(loss, {eta})[0].item();
      const double score = family == m::GlmFamily::kPoisson ? mu - y
                           : family == m::GlmFamily::kGamma ? shape * (1 - y / mu)
                                                            : shape * (mu - y) / (shape + mu);
      const double exact = family == m::GlmFamily::kPoisson ? mu - y * value + std::lgamma(y + 1)
                           : family == m::GlmFamily::kGamma
                               ? shape * (value + y / mu) - (shape - 1) * std::log(y) +
                                     std::lgamma(shape) - shape * std::log(shape)
                               : std::lgamma(shape) + std::lgamma(y + 1) - std::lgamma(y + shape) +
                                     (y + shape) * std::log(shape + mu) - shape * std::log(shape) -
                                     y * value;
      EXPECT_NEAR(g, score, 1e-11);
      EXPECT_NEAR(loss.item(), exact, 1e-11);
    }
}
TEST(NnClassicalWorkflows, MixtureResponsibilitiesAndMonotoneLikelihood) {
  const auto              type = n::DType::Float64;
  const auto              x    = n::tensor({-2, -1, 0, 1, 2}, {5, 1}, type);
  m::GaussianMixtureState s{
      n::tensor({.5, .5}, {2}, type), n::tensor({-1, 1}, {2, 1}, type), n::ones({2, 1, 1}, type)};
  m::GaussianMixture model(2, 1, .01);
  const auto         first = model.Step(x, s);
  for (int i = 0; i < 5; ++i)
    EXPECT_NEAR(first.responsibilities.at({i, 1}), 1 / (1 + std::exp(-2 * x.at({i, 0}))), 1e-12);
  double previous = -1e100;
  for (int i = 0; i < 20; ++i) {
    const auto step = model.Step(x, s);
    EXPECT_GE(step.log_likelihood + 1e-10, previous);
    previous = step.log_likelihood;
    s        = step.next;
  }
}
TEST(NnClassicalWorkflows, MixtureCovarianceFloorAndInvalidWeights) {
  const auto              type = n::DType::Float64;
  m::GaussianMixtureState s{n::ones({1}, type), n::zeros({1, 1}, type), n::ones({1, 1, 1}, type)};
  const auto              step = m::GaussianMixture(1, 1, .001).Step(n::ones({4, 1}, type), s);
  EXPECT_NEAR(step.next.covariances.item(), .001, 1e-12);
  s.weights = n::full({1}, 2, type);
  EXPECT_THROW(m::GaussianMixture(1, 1).Step(n::ones({4, 1}, type), s), std::invalid_argument);
}
TEST(NnClassicalWorkflows, MultinomialAndBernoulliScoresMatchCounts) {
  const auto x      = n::tensor({2, 0, 1, 0, 0, 1, 0, 2}, {4, 2}, n::DType::Float64),
             labels = n::from_vector(std::vector<int32_t>{0, 0, 1, 1});
  const m::NaiveBayes count(x, labels, 2, m::NaiveBayesKind::kMultinomial);
  EXPECT_NEAR(count.probability.at({0, 0}), .8, 1e-12);
  const auto empty = count.Scores(n::zeros({1, 2}, n::DType::Float64));
  EXPECT_NEAR(empty.at({0, 0}), std::log(.5), 1e-12);
  const m::NaiveBayes binary(
      n::minimum(x, n::ones_like(x)), labels, 2, m::NaiveBayesKind::kBernoulli);
  const auto score = binary.Scores(n::tensor({1, 0}, {1, 2}, n::DType::Float64));
  EXPECT_NEAR(score.at({0, 0}), std::log(.5) + 2 * std::log(.75), 1e-12);
  EXPECT_NEAR(score.at({0, 1}), std::log(.5) + 2 * std::log(.25), 1e-12);
}
