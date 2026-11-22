#include <datamunge/optim/joint_bayesian_search.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

namespace o = datamunge::optim;

TEST(JointGaussianPosterior, IdenticalQueriesShareLatentVarianceAndCanBeCorrelated) {
  o::RBFGaussianProcessSurrogate gp(1, 0.1);
  gp.fit({{0}}, {1});
  const auto posterior = gp.joint_posterior({{0.2}, {0.21}, {0.2}});
  ASSERT_EQ(posterior.covariance.size(), 9);
  EXPECT_EQ(posterior.mean[0], posterior.mean[2]);
  EXPECT_EQ(posterior.covariance[0], posterior.covariance[2]);
  EXPECT_EQ(posterior.covariance[0], posterior.covariance[8]);
  EXPECT_NEAR(posterior.covariance[0], std::pow(gp.predict({0.2}).sd, 2), 1e-15);
  EXPECT_EQ(posterior.covariance[1], posterior.covariance[3]);
  EXPECT_GT(posterior.covariance[1], 0.9 * posterior.covariance[0]);
  EXPECT_THROW((void)gp.joint_posterior({{0, 1}}), std::invalid_argument);
}

TEST(JointBayesianSearch, ReplicatesCountAgainstTheSameBudgetAndReplay) {
  struct Model : o::ArbitraryFunction {
    std::size_t calls = 0;
    double evaluate(const std::vector<double>& x) override {
      const auto noise = calls++ % 2 == 0 ? -0.1 : 0.1;
      return (x[0] - 0.4)*(x[0] - 0.4) + noise;
    }
  };
  for (const auto acquisition : {o::JointAcquisition::thompson, o::JointAcquisition::noisy_improvement}) {
    Model f, replay;
    o::JointBayesianOptions options;
    options.evaluations = 13;
    options.initial_samples = 4;
    options.candidates = 7;
    options.posterior_draws = 20;
    options.replicate_every = 3;
    options.gaussian.fit_hyperparameters = false;
    options.gaussian.noise = 0.05;
    options.acquisition = acquisition;
    const auto result = o::joint_bayesian_search(f, {0}, {1}, options);
    const auto again = o::joint_bayesian_search(replay, {0}, {1}, options);
    EXPECT_EQ(f.calls, 13);
    EXPECT_EQ(result.replicates, 3);
    EXPECT_EQ(result.observed.values, again.observed.values);
    EXPECT_EQ(result.observed.points, again.observed.points);
    EXPECT_EQ(result.recommended_point, again.recommended_point);
    EXPECT_EQ(result.observed.best_value, *std::ranges::min_element(result.observed.values));
    ASSERT_EQ(result.observed.acquisition_values.size(), 9);
    for (std::size_t i = 0; i < result.observed.acquisition_values.size(); ++i) {
      EXPECT_EQ(std::isnan(result.observed.acquisition_values[i]), (i + 1) % 3 == 0);
    }
  }
}
