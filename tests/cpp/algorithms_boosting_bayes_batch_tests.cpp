#include <gtest/gtest.h>

#include <datamunge/algorithms/logitboost.hpp>
#include <datamunge/algorithms/nested_sampling.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using datamunge::algorithms::logitboost_predict;
using datamunge::algorithms::logitboost_probability;
using datamunge::algorithms::logitboost_train;
using datamunge::algorithms::nested_sampling;

TEST(LogitBoost, LearnsNonlinearBoundary) {
    // Diagonally separable data (label = [x0 + x1 > 0]); no single stump can do it.
    std::mt19937_64                        rng(1);
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    std::vector<std::vector<double>>       X;
    std::vector<int>                       y;
    for (int i = 0; i < 200; ++i) {
        double a = u(rng), b = u(rng);
        if (std::abs(a + b) < 0.3) continue;
        X.push_back({a, b});
        y.push_back((a + b > 0.0) ? 1 : 0);
    }
    auto model = logitboost_train(X, y, 40);

    // Log-likelihood should improve over rounds (Newton steps on the logistic loss).
    ASSERT_GE(model.log_likelihood.size(), 2u);
    EXPECT_GT(model.log_likelihood.back(), model.log_likelihood.front());

    std::size_t correct = 0;
    for (std::size_t i = 0; i < X.size(); ++i) correct += (logitboost_predict(model, X[i]) == y[i]);
    EXPECT_GT(static_cast<double>(correct) / X.size(), 0.95);

    // Probabilities are calibrated toward the right side of the boundary.
    EXPECT_GT(logitboost_probability(model, {3.0, 3.0}), 0.8);
    EXPECT_LT(logitboost_probability(model, {-3.0, -3.0}), 0.2);
}

TEST(NestedSampling, RecoversGaussianEvidence) {
    // Likelihood = exp(-||theta - mu||^2 / 2) over the box [-10,10]^2 with a uniform prior.
    // Analytic evidence: Z = (2*pi*sigma^2) / V, V = 20*20 = 400, sigma = 1.
    //   log Z = log(2*pi) - log(400) = 1.837877 - 5.991465 = -4.153589.
    const std::vector<double> mu = {2.0, -1.0};
    auto log_like = [&](const std::vector<double>& t) {
        double s = 0.0;
        for (std::size_t k = 0; k < t.size(); ++k) s += (t[k] - mu[k]) * (t[k] - mu[k]);
        return -0.5 * s;
    };
    const std::vector<double> lo = {-10.0, -10.0}, hi = {10.0, 10.0};

    auto r = nested_sampling(log_like, lo, hi, 200, 5000, 7, 40);

    const double analytic = std::log(2.0 * M_PI) - std::log(400.0);
    EXPECT_NEAR(r.log_evidence, analytic, 0.3);
    EXPECT_GT(r.iterations, 100u);

    // Posterior weights form a distribution summing to ~1.
    double wsum = 0.0;
    for (double lw : r.posterior_log_weights) wsum += std::exp(lw);
    EXPECT_NEAR(wsum, 1.0, 1e-6);

    // Posterior-weighted mean should recover mu.
    std::vector<double> pm(2, 0.0);
    for (std::size_t i = 0; i < r.posterior_samples.size(); ++i) {
        double w = std::exp(r.posterior_log_weights[i]);
        pm[0] += w * r.posterior_samples[i][0];
        pm[1] += w * r.posterior_samples[i][1];
    }
    EXPECT_NEAR(pm[0], mu[0], 0.4);
    EXPECT_NEAR(pm[1], mu[1], 0.4);
}
