#include <datamunge/optim/bayesian_optimization.hpp>
#include <datamunge/optim/rbf_gaussian_process_surrogate.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

namespace op = datamunge::optim;

namespace {

/// Branin-Hoo on [-5, 10] x [0, 15]; global minimum 0.397887 at three points.
struct Branin : op::ArbitraryFunction {
    double evaluate(const std::vector<double>& x) override {
        const double a = 1.0, b = 5.1 / (4.0 * std::numbers::pi * std::numbers::pi), c = 5.0 / std::numbers::pi;
        const double r = 6.0, s = 10.0, t = 1.0 / (8.0 * std::numbers::pi);
        return a * std::pow(x[1] - b * x[0] * x[0] + c * x[0] - r, 2) + s * (1.0 - t) * std::cos(x[0]) + s;
    }
};

} // namespace

TEST(RBFGaussianProcessSurrogate, TwoPointPosteriorByHand) {
    op::RBFGaussianProcessSurrogate gp(1.0, 0.0);
    gp.fit({{0.0}, {2.0}}, {1.0, -1.0});
    // K = [[1, e^-2], [e^-2, 1]], k* = e^-1/2 (1, 1): the mean is 0 by antisymmetry and the
    // variance is 1 - 2 e^-1 / (1 + e^-2).
    const auto p = gp.predict({1.0});
    EXPECT_NEAR(p.mean, 0.0, 1e-14);
    EXPECT_NEAR(p.sd, std::sqrt(1.0 - 2.0 * std::exp(-1.0) / (1.0 + std::exp(-2.0))), 1e-14);
    // At x = 0.5: k* = (e^-1/8, e^-9/8), mean = k*' K^-1 y.
    const double e2 = std::exp(-2.0), k1 = std::exp(-0.125), k2 = std::exp(-1.125), det = 1.0 - e2 * e2;
    const double a1 = (1.0 + e2) / det, a2 = -(1.0 + e2) / det;  // K^-1 (1, -1)
    EXPECT_NEAR(gp.predict({0.5}).mean, k1 * a1 + k2 * a2, 1e-13);
    // Log marginal likelihood: -y'K^-1y/2 - log|K|/2 - log(2 pi).
    EXPECT_NEAR(gp.log_marginal_likelihood(), -0.5 * (a1 - a2) - 0.5 * std::log(det) - std::log(2.0 * std::numbers::pi), 1e-13);
}

TEST(RBFGaussianProcessSurrogate, InterpolatesWithTinyNoise) {
    op::RBFGaussianProcessSurrogate gp(0.7, 1e-10);
    const std::vector<std::vector<double>> x{{0.0, 0.0}, {1.0, 0.5}, {-0.5, 2.0}, {2.0, 1.0}};
    const std::vector<double>              y{1.0, -2.0, 0.5, 3.0};
    gp.fit(x, y);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(gp.predict(x[i]).mean, y[i], 1e-6);
        EXPECT_LT(gp.predict(x[i]).sd, 1e-4);
    }
    EXPECT_NEAR(gp.predict({30.0, 30.0}).sd, 1.0, 1e-12);  // far away: the prior
}

TEST(RBFGaussianProcessSurrogate, NormalisationIsAffineEquivariant) {
    const std::vector<std::vector<double>> x{{0.0}, {0.4}, {1.1}, {2.0}, {2.6}};
    const std::vector<double>              y{0.3, -0.2, 0.9, 1.4, -0.7};
    std::vector<double>                    z;
    for (const double v : y) z.push_back(50.0 * v - 7.0);
    op::RBFGaussianProcessSurrogate a({.length_scale = 0.8, .fit_hyperparameters = false});
    op::RBFGaussianProcessSurrogate b({.length_scale = 0.8, .fit_hyperparameters = false});
    a.fit(x, y), b.fit(x, z);
    for (const double q : {-0.5, 0.7, 1.9, 3.5}) {
        EXPECT_NEAR(b.predict({q}).mean, 50.0 * a.predict({q}).mean - 7.0, 1e-10);
        EXPECT_NEAR(b.predict({q}).sd, 50.0 * a.predict({q}).sd, 1e-10);
    }
    EXPECT_NEAR(b.log_marginal_likelihood(), a.log_marginal_likelihood() - 5.0 * std::log(50.0), 1e-10);
}

TEST(RBFGaussianProcessSurrogate, FittedHyperparametersMaximiseTheLikelihood) {
    std::vector<std::vector<double>> x;
    std::vector<double>              y;
    for (int i = 0; i < 15; ++i) {
        const double t = 0.37 * i;
        x.push_back({t}), y.push_back(std::sin(1.3 * t) + 0.2 * t);
    }
    op::RBFGaussianProcessSurrogate fitted({.fit_hyperparameters = true});
    fitted.fit(x, y);
    const double ll = fitted.log_marginal_likelihood();
    auto at = [&](double ell, double s) {
        op::RBFGaussianProcessSurrogate g({.length_scale = ell, .signal_variance = s, .noise = 1e-6 * s, .fit_hyperparameters = false});
        g.fit(x, y);
        return g.log_marginal_likelihood();
    };
    const double ell = fitted.length_scale(), s = fitted.signal_variance();
    EXPECT_NEAR(at(ell, s), ll, 1e-8);
    for (const double f : {0.9, 1.1}) {
        EXPECT_LT(at(ell * f, s), ll);
        EXPECT_LT(at(ell, s * f), ll);
    }
}

TEST(RBFGaussianProcessSurrogate, AcquisitionFunctionsMatchTheirFormulas) {
    // EI against numerical integration of max(t - y, 0) under N(mu, sd^2).
    const double mu = 0.3, sd = 0.8, inc = 0.1;
    double       num = 0.0;
    const int    m   = 200000;
    for (int i = 0; i < m; ++i) {
        const double y = mu - 10.0 * sd + 20.0 * sd * (i + 0.5) / m;
        num += std::max(inc - y, 0.0) * std::exp(-0.5 * std::pow((y - mu) / sd, 2)) / (sd * std::sqrt(2.0 * std::numbers::pi)) *
               (20.0 * sd / m);
    }
    EXPECT_NEAR(op::expected_improvement(mu, sd, inc), num, 1e-9);
    EXPECT_NEAR(op::expected_improvement(0.0, 1.0, 0.0), 1.0 / std::sqrt(2.0 * std::numbers::pi), 1e-15);
    EXPECT_DOUBLE_EQ(op::expected_improvement(1.0, 0.0, 3.0), 2.0);
    EXPECT_NEAR(op::probability_of_improvement(1.0, 1.0, 0.0), 0.15865525393145707, 1e-14);

    const std::vector<std::vector<double>> x{{0.0}, {1.0}, {2.5}};
    const std::vector<double>              y{1.0, -0.5, 2.0};
    op::RBFGaussianProcessSurrogate ei({.fit_hyperparameters = false, .exploration = 0.1});
    op::RBFGaussianProcessSurrogate pi({.fit_hyperparameters = false, .acquisition = op::Acquisition::ProbabilityOfImprovement});
    op::RBFGaussianProcessSurrogate lcb({.fit_hyperparameters = false, .acquisition = op::Acquisition::LowerConfidenceBound, .kappa = 1.5});
    ei.fit(x, y), pi.fit(x, y), lcb.fit(x, y);
    const auto p = ei.predict({1.7});
    EXPECT_NEAR(ei.acquisition({1.7}, -0.5), op::expected_improvement(p.mean, p.sd, -0.5, 0.1), 1e-14);
    EXPECT_NEAR(pi.acquisition({1.7}, -0.5), op::probability_of_improvement(p.mean, p.sd, -0.5), 1e-14);
    EXPECT_NEAR(lcb.acquisition({1.7}, -0.5), -(p.mean - 1.5 * p.sd), 1e-14);
}

TEST(RBFGaussianProcessSurrogate, RejectsBadInput) {
    EXPECT_THROW(op::RBFGaussianProcessSurrogate(0.0), std::invalid_argument);
    EXPECT_THROW(op::RBFGaussianProcessSurrogate({.min_length_scale = 2.0, .max_length_scale = 1.0}), std::invalid_argument);
    op::RBFGaussianProcessSurrogate gp;
    EXPECT_THROW((void)gp.predict({0.0}), std::logic_error);
    EXPECT_THROW(gp.fit({{0.0}, {1.0, 2.0}}, {1.0, 2.0}), std::invalid_argument);
}

TEST(BayesianOptimization, RunRecordsTheWholeHistory) {
    Branin f;
    op::RBFGaussianProcessSurrogate gp({.fit_hyperparameters = true});
    const op::BayesianOptimization bo({.initial_samples = 5, .max_iterations = 7, .seed = 3});
    const auto r = bo.run(f, {-5.0, 0.0}, {10.0, 15.0}, gp);
    ASSERT_EQ(r.points.size(), 12u);
    ASSERT_EQ(r.values.size(), 12u);
    ASSERT_EQ(r.best_values.size(), 12u);
    ASSERT_EQ(r.acquisition_values.size(), 7u);
    double running = r.values[0];
    for (std::size_t i = 0; i < r.values.size(); ++i) {
        running = std::min(running, r.values[i]);
        EXPECT_EQ(r.best_values[i], running);
        EXPECT_DOUBLE_EQ(r.values[i], f.evaluate(r.points[i]));
        EXPECT_GE(r.points[i][0], -5.0);
        EXPECT_LE(r.points[i][0], 10.0);
        EXPECT_GE(r.points[i][1], 0.0);
        EXPECT_LE(r.points[i][1], 15.0);
    }
    EXPECT_EQ(r.best_value, running);
    std::vector<double> x(2);
    op::RBFGaussianProcessSurrogate gp2({.fit_hyperparameters = true});
    EXPECT_EQ(bo.optimize(f, x, {-5.0, 0.0}, {10.0, 15.0}, gp2), r.best_value);
    EXPECT_EQ(x, r.best_point);
}

TEST(BayesianOptimization, FindsTheBraninMinimumWithFortyEvaluations) {
    Branin                          f;
    op::RBFGaussianProcessSurrogate gp({.fit_hyperparameters = true});
    const auto r = op::BayesianOptimization({.initial_samples = 8, .max_iterations = 32, .seed = 11}).run(f, {-5.0, 0.0}, {10.0, 15.0}, gp);
    EXPECT_LT(r.best_value, 0.397887 + 0.02);
}

TEST(BayesianOptimization, RejectsBadBounds) {
    Branin                          f;
    op::RBFGaussianProcessSurrogate gp;
    const op::BayesianOptimization  bo;
    EXPECT_THROW((void)bo.run(f, {0.0, 1.0}, {1.0}, gp), std::invalid_argument);
    EXPECT_THROW((void)bo.run(f, {0.0, 1.0}, {1.0, 1.0}, gp), std::invalid_argument);
    std::vector<double> x(1);
    EXPECT_THROW((void)bo.optimize(f, x, {0.0, 0.0}, {1.0, 1.0}, gp), std::invalid_argument);
}
