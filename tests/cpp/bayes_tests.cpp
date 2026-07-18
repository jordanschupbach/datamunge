#include <gtest/gtest.h>

#include <datamunge/bayes/bayes.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/glm.hpp>

#include <cmath>
#include <random>

using datamunge::autodiff::Tape;
using datamunge::autodiff::Var;
using datamunge::bayes::AutodiffModel;
using datamunge::bayes::HMC;
using datamunge::bayes::HMCOptions;
using datamunge::bayes::MAP;
using datamunge::bayes::NUTS;
using datamunge::bayes::NUTSOptions;
using datamunge::dstruct::DataFrame;
using datamunge::stats::GLM;
using datamunge::stats::GLMFamily;
using datamunge::stats::GLMOptions;

using datamunge::bayes::bernoulli_logit_lpmf;
using datamunge::bayes::cauchy_lpdf;
using datamunge::bayes::exponential_lpdf;
using datamunge::bayes::normal_lpdf;
using datamunge::bayes::poisson_log_lpmf;

namespace {

// A conjugate Normal-Normal model: y_i ~ N(mu, sigma) iid, mu ~ N(mu0, tau0). Has an exact,
// closed-form posterior, so it needs no external oracle -- a strong, self-contained check.
struct ConjugateSetup {
    std::vector<double> y;
    double sigma;
    double mu0;
    double tau0;

    [[nodiscard]] double posterior_mean() const {
        const double n = static_cast<double>(y.size());
        double sum_y = 0.0;
        for (const double v : y) sum_y += v;
        const double precision = 1.0 / (tau0 * tau0) + n / (sigma * sigma);
        return (mu0 / (tau0 * tau0) + sum_y / (sigma * sigma)) / precision;
    }
    [[nodiscard]] double posterior_sd() const {
        const double n = static_cast<double>(y.size());
        const double precision = 1.0 / (tau0 * tau0) + n / (sigma * sigma);
        return std::sqrt(1.0 / precision);
    }
    [[nodiscard]] AutodiffModel model() const {
        return AutodiffModel([this](Tape&, const std::vector<Var>& params) {
            Var lp = normal_lpdf(params[0], mu0, tau0);
            for (const double yi : y) lp = lp + normal_lpdf(yi, params[0], sigma);
            return lp;
        });
    }
};

ConjugateSetup make_conjugate_setup() {
    return ConjugateSetup{{2.1, 1.8, 2.5, 2.0, 1.9, 2.3, 2.2, 1.7, 2.4, 1.95}, 1.0, 0.0, 5.0};
}

} // namespace

// ---- Distribution log-densities: spot-check against hand-computed values ----

TEST(BayesDistributions, NormalLpdfMatchesKnownValue) {
    EXPECT_NEAR(normal_lpdf(0.0, 0.0, 1.0), -0.5 * std::log(2.0 * M_PI), 1e-12);
    EXPECT_NEAR(normal_lpdf(1.0, 0.0, 1.0), -0.5 * std::log(2.0 * M_PI) - 0.5, 1e-12);
}

TEST(BayesDistributions, BernoulliLogitLpmfMatchesKnownValue) {
    // logit_p = 0 -> p = 0.5 for either outcome.
    EXPECT_NEAR(bernoulli_logit_lpmf(1.0, 0.0), std::log(0.5), 1e-12);
    EXPECT_NEAR(bernoulli_logit_lpmf(0.0, 0.0), std::log(0.5), 1e-12);
}

TEST(BayesDistributions, PoissonLogLpmfMatchesKnownValue) {
    // log_rate = 0 -> rate = 1 -> P(0) = exp(-1).
    EXPECT_NEAR(poisson_log_lpmf(0.0, 0.0), -1.0, 1e-12);
}

TEST(BayesDistributions, ExponentialAndCauchyLpdfMatchKnownValues) {
    EXPECT_NEAR(exponential_lpdf(0.0, 2.0), std::log(2.0), 1e-12);
    EXPECT_NEAR(cauchy_lpdf(0.0, 0.0, 1.0), -std::log(M_PI), 1e-12);
}

// ---- AutodiffModel: verify its automatic gradient against a hand-derived analytic one ----

TEST(AutodiffModel, GradientMatchesAnalyticFormula) {
    // f(mu) = normal_lpdf(2.0, mu, 1.0); d/dmu = (2.0 - mu) (standard normal score function).
    AutodiffModel model([](Tape&, const std::vector<Var>& params) { return normal_lpdf(2.0, params[0], 1.0); });
    const std::vector<double> params{0.3};
    const double value = model.evaluate(params);
    const auto grad = model.gradient(params);
    EXPECT_NEAR(value, normal_lpdf(0.3, 2.0, 1.0), 1e-12);
    ASSERT_EQ(grad.size(), 1u);
    EXPECT_NEAR(grad[0], 2.0 - 0.3, 1e-9);
}

// ---- MAP ----

TEST(BayesMAP, RecoversConjugateNormalPosteriorMode) {
    const auto setup = make_conjugate_setup();
    auto model = setup.model();
    std::vector<double> coords{0.0};
    const MAP map;
    const double log_posterior_at_mode = map.optimize(model, coords);
    EXPECT_NEAR(coords[0], setup.posterior_mean(), 1e-4);
    EXPECT_TRUE(std::isfinite(log_posterior_at_mode));
}

TEST(BayesMAP, LogisticRegressionMatchesGlmMleUnderWeakPrior) {
    std::mt19937_64 rng(2024);
    std::uniform_real_distribution<double> x_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    constexpr double true_b0 = -0.5, true_b1 = 1.2;

    std::vector<double> x(400), y(400);
    for (int i = 0; i < 400; ++i) {
        x[i] = x_dist(rng);
        const double p = 1.0 / (1.0 + std::exp(-(true_b0 + true_b1 * x[i])));
        y[i] = unif01(rng) < p ? 1.0 : 0.0;
    }
    DataFrame df;
    df.add_column("x", x);
    df.add_column("y", y);
    GLMOptions glm_options;
    glm_options.family = GLMFamily::Binomial;
    const GLM glm(df, "y ~ x", glm_options);

    AutodiffModel model([&](Tape&, const std::vector<Var>& params) {
        Var lp = normal_lpdf(params[0], 0.0, 10.0) + normal_lpdf(params[1], 0.0, 10.0);
        for (std::size_t i = 0; i < x.size(); ++i) {
            const Var eta = params[0] + params[1] * x[i];
            lp = lp + bernoulli_logit_lpmf(y[i], eta);
        }
        return lp;
    });
    std::vector<double> coords{0.0, 0.0};
    const MAP map;
    map.optimize(model, coords);

    EXPECT_NEAR(coords[0], glm.coefficients()[0], 0.15);
    EXPECT_NEAR(coords[1], glm.coefficients()[1], 0.15);
}

// ---- HMC ----

TEST(BayesHMC, RecoversConjugateNormalPosteriorMoments) {
    const auto setup = make_conjugate_setup();
    auto model = setup.model();
    HMCOptions options;
    options.num_warmup = 500;
    options.num_samples = 3000;
    options.num_leapfrog_steps = 15;
    options.initial_step_size = 0.3;
    const HMC hmc(options);
    const auto result = hmc.sample(model, {0.0});

    ASSERT_EQ(result.samples.size(), options.num_samples);
    double mean = 0.0;
    for (const auto& s : result.samples) mean += s[0];
    mean /= static_cast<double>(result.samples.size());
    double var = 0.0;
    for (const auto& s : result.samples) var += (s[0] - mean) * (s[0] - mean);
    var /= static_cast<double>(result.samples.size() - 1);

    EXPECT_NEAR(mean, setup.posterior_mean(), 0.05);
    EXPECT_NEAR(std::sqrt(var), setup.posterior_sd(), 0.05);
    EXPECT_GT(result.accept_rate, 0.5);
}

TEST(BayesHMC, RecoversIndependentStandardNormalTarget) {
    AutodiffModel model([](Tape&, const std::vector<Var>& params) {
        return normal_lpdf(params[0], 0.0, 1.0) + normal_lpdf(params[1], 0.0, 1.0);
    });
    HMCOptions options;
    options.num_warmup = 1000;
    options.num_samples = 5000;
    options.num_leapfrog_steps = 15;
    options.initial_step_size = 0.3;
    const HMC hmc(options);
    const auto result = hmc.sample(model, {2.0, -2.0}); // start far from the target's mean

    double mean0 = 0.0, mean1 = 0.0;
    for (const auto& s : result.samples) {
        mean0 += s[0];
        mean1 += s[1];
    }
    mean0 /= static_cast<double>(result.samples.size());
    mean1 /= static_cast<double>(result.samples.size());
    EXPECT_NEAR(mean0, 0.0, 0.15);
    EXPECT_NEAR(mean1, 0.0, 0.15);
    EXPECT_GT(result.accept_rate, 0.3);
}

// ---- NUTS ----

TEST(BayesNUTS, RecoversConjugateNormalPosteriorMoments) {
    const auto setup = make_conjugate_setup();
    auto model = setup.model();
    NUTSOptions options;
    options.num_warmup = 500;
    options.num_samples = 2000;
    options.max_tree_depth = 8;
    options.initial_step_size = 0.3;
    const NUTS nuts(options);
    const auto result = nuts.sample(model, {0.0});

    ASSERT_EQ(result.samples.size(), options.num_samples);
    double mean = 0.0;
    for (const auto& s : result.samples) mean += s[0];
    mean /= static_cast<double>(result.samples.size());
    double var = 0.0;
    for (const auto& s : result.samples) var += (s[0] - mean) * (s[0] - mean);
    var /= static_cast<double>(result.samples.size() - 1);

    EXPECT_NEAR(mean, setup.posterior_mean(), 0.05);
    EXPECT_NEAR(std::sqrt(var), setup.posterior_sd(), 0.05);
    EXPECT_LT(result.num_divergences, options.num_samples / 20); // well-behaved target: few divergences
}

TEST(BayesNUTS, RecoversCorrelatedBivariateNormalCovariance) {
    // A target with a known off-diagonal covariance: build it via a linear transform of two
    // independent standard normals, y1 = z1, y2 = rho*z1 + sqrt(1-rho^2)*z2, so its inverse
    // (the log-density used here) is just the standard bivariate normal with correlation rho.
    constexpr double rho = 0.8;
    AutodiffModel model([](Tape&, const std::vector<Var>& params) {
        const Var& x = params[0];
        const Var& y = params[1];
        // log density of a standard bivariate normal with correlation rho (up to a constant).
        const Var quad = (x * x - (2.0 * rho) * x * y + y * y) / (2.0 * (1.0 - rho * rho));
        return quad * (-1.0);
    });
    NUTSOptions options;
    options.num_warmup = 500;
    options.num_samples = 3000;
    options.initial_step_size = 0.3;
    const NUTS nuts(options);
    const auto result = nuts.sample(model, {0.0, 0.0});

    double mean_x = 0.0, mean_y = 0.0;
    for (const auto& s : result.samples) {
        mean_x += s[0];
        mean_y += s[1];
    }
    mean_x /= static_cast<double>(result.samples.size());
    mean_y /= static_cast<double>(result.samples.size());

    double cov = 0.0, var_x = 0.0, var_y = 0.0;
    for (const auto& s : result.samples) {
        cov += (s[0] - mean_x) * (s[1] - mean_y);
        var_x += (s[0] - mean_x) * (s[0] - mean_x);
        var_y += (s[1] - mean_y) * (s[1] - mean_y);
    }
    const std::size_t n = result.samples.size();
    cov /= static_cast<double>(n - 1);
    var_x /= static_cast<double>(n - 1);
    var_y /= static_cast<double>(n - 1);
    const double sample_corr = cov / std::sqrt(var_x * var_y);

    EXPECT_NEAR(sample_corr, rho, 0.1);
    EXPECT_NEAR(var_x, 1.0, 0.2);
    EXPECT_NEAR(var_y, 1.0, 0.2);
}
