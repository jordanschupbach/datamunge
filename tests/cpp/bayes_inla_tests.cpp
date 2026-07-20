#include <gtest/gtest.h>

#include <datamunge/bayes/bayes.hpp>
#include <datamunge/bayes/inla.hpp>

#include <cmath>
#include <functional>
#include <random>

using namespace datamunge::bayes;
using datamunge::autodiff::Tape;
using datamunge::autodiff::Var;
using datamunge::linalg::DenseMatrix;

namespace {
double sum(const std::vector<double>& v) {
    double s = 0.0;
    for (const double x : v) s += x;
    return s;
}

// A single scalar latent field x loaded on by every observation (design is an n x 1 column
// of ones) with a log-precision-parameterized Gaussian prior Q(theta) = exp(theta[0]).
DenseMatrix<double> ones_column(std::size_t n) { return DenseMatrix<double>(n, 1, 1.0); }

std::function<double(const std::vector<double>&)> fixed_sigma(double sigma) {
    return [sigma](const std::vector<double>&) { return sigma; };
}

INLAPrecisionFn scalar_log_precision() {
    return [](const std::vector<double>& theta) {
        DenseMatrix<double> Q(1, 1, 0.0);
        Q(0, 0) = std::exp(theta[0]);
        return Q;
    };
}
} // namespace

// ---- Exact conjugate check: pins theta via a near-zero-width box so Newton-Raphson's mode
// and Hessian can be checked directly against the closed-form Normal-Normal posterior. ----

TEST(INLA, ConjugateNormalMatchesClosedFormAtPinnedTheta) {
    const std::vector<double> y{4.1, 5.3, 4.8, 5.6, 4.4, 5.9, 5.0, 4.6, 5.2, 5.0,
                                 4.9, 5.4, 5.8, 3.9, 5.1, 4.7, 5.0, 5.3, 5.6, 4.4};
    const double sigma = 1.0;
    const double tau = 2.0;
    const std::size_t n = y.size();

    INLAOptions options;
    options.strategy = INLAIntegrationStrategy::EmpiricalBayes;
    options.mode_population_size = 20;
    options.mode_max_generations = 60;
    const INLA inla(options);

    const double theta_true = std::log(tau);
    const auto result = inla.fit(ones_column(n), y, inla_gaussian_likelihood(fixed_sigma(sigma)), scalar_log_precision(),
                                  [](const std::vector<double>&) { return 0.0; }, {theta_true - 1e-6},
                                  {theta_true + 1e-6}, {theta_true});

    const double var_star = 1.0 / (tau + static_cast<double>(n) / (sigma * sigma));
    const double mean_star = var_star * sum(y) / (sigma * sigma);

    ASSERT_EQ(result.latent_mean.size(), 1u);
    EXPECT_NEAR(result.latent_mean[0], mean_star, 1e-3);
    EXPECT_NEAR(result.latent_sd[0], std::sqrt(var_star), 1e-3);
}

// ---- Grid integration should account for theta's own posterior spread, so its marginal sd
// for the latent field must be at least as large as EmpiricalBayes's mode-only sd. ----

TEST(INLA, GridSdIsAtLeastEmpiricalBayesSd) {
    const std::vector<double> y{4.1, 5.3, 4.8, 5.6, 4.4, 5.9, 5.0, 4.6, 5.2, 5.0,
                                 4.9, 5.4, 5.8, 3.9, 5.1, 4.7, 5.0, 5.3, 5.6, 4.4};
    const std::size_t n = y.size();
    const auto log_prior = [](const std::vector<double>& theta) { return normal_lpdf(theta[0], 0.0, 2.0); };

    INLAOptions eb_options;
    eb_options.strategy = INLAIntegrationStrategy::EmpiricalBayes;
    eb_options.mode_population_size = 20;
    eb_options.mode_max_generations = 60;
    const INLA eb_inla(eb_options);
    const auto eb_result =
        eb_inla.fit(ones_column(n), y, inla_gaussian_likelihood(fixed_sigma(1.0)), scalar_log_precision(), log_prior, {-5.0}, {5.0}, {0.0});

    INLAOptions grid_options = eb_options;
    grid_options.strategy = INLAIntegrationStrategy::Grid;
    grid_options.grid_points_per_dim = 9;
    const INLA grid_inla(grid_options);
    const auto grid_result = grid_inla.fit(ones_column(n), y, inla_gaussian_likelihood(fixed_sigma(1.0)), scalar_log_precision(), log_prior,
                                            {-5.0}, {5.0}, {0.0});

    EXPECT_GE(grid_result.latent_sd[0], eb_result.latent_sd[0] - 1e-6);
    EXPECT_NEAR(grid_result.latent_mean[0], eb_result.latent_mean[0], 0.2);
    EXPECT_TRUE(std::isfinite(grid_result.log_marginal_likelihood));

    double weight_sum = 0.0;
    for (const double w : grid_result.theta_grid_weights) weight_sum += w;
    EXPECT_NEAR(weight_sum, 1.0, 1e-9);
    EXPECT_EQ(grid_result.theta_grid.size(), grid_result.theta_grid_weights.size()); // theta_dim == 1
}

// ---- Qualitative Bayesian sanity check: more data should sharpen theta's posterior. ----

TEST(INLA, MoreDataShrinksThetaPosteriorUncertainty) {
    std::mt19937_64 rng(123);
    std::normal_distribution<double> noise(0.0, 1.0);
    const double true_x = 0.7;
    auto make_y = [&](std::size_t n) {
        std::vector<double> y(n);
        for (auto& v : y) v = true_x + noise(rng);
        return y;
    };

    const auto log_prior = [](const std::vector<double>& theta) { return normal_lpdf(theta[0], 0.0, 2.0); };
    INLAOptions options;
    options.strategy = INLAIntegrationStrategy::Grid;
    options.grid_points_per_dim = 9;
    options.mode_population_size = 20;
    options.mode_max_generations = 60;
    const INLA inla(options);

    const auto small = inla.fit(ones_column(20), make_y(20), inla_gaussian_likelihood(fixed_sigma(1.0)), scalar_log_precision(), log_prior,
                                 {-5.0}, {5.0}, {0.0});
    const auto large = inla.fit(ones_column(400), make_y(400), inla_gaussian_likelihood(fixed_sigma(1.0)), scalar_log_precision(), log_prior,
                                 {-5.0}, {5.0}, {0.0});

    EXPECT_LT(large.theta_sd[0], small.theta_sd[0]);
}

// ---- Full-pipeline cross-check against NUTS on a genuinely non-Gaussian (Binomial/logit)
// likelihood -- the strongest available correctness check for the Laplace approximation
// itself, not just the closed-form-checkable Gaussian case above. ----

TEST(INLA, BinomialSingleInterceptMatchesNUTSPosterior) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    const double true_x = 0.4;
    const double p_true = 1.0 / (1.0 + std::exp(-true_x));
    const std::size_t n = 300;
    std::vector<double> y(n);
    for (auto& v : y) v = (unif(rng) < p_true) ? 1.0 : 0.0;

    const double tau0 = 2.0; // fixed prior sd on the intercept
    const auto precision = [tau0](const std::vector<double>&) {
        DenseMatrix<double> Q(1, 1, 0.0);
        Q(0, 0) = 1.0 / (tau0 * tau0);
        return Q;
    };

    INLAOptions options;
    options.strategy = INLAIntegrationStrategy::EmpiricalBayes;
    options.mode_population_size = 10;
    options.mode_max_generations = 30;
    const INLA inla(options);
    const auto result = inla.fit(ones_column(n), y, inla_binomial_logit_likelihood(), precision,
                                  [](const std::vector<double>&) { return 0.0; }, {0.0}, {1.0}, {0.5});

    AutodiffModel model([&](Tape&, const std::vector<Var>& params) {
        Var lp = normal_lpdf(params[0], 0.0, tau0);
        for (const double yi : y) lp = lp + bernoulli_logit_lpmf(yi, params[0]);
        return lp;
    });
    NUTSOptions nuts_options;
    nuts_options.num_warmup = 500;
    nuts_options.num_samples = 2000;
    nuts_options.initial_step_size = 0.3;
    const NUTS nuts(nuts_options);
    const auto nuts_result = nuts.sample(model, {0.0});

    double mean = 0.0;
    for (const auto& s : nuts_result.samples) mean += s[0];
    mean /= static_cast<double>(nuts_result.samples.size());
    double var = 0.0;
    for (const auto& s : nuts_result.samples) var += (s[0] - mean) * (s[0] - mean);
    var /= static_cast<double>(nuts_result.samples.size() - 1);

    EXPECT_NEAR(result.latent_mean[0], mean, 0.05);
    EXPECT_NEAR(result.latent_sd[0], std::sqrt(var), 0.05);
}

// ---- Poisson smoke test: no closed form, but the implied rate should land near the true
// generating rate under a near-flat prior. ----

TEST(INLA, PoissonSingleInterceptRecoversApproximateRate) {
    std::mt19937_64 rng(11);
    std::poisson_distribution<int> pois(6);
    std::vector<double> y(500);
    for (auto& v : y) v = static_cast<double>(pois(rng));

    const auto precision = [](const std::vector<double>&) { return DenseMatrix<double>(1, 1, 1e-4); };
    INLAOptions options;
    options.strategy = INLAIntegrationStrategy::EmpiricalBayes;
    const INLA inla(options);
    const auto result = inla.fit(ones_column(y.size()), y, inla_poisson_log_likelihood(), precision,
                                  [](const std::vector<double>&) { return 0.0; }, {0.0}, {1.0}, {0.5});

    EXPECT_NEAR(std::exp(result.latent_mean[0]), 6.0, 0.5);
}

// ---- Validation ----

TEST(INLA, RejectsMismatchedDesignAndResponseSizes) {
    const INLA inla;
    EXPECT_THROW((void)inla.fit(ones_column(5), std::vector<double>(4, 0.0), inla_gaussian_likelihood(fixed_sigma(1.0)), scalar_log_precision(),
                           [](const std::vector<double>&) { return 0.0; }, {0.0}, {1.0}, {0.5}),
                 std::invalid_argument);
}

TEST(INLA, RejectsEmptyTheta) {
    const INLA inla;
    EXPECT_THROW((void)inla.fit(ones_column(5), std::vector<double>(5, 0.0), inla_gaussian_likelihood(fixed_sigma(1.0)), scalar_log_precision(),
                           [](const std::vector<double>&) { return 0.0; }, {}, {}, {}),
                 std::invalid_argument);
}

TEST(INLA, GridStrategyRejectsTooManyThetaDimensions) {
    INLAOptions options; // default strategy is Grid
    const INLA inla(options);
    const std::vector<double> lower(5, -1.0), upper(5, 1.0), init(5, 0.0);
    EXPECT_THROW((void)inla.fit(ones_column(5), std::vector<double>(5, 0.0), inla_gaussian_likelihood(fixed_sigma(1.0)), scalar_log_precision(),
                           [](const std::vector<double>&) { return 0.0; }, lower, upper, init),
                 std::invalid_argument);
}
