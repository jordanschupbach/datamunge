#include <gtest/gtest.h>

#include <datamunge/bayes/bayes.hpp>

#include <cmath>

using datamunge::autodiff::Tape;
using datamunge::autodiff::Var;
using datamunge::bayes::AutodiffModel;
using datamunge::bayes::GibbsOptions;
using datamunge::bayes::GibbsSampler;
using datamunge::bayes::ImportanceSampling;
using datamunge::bayes::ImportanceSamplingOptions;
using datamunge::bayes::RandomWalkMetropolis;
using datamunge::bayes::RWMOptions;
using datamunge::bayes::normal_lpdf;

namespace {

// Same conjugate Normal-Normal setup as bayes_tests.cpp (duplicated locally -- it's in that
// file's anonymous namespace, not exported): y_i ~ N(mu, sigma) iid, mu ~ N(mu0, tau0). Exact
// closed-form posterior, so no external oracle is needed.
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

// A standard bivariate normal with known correlation rho -- the same target NUTS's own
// "RecoversCorrelatedBivariateNormalCovariance" test uses, reused here so RWM/Gibbs's
// correlation recovery is checked against the identical known answer.
AutodiffModel correlated_bivariate_model(const double rho) {
    return AutodiffModel([rho](Tape&, const std::vector<Var>& params) {
        const Var& x = params[0];
        const Var& y = params[1];
        const Var quad = (x * x - (2.0 * rho) * x * y + y * y) / (2.0 * (1.0 - rho * rho));
        return quad * (-1.0);
    });
}

} // namespace

// ---- RandomWalkMetropolis ----

TEST(RandomWalkMetropolis, RecoversConjugateNormalPosteriorMoments) {
    const auto setup = make_conjugate_setup();
    auto model = setup.model();
    RWMOptions options;
    options.num_warmup = 2000;
    options.num_samples = 8000;
    options.initial_step_size = 1.0;
    const RandomWalkMetropolis rwm(options);
    const auto result = rwm.sample(model, {0.0});

    ASSERT_EQ(result.samples.size(), options.num_samples);
    double mean = 0.0;
    for (const auto& s : result.samples) mean += s[0];
    mean /= static_cast<double>(result.samples.size());
    double var = 0.0;
    for (const auto& s : result.samples) var += (s[0] - mean) * (s[0] - mean);
    var /= static_cast<double>(result.samples.size() - 1);

    EXPECT_NEAR(mean, setup.posterior_mean(), 0.05);
    EXPECT_NEAR(std::sqrt(var), setup.posterior_sd(), 0.05);
    EXPECT_GT(result.accept_rate, 0.1);
    EXPECT_LT(result.accept_rate, 0.6);
}

TEST(RandomWalkMetropolis, RecoversCorrelatedBivariateNormalCovariance) {
    constexpr double rho = 0.8;
    auto model = correlated_bivariate_model(rho);
    RWMOptions options;
    options.num_warmup = 2000;
    options.num_samples = 8000;
    options.initial_step_size = 0.5;
    const RandomWalkMetropolis rwm(options);
    const auto result = rwm.sample(model, {0.0, 0.0});

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
    EXPECT_NEAR(var_x, 1.0, 0.25);
    EXPECT_NEAR(var_y, 1.0, 0.25);
}

// ---- GibbsSampler (Metropolis-within-Gibbs) ----

TEST(GibbsSampler, RecoversConjugateNormalPosteriorMoments) {
    const auto setup = make_conjugate_setup();
    auto model = setup.model();
    GibbsOptions options;
    options.num_warmup = 2000;
    options.num_samples = 8000;
    options.initial_step_sizes = {1.0};
    const GibbsSampler gibbs(options);
    const auto result = gibbs.sample(model, {0.0});

    ASSERT_EQ(result.samples.size(), options.num_samples);
    double mean = 0.0;
    for (const auto& s : result.samples) mean += s[0];
    mean /= static_cast<double>(result.samples.size());
    double var = 0.0;
    for (const auto& s : result.samples) var += (s[0] - mean) * (s[0] - mean);
    var /= static_cast<double>(result.samples.size() - 1);

    EXPECT_NEAR(mean, setup.posterior_mean(), 0.05);
    EXPECT_NEAR(std::sqrt(var), setup.posterior_sd(), 0.05);
    ASSERT_EQ(result.accept_rates.size(), 1u);
    EXPECT_GT(result.accept_rates[0], 0.2);
    EXPECT_LT(result.accept_rates[0], 0.7);
}

TEST(GibbsSampler, RecoversCorrelatedBivariateNormalCovariance) {
    // Coordinate-wise updates mix slowly under strong correlation -- compensate with a much
    // longer chain rather than a looser tolerance, so this is still checking the same
    // known-exact target NUTS/RWM check against.
    constexpr double rho = 0.8;
    auto model = correlated_bivariate_model(rho);
    GibbsOptions options;
    options.num_warmup = 4000;
    options.num_samples = 40000;
    options.initial_step_sizes = {1.0};
    options.seed = 7;
    const GibbsSampler gibbs(options);
    const auto result = gibbs.sample(model, {0.0, 0.0});

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
    EXPECT_NEAR(var_x, 1.0, 0.25);
    EXPECT_NEAR(var_y, 1.0, 0.25);
}

TEST(GibbsSampler, RejectsMismatchedStepSizeLength) {
    const auto setup = make_conjugate_setup();
    auto model = setup.model();
    GibbsOptions options;
    options.initial_step_sizes = {1.0, 2.0}; // model has 1 parameter
    const GibbsSampler gibbs(options);
    EXPECT_THROW((void)gibbs.sample(model, {0.0}), std::invalid_argument);
}

// ---- ImportanceSampling ----

TEST(ImportanceSampling, ExactProposalGivesUniformWeightsAndFullEss) {
    // When the proposal IS the true posterior (exact for this conjugate model), every
    // importance weight target(x)/proposal(x) = p(x,y)/p(x|y) = p(y) is the SAME constant
    // regardless of x -- an exact, not just approximate, check.
    const auto setup = make_conjugate_setup();
    auto model = setup.model();

    ImportanceSamplingOptions options;
    options.num_samples = 2000;
    const ImportanceSampling is(options);
    const std::vector<std::vector<double>> cov{{setup.posterior_sd() * setup.posterior_sd()}};
    const auto result = is.sample(model, {setup.posterior_mean()}, cov);

    ASSERT_EQ(result.normalized_weights.size(), options.num_samples);
    for (const double w : result.normalized_weights) EXPECT_NEAR(w, 1.0 / static_cast<double>(options.num_samples), 1e-9);
    EXPECT_NEAR(result.effective_sample_size, static_cast<double>(options.num_samples), 1e-6);
    EXPECT_TRUE(std::isfinite(result.log_evidence));
}

TEST(ImportanceSampling, MismatchedProposalStillRecoversPosteriorMomentsWithLowerEss) {
    const auto setup = make_conjugate_setup();
    auto model = setup.model();

    ImportanceSamplingOptions options;
    options.num_samples = 20000;
    const ImportanceSampling is(options);
    // A wider-than-necessary but still reasonably centered proposal.
    const std::vector<std::vector<double>> cov{{4.0 * setup.posterior_sd() * setup.posterior_sd()}};
    const auto result = is.sample(model, {setup.posterior_mean() + 0.3}, cov);

    double weighted_mean = 0.0;
    for (std::size_t i = 0; i < result.samples.size(); ++i) weighted_mean += result.normalized_weights[i] * result.samples[i][0];
    double weighted_var = 0.0;
    for (std::size_t i = 0; i < result.samples.size(); ++i) {
        const double d = result.samples[i][0] - weighted_mean;
        weighted_var += result.normalized_weights[i] * d * d;
    }

    EXPECT_NEAR(weighted_mean, setup.posterior_mean(), 0.05);
    EXPECT_NEAR(std::sqrt(weighted_var), setup.posterior_sd(), 0.1);
    // A mismatched (too-wide, off-center) proposal must be strictly less efficient than the
    // exact-proposal case above, which achieved ESS == num_samples.
    EXPECT_LT(result.effective_sample_size, static_cast<double>(options.num_samples));
}

TEST(ImportanceSampling, RejectsNonPositiveDefiniteCovariance) {
    const auto setup = make_conjugate_setup();
    auto model = setup.model();
    const ImportanceSampling is;
    const std::vector<std::vector<double>> bad_cov{{-1.0}};
    EXPECT_THROW((void)is.sample(model, {0.0}, bad_cov), std::invalid_argument);
}

TEST(ImportanceSampling, RejectsMismatchedCovarianceDimension) {
    const auto setup = make_conjugate_setup();
    auto model = setup.model();
    const ImportanceSampling is;
    const std::vector<std::vector<double>> wrong_size_cov{{0.0, 0.0}, {0.0, 0.0}};
    EXPECT_THROW((void)is.sample(model, {0.0}, wrong_size_cov), std::invalid_argument);
}
