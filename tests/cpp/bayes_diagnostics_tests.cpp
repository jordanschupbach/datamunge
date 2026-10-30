#include <gtest/gtest.h>

#include <datamunge/bayes/bayes.hpp>

#include <cmath>
#include <random>

namespace bayes = datamunge::bayes;
using datamunge::autodiff::Tape;
using datamunge::autodiff::Var;

namespace {

std::vector<std::vector<double>> ar1_chains(const double phi, const std::size_t m, const std::size_t n,
                                            const std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    std::vector<std::vector<double>> chains(m);
    for (auto& c : chains) {
        double x = z(rng) / std::sqrt(1.0 - phi * phi);
        for (std::size_t i = 0; i < n; ++i) {
            x = phi * x + z(rng);
            c.push_back(x);
        }
    }
    return chains;
}

} // namespace

TEST(BayesDiagnostics, EssMatchesAr1Theory) {
    const auto iid = ar1_chains(0.0, 4, 5000, 1);
    EXPECT_NEAR(bayes::effective_sample_size(iid) / 20000.0, 1.0, 0.1);
    const auto ar = ar1_chains(0.5, 4, 20000, 2);
    // Integrated autocorrelation time of AR(1) is (1+phi)/(1-phi) = 3.
    EXPECT_NEAR(bayes::effective_sample_size(ar) / 80000.0, 1.0 / 3.0, 0.03);
}

TEST(BayesDiagnostics, RhatDetectsDisagreeingChains) {
    auto chains = ar1_chains(0.3, 4, 2000, 3);
    EXPECT_LT(bayes::split_rhat(chains), 1.01);
    for (double& v : chains[0]) v += 3.0;
    EXPECT_GT(bayes::split_rhat(chains), 1.2);
    // A drift inside one chain is caught by splitting it.
    std::vector<std::vector<double>> drift{std::vector<double>(1000)};
    for (std::size_t i = 0; i < 1000; ++i) drift[0][i] = static_cast<double>(i) / 100.0;
    EXPECT_GT(bayes::split_rhat(drift), 1.5);
    EXPECT_THROW(bayes::split_rhat({{1.0, 2.0}}), std::invalid_argument);
}

TEST(BayesDiagnostics, EnergyBfmiOfIndependentEnergiesIsTwo) {
    const auto iid = ar1_chains(0.0, 1, 50000, 4);
    EXPECT_NEAR(bayes::energy_bfmi(iid[0]), 2.0, 0.05);
    EXPECT_THROW(bayes::energy_bfmi({1.0, 1.0}), std::invalid_argument);
}

TEST(BayesDiagnostics, SummaryAndConjugatePosterior) {
    const auto chains = ar1_chains(0.0, 2, 20000, 5);
    const auto s = bayes::summarize_draws(chains);
    EXPECT_NEAR(s.mean, 0.0, 0.03);
    EXPECT_NEAR(s.sd, 1.0, 0.02);
    EXPECT_NEAR(s.q05, -1.6449, 0.05);
    EXPECT_NEAR(s.q95, 1.6449, 0.05);
    EXPECT_NEAR(s.mcse, s.sd / std::sqrt(s.ess), 1e-12);

    const auto p = bayes::normal_mean_posterior({1.0, 3.0}, {1.0, 2.0}, 0.0, 1.0);
    // precision 1 + 1 + 1/4 = 2.25; weighted sum 0 + 1 + 3/4 = 1.75.
    EXPECT_NEAR(p.mean, 1.75 / 2.25, 1e-12);
    EXPECT_NEAR(p.sd, std::sqrt(1.0 / 2.25), 1e-12);
    const auto flat = bayes::normal_mean_posterior({1.0, 3.0}, {1.0, 1.0}, 0.0, INFINITY);
    EXPECT_NEAR(flat.mean, 2.0, 1e-12);
    EXPECT_NEAR(flat.sd, std::sqrt(0.5), 1e-12);
}

TEST(BayesDiagnostics, NormalLpdfWithParameterScale) {
    bayes::AutodiffModel model([](Tape&, const std::vector<Var>& p) {
        return bayes::normal_lpdf(p[0], p[1], p[2]);
    });
    const std::vector<double> at{0.7, 0.2, 1.3};
    EXPECT_NEAR(model.evaluate(at), bayes::normal_lpdf(0.7, 0.2, 1.3), 1e-12);
    const auto g = model.gradient(at);
    const double z = 0.5 / 1.3;
    EXPECT_NEAR(g[0], -z / 1.3, 1e-10);
    EXPECT_NEAR(g[1], z / 1.3, 1e-10);
    EXPECT_NEAR(g[2], -1.0 / 1.3 + z * z / 1.3, 1e-10);
}

TEST(BayesDiagnostics, NutsRecordsPerDrawDiagnosticsAndChains) {
    bayes::AutodiffModel model([](Tape&, const std::vector<Var>& p) {
        return bayes::normal_lpdf(p[0], 0.0, 1.0) + bayes::normal_lpdf(p[1], 0.0, 3.0);
    });
    bayes::NUTSOptions opts;
    opts.num_warmup = 300;
    opts.num_samples = 400;
    const auto chains = bayes::NUTS(opts).sample_chains(model, {0.0, 0.0}, 3);
    ASSERT_EQ(chains.size(), 3U);
    for (const auto& c : chains) {
        ASSERT_EQ(c.divergent.size(), 400U);
        ASSERT_EQ(c.energy.size(), 400U);
        std::size_t divergences = 0;
        for (std::size_t i = 0; i < 400; ++i) {
            EXPECT_GE(c.n_leapfrog[i], 1);
            EXPECT_LE(c.n_leapfrog[i], (1 << c.tree_depth[i]) - 1);
            divergences += static_cast<std::size_t>(c.divergent[i]);
        }
        EXPECT_LE(divergences, c.num_divergences);
    }
    EXPECT_NE(chains[0].samples[10][0], chains[1].samples[10][0]);
    const auto s = bayes::summarize_draws(bayes::chains_of(chains, 1));
    EXPECT_NEAR(s.sd, 3.0, 0.4);
    EXPECT_LT(s.rhat, 1.05);
    EXPECT_GT(bayes::energy_bfmi(chains[0].energy), 0.3);
    EXPECT_THROW(bayes::NUTS(opts).sample_chains(model, {0.0, 0.0}, 0), std::invalid_argument);
}

TEST(BayesNormalModels, EightSchoolsMatchesIndependentQuadrature) {
    const std::vector<double> y{28, 8, -3, 7, -1, 1, 18, 12}, sigma{15, 10, 16, 11, 9, 11, 10, 18};
    // References from scipy.integrate.quad over mu and tau.
    EXPECT_NEAR(bayes::hierarchical_normal_log_marginal(y, sigma, 0.0, 5.0, 0.0), -30.8442381260, 1e-8);
    EXPECT_NEAR(bayes::hierarchical_normal_log_marginal(y, sigma, 0.0, 5.0, 3.0), -30.9260835343, 1e-8);
    const auto post = bayes::hierarchical_normal_posterior(y, sigma, 0.0, 5.0, 5.0, 20000);
    EXPECT_NEAR(post.log_evidence, -31.3113312056, 1e-4);
    EXPECT_NEAR(post.mu_mean, 4.3968, 1e-3);
    EXPECT_NEAR(post.mu_sd, 3.3177, 1e-3);
    EXPECT_NEAR(post.tau_mean, 3.5977, 2e-3);
    EXPECT_NEAR(post.tau_sd, 3.2200, 2e-3);
    EXPECT_NEAR(post.tau_q05, 0.2464, 2e-3);
    EXPECT_NEAR(post.tau_q50, 2.7488, 2e-3);
    EXPECT_NEAR(post.tau_q95, 9.8429, 5e-3);
    EXPECT_NEAR(post.theta_mean[0], 6.212, 2e-3);
    EXPECT_NEAR(post.theta_sd[0], 5.593, 2e-3);
    EXPECT_NEAR(post.theta_mean[2], 3.927, 2e-3);
    for (const double b : post.shrinkage) {
        EXPECT_GT(b, 0.0);
        EXPECT_LT(b, 1.0);
    }
    EXPECT_THROW(bayes::hierarchical_normal_posterior(y, {1.0}, 0.0, 5.0, 5.0), std::invalid_argument);
}

TEST(BayesHmc, LeapfrogGaussianAcceptanceMatchesMonteCarlo) {
    // Monte Carlo references (2e6 equilibrium starts, numpy) for L = 10, s = 3.15736.
    const double s = 3.157360445642214;
    EXPECT_NEAR(bayes::leapfrog_gaussian_acceptance(1.0, 10, s), 0.99969, 1e-3);
    EXPECT_NEAR(bayes::leapfrog_gaussian_acceptance(4.0, 10, s), 0.85217, 2e-3);
    EXPECT_NEAR(bayes::leapfrog_gaussian_acceptance(6.0, 10, s), 0.94705, 2e-3);
    EXPECT_NEAR(bayes::leapfrog_gaussian_acceptance(6.2, 10, s), 0.35750, 2e-3);
    // Beyond the stability limit 2s the trajectory grows exponentially.
    EXPECT_LT(bayes::leapfrog_gaussian_acceptance(7.0, 10, s), 1e-3);
    EXPECT_THROW(bayes::leapfrog_gaussian_acceptance(1.0, 0, s), std::invalid_argument);
}

TEST(BayesDiagnostics, PosteriorHistogramIsADensity) {
    const auto h = bayes::posterior_histogram({0.1, 0.2, 0.6, 0.9, 1.0, 5.0}, 0.0, 1.0, 2);
    ASSERT_EQ(h.centers.size(), 2U);
    EXPECT_DOUBLE_EQ(h.width, 0.5);
    EXPECT_DOUBLE_EQ(h.centers[0], 0.25);
    // 2 of 6 draws in [0, 0.5), 3 in [0.5, 1]; 5.0 is outside but counts in the total.
    EXPECT_DOUBLE_EQ(h.density[0], 2.0 / (6 * 0.5));
    EXPECT_DOUBLE_EQ(h.density[1], 3.0 / (6 * 0.5));
    EXPECT_THROW(bayes::posterior_histogram({}, 0.0, 1.0, 2), std::invalid_argument);
}

TEST(BayesGridPosterior, CorrelatedGaussianMomentsAndEvidence) {
    // log density -x'Px/2 with P = [[2, 1], [1, 2]]: Z = 2 pi / sqrt(det P), covariance P^{-1}.
    bayes::LogDensity f([](const std::vector<double>& x) {
        return -0.5 * (2 * x[0] * x[0] + 2 * x[0] * x[1] + 2 * x[1] * x[1]) + 0.0;
    });
    const auto g = bayes::grid_posterior(f, {-8.0, -8.0}, {8.0, 8.0}, 400);
    EXPECT_NEAR(g.log_evidence, std::log(2 * M_PI / std::sqrt(3.0)), 1e-6);
    EXPECT_NEAR(g.mean[0], 0.0, 1e-9);
    EXPECT_NEAR(g.covariance[0][0], 2.0 / 3.0, 1e-4);
    EXPECT_NEAR(g.covariance[0][1], -1.0 / 3.0, 1e-4);
    EXPECT_NEAR(g.sd[1], std::sqrt(2.0 / 3.0), 1e-4);
    EXPECT_LT(g.edge_mass, 1e-10);
    // Marginal of x0 is N(0, 2/3).
    double peak = 0.0;
    for (const double v : g.marginal_density[0]) peak = std::max(peak, v);
    EXPECT_NEAR(peak, 1.0 / std::sqrt(2 * M_PI * 2.0 / 3.0), 2e-3);
    EXPECT_THROW(bayes::grid_posterior(f, {0.0}, {1.0, 2.0}, 10), std::invalid_argument);
}

TEST(BayesRwm, CovarianceAdaptationLearnsScalesAndMixesBetter) {
    // Independent N(0, 10^2) x N(0, 0.1^2): an isotropic walk is limited by the small scale.
    bayes::LogDensity f([](const std::vector<double>& x) {
        return -0.5 * (x[0] * x[0] / 100.0 + x[1] * x[1] / 0.01);
    });
    bayes::RWMOptions o;
    o.num_warmup = 2000;
    o.num_samples = 4000;
    const auto plain = bayes::RandomWalkMetropolis(o).sample(f, {0.0, 0.0});
    EXPECT_TRUE(plain.proposal_covariance.empty());
    o.adapt_covariance = true;
    const auto shaped = bayes::RandomWalkMetropolis(o).sample(f, {0.0, 0.0});
    ASSERT_EQ(shaped.proposal_covariance.size(), 2U);
    EXPECT_NEAR(std::sqrt(shaped.proposal_covariance[1][1]), 0.1, 0.05);
    EXPECT_GT(shaped.proposal_covariance[0][0], 1.0);
    const double ess_plain = bayes::effective_sample_size({bayes::draws_of(plain.samples, 0)});
    const double ess_shaped = bayes::effective_sample_size({bayes::draws_of(shaped.samples, 0)});
    EXPECT_GT(ess_shaped, 5 * ess_plain);
    EXPECT_NEAR(shaped.accept_rate, 0.234, 0.1);
}

TEST(BayesDistributions, LaplaceAndDrawMoments) {
    EXPECT_NEAR(bayes::laplace_lpdf(1.5, 1.0, 2.0), -std::log(4.0) - 0.25, 1e-12);
    bayes::AutodiffModel m([](Tape&, const std::vector<Var>& p) { return bayes::laplace_lpdf(3.0, p[0], 2.0); });
    EXPECT_NEAR(m.gradient({1.0})[0], 0.5, 1e-12);   // below the observation the slope is +1/b
    EXPECT_NEAR(m.gradient({4.0})[0], -0.5, 1e-12);
    const std::vector<std::vector<double>> s{{1, 2}, {3, 6}, {5, 10}};
    const auto mu = bayes::draws_mean(s);
    EXPECT_DOUBLE_EQ(mu[1], 6.0);
    const auto c = bayes::draws_covariance(s);
    EXPECT_DOUBLE_EQ(c[0][0], 4.0);
    EXPECT_DOUBLE_EQ(c[0][1], 8.0);
    EXPECT_THROW(bayes::draws_covariance({{1.0}}), std::invalid_argument);
}

TEST(BayesImportance, GaussianEfficiencyMatchesSimulation) {
    EXPECT_DOUBLE_EQ(bayes::gaussian_importance_efficiency(1.0, 3), 1.0);
    EXPECT_DOUBLE_EQ(bayes::gaussian_importance_efficiency(0.7, 1), 0.0);
    bayes::LogDensity target([](const std::vector<double>& x) { return -0.5 * (x[0] * x[0] + x[1] * x[1]); });
    bayes::ImportanceSamplingOptions o;
    o.num_samples = 40000;
    const auto r = bayes::ImportanceSampling(o).sample(target, {0.0, 0.0}, {{2.25, 0.0}, {0.0, 2.25}});
    EXPECT_NEAR(r.effective_sample_size / 40000.0, bayes::gaussian_importance_efficiency(1.5, 2), 0.03);
    EXPECT_NEAR(r.log_evidence, std::log(2 * M_PI), 0.02);
}
