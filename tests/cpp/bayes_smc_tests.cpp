#include <datamunge/bayes/smc.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::bayes;
using datamunge::random::SplitMix64;

const double log2pi = std::log(2 * std::numbers::pi);

// Normal means model: theta ~ N(0, tau^2 I_d), y_j ~ N(theta_j, 1). Exact evidence: y_j ~ N(0, 1 + tau^2).
struct NormalMeans {
    std::vector<double> y;
    double tau;
    double log_prior(const std::vector<double>& t) const {
        double s = 0;
        for (const double v : t)
            s += -0.5 * v * v / (tau * tau) - 0.5 * log2pi - std::log(tau);
        return s;
    }
    double log_lik(const std::vector<double>& t) const {
        double s = 0;
        for (std::size_t j = 0; j < y.size(); ++j)
            s += -0.5 * (y[j] - t[j]) * (y[j] - t[j]) - 0.5 * log2pi;
        return s;
    }
    std::vector<double> draw(SplitMix64& g) const {
        std::vector<double> t(y.size());
        for (double& v : t)
            v = tau * g.normal();
        return t;
    }
    double exact() const {
        double s = 0;
        for (const double v : y)
            s += -0.5 * v * v / (1 + tau * tau) - 0.5 * log2pi - 0.5 * std::log(1 + tau * tau);
        return s;
    }
};

SmcResult run(const NormalMeans& m, const SmcOptions& o) {
    return sequential_monte_carlo([&](const std::vector<double>& t) { return m.log_prior(t); },
                                  [&](const std::vector<double>& t) { return m.log_lik(t); },
                                  [&](SplitMix64& g) { return m.draw(g); }, o);
}
} // namespace

TEST(Smc, EvidenceAndPosteriorMeanMatchConjugateModel) {
    const NormalMeans m{{2.5, -1.0, 0.7, 3.2}, 3.0};
    const auto r = run(m, {.particles = 4000, .seed = 3});
    EXPECT_NEAR(r.log_evidence, m.exact(), 0.08);
    for (std::size_t j = 0; j < m.y.size(); ++j) {
        double mean = 0;
        for (std::size_t i = 0; i < r.particles.size(); ++i)
            mean += std::exp(r.log_weights[i]) * r.particles[i][j];
        EXPECT_NEAR(mean, m.y[j] * 9.0 / 10.0, 0.06) << j;
    }
    EXPECT_GT(r.stages.size(), 2u);
    EXPECT_DOUBLE_EQ(r.stages.back().temperature, 1.0);
    for (std::size_t s = 1; s < r.stages.size(); ++s)
        EXPECT_GT(r.stages[s].temperature, r.stages[s - 1].temperature);
}

TEST(Smc, RepeatedRunsAreUnbiasedForTheEvidence) {
    const NormalMeans m{{1.5, -2.0}, 2.0};
    double mean = 0;
    const int runs = 30;
    for (int k = 0; k < runs; ++k)
        mean += run(m, {.particles = 500, .seed = 100u + static_cast<unsigned>(k)}).log_evidence / runs;
    EXPECT_NEAR(mean, m.exact(), 0.06);
}

TEST(Smc, FixedScheduleAndReproducibility) {
    const NormalMeans m{{1.0}, 1.0};
    const SmcOptions o{.particles = 2000, .temperatures = {0.1, 0.3, 0.6, 1.0}, .seed = 9};
    const auto a = run(m, o), b = run(m, o);
    ASSERT_EQ(a.stages.size(), 4u);
    EXPECT_EQ(a.log_evidence, b.log_evidence);
    EXPECT_NEAR(a.log_evidence, m.exact(), 0.05);
}

TEST(Smc, RecoversBothModesOfABimodalPosterior) {
    // Prior N(0, 3^2); likelihood exp(-2 (theta^2 - 4)^2): symmetric modes at +-2.
    const auto lp = [](const std::vector<double>& t) {
        return -0.5 * t[0] * t[0] / 9 - 0.5 * log2pi - std::log(3.0);
    };
    const auto ll = [](const std::vector<double>& t) {
        const double u = t[0] * t[0] - 4;
        return -2 * u * u;
    };
    double exact = 0;
    for (double t = -8; t <= 8; t += 1e-4)
        exact += std::exp(lp({t}) + ll({t})) * 1e-4;
    const auto r = sequential_monte_carlo(lp, ll, [](SplitMix64& g) { return std::vector<double>{3 * g.normal()}; },
                                          {.particles = 4000, .seed = 5});
    double right = 0;
    for (std::size_t i = 0; i < r.particles.size(); ++i)
        right += std::exp(r.log_weights[i]) * (r.particles[i][0] > 0);
    EXPECT_NEAR(right, 0.5, 0.05);
    EXPECT_NEAR(r.log_evidence, std::log(exact), 0.06);
}

TEST(Smc, ValidatesInputs) {
    const NormalMeans m{{1.0}, 1.0};
    EXPECT_THROW(run(m, {.particles = 1}), std::invalid_argument);
    EXPECT_THROW(run(m, {.target_cess = 1.0}), std::invalid_argument);
    EXPECT_THROW(run(m, {.temperatures = {0.5, 0.9}}), std::invalid_argument);
    EXPECT_THROW(run(m, {.temperatures = {0.5, 0.4, 1.0}}), std::invalid_argument);
    EXPECT_THROW(sequential_monte_carlo([](const std::vector<double>&) { return 0.0; },
                                        [](const std::vector<double>&) { return std::nan(""); },
                                        [](SplitMix64&) { return std::vector<double>{0.0}; }),
                 std::runtime_error);
}

TEST(Smc, DiagonalProposalKeepsHighDimensionalEvidenceUnbiased) {
    NormalMeans m{std::vector<double>(20), 3.0};
    SplitMix64 g(4);
    for (double& v : m.y)
        v = 3 * g.normal() + g.normal();
    double mean = 0;
    const int runs = 20;
    for (int k = 0; k < runs; ++k)
        mean += run(m, {.particles = 1000, .mcmc_steps = 20, .proposal = SmcProposal::diagonal,
                        .seed = 300u + static_cast<unsigned>(k)})
                    .log_evidence /
                runs;
    EXPECT_NEAR(mean, m.exact(), 0.15);
}
