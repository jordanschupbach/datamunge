#include <datamunge/bayes/advi.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {
using namespace datamunge::bayes;

// Bivariate normal with means (1, -2), SDs (1, 2) and correlation 0.8.
struct Bivariate : datamunge::optim::DifferentiableFunction {
    // Precision matrix of [[1, 1.6], [1.6, 4]].
    const double det = 4 - 1.6 * 1.6, p00 = 4 / det, p01 = -1.6 / det, p11 = 1 / det;
    double evaluate(const std::vector<double>& t) override {
        const double a = t[0] - 1, b = t[1] + 2;
        return -0.5 * (p00 * a * a + 2 * p01 * a * b + p11 * b * b);
    }
    std::vector<double> gradient(const std::vector<double>& t) override {
        const double a = t[0] - 1, b = t[1] + 2;
        return {-(p00 * a + p01 * b), -(p01 * a + p11 * b)};
    }
};

// Log-normal on theta > 0: log theta ~ N(m, s^2).
struct LogNormal : datamunge::optim::DifferentiableFunction {
    double m = 0.7, s = 0.4;
    double evaluate(const std::vector<double>& t) override {
        const double u = std::log(t[0]);
        return -u - 0.5 * (u - m) * (u - m) / (s * s);
    }
    std::vector<double> gradient(const std::vector<double>& t) override {
        const double u = std::log(t[0]);
        return {(-1 - (u - m) / (s * s)) / t[0]};
    }
};

// Logit-normal on (2, 5): logit((theta - 2) / 3) ~ N(m, s^2).
struct LogitNormal : datamunge::optim::DifferentiableFunction {
    double m = -0.5, s = 0.8;
    double evaluate(const std::vector<double>& t) override {
        const double p = (t[0] - 2) / 3, u = std::log(p / (1 - p));
        return -std::log(p) - std::log(1 - p) - 0.5 * (u - m) * (u - m) / (s * s);
    }
    std::vector<double> gradient(const std::vector<double>& t) override {
        const double p = (t[0] - 2) / 3, u = std::log(p / (1 - p));
        const double dlogp = -1 / p + 1 / (1 - p) - (u - m) / (s * s) * (1 / p + 1 / (1 - p));
        return {dlogp / 3};
    }
};

const AdviOptions precise{.gradient_samples = 10, .max_iterations = 20000, .tolerance = 1e-4, .seed = 11};
} // namespace

TEST(Advi, FullRankRecoversAGaussianPosterior) {
    Bivariate target;
    AdviOptions o = precise;
    o.family = AdviFamily::full_rank;
    const auto r = advi(target, {0.0, 0.0}, o);
    EXPECT_NEAR(r.mean[0], 1, 0.05);
    EXPECT_NEAR(r.mean[1], -2, 0.1);
    const auto& L = r.cholesky;
    EXPECT_NEAR(L[0] * L[0], 1, 0.1);
    EXPECT_NEAR(L[2] * L[0], 1.6, 0.16);
    EXPECT_NEAR(L[2] * L[2] + L[3] * L[3], 4, 0.4);
    // log Z of the unnormalised density; the final ELBO is a 100-draw Monte Carlo estimate.
    EXPECT_NEAR(r.elbo, std::log(2 * 3.14159265358979 * std::sqrt(4 - 1.6 * 1.6)), 0.25);
}

TEST(Advi, MeanFieldMatchesConditionalVariances) {
    Bivariate target;
    const auto r = advi(target, {0.0, 0.0}, precise);
    EXPECT_NEAR(r.mean[0], 1, 0.05);
    EXPECT_NEAR(r.mean[1], -2, 0.1);
    // KL(q || p) optimum: variances 1 / precision diagonal = (1 - rho^2) sigma^2.
    EXPECT_NEAR(r.cholesky[0], 0.6, 0.04);
    EXPECT_NEAR(r.cholesky[3], 1.2, 0.08);
    EXPECT_EQ(r.cholesky[2], 0.0);
}

TEST(Advi, TransformsRecoverConstrainedTargets) {
    LogNormal ln;
    AdviOptions o = precise;
    o.supports = {ParameterSupport::positive()};
    const auto r = advi(ln, {1.0}, o);
    EXPECT_NEAR(r.mean[0], 0.7, 0.03);
    EXPECT_NEAR(r.cholesky[0], 0.4, 0.03);
    for (const auto& d : r.draws)
        EXPECT_GT(d[0], 0);
    LogitNormal lg;
    o.supports = {{SupportKind::interval, 2.0, 5.0}};
    const auto q = advi(lg, {3.0}, o);
    EXPECT_NEAR(q.mean[0], -0.5, 0.05);
    EXPECT_NEAR(q.cholesky[0], 0.8, 0.05);
    for (const auto& d : q.draws) {
        EXPECT_GT(d[0], 2);
        EXPECT_LT(d[0], 5);
    }
}

TEST(Advi, IsReproducibleAndValidates) {
    Bivariate target;
    const auto a = advi(target, {0.0, 0.0}, {.seed = 4}), b = advi(target, {0.0, 0.0}, {.seed = 4});
    EXPECT_EQ(a.mean, b.mean);
    EXPECT_EQ(a.elbo_history, b.elbo_history);
    EXPECT_GT(a.step_size, 0);
    EXPECT_THROW(advi(target, {}, {}), std::invalid_argument);
    EXPECT_THROW(advi(target, {0.0, 0.0}, {.supports = {ParameterSupport::positive()}}), std::invalid_argument);
    EXPECT_THROW(advi(target, {0.0, 0.0}, {.supports = {ParameterSupport::positive(), {}}}),
                 std::invalid_argument);
    EXPECT_THROW(advi(target, {0.0, 0.0}, {.gradient_samples = 0}), std::invalid_argument);
}
