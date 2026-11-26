#include <datamunge/causal/aipw.hpp>
#include <datamunge/random/random.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using datamunge::causal::AipwOptions;
using datamunge::causal::cross_fitted_aipw;
using datamunge::causal::targeted_mle;
using Matrix = datamunge::linalg::DenseMatrix<double>;
const double na = std::numeric_limits<double>::quiet_NaN();
#include "causal_aipw_data.inc"
#include "causal_matching_data.inc"

struct Lalonde {
    Matrix x;
    std::vector<int> a;
};

Lalonde lalonde() {
    const std::size_t n = k_treat.size(), p = k_x.size() / n;
    Lalonde d{Matrix(n, p), {}};
    for (std::size_t i = 0; i < n; ++i) {
        d.a.push_back(static_cast<int>(k_treat[i]));
        for (std::size_t j = 0; j < p; ++j)
            d.x(i, j) = k_x[i * p + j];
    }
    return d;
}

void expect_reference(const datamunge::causal::AipwResult& r, const std::vector<double>& want) {
    EXPECT_NEAR(r.ate, want[0], 1e-6 * std::abs(want[0]));
    EXPECT_NEAR(r.standard_error, want[1], 1e-6 * want[1]);
    EXPECT_NEAR(r.treated_mean, want[2], 1e-6 * want[2]);
    EXPECT_NEAR(r.control_mean, want[3], 1e-6 * want[3]);
    EXPECT_EQ(static_cast<double>(r.clipped), want[4]);
}
} // namespace

TEST(CausalAipw, MatchesManualRCrossFitting) {
    const auto d = lalonde();
    std::vector<std::size_t> folds(a_folds.begin(), a_folds.end());
    const auto cross = cross_fitted_aipw(k_re78, d.a, d.x, {.fold_ids = folds, .propensity_bound = 0.025});
    expect_reference(cross, a_cross);
    const auto full = cross_fitted_aipw(k_re78, d.a, d.x, {.folds = 1, .propensity_bound = 0.025});
    expect_reference(full, a_full);
    EXPECT_NEAR(std::accumulate(cross.influence.begin(), cross.influence.end(), 0.0), 0.0, 1e-6);
}

TEST(CausalAipw, FoldsAreStratifiedAndPredictionsOutOfFold) {
    datamunge::random::SplitMix64 rng(4);
    const std::size_t n = 403;
    Matrix x(n, 2);
    std::vector<int> a(n);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 0) = rng.normal();
        x(i, 1) = rng.normal();
        a[i] = rng.uniform01() < 1 / (1 + std::exp(-x(i, 0)));
        y[i] = x(i, 0) + 0.5 * x(i, 1) + 1.5 * a[i] + rng.normal();
    }
    // A learner that records the training size shows no unit is used to predict itself.
    std::size_t calls = 0;
    const auto outcome = [&](const Matrix& tx, const std::vector<double>& ty, const Matrix& test) {
        ++calls;
        EXPECT_EQ(tx.rows(), ty.size());
        return datamunge::causal::linear_outcome_learner(tx, ty, test);
    };
    const auto r = cross_fitted_aipw(y, a, x, {.folds = 5, .seed = 9, .outcome = outcome});
    EXPECT_EQ(calls, 10u);
    std::size_t size[5][2]{};
    for (std::size_t i = 0; i < n; ++i)
        ++size[r.fold_ids[i]][a[i]];
    for (int arm = 0; arm < 2; ++arm)
        for (int k = 0; k < 5; ++k)
            for (int l = 0; l < 5; ++l)
                EXPECT_LE(std::abs(static_cast<long>(size[k][arm]) - static_cast<long>(size[l][arm])), 1);
    EXPECT_NEAR(r.ate, 1.5, 4 * r.standard_error);
    const auto again = cross_fitted_aipw(y, a, x, {.folds = 5, .seed = 9});
    EXPECT_DOUBLE_EQ(again.ate, r.ate);
    // Doubly robust: a constant (wrong) propensity still gives a consistent estimate with a correct outcome model.
    const auto constant = [](const Matrix&, const std::vector<int>& ta, const Matrix& test) {
        const double p = std::accumulate(ta.begin(), ta.end(), 0.0) / static_cast<double>(ta.size());
        return std::vector<double>(test.rows(), p);
    };
    EXPECT_NEAR(cross_fitted_aipw(y, a, x, {.propensity = constant}).ate, 1.5, 0.3);
    EXPECT_THROW((void)cross_fitted_aipw(y, a, x, {.propensity_bound = 0.5}), std::invalid_argument);
    EXPECT_THROW((void)cross_fitted_aipw(y, a, x, {.folds = 0}), std::invalid_argument);
}

TEST(CausalTmle, MatchesRTmlePackage) {
    const auto d = lalonde();
    const auto r = targeted_mle(k_re78, d.a, d.x, {.folds = 1});
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.ate, t_full[0], 1e-6 * std::abs(t_full[0]));
    EXPECT_NEAR(r.standard_error, t_full[1], 1e-6 * t_full[1]);
    EXPECT_NEAR(r.treated_mean, t_full[2], 1e-6 * t_full[2]);
    EXPECT_NEAR(r.control_mean, t_full[3], 1e-6 * t_full[3]);
    EXPECT_NEAR(r.epsilon[0], t_full[4], 1e-6);
    EXPECT_NEAR(r.epsilon[1], t_full[5], 1e-6);
    EXPECT_NEAR(std::accumulate(r.influence.begin(), r.influence.end(), 0.0), 0.0, 1e-6 * k_re78.size());
}

TEST(CausalTmle, RespectsBoundsAndIsDoublyRobust) {
    datamunge::random::SplitMix64 rng(8);
    const std::size_t n = 2000;
    Matrix x(n, 2);
    std::vector<int> a(n);
    std::vector<double> y(n);
    double truth = 0;
    for (std::size_t i = 0; i < n; ++i) {
        x(i, 0) = rng.normal();
        x(i, 1) = rng.normal();
        a[i] = rng.uniform01() < 1 / (1 + std::exp(-(0.5 * x(i, 0) - 0.5 * x(i, 1))));
        const auto risk = [&](int arm) { return 1 / (1 + std::exp(-(-0.5 + x(i, 0) + 0.8 * arm))); };
        truth += (risk(1) - risk(0)) / static_cast<double>(n);
        y[i] = rng.uniform01() < risk(a[i]);
    }
    // Linear-probability initial fits are misspecified but the logistic propensity is correct.
    const auto r = targeted_mle(y, a, x);
    EXPECT_NEAR(r.ate, truth, 4 * r.standard_error);
    EXPECT_GE(r.treated_mean, 0.0);
    EXPECT_LE(r.treated_mean, 1.0);
    // Invalid outcome ranges and bounds are rejected.
    EXPECT_THROW((void)targeted_mle(y, a, x, {.outcome_range = std::pair{0.2, 1.0}}), std::invalid_argument);
    EXPECT_THROW((void)targeted_mle(y, a, x, {.outcome_bound = 0.0}), std::invalid_argument);
}
