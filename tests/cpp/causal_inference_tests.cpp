#include <datamunge/bayes/discrete_network.hpp>
#include <datamunge/causal/causal.hpp>

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <random>

namespace {
namespace causal = datamunge::causal;
namespace bayes = datamunge::bayes;
using Matrix = datamunge::linalg::DenseMatrix<double>;
using Cases = datamunge::linalg::DenseMatrix<std::size_t>;

TEST(CausalDAG, SeparatesColliderAndFindsBackdoorAdjustment) {
    const causal::DAG collider(4, {{0, 2}, {1, 2}, {2, 3}});
    EXPECT_TRUE(collider.d_separated(0, 1));
    EXPECT_FALSE(collider.d_separated(0, 1, {2}));
    EXPECT_FALSE(collider.d_separated(0, 1, {3}));
    const causal::DAG confounded(4, {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {3, 2}});
    ASSERT_TRUE(confounded.backdoor_adjustment(1, 2));
    EXPECT_EQ(*confounded.backdoor_adjustment(1, 2), (std::vector<std::size_t>{0}));
    EXPECT_FALSE(confounded.is_backdoor_adjustment(1, 2, {0, 3}));
    EXPECT_FALSE(confounded.backdoor_adjustment(1, 2, 0));
    EXPECT_THROW((void)confounded.backdoor_adjustment(1, 2, 1, 1), std::length_error);
    EXPECT_THROW((causal::DAG(2, {{0, 1}, {1, 0}})), std::invalid_argument);
}

TEST(CausalEffects, CombinesSuppliedNuisancePredictions) {
    const auto result = causal::binary_ate({2, 5, 4, 9}, {0, 1, 0, 1},
                                           {.25, .25, .75, .75}, {1, 1, 3, 3}, {4, 4, 8, 8});
    EXPECT_DOUBLE_EQ(result.difference_in_means, 4.0);
    EXPECT_DOUBLE_EQ(result.g_computation, 4.0);
    EXPECT_DOUBLE_EQ(result.ipw, 2.5);
    EXPECT_DOUBLE_EQ(result.aipw, 4.0);
    EXPECT_DOUBLE_EQ(result.min_propensity, .25);
    EXPECT_THROW((void)causal::binary_ate({1, 2}, {0, 1}, {0, .5}, {1, 1}, {2, 2}), std::invalid_argument);
}

TEST(CausalEffects, FitsLogisticAndSeparateOutcomeRegressions) {
    const Matrix covariate(8, 1, {-1, -1, -1, -1, 1, 1, 1, 1});
    const auto fit = causal::fit_binary_ate({-1, -1, -1, 1, 1, 3, 3, 3},
                                             {0, 0, 0, 1, 0, 1, 1, 1}, covariate);
    EXPECT_NEAR(fit.difference_in_means, 3.0, 1e-12);
    EXPECT_NEAR(fit.g_computation, 2.0, 1e-12);
    EXPECT_NEAR(fit.ipw, 2.0, 1e-7);
    EXPECT_NEAR(fit.aipw, 2.0, 1e-12);
    EXPECT_NEAR(fit.min_propensity, .25, 1e-7);
    EXPECT_NEAR(fit.max_propensity, .75, 1e-7);
}

TEST(CausalEffects, FitsPanelAndInstrumentalEffect) {
    EXPECT_DOUBLE_EQ(causal::difference_in_differences({0, 4, 3, 8}, {1, 7, 4, 11}, {0, 1, 0, 1}), 2);
    const Matrix z(4, 1, {0, 0, 1, 1});
    const auto fit = causal::two_stage_least_squares({1, 3, 3, 7}, {0, 1, 1, 3}, z);
    EXPECT_NEAR(fit.effect, 2.0, 1e-12);
    EXPECT_NEAR(fit.first_stage_f, 1.8, 1e-12);
}

TEST(LinearSEM, PropagatesMomentsAndCounterfactualDisturbances) {
    const causal::DAG graph(3, {{0, 1}, {1, 2}, {0, 2}});
    Matrix data(8, 3);
    for (std::size_t i = 0; i < data.rows(); ++i) {
        const double x = i & 1 ? 1 : -1;
        const double e = i & 2 ? 1 : -1;
        const double f = i & 4 ? 1 : -1;
        data(i, 0) = x;
        data(i, 1) = 2 * x + e;
        data(i, 2) = 3 * x + 4 * data(i, 1) + f;
    }
    const auto fit = causal::fit_linear_sem(graph, data);
    EXPECT_NEAR(fit.total_effect(0, 2), 11, 1e-12);
    EXPECT_NEAR(fit.moments().covariance(0, 2), 11, 1e-12);
    EXPECT_NEAR(fit.moments().covariance(2, 2), 138, 1e-12);
    const auto intervened = fit.moments({{0, 2}});
    EXPECT_NEAR(intervened.mean[2], 22, 1e-12);
    EXPECT_DOUBLE_EQ(intervened.covariance(0, 2), 0);
    EXPECT_NEAR(intervened.covariance(2, 2), 17, 1e-12);
    const auto changed = fit.counterfactual({-1, -3, -16}, {{0, 2}});
    EXPECT_NEAR(changed[1], 3, 1e-12);
    EXPECT_NEAR(changed[2], 17, 1e-12);
    const auto sampled = fit.sample(4, 52, {{0, 23}});
    for (std::size_t i = 0; i < 4; ++i) EXPECT_DOUBLE_EQ(sampled(i, 0), 23);
}

TEST(CausalDiscovery, FindsColliderAndLeavesEquivalentDirectionUnresolved) {
    Matrix data(3500, 5);
    std::mt19937_64 engine(41);
    std::normal_distribution<double> normal;
    for (std::size_t row = 0; row < data.rows(); ++row) {
        data(row, 0) = normal(engine);
        data(row, 1) = normal(engine);
        data(row, 2) = data(row, 0) + data(row, 1) + normal(engine);
        data(row, 3) = data(row, 2) + normal(engine);
        data(row, 4) = .9 * data(row, 0) + normal(engine);
    }
    const auto fit = causal::pc_stable(data, {.alpha = .001});
    using Edge = std::pair<std::size_t, std::size_t>;
    EXPECT_EQ(fit.directed, (std::vector<Edge>{{0, 2}, {1, 2}, {2, 3}}));
    EXPECT_EQ(fit.undirected, (std::vector<Edge>{{0, 4}}));
    EXPECT_EQ(fit.orientation_conflicts, 0U);
    EXPECT_GT(causal::gaussian_ci_test(data, 0, 1).p_value, .001);
    EXPECT_LT(causal::gaussian_ci_test(data, 0, 1, {2}).p_value, .001);
    EXPECT_THROW((void)causal::pc_stable(data, {.max_tests = 1}), std::length_error);
}

TEST(DiscreteNetwork, DistinguishesEvidenceAndIntervention) {
    const causal::DAG graph(3, {{0, 1}, {0, 2}, {1, 2}});
    const bayes::DiscreteBayesianNetwork model(graph, {2, 2, 2},
        {{.5, .5}, {.8, .2, .2, .8}, {.9, .1, .7, .3, .5, .5, .3, .7}});
    EXPECT_NEAR(model.marginal(2)[1], .4, 1e-12);
    EXPECT_NEAR(model.marginal(2, {{1, 1}})[1], .62, 1e-12);
    EXPECT_NEAR(model.marginal(2, {}, {{1, 1}})[1], .5, 1e-12);
    EXPECT_NEAR(model.marginal(2, {{0, 0}}, {{1, 1}}, 2)[1], .3, 1e-12);
    EXPECT_THROW((void)model.marginal(2, {{1, 0}}, {{1, 1}}), std::domain_error);
    EXPECT_THROW((void)model.marginal(2, {}, {}, 7), std::length_error);
    EXPECT_DOUBLE_EQ(model.marginal(1, {}, {{1, 1}}, 1)[1], 1);
    const auto draws = model.sample(20, 379, {{1, 0}});
    for (std::size_t i = 0; i < draws.rows(); ++i) EXPECT_EQ(draws(i, 1), 0U);
    const bayes::DiscreteBayesianNetwork deterministic(causal::DAG(1), {2}, {{1, 0}});
    EXPECT_THROW((void)deterministic.marginal(0, {{0, 1}}), std::domain_error);
}

TEST(DiscreteNetwork, LearnsPosteriorTablesAndChowLiuTree) {
    const Cases data(4, 3, {0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1});
    const auto fit = bayes::fit_discrete_network(causal::DAG(3, {{0, 1}}), {2, 2, 2}, data);
    EXPECT_NEAR(fit.marginal(1, {{0, 1}})[1], .75, 1e-12);
    const auto tree = bayes::fit_chow_liu(data, {2, 2, 2}, 1);
    EXPECT_EQ(tree.graph().parents(0), (std::vector<std::size_t>{1}));
    EXPECT_TRUE(tree.graph().parents(1).empty());
    EXPECT_EQ(tree.graph().parents(2), (std::vector<std::size_t>{1}));
}
} // namespace
