#include <gtest/gtest.h>

#include <datamunge/algorithms/hopfield_network.hpp>
#include <datamunge/algorithms/rbf_network.hpp>
#include <datamunge/algorithms/self_organizing_map.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using datamunge::algorithms::hopfield_energy;
using datamunge::algorithms::hopfield_hamming;
using datamunge::algorithms::hopfield_recall;
using datamunge::algorithms::hopfield_train;
using datamunge::algorithms::rbf_predict;
using datamunge::algorithms::rbf_train;
using datamunge::algorithms::som_bmu;
using datamunge::algorithms::som_quantization_error;
using datamunge::algorithms::som_train;
using datamunge::algorithms::SOMParameters;

namespace {

std::vector<int> random_bipolar(std::mt19937_64& rng, std::size_t n) {
    std::bernoulli_distribution bit(0.5);
    std::vector<int>            p(n);
    for (auto& v : p) v = bit(rng) ? 1 : -1;
    return p;
}

}  // namespace

TEST(Hopfield, StoredPatternsAreFixedPoints) {
    std::mt19937_64               rng(1);
    std::vector<std::vector<int>> patterns;
    for (int i = 0; i < 3; ++i) patterns.push_back(random_bipolar(rng, 60));
    auto net = hopfield_train(patterns);
    for (const auto& p : patterns) {
        auto recalled = hopfield_recall(net, p, 100, 7);
        EXPECT_EQ(hopfield_hamming(recalled, p), 0u);
    }
}

TEST(Hopfield, RecoversFromCorruptedProbe) {
    std::mt19937_64               rng(2);
    std::vector<std::vector<int>> patterns;
    for (int i = 0; i < 3; ++i) patterns.push_back(random_bipolar(rng, 60));
    auto net = hopfield_train(patterns);

    // Corrupt the first pattern by flipping 4 of 60 bits.
    auto                                       probe = patterns[0];
    std::uniform_int_distribution<std::size_t> pos(0, 59);
    for (int f = 0; f < 4; ++f) {
        std::size_t i = pos(rng);
        probe[i]      = -probe[i];
    }
    EXPECT_GT(hopfield_hamming(probe, patterns[0]), 0u);
    EXPECT_LT(hopfield_energy(net, patterns[0]), hopfield_energy(net, probe));  // stored is lower energy

    auto recalled = hopfield_recall(net, probe, 100, 3);
    EXPECT_EQ(hopfield_hamming(recalled, patterns[0]), 0u);  // clean recovery
}

TEST(SelfOrganizingMap, LearnsTopologyOfSeparatedBlobs) {
    std::mt19937_64                        rng(5);
    std::normal_distribution<double>       jitter(0.0, 0.3);
    std::vector<std::vector<double>>       data;
    const std::vector<std::vector<double>> centers = {{0, 0}, {10, 0}, {0, 10}};
    for (const auto& c : centers)
        for (int i = 0; i < 60; ++i) data.push_back({c[0] + jitter(rng), c[1] + jitter(rng)});

    SOMParameters params;
    params.rows = params.cols = 5;
    params.iterations         = 3000;
    params.seed               = 1;
    auto som                  = som_train(data, params);

    // Quantization error should be small relative to the inter-blob distance (~10).
    EXPECT_LT(som_quantization_error(som, data), 1.0);

    // The three blob centers map to three distinct grid units (topology separation).
    auto b0 = som_bmu(som, {0, 0});
    auto b1 = som_bmu(som, {10, 0});
    auto b2 = som_bmu(som, {0, 10});
    EXPECT_TRUE(b0 != b1 && b0 != b2 && b1 != b2);
}

TEST(RBFNetwork, FitsSmoothNonlinearFunction) {
    std::vector<std::vector<double>> X;
    std::vector<double>              y;
    for (int i = 0; i <= 50; ++i) {
        double x = 2.0 * M_PI * i / 50.0;
        X.push_back({x});
        y.push_back(std::sin(x));
    }
    auto net = rbf_train(X, y, 10, 1e-6, 0.0, 30, 1);

    // Low training RMSE.
    double se = 0.0;
    for (std::size_t i = 0; i < X.size(); ++i) {
        double e = rbf_predict(net, X[i]) - y[i];
        se += e * e;
    }
    EXPECT_LT(std::sqrt(se / X.size()), 0.1);

    // Reasonable at interior points not exactly on the grid.
    EXPECT_NEAR(rbf_predict(net, {M_PI / 2.0}), std::sin(M_PI / 2.0), 0.15);
    EXPECT_NEAR(rbf_predict(net, {M_PI}), std::sin(M_PI), 0.15);
}
