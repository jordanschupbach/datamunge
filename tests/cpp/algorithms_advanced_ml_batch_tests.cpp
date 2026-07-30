#include <gtest/gtest.h>

#include <datamunge/algorithms/alopex.hpp>
#include <datamunge/algorithms/rvm.hpp>
#include <datamunge/algorithms/structured_svm.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using datamunge::algorithms::alopex_minimize;
using datamunge::algorithms::AlopexParameters;
using datamunge::algorithms::rvm_predict;
using datamunge::algorithms::rvm_train;
using datamunge::algorithms::structured_svm_predict;
using datamunge::algorithms::structured_svm_train;
using datamunge::algorithms::StructuredSVMParameters;

TEST(Alopex, MinimizesQuadraticBowl) {
    const std::vector<double> target = {1.5, -2.0, 0.5};
    auto cost = [&](const std::vector<double>& w) {
        double s = 0.0;
        for (std::size_t i = 0; i < w.size(); ++i) s += (w[i] - target[i]) * (w[i] - target[i]);
        return s;
    };
    AlopexParameters p;
    p.step        = 0.02;
    p.temperature = 0.0;  // auto-scale from correlations
    p.iterations  = 4000;
    p.seed        = 1;
    auto r = alopex_minimize(cost, 3, {0.0, 0.0, 0.0}, p);
    EXPECT_LT(r.best_cost, 0.05);
    for (std::size_t i = 0; i < 3; ++i) EXPECT_NEAR(r.best_parameters[i], target[i], 0.2);
}

TEST(RVM, FitsSineSparsely) {
    std::vector<std::vector<double>> X;
    std::vector<double>              t;
    for (int i = 0; i <= 40; ++i) {
        double x = 2.0 * M_PI * i / 40.0;
        X.push_back({x});
        t.push_back(std::sin(x));
    }
    auto model = rvm_train(X, t, 0.5, 300, 1e9, 1e-4);

    double se = 0.0;
    for (std::size_t i = 0; i < X.size(); ++i) { double e = rvm_predict(model, X[i]) - t[i]; se += e * e; }
    EXPECT_LT(std::sqrt(se / X.size()), 0.1);
    // Sparsity: far fewer relevance vectors than data points.
    EXPECT_LT(model.num_relevance_vectors, X.size());
    EXPECT_GT(model.num_relevance_vectors, 0u);
}

TEST(StructuredSVM, SeparatesThreeClasses) {
    std::mt19937_64                  rng(3);
    std::normal_distribution<double> j(0.0, 0.4);
    std::vector<std::vector<double>> X;
    std::vector<int>                 y;
    const std::vector<std::vector<double>> c = {{0, 0}, {6, 0}, {3, 6}};
    for (int cls = 0; cls < 3; ++cls)
        for (int i = 0; i < 60; ++i) { X.push_back({c[cls][0] + j(rng), c[cls][1] + j(rng)}); y.push_back(cls); }

    StructuredSVMParameters p;
    p.lambda = 0.01;
    p.epochs = 50;
    p.seed   = 1;
    auto model = structured_svm_train(X, y, 3, p);

    std::size_t correct = 0;
    for (std::size_t i = 0; i < X.size(); ++i) correct += (static_cast<int>(structured_svm_predict(model, X[i])) == y[i]);
    EXPECT_GT(static_cast<double>(correct) / X.size(), 0.95);
    // Clear test points land in the right class.
    EXPECT_EQ(structured_svm_predict(model, {0, 0}), 0u);
    EXPECT_EQ(structured_svm_predict(model, {6, 0}), 1u);
    EXPECT_EQ(structured_svm_predict(model, {3, 6}), 2u);
}
