#include <gtest/gtest.h>

#include <datamunge/algorithms/online_linear_classifiers.hpp>

#include <cstddef>
#include <random>
#include <vector>

using datamunge::algorithms::perceptron_predict;
using datamunge::algorithms::perceptron_train;
using datamunge::algorithms::winnow_predict;
using datamunge::algorithms::winnow_train;

TEST(Perceptron, ConvergesOnLinearlySeparableData) {
    // Two clouds separated by the line x0 + x1 = 0.
    std::vector<std::vector<double>> X = {{2, 2}, {3, 1}, {2, 3}, {1, 2}, {-2, -1}, {-1, -2}, {-3, -1}, {-1, -3}};
    std::vector<int>                 y = {1, 1, 1, 1, -1, -1, -1, -1};

    auto model = perceptron_train(X, y, 100, 1.0);
    EXPECT_TRUE(model.converged);
    for (std::size_t i = 0; i < X.size(); ++i) EXPECT_EQ(perceptron_predict(model, X[i]), y[i]);
    // A clearly positive and clearly negative test point.
    EXPECT_EQ(perceptron_predict(model, {5.0, 5.0}), 1);
    EXPECT_EQ(perceptron_predict(model, {-5.0, -5.0}), -1);
}

TEST(Perceptron, MistakeBoundHolds) {
    // Novikoff: mistakes <= (R/gamma)^2. With a comfortable margin this is small.
    std::vector<std::vector<double>> X = {{3, 0}, {0, 3}, {-3, 0}, {0, -3}};
    std::vector<int>                 y = {1, 1, -1, -1};
    auto                             model = perceptron_train(X, y, 100, 1.0);
    EXPECT_TRUE(model.converged);
    EXPECT_LT(model.total_mistakes, 50u);
}

TEST(Winnow, LearnsDisjunctionAmongIrrelevantFeatures) {
    // Target concept: y = x[2] OR x[5], over 10 boolean features (8 irrelevant).
    std::mt19937_64                    rng(2024);
    std::bernoulli_distribution        bit(0.5);
    const std::size_t                  dim = 10;
    std::vector<std::vector<int>>      X;
    std::vector<int>                   y;
    for (int i = 0; i < 400; ++i) {
        std::vector<int> row(dim);
        for (std::size_t j = 0; j < dim; ++j) row[j] = bit(rng) ? 1 : 0;
        X.push_back(row);
        y.push_back((row[2] || row[5]) ? 1 : 0);
    }

    auto model = winnow_train(X, y, 2.0, -1.0, 100);
    EXPECT_TRUE(model.converged);
    // Perfect on training data once converged.
    for (std::size_t i = 0; i < X.size(); ++i) EXPECT_EQ(winnow_predict(model, X[i]), y[i]);

    // Held-out checks of the learned disjunction.
    std::vector<int> only2(dim, 0); only2[2] = 1;
    std::vector<int> only5(dim, 0); only5[5] = 1;
    std::vector<int> none(dim, 0);
    std::vector<int> irrelevant(dim, 0); irrelevant[0] = irrelevant[7] = 1;
    EXPECT_EQ(winnow_predict(model, only2), 1);
    EXPECT_EQ(winnow_predict(model, only5), 1);
    EXPECT_EQ(winnow_predict(model, none), 0);
    EXPECT_EQ(winnow_predict(model, irrelevant), 0);
}
