#include <gtest/gtest.h>

#include <datamunge/algorithms/adaboost.hpp>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

using datamunge::algorithms::adaboost_predict;
using datamunge::algorithms::adaboost_train;

namespace {

// A diagonally separable set (label = sign(x0 + x1)) that NO single axis-aligned
// stump can classify perfectly, but a weighted vote of stumps can.
void diagonal_data(std::vector<std::vector<double>>& X, std::vector<int>& y) {
    std::mt19937_64                        rng(99);
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    for (int i = 0; i < 200; ++i) {
        double a = u(rng), b = u(rng);
        if (std::abs(a + b) < 0.3) continue;  // keep a clear margin around the boundary
        X.push_back({a, b});
        y.push_back((a + b > 0.0) ? 1 : -1);
    }
}

}  // namespace

TEST(AdaBoost, DrivesTrainingErrorDown) {
    std::vector<std::vector<double>> X;
    std::vector<int>                 y;
    diagonal_data(X, y);

    auto model = adaboost_train(X, y, 60);
    ASSERT_GE(model.stumps.size(), 2u);  // needed more than one weak learner

    // Every weak learner beats chance.
    for (double e : model.weak_error) EXPECT_LT(e, 0.5);

    // Ensemble training error drops well below the first weak learner's and is near zero.
    ASSERT_FALSE(model.training_error.empty());
    EXPECT_LE(model.training_error.back(), model.training_error.front());
    EXPECT_LE(model.training_error.back(), 0.02);
}

TEST(AdaBoost, PredictsClearPoints) {
    std::vector<std::vector<double>> X;
    std::vector<int>                 y;
    diagonal_data(X, y);
    auto model = adaboost_train(X, y, 60);
    EXPECT_EQ(adaboost_predict(model, {3.0, 3.0}), 1);
    EXPECT_EQ(adaboost_predict(model, {-3.0, -3.0}), -1);
    EXPECT_EQ(adaboost_predict(model, {2.5, 1.0}), 1);
    EXPECT_EQ(adaboost_predict(model, {-2.5, -1.0}), -1);
}

TEST(AdaBoost, SingleStumpSeparableIsSolvedFast) {
    // Label depends only on x0 -> the first stump already separates; error 0 in one round.
    std::vector<std::vector<double>> X = {{-2, 5}, {-1, -5}, {1, 3}, {2, -3}, {3, 9}, {-3, 0}};
    std::vector<int>                 y = {-1, -1, 1, 1, 1, -1};
    auto                             model = adaboost_train(X, y, 20);
    ASSERT_FALSE(model.training_error.empty());
    EXPECT_DOUBLE_EQ(model.training_error.back(), 0.0);
    for (std::size_t i = 0; i < X.size(); ++i) EXPECT_EQ(adaboost_predict(model, X[i]), y[i]);
}
