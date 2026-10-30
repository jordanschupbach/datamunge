#include <datamunge/algorithms/almeida_pineda.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/almeida_pineda_classifier.hpp>
#include <datamunge/stats/train_test_split.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace alg = datamunge::algorithms;

TEST(AlmeidaPineda, GradientMatchesFiniteDifferences) {
  auto net = alg::almeida_pineda_network(8, 3, {6, 7}, 0.4, 3);
  const std::vector<double> input{1.0, 0.5, -0.8}, target{0.6, -0.4};
  const alg::AlmeidaPinedaSettleOptions opt{5000, 1e-14, 0.5};
  const auto g = alg::almeida_pineda_gradient(net, input, target, opt);
  ASSERT_TRUE(g.converged);
  const double h = 1e-6;
  double worst = 0.0;
  for (std::size_t i = 3; i < 8; ++i)
    for (std::size_t j = 0; j < 8; ++j) {
      if (i == j) continue;
      auto plus = net, minus = net;
      plus.W[i][j] += h;
      minus.W[i][j] -= h;
      const double fd = (alg::almeida_pineda_gradient(plus, input, target, opt).error -
                         alg::almeida_pineda_gradient(minus, input, target, opt).error) / (2 * h);
      worst = std::max(worst, std::abs(fd - g.dW[i][j]));
    }
  EXPECT_LT(worst, 1e-7);
}

TEST(AlmeidaPineda, AdjointConvergesAtTheForwardRate) {
  auto net = alg::almeida_pineda_network(10, 2, {9}, 0.6, 7);
  const auto g = alg::almeida_pineda_gradient(net, {1.0, -1.0}, {0.5}, {5000, 1e-13, 1.0});
  ASSERT_TRUE(g.converged);
  // Asymptotic contraction factor from the residual ratios late in each relaxation.
  // Contraction per sweep over ten sweeps after the residual first drops below 1e-4.
  auto rate = [](const std::vector<double>& r) {
    std::size_t k = 0;
    while (r[k] >= 1e-4) ++k;
    return std::pow(r[k + 10] / r[k], 0.1);
  };
  EXPECT_NEAR(rate(g.forward.residuals), rate(g.adjoint_residuals), 0.02);
}

TEST(AlmeidaPineda, ClassifierLearnsIris) {
  const auto [train, test] = datamunge::stats::train_test_split(datamunge::datasets::iris(), 0.3, 1, "Species");
  const datamunge::stats::AlmeidaPinedaClassifier model(train, "Species ~ .", {.epochs = 150});
  const auto history = model.history();
  EXPECT_LT(history.double_at("loss", history.nrows() - 1), 0.5 * history.double_at("loss", 0));
  EXPECT_LT(model.error(test), 0.15);
  EXPECT_EQ(model.confusion(test).nrows(), 3U);
  EXPECT_EQ(model.inputs(test).front().size(), 5U);  // bias + 4 predictors
}
