#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/tabular_classification.hpp>
#include <numbers>
#include <numeric>
#include <random>
using namespace datamunge::nn;
using namespace datamunge::nn::study;
TEST(TabularClassification, OriginalGeneratorStreamsLabelsAndSplits) {
  auto                             x = xor_classification_data();
  std::mt19937_64                  rng(1);
  std::normal_distribution<double> noise(0., .12);
  std::size_t                      row = 0;
  for (auto c : std::vector<std::pair<int, int>>{{0, 0}, {0, 1}, {1, 0}, {1, 1}})
    for (int i = 0; i < 150; ++i) {
      EXPECT_EQ(x.features.at({std::int64_t(row), 0}), float(c.first + noise(rng)));
      EXPECT_EQ(x.features.at({std::int64_t(row), 1}), float(c.second + noise(rng)));
      EXPECT_EQ(x.labels.at({std::int64_t(row)}), c.first ^ c.second);
      ++row;
    }
  auto                             m = moon_classification_data();
  std::mt19937_64                  mrng(3);
  std::normal_distribution<double> jitter(0., .12);
  for (int i = 0; i < 250; ++i) {
    double              t = std::numbers::pi * i / 249.;
    std::vector<double> v{
        std::cos(t) + jitter(mrng),
        std::sin(t) + jitter(mrng),
        1. - std::cos(t) + jitter(mrng),
        .5 - std::sin(t) + jitter(mrng)};
    for (int k = 0; k < 4; ++k)
      EXPECT_EQ(m.features.at({2 * i + k / 2, k % 2}), float(v[k]));
    EXPECT_EQ(m.labels.at({2 * i}), 0);
    EXPECT_EQ(m.labels.at({2 * i + 1}), 1);
  }
  auto iris = iris_petal_classification_data();
  auto df   = datamunge::datasets::iris();
  EXPECT_EQ(iris.features.shape(), (Shape{150, 2}));
  for (int i = 0; i < 150; ++i) {
    EXPECT_EQ(iris.features.at({i, 0}), float(df.double_at("Petal.Length", i)));
    EXPECT_EQ(iris.features.at({i, 1}), float(df.double_at("Petal.Width", i)));
    EXPECT_EQ(iris.classes[std::size_t(iris.labels.at({i}))], df.string_at("Species", i));
  }
  for (auto [n, seed] : std::vector<std::pair<int, int>>{{500, 11}, {150, 2024}}) {
    auto                      s = tabular_classification_split(n, .7, seed);
    std::vector<std::int64_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937_64 r(seed);
    std::shuffle(ids.begin(), ids.end(), r);
    std::vector<std::int64_t> joined = s.training;
    joined.insert(joined.end(), s.heldout.begin(), s.heldout.end());
    EXPECT_EQ(joined, ids);
    EXPECT_EQ(s.training.size(), std::size_t(.7 * n));
  }
}
TEST(TabularClassification, TrainOnlyPopulationScalingAndDifferentiableInference) {
  auto training = from_vector(std::vector<double>{0, 5, 2, 5, 4, 5}, {3, 2});
  auto s        = fit_tabular_standardization(training);
  EXPECT_DOUBLE_EQ(s.mean.at({0, 0}), 2.);
  EXPECT_DOUBLE_EQ(s.mean.at({0, 1}), 5.);
  EXPECT_NEAR(s.scale.at({0, 0}), std::sqrt(8. / 3.), 1e-15);
  EXPECT_DOUBLE_EQ(s.scale.at({0, 1}), 1.);
  auto heldout = from_vector(std::vector<double>{100, 7}, {1, 2});
  heldout.requires_grad_(true);
  auto z = transform_tabular_features(heldout, s);
  EXPECT_NEAR(z.at({0, 0}), 98 / std::sqrt(8. / 3.), 1e-13);
  EXPECT_DOUBLE_EQ(z.at({0, 1}), 2.);
  auto g = grad(sum(z), {heldout});
  EXPECT_NEAR(g[0].at({0, 0}), 1 / std::sqrt(8. / 3.), 1e-15);
  EXPECT_EQ(s.mean.at({0, 0}), 2.);
  EXPECT_THROW(fit_tabular_standardization(zeros({0, 2})), std::invalid_argument);
  EXPECT_THROW(
      fit_tabular_standardization(from_vector(std::vector<std::int64_t>{1, 2}, {1, 2})),
      std::invalid_argument);
  EXPECT_THROW(transform_tabular_features(zeros({1, 3}, DType::Float64), s), std::invalid_argument);
}
TEST(TabularClassification, ActualGridCallsAndCompleteGuards) {
  auto d      = xor_classification_data(1);
  int  calls  = 0;
  auto logits = [&](const Tensor& q) {
    ++calls;
    EXPECT_EQ(q.shape(), (Shape{25, 2}));
    EXPECT_FALSE(q.requires_grad());
    return q;
  };
  auto p = tabular_decision_regions(d.features, d.labels, d.classes, logits, 5);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(p.series().back().x.size(), 2);
  EXPECT_EQ(p.series()[p.series().size()-2].x.size(), 2);
  EXPECT_THROW(
      tabular_decision_regions(d.features, d.labels, d.classes, logits, 1), std::invalid_argument);
  EXPECT_EQ(calls, 1);
  EXPECT_THROW(
      tabular_decision_regions(
          d.features, d.labels, d.classes, [](const auto&) { return zeros({1, 2}); }, 5),
      std::invalid_argument);
  EXPECT_THROW(xor_classification_data(0), std::invalid_argument);
  EXPECT_THROW(moon_classification_data(1), std::invalid_argument);
  EXPECT_THROW(tabular_classification_split(2, .1, 1), std::invalid_argument);
  EXPECT_THROW(tabular_classification_split(2, 1., 1), std::invalid_argument);
}
