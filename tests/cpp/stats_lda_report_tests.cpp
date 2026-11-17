#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/lda.hpp>
#include <datamunge/stats/train_test_split.hpp>
using namespace datamunge;
TEST(LDAReport, TinySymmetricBayesScore) {
  dstruct::DataFrame d;
  d.add_column("x", std::vector<double>{-3, -2, -1, 1, 2, 3});
  d.add_column("y", std::vector<std::string>{"A", "A", "A", "B", "B", "B"});
  stats::LDA         model(d, "y ~ x");
  dstruct::DataFrame q;
  q.add_column("x", std::vector<double>{-1, 0, 1});
  auto p = model.predict_detail(q);
  EXPECT_EQ(p.class_label, (std::vector<std::string>{"A", "A", "B"}));
  // Unbiased pooled variance=(1+0+1+1+0+1)/(6-2)=1; muA=-2, muB=2.
  for (int i = 0; i < 3; ++i) {
    double expected = 1 / (1 + std::exp(-4. * (i - 1)));
    EXPECT_NEAR(p.posterior[i][1], expected, 1e-14);
  }
}
TEST(LDAReport, IndependentTwoFeatureCovarianceAndPosterior) {
  auto iris          = datasets::iris();
  auto [train, test] = stats::train_test_split(iris, .2, 42, "Species");
  stats::LDA model(train, "Species ~ Petal.Length + Petal.Width");
  EXPECT_EQ(train.nrows(), 120);
  EXPECT_EQ(test.nrows(), 30);
  auto p     = model.predict_detail(test);
  auto x     = train.numeric_column("Petal.Length").values(),
       y     = train.numeric_column("Petal.Width").values();
  double s00 = 0, s01 = 0, s11 = 0;
  for (std::size_t i = 0; i < train.nrows(); ++i) {
    auto label = train.string_column("Species").at(i);
    auto c =
        std::find(model.classes().begin(), model.classes().end(), label) - model.classes().begin();
    double a = x[i] - model.group_means()[c][0], b = y[i] - model.group_means()[c][1];
    s00 += a * a;
    s01 += a * b;
    s11 += b * b;
  }
  s00 /= 117;
  s01 /= 117;
  s11 /= 117;
  double det = s00 * s11 - s01 * s01;
  auto   tx  = test.numeric_column("Petal.Length").values(),
       ty    = test.numeric_column("Petal.Width").values();
  for (std::size_t i = 0; i < test.nrows(); ++i) {
    std::vector<double> score;
    for (std::size_t c = 0; c < 3; ++c) {
      auto   m  = model.group_means()[c];
      double dx = tx[i] - m[0], dy = ty[i] - m[1];
      score.push_back(
          -.5 * (s11 * dx * dx - 2 * s01 * dx * dy + s00 * dy * dy) / det +
          std::log(model.priors()[c]));
    }
    double mx = *std::max_element(score.begin(), score.end()), sum = 0;
    for (auto& v : score) {
      v = std::exp(v - mx);
      sum += v;
    }
    for (std::size_t c = 0; c < 3; ++c)
      EXPECT_NEAR(p.posterior[i][c], score[c] / sum, 2e-13);
  }
  auto before = model.group_means();
  auto labels = model.fitted_classes();
  model.predict_detail(test.head(1));
  EXPECT_EQ(model.group_means(), before);
  EXPECT_EQ(model.fitted_classes(), labels);
}
TEST(LDAReport, PriorsMissingRowsAndPredictorPlotContract) {
  auto       d = datasets::iris();
  stats::LDA m(
      d, "Species ~ Petal.Length + Petal.Width", {.priors = std::vector<double>{.8, .1, .1}});
  EXPECT_EQ(m.priors()[0], .8);
  dstruct::DataFrame q;
  q.add_column("Petal.Length", std::vector<double>{1.4});
  q.add_column("Petal.Width", std::vector<double>{.2});
  EXPECT_EQ(m.predict(q)[0], "setosa");
  stats::LDA full(d, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
  EXPECT_THROW(full.plot_decision_regions(), std::logic_error);
  EXPECT_THROW(
      stats::LDA(
          d, "Species ~ Petal.Length + Petal.Width", {.priors = std::vector<double>{0, .5, .5}}),
      std::invalid_argument);
}
