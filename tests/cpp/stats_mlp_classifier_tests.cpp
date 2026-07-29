#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/mlp_classifier.hpp>

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using datamunge::dstruct::DataFrame;
using datamunge::stats::MLPClassifier;
using datamunge::stats::MLPClassifierOptions;

namespace {

// Balanced, jittered XOR: 150 points at each of the four corners, labeled by parity.
DataFrame make_xor(std::size_t per = 150, unsigned seed = 1) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> j(0.0, 0.12);
  std::vector<double> x, y;
  std::vector<std::string> label;
  const int corners[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  for (auto& c : corners)
    for (std::size_t i = 0; i < per; ++i) {
      x.push_back(c[0] + j(rng));
      y.push_back(c[1] + j(rng));
      label.push_back((c[0] ^ c[1]) ? "B" : "A");
    }
  DataFrame d;
  d.add_column("x", x);
  d.add_column("y", y);
  d.add_column("class", label);
  return d;
}

// Two linearly separable Gaussian blobs.
DataFrame make_blobs(std::size_t per = 100, unsigned seed = 2) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> j(0.0, 0.4);
  std::vector<double> x, y;
  std::vector<std::string> label;
  for (std::size_t i = 0; i < per; ++i) {
    x.push_back(-2.0 + j(rng)); y.push_back(0.0 + j(rng)); label.push_back("left");
    x.push_back(2.0 + j(rng));  y.push_back(0.0 + j(rng)); label.push_back("right");
  }
  DataFrame d;
  d.add_column("x", x);
  d.add_column("y", y);
  d.add_column("class", label);
  return d;
}

} // namespace

TEST(MLPClassifier, HiddenLayerSolvesXorLinearCannot) {
  const auto xor_data = make_xor();
  MLPClassifier linear(xor_data, "class ~ x + y", MLPClassifierOptions{{}, "tanh", 0.1, 800, 0, 0.0, 7});
  MLPClassifier mlp(xor_data, "class ~ x + y", MLPClassifierOptions{{8}, "tanh", 0.1, 800, 0, 0.0, 7});
  EXPECT_LT(linear.training_accuracy(), 0.6);   // at chance -- cannot represent XOR
  EXPECT_NEAR(linear.loss_curve().back(), std::log(2.0), 0.02);
  EXPECT_GT(mlp.training_accuracy(), 0.95);      // a hidden layer solves it
}

TEST(MLPClassifier, ArchitectureAndParameterCount) {
  const auto xor_data = make_xor();
  MLPClassifier mlp(xor_data, "class ~ x + y", MLPClassifierOptions{{16}, "relu", 0.05, 50, 32, 0.0, 1});
  EXPECT_EQ(mlp.architecture(), (std::vector<std::size_t>{2, 16, 2}));
  // 2->16: 16*2 + 16 = 48; 16->2: 2*16 + 2 = 34; total 82.
  EXPECT_EQ(mlp.n_parameters(), 82u);

  MLPClassifier linear(xor_data, "class ~ x + y", MLPClassifierOptions{{}, "relu", 0.05, 50, 32, 0.0, 1});
  EXPECT_EQ(linear.architecture(), (std::vector<std::size_t>{2, 2}));
  EXPECT_EQ(linear.n_parameters(), 6u);  // 2*2 + 2
}

TEST(MLPClassifier, TrainingReducesLoss) {
  const auto blobs = make_blobs();
  MLPClassifier m(blobs, "class ~ x + y", MLPClassifierOptions{{8}, "relu", 0.05, 200, 32, 0.0, 3});
  ASSERT_EQ(m.loss_curve().size(), 200u);
  EXPECT_LT(m.loss_curve().back(), m.loss_curve().front());
  EXPECT_GT(m.training_accuracy(), 0.95);  // blobs are separable
}

TEST(MLPClassifier, ProbabilitiesAreValid) {
  const auto blobs = make_blobs();
  MLPClassifier m(blobs, "class ~ x + y", MLPClassifierOptions{{8}, "relu", 0.05, 200, 32, 0.0, 3});
  const auto detail = m.predict_detail(blobs);
  ASSERT_EQ(detail.probabilities.size(), blobs.nrows());
  for (const auto& probs : detail.probabilities) {
    ASSERT_EQ(probs.size(), m.classes().size());
    double sum = 0.0;
    for (double p : probs) {
      EXPECT_GE(p, 0.0);
      EXPECT_LE(p, 1.0);
      sum += p;
    }
    EXPECT_NEAR(sum, 1.0, 1e-9);
  }
}

TEST(MLPClassifier, DeterministicGivenSeed) {
  const auto blobs = make_blobs();
  MLPClassifier a(blobs, "class ~ x + y", MLPClassifierOptions{{8}, "relu", 0.05, 100, 32, 0.0, 5});
  MLPClassifier b(blobs, "class ~ x + y", MLPClassifierOptions{{8}, "relu", 0.05, 100, 32, 0.0, 5});
  EXPECT_EQ(a.fitted_classes(), b.fitted_classes());
  EXPECT_EQ(a.loss_curve(), b.loss_curve());
}

TEST(MLPClassifier, ClassifiesIris) {
  const auto iris = datamunge::datasets::iris();
  MLPClassifier m(iris, "Species ~ Petal.Length + Petal.Width",
                  MLPClassifierOptions{{16}, "relu", 0.05, 400, 32, 0.0, 42});
  EXPECT_EQ(m.classes().size(), 3u);
  EXPECT_GT(m.training_accuracy(), 0.9);
  const auto cm = m.confusion_matrix();
  EXPECT_EQ(cm.rows(), 3u);
  double diag = 0.0, total = 0.0;
  for (std::size_t r = 0; r < 3; ++r)
    for (std::size_t c = 0; c < 3; ++c) { total += cm(r, c); if (r == c) diag += cm(r, c); }
  EXPECT_EQ(total, static_cast<double>(iris.nrows()));
  EXPECT_GT(diag / total, 0.9);
}

TEST(MLPClassifier, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0, 6.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0, 3.0});
  frame.add_column("class", std::vector<std::string>{"a", "b", "a", "b", "a", "b"});
  MLPClassifier m(frame, "class ~ x + y", MLPClassifierOptions{{4}, "relu", 0.05, 50, 0, 0.0, 1});
  EXPECT_EQ(m.observations(), 5u);
}

TEST(MLPClassifier, RejectsInvalidOptions) {
  const auto blobs = make_blobs(20);
  EXPECT_THROW(MLPClassifier(blobs, "class ~ x + y", MLPClassifierOptions{{4}, "bogus", 0.05, 50, 32, 0.0, 1}),
               std::invalid_argument);
  EXPECT_THROW(MLPClassifier(blobs, "class ~ x + y", MLPClassifierOptions{{4}, "relu", 0.05, 0, 32, 0.0, 1}),
               std::invalid_argument);
  EXPECT_THROW(MLPClassifier(blobs, "class ~ x + y", MLPClassifierOptions{{4}, "relu", -1.0, 50, 32, 0.0, 1}),
               std::invalid_argument);
}

TEST(MLPClassifier, DecisionRegionsGuardsTwoPredictors) {
  const auto blobs = make_blobs();
  MLPClassifier m(blobs, "class ~ x + y", MLPClassifierOptions{{4}, "relu", 0.05, 50, 32, 0.0, 1});
  EXPECT_FALSE(m.plot_decision_regions("x", "y").series().empty());
  EXPECT_FALSE(m.plot_loss_curve().series().empty());

  const auto iris = datamunge::datasets::iris();
  MLPClassifier three(iris, "Species ~ Petal.Length + Petal.Width + Sepal.Length",
                      MLPClassifierOptions{{4}, "relu", 0.05, 20, 32, 0.0, 1});
  EXPECT_THROW(three.plot_decision_regions("Petal.Length", "Petal.Width"), std::invalid_argument);
}
