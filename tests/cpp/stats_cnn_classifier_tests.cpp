#include <gtest/gtest.h>

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/cnn_classifier.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using datamunge::linalg::DenseMatrix;
using datamunge::stats::CNNClassifier;
using datamunge::stats::CNNClassifierOptions;

namespace {

constexpr std::size_t H = 12, W = 12;

DenseMatrix<double> make_bar(bool horizontal, std::size_t pos, std::mt19937_64& rng) {
  std::normal_distribution<double> noise(0.0, 0.08);
  DenseMatrix<double> img(H, W, 0.0);
  for (std::size_t i = 0; i < H; ++i)
    for (std::size_t j = 0; j < W; ++j) {
      double v = 0.0;
      if (horizontal && (i == pos || i == pos + 1)) v = 1.0;
      if (!horizontal && (j == pos || j == pos + 1)) v = 1.0;
      img(i, j) = std::clamp(v + noise(rng), 0.0, 1.0);
    }
  return img;
}

// Random-position bars, `per` of each class.
void make_bars(std::size_t per, unsigned seed, std::vector<DenseMatrix<double>>& imgs,
               std::vector<std::string>& labels) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::size_t> pos(0, H - 2);
  for (std::size_t i = 0; i < per; ++i) {
    imgs.push_back(make_bar(true, pos(rng), rng));
    labels.push_back("horizontal");
    imgs.push_back(make_bar(false, pos(rng), rng));
    labels.push_back("vertical");
  }
}

} // namespace

TEST(CNNClassifier, ClassifiesBars) {
  std::vector<DenseMatrix<double>> imgs;
  std::vector<std::string> labels;
  make_bars(80, 1, imgs, labels);
  CNNClassifier cnn(imgs, labels, CNNClassifierOptions{6, 3, 2, 16, 0.05, 25, 16, 0.0, 1});
  EXPECT_EQ(cnn.classes().size(), 2u);
  EXPECT_EQ(cnn.image_height(), H);
  EXPECT_EQ(cnn.image_width(), W);
  EXPECT_GT(cnn.training_accuracy(), 0.9);

  std::vector<DenseMatrix<double>> test_i;
  std::vector<std::string> test_l;
  make_bars(30, 999, test_i, test_l);
  EXPECT_GT(cnn.accuracy(test_i, test_l), 0.85);
}

TEST(CNNClassifier, ParameterCountAndFilters) {
  std::vector<DenseMatrix<double>> imgs;
  std::vector<std::string> labels;
  make_bars(20, 2, imgs, labels);
  CNNClassifier cnn(imgs, labels, CNNClassifierOptions{4, 3, 2, 8, 0.05, 5, 16, 0.0, 1});
  // conv 4*9+4=40; flat=4*5*5=100; W1 8*100+8=808; W2 2*8+2=18; total 866.
  EXPECT_EQ(cnn.n_parameters(), 866u);
  ASSERT_EQ(cnn.filters().size(), 4u);
  for (const auto& f : cnn.filters()) {
    EXPECT_EQ(f.rows(), 3u);
    EXPECT_EQ(f.cols(), 3u);
  }
}

TEST(CNNClassifier, GlobalPoolingGeneralizesToShiftedPositions) {
  // Train with bars only in the top-left; test with bars only in the bottom-right.
  std::mt19937_64 rng(7);
  std::uniform_int_distribution<std::size_t> top(0, 3), bottom(8, 10);
  std::vector<DenseMatrix<double>> tr, te;
  std::vector<std::string> trl, tel;
  for (std::size_t i = 0; i < 120; ++i) {
    tr.push_back(make_bar(true, top(rng), rng));    trl.push_back("horizontal");
    tr.push_back(make_bar(false, top(rng), rng));   trl.push_back("vertical");
    te.push_back(make_bar(true, bottom(rng), rng)); tel.push_back("horizontal");
    te.push_back(make_bar(false, bottom(rng), rng)); tel.push_back("vertical");
  }
  CNNClassifier local(tr, trl, CNNClassifierOptions{8, 3, 2, 16, 0.05, 40, 16, 0.0, 1});
  CNNClassifier global(tr, trl, CNNClassifierOptions{8, 3, 10, 16, 0.05, 40, 16, 0.0, 1});
  // Both fit training; only global pooling transfers to unseen positions.
  EXPECT_GT(local.training_accuracy(), 0.95);
  EXPECT_GT(global.training_accuracy(), 0.95);
  EXPECT_GT(global.accuracy(te, tel), 0.7);
  EXPECT_GT(global.accuracy(te, tel), local.accuracy(te, tel) + 0.3);
}

TEST(CNNClassifier, TrainingReducesLossAndProbabilitiesAreValid) {
  std::vector<DenseMatrix<double>> imgs;
  std::vector<std::string> labels;
  make_bars(40, 3, imgs, labels);
  CNNClassifier cnn(imgs, labels, CNNClassifierOptions{4, 3, 2, 8, 0.05, 20, 16, 0.0, 1});
  ASSERT_EQ(cnn.loss_curve().size(), 20u);
  EXPECT_LT(cnn.loss_curve().back(), cnn.loss_curve().front());

  const auto detail = cnn.predict_detail(imgs);
  ASSERT_EQ(detail.probabilities.size(), imgs.size());
  for (const auto& p : detail.probabilities) {
    ASSERT_EQ(p.size(), 2u);
    double s = 0.0;
    for (double v : p) { EXPECT_GE(v, 0.0); s += v; }
    EXPECT_NEAR(s, 1.0, 1e-9);
  }
}

TEST(CNNClassifier, DeterministicGivenSeed) {
  std::vector<DenseMatrix<double>> imgs;
  std::vector<std::string> labels;
  make_bars(30, 4, imgs, labels);
  CNNClassifier a(imgs, labels, CNNClassifierOptions{4, 3, 2, 8, 0.05, 10, 16, 0.0, 1});
  CNNClassifier b(imgs, labels, CNNClassifierOptions{4, 3, 2, 8, 0.05, 10, 16, 0.0, 1});
  EXPECT_EQ(a.fitted_classes(), b.fitted_classes());
  EXPECT_EQ(a.loss_curve(), b.loss_curve());
}

TEST(CNNClassifier, RejectsInvalidInput) {
  std::vector<DenseMatrix<double>> imgs;
  std::vector<std::string> labels;
  make_bars(10, 5, imgs, labels);

  EXPECT_THROW(CNNClassifier({}, {}), std::invalid_argument);
  std::vector<std::string> short_labels(labels.begin(), labels.end() - 1);
  EXPECT_THROW(CNNClassifier(imgs, short_labels), std::invalid_argument);
  EXPECT_THROW(CNNClassifier(imgs, labels, CNNClassifierOptions{4, 99, 2, 8, 0.05, 5, 16, 0.0, 1}),
               std::invalid_argument);  // kernel bigger than image
  EXPECT_THROW(CNNClassifier(imgs, labels, CNNClassifierOptions{4, 3, 2, 8, 0.05, 0, 16, 0.0, 1}),
               std::invalid_argument);  // zero epochs

  std::vector<std::string> one_class(labels.size(), "only");
  EXPECT_THROW(CNNClassifier(imgs, one_class), std::invalid_argument);
}

TEST(CNNClassifier, RelevanceConcentratesOnTheBar) {
  std::vector<DenseMatrix<double>> imgs;
  std::vector<std::string> labels;
  make_bars(100, 1, imgs, labels);
  CNNClassifier cnn(imgs, labels, CNNClassifierOptions{8, 3, 2, 32, 0.05, 40, 16, 0.0, 1});

  std::mt19937_64 rng(99);
  const std::size_t bar_row = 5;
  const auto hbar = make_bar(true, bar_row, rng);
  const auto R = cnn.relevance(hbar);
  ASSERT_EQ(R.rows(), H);
  ASSERT_EQ(R.cols(), W);

  double total = 0.0, on_bar = 0.0, min_val = 0.0;
  for (std::size_t i = 0; i < H; ++i)
    for (std::size_t j = 0; j < W; ++j) {
      total += R(i, j);
      min_val = std::min(min_val, R(i, j));
      if (i == bar_row || i == bar_row + 1) on_bar += R(i, j);
    }
  EXPECT_GE(min_val, -1e-9);          // z+ relevance is non-negative
  EXPECT_GT(total, 0.0);
  EXPECT_GT(on_bar / total, 0.6);     // most relevance sits on the bar's two rows

  // Explaining a specific class index works and differs from the predicted-class explanation only
  // in which class is targeted.
  const auto R0 = cnn.relevance(hbar, 0);
  EXPECT_EQ(R0.rows(), H);
}

TEST(CNNClassifier, ConfusionMatrixSumsToObservations) {
  std::vector<DenseMatrix<double>> imgs;
  std::vector<std::string> labels;
  make_bars(30, 6, imgs, labels);
  CNNClassifier cnn(imgs, labels, CNNClassifierOptions{4, 3, 2, 8, 0.05, 15, 16, 0.0, 1});
  const auto cm = cnn.confusion_matrix();
  ASSERT_EQ(cm.rows(), 2u);
  double total = 0.0;
  for (std::size_t r = 0; r < 2; ++r)
    for (std::size_t c = 0; c < 2; ++c) total += cm(r, c);
  EXPECT_EQ(total, static_cast<double>(imgs.size()));
}
