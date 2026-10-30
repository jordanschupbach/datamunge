#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/train_test_split.hpp>
#include <datamunge/stats/xgboost_diagnostics.hpp>
#include <limits>
#include <numeric>
namespace stats = datamunge::stats;
using DF        = datamunge::dstruct::DataFrame;

TEST(XGBoostEvaluation, MatchesIndependentPrefixRefitsAndProbabilityScores) {
  const auto iris          = datamunge::datasets::iris();
  const auto [train, test] = stats::train_test_split(iris, 0.2, 42, "Species");
  stats::XGBoostClassifierOptions options;
  options.n_trees          = 5;
  options.max_depth        = 2;
  options.subsample        = 0.8;
  options.colsample_bytree = 0.5;
  const stats::XGBoostClassifier model(train, "Species ~ .", options);
  const auto                     curve = model.staged_evaluation(test);
  ASSERT_EQ(curve.nrows(), 6);
  EXPECT_NEAR(curve.double_at("mean_log_loss", 0), std::log(3.0), 1e-14);
  EXPECT_NEAR(curve.double_at("brier", 0), 2.0 / 3, 1e-14);
  EXPECT_NEAR(curve.double_at("accuracy", 0), 1.0 / 3, 1e-14);
  for (std::size_t m = 1; m <= 5; ++m) {
    auto prefix_options    = options;
    prefix_options.n_trees = m;
    const stats::XGBoostClassifier prefix(train, "Species ~ .", prefix_options);
    const auto                     probabilities = prefix.predict_detail(test);
    double                         correct = 0, loss = 0, brier = 0;
    for (std::size_t i = 0; i < test.nrows(); ++i) {
      const auto actual = test.string_at("Species", i);
      correct += probabilities.class_label[i] == actual;
      for (std::size_t k = 0; k < 3; ++k) {
        const bool   y = prefix.classes()[k] == actual;
        const double p = probabilities.probability[i][k];
        if (y)
          loss -= std::log(std::max(p, 1e-15));
        brier += (p - y) * (p - y);
      }
    }
    EXPECT_NEAR(curve.double_at("mean_log_loss", m), loss / test.nrows(), 1e-14);
    EXPECT_NEAR(curve.double_at("accuracy", m), correct / test.nrows(), 1e-14);
    EXPECT_NEAR(curve.double_at("brier", m), brier / test.nrows(), 1e-14);
  }
  const auto train_curve = model.staged_evaluation(train);
  for (std::size_t m = 1; m <= 5; ++m)
    EXPECT_DOUBLE_EQ(train_curve.double_at("mean_log_loss", m), model.training_deviance()[m - 1]);
  EXPECT_DOUBLE_EQ(
      model.evaluation(test).double_at("mean_log_loss", 0), curve.double_at("mean_log_loss", 5));
  const auto confusion = model.confusion(test);
  double     total = 0, diagonal = 0;
  for (std::size_t k = 0; k < 3; ++k)
    for (std::size_t j = 0; j < 3; ++j) {
      const double count = confusion.double_at("pred_" + model.classes()[k], j);
      total += count;
      if (k == j)
        diagonal += count;
    }
  EXPECT_EQ(total, test.nrows());
  EXPECT_NEAR(diagonal / total, curve.double_at("accuracy", 5), 1e-14);
}
TEST(XGBoostEvaluation, DropsOnlyIncompleteRowsAndDoesNotRefit) {
  const auto                      iris = datamunge::datasets::iris();
  stats::XGBoostClassifierOptions options;
  options.n_trees = 5;
  const stats::XGBoostClassifier model(iris, "Species ~ Petal.Length", options);
  auto                           altered = iris;
  altered.set_null("Petal.Length", 0);
  altered.set_null("Species", 1);
  const auto metrics = model.evaluation(altered);
  EXPECT_EQ(metrics.double_at("input_rows", 0), 150);
  EXPECT_EQ(metrics.double_at("evaluated_rows", 0), 148);
  EXPECT_EQ(metrics.double_at("skipped_rows", 0), 2);
  EXPECT_EQ(model.observations(), 150);
  auto unknown = iris.head(1);
  unknown.replace_column("Species", DF::string_column_type(std::vector<std::string>{"absent"}));
  EXPECT_THROW((void)model.evaluation(unknown), std::invalid_argument);
  EXPECT_THROW((void)model.confusion(unknown), std::invalid_argument);
  auto missing = iris.head(1);
  missing.set_null("Species", 0);
  EXPECT_THROW((void)model.evaluation(missing), std::runtime_error);
  auto infinite = iris.head(1);
  infinite.replace_column(
      "Petal.Length",
      DF::numeric_column_type(std::vector<double>{std::numeric_limits<double>::infinity()}));
  EXPECT_THROW((void)model.evaluation(infinite), std::invalid_argument);
  EXPECT_THROW((void)model.predict_detail(infinite), std::invalid_argument);
}
TEST(XGBoostValidationSearch, MatchesManualSelectionAndRetainsDeterministicTies) {
  const auto [train, validation] =
      stats::train_test_split(datamunge::datasets::iris(), 0.2, 42, "Species");
  stats::XGBoostClassifierOptions options;
  options.n_trees   = 8;
  options.max_depth = 2;
  const std::vector<double> lambdas{0.1, 1, 10};
  const auto                result =
      stats::xgboost_validation_search(train, validation, "Species ~ .", options, lambdas);
  double      best        = std::numeric_limits<double>::infinity();
  std::size_t best_round  = 0;
  double      best_lambda = 0;
  for (std::size_t i = 0; i < lambdas.size(); ++i) {
    auto current   = options;
    current.lambda = lambdas[i];
    const stats::XGBoostClassifier model(train, "Species ~ .", current);
    const auto                     curve       = model.staged_evaluation(validation);
    double                         local       = std::numeric_limits<double>::infinity();
    std::size_t                    local_round = 0;
    for (std::size_t m = 1; m <= options.n_trees; ++m) {
      const double loss = curve.double_at("mean_log_loss", m);
      if (loss < local) {
        local       = loss;
        local_round = m;
      }
      if (loss < best) {
        best        = loss;
        best_round  = m;
        best_lambda = lambdas[i];
      }
    }
    EXPECT_DOUBLE_EQ(result.candidates.double_at("mean_log_loss", i), local);
    EXPECT_EQ(result.candidates.double_at("round", i), local_round);
  }
  EXPECT_EQ(result.selected_options.n_trees, best_round);
  EXPECT_EQ(result.selected_options.lambda, best_lambda);
  options.max_depth = 0;
  options.alpha     = 1e6;
  const auto ties =
      stats::xgboost_validation_search(train, validation, "Species ~ .", options, {10, 1});
  EXPECT_EQ(ties.selected_options.n_trees, 1);
  EXPECT_EQ(ties.selected_options.lambda, 10);
  EXPECT_THROW(
      (void)stats::xgboost_validation_search(train, validation, "Species ~ .", options, {}),
      std::invalid_argument);
  EXPECT_THROW(
      (void)stats::xgboost_validation_search(train, validation, "Species ~ .", options, {-1}),
      std::invalid_argument);
  EXPECT_THROW(
      (void)stats::xgboost_validation_search(
          train, validation, "Species ~ .", options, {std::numeric_limits<double>::quiet_NaN()}),
      std::invalid_argument);
}
TEST(XGBoostOptions, RejectsInvalidFloatingSettingsAndLeafCounts) {
  const auto iris = datamunge::datasets::iris();
  for (auto member :
       {&stats::XGBoostClassifierOptions::lambda,
        &stats::XGBoostClassifierOptions::alpha,
        &stats::XGBoostClassifierOptions::gamma,
        &stats::XGBoostClassifierOptions::min_child_weight,
        &stats::XGBoostClassifierOptions::learning_rate,
        &stats::XGBoostClassifierOptions::subsample,
        &stats::XGBoostClassifierOptions::colsample_bytree}) {
    for (double value :
         {-1.0,
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity()}) {
      stats::XGBoostClassifierOptions options;
      options.*member = value;
      EXPECT_THROW(
          (void)stats::XGBoostClassifier(iris, "Species ~ Petal.Length", options),
          std::invalid_argument);
    }
  }
  for (std::size_t count : {std::size_t(0), std::numeric_limits<std::size_t>::max()}) {
    stats::XGBoostClassifierOptions options;
    options.min_samples_leaf = count;
    EXPECT_THROW(
        (void)stats::XGBoostClassifier(iris, "Species ~ Petal.Length", options),
        std::invalid_argument);
  }
  auto infinite = iris;
  infinite.replace_column(
      "Petal.Length",
      DF::numeric_column_type(std::vector<double>(150, std::numeric_limits<double>::infinity())));
  EXPECT_THROW(
      (void)stats::XGBoostClassifier(infinite, "Species ~ Petal.Length"), std::invalid_argument);
}
