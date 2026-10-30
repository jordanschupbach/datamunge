#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>
#include <limits>

using datamunge::dstruct::DataFrame;
using datamunge::stats::GBMClassifier;
using datamunge::stats::GBMClassifierOptions;
using datamunge::stats::GBMRegressor;
using datamunge::stats::GBMRegressorOptions;

// Reference ballpark computed with scikit-learn's GradientBoostingClassifier
// / GradientBoostingRegressor (n_estimators=100, learning_rate=0.1,
// max_depth=3, random_state=42) on the same embedded iris dataset:
// classifier train accuracy 0.9933, deviance 0.917 -> 0.013; regressor
// train R^2 0.9932, deviance (MSE) 2.526 -> 0.021. sklearn's tree-building
// and multiclass initialization details are not identical to this
// implementation's, so exact agreement isn't expected -- these tests check
// the same ballpark and the boosting procedure's defining property
// (training loss goes down) instead.

TEST(GBMClassifier, IrisAchievesHighTrainAccuracy) {
  GBMClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  EXPECT_GT(model.training_accuracy(), 0.95);
}

TEST(GBMClassifier, TrainingDevianceDecreasesOverBoostingRounds) {
  GBMClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  const auto&   deviance = model.training_deviance();
  ASSERT_EQ(deviance.size(), 100u);
  // Not strictly monotonic round to round, but the ensemble should fit
  // the training data dramatically better by the last round than the first.
  EXPECT_LT(deviance.back(), deviance.front() * 0.1);
  EXPECT_LT(deviance.back(), deviance[deviance.size() / 2]);
}

TEST(GBMClassifier, FeatureImportanceSumsToOneAndIsNonNegative) {
  GBMClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  const auto    importance = model.feature_importance();
  ASSERT_EQ(importance.size(), 2u);
  double sum = 0.0;
  for (const double v : importance) {
    EXPECT_GE(v, 0.0);
    sum += v;
  }
  EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(GBMClassifier, ConfusionMatrixDiagonalDominatesForIris) {
  GBMClassifier model(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  const auto    cm = model.confusion_matrix();
  ASSERT_EQ(cm.rows(), 3u);
  for (std::size_t i = 0; i < 3; ++i) {
    double row_total = 0.0;
    for (std::size_t j = 0; j < 3; ++j)
      row_total += cm(i, j);
    EXPECT_GT(cm(i, i) / row_total, 0.85);
  }
}

TEST(GBMClassifier, PredictMatchesFittedClassesOnTrainingData) {
  const auto    iris = datamunge::datasets::iris();
  GBMClassifier model(iris, "Species ~ Petal.Length + Petal.Width");
  const auto    preds = model.predict(iris);
  ASSERT_EQ(preds, model.fitted_classes());
}

TEST(GBMClassifier, FewerTreesFitsTrainingDataLessWell) {
  GBMClassifierOptions few;
  few.n_trees = 5;
  GBMClassifier small(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", few);
  GBMClassifier large(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  EXPECT_LE(small.training_accuracy(), large.training_accuracy() + 1e-9);
  EXPECT_GT(small.training_deviance().back(), large.training_deviance().back());
}

TEST(GBMClassifier, RejectsZeroTrees) {
  GBMClassifierOptions options;
  options.n_trees = 0;
  EXPECT_THROW(
      GBMClassifier(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options),
      std::invalid_argument);
}

TEST(GBMClassifier, RejectsSubsampleOutOfRange) {
  GBMClassifierOptions options;
  options.subsample = 1.5;
  EXPECT_THROW(
      GBMClassifier(datamunge::datasets::iris(), "Species ~ Petal.Length + Petal.Width", options),
      std::invalid_argument);
}

TEST(GBMClassifier, RejectsSingleClassResponse) {
  DataFrame df;
  df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  df.add_column("label", std::vector<std::string>{"a", "a", "a", "a", "a", "a"});
  EXPECT_THROW(GBMClassifier(df, "label ~ x"), std::invalid_argument);
}

TEST(GBMClassifier, PredictThrowsOnUnseenCategoricalLevel) {
  auto penguins =
      datamunge::datasets::penguins().drop_nulls({"bill_length_mm", "bill_depth_mm", "island"});
  GBMClassifier model(penguins, "species ~ bill_length_mm + bill_depth_mm + island");

  DataFrame bad;
  bad.add_column("bill_length_mm", std::vector<double>{40.0});
  bad.add_column("bill_depth_mm", std::vector<double>{18.0});
  bad.add_column("island", std::vector<std::string>{"Atlantis"});
  EXPECT_THROW((void)model.predict(bad), std::runtime_error);
}

TEST(GBMClassifier, ProbabilitiesSumToOne) {
  const auto    iris = datamunge::datasets::iris();
  GBMClassifier model(iris, "Species ~ Petal.Length + Petal.Width");

  DataFrame newdata;
  newdata.add_column("Petal.Length", std::vector<double>{4.5});
  newdata.add_column("Petal.Width", std::vector<double>{1.5});
  const auto detail = model.predict_detail(newdata);
  ASSERT_EQ(detail.probability.size(), 1u);
  double sum = 0.0;
  for (const double v : detail.probability[0])
    sum += v;
  EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(GBMClassifier, PlotDecisionRegionsRequiresExactlyTwoPredictors) {
  GBMClassifierOptions options;
  options.n_trees = 5;
  GBMClassifier model(
      datamunge::datasets::iris(),
      "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width",
      options);
  EXPECT_THROW(model.plot_decision_regions("Sepal.Length", "Sepal.Width"), std::invalid_argument);
}

// --- Regressor -------------------------------------------------------

TEST(GBMRegressor, IrisAchievesHighRSquared) {
  GBMRegressor model(
      datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
  EXPECT_GT(model.r_squared(), 0.95);
}

TEST(GBMRegressor, TrainingDevianceDecreasesOverBoostingRounds) {
  GBMRegressor model(
      datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
  const auto& deviance = model.training_deviance();
  ASSERT_EQ(deviance.size(), 100u);
  EXPECT_LT(deviance.back(), deviance.front() * 0.1);
}

TEST(GBMRegressor, PredictorNamesExcludeIntercept) {
  GBMRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");
  const auto&  names = model.predictor_names();
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "Sepal.Length");
  EXPECT_EQ(names[1], "Sepal.Width");
}

TEST(GBMRegressor, FeatureImportanceSumsToOne) {
  GBMRegressor model(
      datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
  const auto importance = model.feature_importance();
  double     sum        = 0.0;
  for (const double v : importance) {
    EXPECT_GE(v, 0.0);
    sum += v;
  }
  EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(GBMRegressor, PredictMatchesFittedValuesOnTrainingData) {
  const auto   iris = datamunge::datasets::iris();
  GBMRegressor model(iris, "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width");
  const auto   preds = model.predict(iris);
  ASSERT_EQ(preds.size(), model.fitted_values().size());
  for (std::size_t i = 0; i < preds.size(); ++i)
    EXPECT_NEAR(preds[i], model.fitted_values()[i], 1e-9);
}

TEST(GBMRegressor, PredictReturnsNanForMissingPredictors) {
  GBMRegressor model(datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width");

  DataFrame newdata;
  newdata.add_column("Sepal.Length", std::vector<std::optional<double>>{5.0, std::nullopt});
  newdata.add_column("Sepal.Width", std::vector<std::optional<double>>{3.0, 3.0});
  const auto preds = model.predict(newdata);
  ASSERT_EQ(preds.size(), 2u);
  EXPECT_FALSE(std::isnan(preds[0]));
  EXPECT_TRUE(std::isnan(preds[1]));
}

TEST(GBMRegressor, RejectsZeroTrees) {
  GBMRegressorOptions options;
  options.n_trees = 0;
  EXPECT_THROW(
      GBMRegressor(
          datamunge::datasets::iris(), "Petal.Length ~ Sepal.Length + Sepal.Width", options),
      std::invalid_argument);
}

TEST(GBMRegressor, SubsampleStillProducesAValidFit) {
  GBMRegressorOptions options;
  options.subsample = 0.7;
  options.seed      = 7;
  GBMRegressor model(
      datamunge::datasets::iris(),
      "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width",
      options);
  EXPECT_GT(model.r_squared(), 0.8);
}

namespace {
DataFrame gbm_binary_example() {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 2, 3});
  d.add_column("label", std::vector<std::string>{"a", "a", "b", "b"});
  return d;
}
} // namespace
TEST(GBMClassifier, ExactBinaryPriorResidualAndFirstUpdate) {
  const auto          d = gbm_binary_example();
  const GBMClassifier m(d, "label ~ x", {1, .5, 1});
  const auto          before = m.predict_detail(d, 0), after = m.predict_detail(d, 1);
  const auto          raw  = m.raw_scores(d, 1);
  const auto          step = m.training_step(1);
  EXPECT_EQ(m.classes(), (std::vector<std::string>{"a", "b"}));
  ASSERT_EQ(step.nrows(), 8);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(before.probability[i][0], .5, 1e-15);
    const double sign = i < 2 ? 1 : -1;
    EXPECT_NEAR(raw(i, 0), std::log(.5) + sign * .25, 1e-15);
    EXPECT_NEAR(after.probability[i][0], 1 / (1 + std::exp(-sign * .5)), 1e-15);
    EXPECT_NEAR(step.double_at("pseudo_residual", 2 * i), sign * .5, 1e-15);
    EXPECT_NEAR(step.double_at("tree_value", 2 * i), sign * .5, 1e-15);
    EXPECT_EQ(step.double_at("in_bag", 2 * i), 1);
  }
  EXPECT_NEAR(m.training_deviance()[0], std::log1p(std::exp(-.5)), 1e-15);
  EXPECT_EQ(m.round_sample_rows(1), (std::vector<std::size_t>{0, 1, 2, 3}));
}
TEST(GBMClassifier, MulticlassTargetsUseTheSamePreRoundProbabilities) {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 2, 3, 4, 5});
  d.add_column("label", std::vector<std::string>{"a", "a", "b", "b", "c", "c"});
  const GBMClassifier m(d, "label ~ x", {1, .5, 1});
  const auto          step = m.training_step(1);
  for (std::size_t i = 0; i < step.nrows(); ++i)
    EXPECT_NEAR(step.double_at("before_probability", i), 1. / 3, 1e-15);
  EXPECT_NEAR(step.double_at("tree_value", 0), 2. / 3, 1e-15);
  EXPECT_NEAR(step.double_at("tree_value", 1), -1. / 3, 1e-15);
  EXPECT_NEAR(step.double_at("tree_value", 3 * 2 + 1), 1. / 6, 1e-15);
  EXPECT_EQ(m.n_trees(), 1);
}
TEST(GBMClassifier, PrefixesAndSubsamplingMatchIndependentRefits) {
  DataFrame                d;
  std::vector<double>      x;
  std::vector<std::string> y;
  for (int i = 0; i < 12; ++i) {
    x.push_back(i);
    y.push_back(i < 4 ? "a" : i < 8 ? "b" : "c");
  }
  d.add_column("x", x);
  d.add_column("label", y);
  for (double fraction : {1., .6}) {
    GBMClassifierOptions o{8, .3, 2};
    o.subsample = fraction;
    o.seed      = 17;
    const GBMClassifier full(d, "label ~ x", o);
    for (std::size_t rounds : {1, 3, 8}) {
      o.n_trees = rounds;
      const GBMClassifier refit(d, "label ~ x", o);
      const auto          a = full.predict_detail(d, rounds), b = refit.predict_detail(d);
      for (std::size_t i = 0; i < 12; ++i)
        for (std::size_t k = 0; k < 3; ++k)
          EXPECT_NEAR(a.probability[i][k], b.probability[i][k], 1e-15);
      EXPECT_EQ(full.round_sample_rows(rounds), refit.round_sample_rows(rounds));
    }
    const auto sample = full.round_sample_rows(1);
    EXPECT_EQ(sample.size(), fraction == 1 ? 12 : 7);
    EXPECT_EQ(std::adjacent_find(sample.begin(), sample.end()), sample.end());
    const auto curve = full.staged_evaluation(d);
    ASSERT_EQ(curve.nrows(), 9);
    for (std::size_t r = 1; r <= 8; ++r)
      EXPECT_NEAR(curve.double_at("mean_log_loss", r), full.training_deviance()[r - 1], 1e-14);
  }
}
TEST(GBMClassifier, LogLossDoesNotClipUnderflowedTrueClassProbabilities) {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 1, 2, 3, 4, 5});
  d.add_column("label", std::vector<std::string>{"a", "a", "b", "b", "a", "b"});
  const GBMClassifier m(d, "label ~ x", {1, 2000, 1});
  const auto          prediction = m.predict_detail(d);
  EXPECT_EQ(prediction.probability[4][0], 0);
  EXPECT_NEAR(m.training_deviance()[0], 1000. / 6, 1e-12);
  EXPECT_NEAR(m.evaluation(d).double_at("mean_log_loss", 0), 1000. / 6, 1e-12);
  EXPECT_GT(m.training_deviance()[0], m.evaluation(d, 0).double_at("mean_log_loss", 0));
}
TEST(GBMClassifier, ValidationSearchCanChooseThePriorAndKeepsTieOrder) {
  const auto train      = gbm_binary_example();
  auto       validation = train.mutate("label", std::vector<std::string>{"b", "b", "a", "a"});
  const auto search =
      datamunge::stats::gbm_validation_search(train, validation, "label ~ x", {4, .1, 1}, {.3, .1});
  EXPECT_EQ(search.selected_rounds, 0);
  EXPECT_EQ(search.selected_options.n_trees, 1);
  EXPECT_EQ(search.selected_options.learning_rate, .3);
  EXPECT_EQ(search.candidates.nrows(), 10);
  const GBMClassifier refit(train, "label ~ x", search.selected_options);
  EXPECT_NEAR(refit.evaluation(validation, 0).double_at("mean_log_loss", 0), std::log(2.), 1e-15);
}
TEST(GBMClassifier, MissingRowMapsAndEvaluationClassChecksAreExplicit) {
  DataFrame d;
  d.add_column("x", std::vector<std::optional<double>>{0., std::nullopt, 1., 2., 3.});
  d.add_column("label", std::vector<std::string>{"a", "b", "a", "b", "b"});
  const GBMClassifier m(d, "label ~ x", {2, .5, 1});
  EXPECT_EQ(m.kept_row_indices(), (std::vector<std::size_t>{0, 2, 3, 4}));
  const auto p = m.predict_detail(d);
  EXPECT_TRUE(p.class_label[1].empty());
  EXPECT_TRUE(p.probability[1].empty());
  EXPECT_TRUE(std::isnan(m.raw_scores(d, 1)(1, 0)));
  const auto score = m.evaluation(d);
  EXPECT_EQ(score.double_at("input_rows", 0), 5);
  EXPECT_EQ(score.double_at("evaluated_rows", 0), 4);
  EXPECT_EQ(score.double_at("skipped_rows", 0), 1);
  auto unknown = d.mutate("label", std::vector<std::string>{"a", "b", "a", "c", "b"});
  EXPECT_THROW((void)m.evaluation(unknown), std::invalid_argument);
  EXPECT_EQ(m.predict_detail(d.take_rows({4})).probability[0], p.probability[4]);
  EXPECT_TRUE(m.predict(DataFrame{}).empty());
}
TEST(GBMClassifier, MeanNormalizedImportanceAllowsConstantClassTrees) {
  DataFrame d;
  d.add_column("x", std::vector<double>{0, 0, 0, 1, 1, 1});
  d.add_column("label", std::vector<std::string>{"a", "a", "c", "b", "b", "c"});
  const GBMClassifier m(d, "label ~ x", {1, .5, 1});
  EXPECT_NEAR(m.feature_importance()[0], 2. / 3, 1e-14);
  EXPECT_EQ(m.importance_table().double_at("relative_share", 0), 1);
  const auto          flat = d.mutate("x", std::vector<double>(6, 1));
  const GBMClassifier constant(flat, "label ~ x", {1, .5, 1});
  EXPECT_EQ(constant.feature_importance()[0], 0);
  EXPECT_EQ(constant.importance_table().double_at("relative_share", 0), 0);
}
TEST(GBMClassifier, InvalidControlsPrefixesAndGridSizesAreRejected) {
  const auto d = gbm_binary_example();
  for (double value :
       {0.,
        -1.,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity()}) {
    GBMClassifierOptions o;
    o.learning_rate = value;
    EXPECT_THROW(GBMClassifier(d, "label ~ x", o), std::invalid_argument);
  }
  GBMClassifierOptions o;
  o.subsample = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(GBMClassifier(d, "label ~ x", o), std::invalid_argument);
  o                  = {};
  o.min_samples_leaf = 0;
  EXPECT_THROW(GBMClassifier(d, "label ~ x", o), std::invalid_argument);
  const GBMClassifier m(d, "label ~ x", {2});
  EXPECT_THROW((void)m.predict_detail(d, 3), std::out_of_range);
  EXPECT_THROW((void)m.training_step(0), std::out_of_range);
  auto bad = d.mutate("x", std::vector<double>{0, 1, std::numeric_limits<double>::infinity(), 3});
  EXPECT_THROW(GBMClassifier(bad, "label ~ x"), std::invalid_argument);
  EXPECT_THROW((void)m.predict(bad), std::invalid_argument);
  const auto          iris = datamunge::datasets::iris();
  const GBMClassifier two(iris, "Species ~ Petal.Length + Petal.Width", {1});
  EXPECT_THROW(
      (void)two.plot_decision_regions("Petal.Length", "Petal.Width", 1), std::invalid_argument);
  EXPECT_THROW(
      (void)two.plot_decision_regions("Petal.Length", "Petal.Length", 10), std::invalid_argument);
}

TEST(GBMClassifier, FullSampleLeafMeansGiveTheQuadraticDescentBound) {
  const auto          d = gbm_binary_example();
  const GBMClassifier m(d, "label ~ x", {8, .5, 1});
  for (std::size_t round = 1; round <= m.n_trees(); ++round) {
    const auto a = m.step_diagnostics(round);
    EXPECT_EQ(a.double_at("full_sample", 0), 1);
    EXPECT_EQ(a.double_at("sample_size", 0), 4);
    EXPECT_NEAR(a.double_at("gradient_tree_inner", 0), a.double_at("tree_squared_norm", 0), 1e-14);
    EXPECT_LE(a.double_at("loss_after", 0), a.double_at("quadratic_bound", 0) + 1e-14);
    EXPECT_LE(a.double_at("loss_after", 0), a.double_at("loss_before", 0));
  }
  EXPECT_THROW((void)m.step_diagnostics(0), std::out_of_range);
  EXPECT_THROW((void)m.step_diagnostics(9), std::out_of_range);
}
