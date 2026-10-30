#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/winnow.hpp>
#include <limits>
#include <random>
namespace alg = datamunge::algorithms;
using DF      = datamunge::dstruct::DataFrame;
namespace {
DF winnow_trace_frame() {
  DF data;
  data.add_column("a", std::vector<double>{1, 0, 0});
  data.add_column("b", std::vector<double>{0, 1, 0});
  data.add_column("target", std::vector<double>{1, 1, 0});
  return data;
}
} // namespace
TEST(WinnowTrace, SharedLoopPreservesWeightsAndSeparatesOnlineFromFinalErrors) {
  const auto data  = winnow_trace_frame();
  const auto fit   = alg::winnow_fit(data);
  const auto plain = alg::winnow_train({{1, 0}, {0, 1}, {0, 0}}, {1, 1, 0});
  EXPECT_EQ(fit.model.weights, plain.weights);
  EXPECT_EQ(fit.model.total_mistakes, 4);
  EXPECT_EQ(fit.model.epochs_run, 3);
  EXPECT_TRUE(fit.model.converged);
  ASSERT_EQ(fit.history.nrows(), 3);
  EXPECT_EQ(fit.history.double_at("mistakes", 0), 2);
  EXPECT_EQ(fit.history.double_at("mistakes", 1), 2);
  EXPECT_EQ(fit.history.double_at("accuracy", 1), 1);
  EXPECT_EQ(fit.history.double_at("mistakes", 2), 0);
  EXPECT_EQ(fit.history.double_at("total_weight", 2), 8);
  EXPECT_EQ(fit.weights.double_at("log2_weight", 0), 2);
  auto reordered = data.select({"target", "b", "a"});
  EXPECT_EQ(alg::winnow_evaluate(fit, reordered).double_at("mistakes", 0), 0);
}
TEST(WinnowTrace, PromotesOtherAttributesAndRecordsDemotions) {
  const auto correlated = alg::winnow_train({{1, 1}, {0, 1}, {1, 0}, {0, 0}}, {1, 0, 1, 0});
  EXPECT_EQ(correlated.weights[1], 2); // irrelevant to y=x0, yet promoted on a positive row
  DF data;
  data.add_column("a", std::vector<double>{1, 0});
  data.add_column("b", std::vector<double>{1, 1});
  data.add_column("c", std::vector<double>{1, 1});
  data.add_column("target", std::vector<double>{1, 0});
  const auto fit = alg::winnow_fit(data);
  EXPECT_EQ(fit.history.double_at("false_positives", 0), 1);
  EXPECT_EQ(fit.history.double_at("false_negatives", 0), 1);
  EXPECT_EQ(fit.model.weights, (std::vector<double>{2, 1, 1}));
  EXPECT_TRUE(fit.model.converged);
}
TEST(WinnowTrace, BooleanDisjunctionBoundHoldsForRepeatedArbitraryRows) {
  std::mt19937                engine(2026);
  std::bernoulli_distribution bit(0.25);
  for (const std::size_t d : {2, 8, 32, 128}) {
    std::vector<std::vector<int>> X(300, std::vector<int>(d));
    std::vector<int>              y;
    for (auto& row : X) {
      for (auto& value : row)
        value = bit(engine);
      y.push_back(row[0] || row[1]);
    }
    const auto model = alg::winnow_train(X, y, 2, -1, 100);
    EXPECT_TRUE(model.converged);
    EXPECT_LT(model.total_mistakes, 2 + 6 * std::log2(2 * d));
    for (std::size_t i = 0; i < X.size(); ++i)
      EXPECT_EQ(alg::winnow_predict(model, X[i]), y[i]);
    EXPECT_LE(model.weights[0], 2 * d);
    EXPECT_LE(model.weights[1], 2 * d);
  }
}
TEST(WinnowValidation, ChecksShapesBooleanValuesParametersAndPredictionDimensions) {
  EXPECT_THROW((void)alg::winnow_train({{1}, {0, 1}}, {1, 0}), std::invalid_argument);
  EXPECT_THROW((void)alg::winnow_train({{1}}, {}), std::invalid_argument);
  EXPECT_THROW((void)alg::winnow_train({{2}}, {1}), std::invalid_argument);
  EXPECT_THROW((void)alg::winnow_train({{1}}, {2}), std::invalid_argument);
  for (double alpha :
       {1.0,
        -1.0,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW((void)alg::winnow_train({{1}}, {1}, alpha), std::invalid_argument);
  EXPECT_THROW(
      (void)alg::winnow_train({{1}}, {1}, 2, std::numeric_limits<double>::infinity()),
      std::invalid_argument);
  const auto initial = alg::winnow_train({{1}}, {1}, 2, -1, 0);
  EXPECT_EQ(initial.weights, (std::vector<double>{1}));
  EXPECT_FALSE(initial.converged);
  EXPECT_EQ(initial.epochs_run, 0);
  EXPECT_EQ(alg::winnow_predict(initial, {1}), 0); // strict threshold tie
  EXPECT_THROW((void)alg::winnow_predict(initial, {}), std::invalid_argument);
  EXPECT_THROW((void)alg::winnow_predict(initial, {2}), std::invalid_argument);
  const auto empty = alg::winnow_train({}, {});
  EXPECT_TRUE(empty.weights.empty());
  EXPECT_FALSE(empty.converged);
}
TEST(WinnowValidation, ReportsOverflowAndUnderflow) {
  EXPECT_THROW((void)alg::winnow_train({{1}}, {1}, 1e300, 1e308, 2), std::overflow_error);
  EXPECT_THROW((void)alg::winnow_train({{1}}, {0}, 1e300, 0, 2), std::overflow_error);
  alg::WinnowModel model;
  model.weights = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  EXPECT_THROW((void)alg::winnow_score(model, {1, 1}), std::overflow_error);
  model.weights = {0};
  EXPECT_THROW((void)alg::winnow_score(model, {0}), std::invalid_argument);
}
TEST(WinnowDataFrame, EncodesStrictCutoffAndKnownOrWithoutModifyingData) {
  DF data;
  data.add_column("a", std::vector<double>{8, 9, 0});
  data.add_column("b", std::vector<double>{0, 8, 16});
  const auto binary = alg::winnow_disjunction_data(data, {"b", "a"}, {"a", "b"});
  EXPECT_EQ(binary.columns(), (std::vector<std::string>{"b", "a", "target"}));
  EXPECT_EQ(binary.get_column("target").values(), (std::vector<double>{0, 1, 1}));
  EXPECT_EQ(data.double_at("a", 0), 8);
  const auto none = alg::winnow_disjunction_data(data, {"a"}, {});
  EXPECT_EQ(none.get_column("target").values(), (std::vector<double>{0, 0, 0}));
  EXPECT_THROW((void)alg::winnow_disjunction_data(data, {"a", "a"}, {"a"}), std::invalid_argument);
  EXPECT_THROW((void)alg::winnow_disjunction_data(data, {"a"}, {"b"}), std::invalid_argument);
  EXPECT_THROW(
      (void)alg::winnow_disjunction_data(data, {"a"}, {"a"}, 8, "a"), std::invalid_argument);
  data.set_null("a", 0);
  EXPECT_THROW((void)alg::winnow_disjunction_data(data, {"a"}, {"a"}), std::invalid_argument);
}
TEST(WinnowDataFrame, ValidatesEmptyAndInvalidFramesAndKeepsFiniteModelMetadata) {
  auto       data      = winnow_trace_frame();
  const auto fit       = alg::winnow_fit(data);
  const auto empty     = data.head(0);
  const auto untrained = alg::winnow_fit(empty);
  EXPECT_EQ(untrained.model.weights, (std::vector<double>{1, 1}));
  EXPECT_EQ(untrained.model.threshold, 2);
  EXPECT_EQ(untrained.weights.nrows(), 2);
  EXPECT_EQ(untrained.history.nrows(), 0);
  const auto result = alg::winnow_evaluate(fit, empty);
  EXPECT_EQ(result.double_at("rows", 0), 0);
  EXPECT_TRUE(std::isnan(result.double_at("accuracy", 0)));
  data.replace_column("a", DF::numeric_column_type(std::vector<double>{0.5, 0, 0}));
  EXPECT_THROW((void)alg::winnow_fit(data), std::invalid_argument);
  EXPECT_THROW((void)alg::winnow_evaluate(fit, data), std::invalid_argument);
  EXPECT_THROW((void)alg::winnow_fit(data, "absent"), std::invalid_argument);
  EXPECT_THROW((void)alg::winnow_fit(data.select({"target"})), std::invalid_argument);
}

TEST(WinnowDataFrame, CounterfactualProbesExposeStrictThresholdTiesWithoutUpdating) {
  alg::WinnowFit fit;
  fit.predictors      = {"a", "b", "c"};
  fit.model.weights   = {2, 2, 3};
  fit.model.threshold = 2;
  const auto before   = fit.model.weights;
  const auto probes   = alg::winnow_disjunction_probes(fit, {"a", "b"});
  ASSERT_EQ(probes.nrows(), 5);
  EXPECT_EQ(probes.double_at("score", 1), 2);
  EXPECT_EQ(probes.double_at("expected", 1), 1);
  EXPECT_EQ(probes.double_at("predicted", 1), 0);
  EXPECT_EQ(probes.double_at("expected", 3), 0);
  EXPECT_EQ(probes.double_at("predicted", 3), 1);
  EXPECT_EQ(fit.model.weights, before);
  EXPECT_THROW((void)alg::winnow_disjunction_probes(fit, {"missing"}), std::invalid_argument);
}
