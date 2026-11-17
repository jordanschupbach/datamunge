#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/one_r.hpp>
#include <dm/dm.hpp>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(OneRTrace, IndependentBruteLabelChoicesEveryBucketFeatureTieAndCapture) {
  std::mt19937_64 rng(264273);
  for (std::size_t n = 1; n <= 15; ++n)
    for (std::size_t d = 1; d <= 5; ++d) {
      std::vector<std::vector<int>> x(n, std::vector<int>(d));
      std::vector<int>              y(n);
      for (auto& row : x)
        for (auto& v : row)
          v = int(rng() % 5) - 2;
      for (auto& v : y)
        v = int(rng() % 3) - 1;
      auto          t = one_r_train_trace(x, y);
      std::set<int> classes(y.begin(), y.end());
      std::size_t   best_error = SIZE_MAX, best_feature = 0, units = n * d + n;
      for (std::size_t f = 0; f < d; ++f) {
        auto& candidate = t.candidates[f];
        EXPECT_EQ(candidate.feature, f);
        std::set<int> values;
        for (auto& row : x)
          values.insert(row[f]);
        std::size_t error = 0, b = 0;
        ++units;
        for (int value : values) {
          auto& bucket = candidate.buckets[b++];
          EXPECT_EQ(bucket.value, value);
          std::size_t                minimum  = SIZE_MAX;
          int                        selected = 0;
          std::map<int, std::size_t> counts;
          for (std::size_t i = 0; i < n; ++i)
            if (x[i][f] == value)
              ++counts[y[i]];
          for (int label : classes) {
            std::size_t errors = 0;
            for (std::size_t i = 0; i < n; ++i)
              errors += x[i][f] == value && y[i] != label;
            if (errors < minimum) {
              minimum  = errors;
              selected = label;
            }
          }
          EXPECT_EQ(bucket.prediction, selected);
          EXPECT_EQ(bucket.errors, minimum);
          EXPECT_EQ(bucket.class_counts, counts);
          error += minimum;
          units += 1 + 2 * counts.size();
        }
        EXPECT_EQ(candidate.errors, error);
        if (error < best_error) {
          best_error   = error;
          best_feature = f;
        }
      }
      EXPECT_EQ(t.model.feature, best_feature);
      EXPECT_EQ(t.model.errors, best_error);
      units += 2 * t.model.value_to_class.size();
      EXPECT_EQ(t.captured_entries, units);
      auto model = one_r_train(x, y);
      EXPECT_EQ(model.value_to_class, t.model.value_to_class);
      EXPECT_EQ(model.feature, t.model.feature);
      auto e = one_r_evaluate(model, x, y);
      EXPECT_EQ(e.correct, n - best_error);
      EXPECT_EQ(e.fallbacks, 0);
      EXPECT_NO_THROW(one_r_train_trace(x, y, units));
      EXPECT_THROW(one_r_train_trace(x, y, units - 1), std::length_error);
    }
}
TEST(OneRTrace, OriginalWeatherShapeGuardsEmptyAndFrozenUnseenFallback) {
  auto d = one_r_weather_data();
  auto t = one_r_train_trace(d.features, d.labels);
  EXPECT_EQ(t.model.feature, 0);
  EXPECT_EQ(t.model.errors, 4);
  EXPECT_EQ(t.model.value_to_class, (std::map<int, int>{{0, 0}, {1, 1}, {2, 1}}));
  auto model = one_r_train({{0}, {1}}, {0, 1});
  EXPECT_EQ(one_r_predict(model, {2}), 0);
  auto e = one_r_evaluate(model, {{2}, {2}, {0}}, {1, 1, 1});
  EXPECT_EQ(e.correct, 0);
  EXPECT_EQ(e.fallbacks, 2);
  EXPECT_EQ(model.default_class, 0);
  EXPECT_EQ(one_r_train({}, {}).errors, 0);
  EXPECT_EQ(one_r_train({}, {}).accuracy(), 0);
  EXPECT_THROW(one_r_predict({}, {}), std::invalid_argument);
  EXPECT_THROW(one_r_train({{0}}, {}), std::invalid_argument);
  EXPECT_THROW(one_r_train({{}}, {0}), std::invalid_argument);
  EXPECT_THROW(one_r_train({{0}, {1, 2}}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(one_r_predict(model, {}), std::invalid_argument);
  EXPECT_TRUE(one_r_evaluate(model, {}, {}).confusion.empty());
}
TEST(OneRTrace, NativeDataFrameTitanicTrainOnlyModelAndRawConfusion) {
  auto data          = dm::datasets::titanic();
  auto [train, test] = dm::stats::train_test_split(data, .3, 2024);
  auto fit           = one_r_dataframe(train, {"Pclass", "SibSp", "Parch"}, "Survived");
  auto held          = one_r_dataframe(test, fit.names, "Survived");
  EXPECT_EQ(fit.labels.size(), 624);
  EXPECT_EQ(held.labels.size(), 267);
  auto model = one_r_train(fit.features, fit.labels);
  auto e     = one_r_evaluate(model, held.features, held.labels);
  EXPECT_EQ(model.total, 624);
  EXPECT_EQ(e.rows, 267);
  std::size_t sum = 0, diagonal = 0;
  for (std::size_t i = 0; i < e.confusion.size(); ++i)
    for (std::size_t j = 0; j < e.confusion[i].size(); ++j) {
      sum += e.confusion[i][j];
      if (i == j)
        diagonal += e.confusion[i][j];
    }
  EXPECT_EQ(sum, 267);
  EXPECT_EQ(diagonal, e.correct);
  for (std::size_t i = 0; i < held.labels.size(); ++i)
    EXPECT_EQ(e.predictions[i], one_r_predict(model, held.features[i]));
  EXPECT_THROW(one_r_dataframe(train, {"Survived"}, "Survived"), std::invalid_argument);
  EXPECT_THROW(one_r_dataframe(train, {"Pclass", "Pclass"}, "Survived"), std::invalid_argument);
  dm::dstruct::DataFrame bad;
  bad.add_column("x", std::vector<double>{.5});
  bad.add_column("y", std::vector<double>{0});
  EXPECT_THROW(one_r_dataframe(bad, {"x"}, "y"), std::invalid_argument);
}
