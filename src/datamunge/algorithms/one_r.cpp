/// @file one_r.cpp
/// @brief Canonical checked single-feature rule fitting, actual bucket histories and fixed-model evaluation.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/one_r.hpp>
#include <limits>
#include <set>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete shape/model admission and one actual training path.
namespace {
/// @brief Add logical element counts without size_t overflow.
/// @param a Existing count.
/// @param b Additional count.
/// @return Checked sum.
/// @throws std::length_error If addition is unrepresentable.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("OneR: count geometry");
  return a + b;
}
/// @brief Validate complete matching rows/labels and rectangular positive-dimensional nonempty data.
/// @param x Complete feature matrix.
/// @param y Complete class vector.
/// @return Common dimension,0 for completely empty data.
/// @throws std::invalid_argument If complete counts/shape are inconsistent.
/// @throws std::length_error If complete numeric matrix count overflows.
std::size_t validate(const std::vector<std::vector<int>>& x, const std::vector<int>& y) {
  if (x.size() != y.size())
    throw std::invalid_argument("OneR: matching rows and labels required");
  auto d = x.empty() ? 0 : x[0].size();
  if (!x.empty() && !d)
    throw std::invalid_argument("OneR: nonempty rows need a feature");
  for (auto& row : x)
    if (row.size() != d)
      throw std::invalid_argument("OneR: complete rectangular features required");
  if (d && x.size() > std::numeric_limits<std::size_t>::max() / d)
    throw std::length_error("OneR: complete matrix geometry");
  return d;
}
/// @brief Validate usable native model metadata before prediction/evaluation.
/// @param model Original owned rule model.
/// @throws std::invalid_argument If model is empty or internally inconsistent.
void validate_model(const OneRModel& model) {
  if (!model.total || model.errors > model.total || model.value_to_class.empty())
    throw std::invalid_argument("OneR: nonempty consistent fitted model required");
}
/// @brief Execute actual candidate map counting/majority selection with optional owned capture.
/// @param x Complete original feature rows.
/// @param y Complete original integer classes.
/// @param history Optional owned actual-history sink.
/// @param cap Actual logical history quota,0 disables.
/// @return Actual canonical minimum-training-error single-feature model.
/// @throws std::invalid_argument If complete shape/row counts fail.
/// @throws std::length_error If complete count/capture geometry or selected quota fails.
OneRModel run(
    const std::vector<std::vector<int>>& x,
    const std::vector<int>&              y,
    OneRTrace*                           history,
    std::size_t                          cap) {
  auto d = validate(x, y);
  auto n = y.size();
  /// @brief Admit complete next retained numeric/record contribution before copying.
  /// @param units Actual next record contribution.
  /// @throws std::length_error If geometry or selected quota fails.
  auto charge = [&](std::size_t units) {
    if (!history)
      return;
    auto next = add(history->captured_entries, units);
    if (cap && next > cap)
      throw std::length_error("OneR: actual capture quota");
    history->captured_entries = next;
  };
  if (history) {
    charge(add(n * d, n));
    history->features = x;
    history->labels   = y;
  }
  OneRModel best;
  if (!n)
    return best;
  best.total  = n;
  best.errors = std::numeric_limits<std::size_t>::max();
  std::map<int, std::size_t> global;
  for (int label : y)
    ++global[label];
  std::size_t largest = 0;
  for (auto [label, count] : global)
    if (count > largest) {
      largest            = count;
      best.default_class = label;
    }
  for (std::size_t f = 0; f < d; ++f) {
    std::map<int, std::map<int, std::size_t>> counts;
    for (std::size_t i = 0; i < n; ++i)
      ++counts[x[i][f]][y[i]];
    OneRFeature candidate;
    candidate.feature = f;
    std::map<int, int> rule;
    if (history)
      charge(1);
    for (auto& [value, classes] : counts) {
      OneRBucket bucket;
      bucket.value         = value;
      std::size_t majority = 0;
      for (auto [label, count] : classes) {
        bucket.total += count;
        if (count > majority) {
          majority          = count;
          bucket.prediction = label;
        }
      }
      bucket.errors = bucket.total - majority;
      candidate.errors += bucket.errors;
      rule[value] = bucket.prediction;
      if (history) {
        charge(add(1, add(classes.size(), classes.size())));
        bucket.class_counts = classes;
        candidate.buckets.push_back(std::move(bucket));
      }
    }
    if (candidate.errors < best.errors) {
      best.errors         = candidate.errors;
      best.feature        = f;
      best.value_to_class = std::move(rule);
    }
    if (history)
      history->candidates.push_back(std::move(candidate));
  }
  if (history)
    charge(add(best.value_to_class.size(), best.value_to_class.size()));
  return best;
}
/// @brief Validate and convert one numeric categorical cell without truncation.
/// @param value Native numeric cell.
/// @return Exact int category.
/// @throws std::invalid_argument If nonfinite, fractional or outside int range.
int category(double value) {
  if (!std::isfinite(value) || value != std::floor(value) ||
      value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    throw std::invalid_argument("OneR: finite integral int-range categories required");
  return static_cast<int>(value);
}
} // namespace
OneRModel one_r_train(const std::vector<std::vector<int>>& x, const std::vector<int>& y) {
  return run(x, y, nullptr, 0);
}
OneRTrace one_r_train_trace(
    const std::vector<std::vector<int>>& x,
    const std::vector<int>&              y,
    std::size_t                          max_entries) {
  OneRTrace t;
  t.model = run(x, y, &t, max_entries);
  return t;
}
int one_r_predict(const OneRModel& model, const std::vector<int>& x) {
  validate_model(model);
  if (model.feature >= x.size())
    throw std::invalid_argument("OneR: selected feature absent");
  auto it = model.value_to_class.find(x[model.feature]);
  return it == model.value_to_class.end() ? model.default_class : it->second;
}
OneREvaluation one_r_evaluate(
    const OneRModel&                     model,
    const std::vector<std::vector<int>>& x,
    const std::vector<int>&              y) {
  auto d = validate(x, y);
  validate_model(model);
  if (!x.empty() && model.feature >= d)
    throw std::invalid_argument("OneR: evaluation selected feature absent");
  OneREvaluation result;
  result.rows = x.size();
  result.predictions.reserve(x.size());
  std::set<int> classes(y.begin(), y.end());
  for (std::size_t i = 0; i < x.size(); ++i) {
    auto prediction = one_r_predict(model, x[i]);
    result.predictions.push_back(prediction);
    result.correct += prediction == y[i];
    result.fallbacks += !model.value_to_class.contains(x[i][model.feature]);
    classes.insert(prediction);
  }
  result.classes.assign(classes.begin(), classes.end());
  auto k = classes.size();
  if (k && k > std::numeric_limits<std::size_t>::max() / k)
    throw std::length_error("OneR: complete confusion geometry");
  result.confusion.assign(k, std::vector<std::size_t>(k));
  for (std::size_t i = 0; i < x.size(); ++i) {
    auto truth = std::lower_bound(result.classes.begin(), result.classes.end(), y[i]) -
                 result.classes.begin(),
         predicted =
             std::lower_bound(result.classes.begin(), result.classes.end(), result.predictions[i]) -
             result.classes.begin();
    ++result.confusion[truth][predicted];
  }
  return result;
}
OneRData one_r_dataframe(
    const dstruct::DataFrame&       frame,
    const std::vector<std::string>& predictors,
    const std::string&              response) {
  if (predictors.empty() || response.empty())
    throw std::invalid_argument("OneR: nonempty predictor/response names required");
  std::set<std::string> names;
  for (auto& name : predictors)
    if (name.empty() || name == response || !names.insert(name).second)
      throw std::invalid_argument("OneR: distinct nonresponse predictor names required");
  auto n = frame.nrows(), d = predictors.size();
  if (n && d > std::numeric_limits<std::size_t>::max() / n)
    throw std::length_error("OneR: complete DataFrame matrix geometry");
  for (auto& name : predictors) {
    const auto& column = frame.get_column(name);
    for (std::size_t i = 0; i < n; ++i)
      category(column.at(i));
  }
  const auto& labels = frame.get_column(response);
  for (std::size_t i = 0; i < n; ++i)
    category(labels.at(i));
  OneRData data;
  data.names = predictors;
  data.features.assign(n, std::vector<int>(d));
  data.labels.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    data.labels[i] = category(labels.at(i));
    for (std::size_t j = 0; j < d; ++j)
      data.features[i][j] = category(frame.get_column(predictors[j]).at(i));
  }
  return data;
}
OneRData one_r_weather_data() {
  return {
      {{0, 0, 0, 0},
       {0, 0, 0, 1},
       {1, 0, 0, 0},
       {2, 1, 0, 0},
       {2, 2, 1, 0},
       {2, 2, 1, 1},
       {1, 2, 1, 1},
       {0, 1, 0, 0},
       {0, 2, 1, 0},
       {2, 1, 1, 0},
       {0, 1, 1, 1},
       {1, 1, 0, 1},
       {1, 0, 1, 0},
       {2, 1, 0, 1}},
      {0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0},
      {"outlook", "temperature", "humidity", "windy"}};
}
} // namespace datamunge::algorithms
