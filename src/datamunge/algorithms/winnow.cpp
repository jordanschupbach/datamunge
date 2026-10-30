/// @file winnow.cpp
/// @brief Checked shared Winnow2 updates, Boolean DataFrame conversion and pass diagnostics.
#include <algorithm>
#include <datamunge/algorithms/winnow.hpp>
#include <limits>
#include <numeric>
#include <set>

namespace datamunge::algorithms {
/// @brief Local shape, Boolean-value and finite-arithmetic helpers.
namespace {
/// @brief Require a rectangular Boolean matrix and an aligned Boolean response.
/// @param features Borrowed input rows.
/// @param labels Borrowed responses.
/// @return Row dimension, or zero when no rows are present.
/// @throws std::invalid_argument If shapes or values are invalid.
std::size_t winnow_validate_rows(
    const std::vector<std::vector<int>>& features,
    const std::vector<int>&              labels) {
  if (features.size() != labels.size())
    throw std::invalid_argument("winnow: row/label size mismatch");
  const std::size_t dimension = features.empty() ? 0 : features.front().size();
  for (std::size_t i = 0; i < features.size(); ++i) {
    if (features[i].size() != dimension)
      throw std::invalid_argument("winnow: ragged feature matrix");
    if (labels[i] != 0 && labels[i] != 1)
      throw std::invalid_argument("winnow: labels must be Boolean");
    for (const int value : features[i])
      if (value != 0 && value != 1)
        throw std::invalid_argument("winnow: features must be Boolean");
  }
  return dimension;
}
/// @brief Sum active weights for already-validated rows and positive weights.
/// @param weights Finite positive weights.
/// @param row Boolean values aligned with weights.
/// @return Finite weighted sum.
/// @throws std::overflow_error If the accumulated score is non-finite.
double winnow_dot(const std::vector<double>& weights, const std::vector<int>& row) {
  double result = 0;
  for (std::size_t j = 0; j < weights.size(); ++j)
    if (row[j])
      result += weights[j];
  if (!std::isfinite(result))
    throw std::overflow_error("winnow: weighted score overflow");
  return result;
}
/// @brief Copy a finite non-null numeric column, reporting a missing name as an argument error.
/// @param data Borrowed frame.
/// @param name Required numeric column.
/// @return Owned numeric values in row order.
/// @throws std::invalid_argument If the column/type/cells are invalid.
std::vector<double> winnow_numeric_column(const dstruct::DataFrame& data, const std::string& name) {
  if (!data.has_column(name))
    throw std::invalid_argument("winnow: missing required column");
  const auto&         column = data.get_column(name);
  std::vector<double> out;
  out.reserve(data.nrows());
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    const double value = column.at(i);
    if (!std::isfinite(value))
      throw std::invalid_argument("winnow: non-finite numeric cell");
    out.push_back(value);
  }
  return out;
}
/// @brief Decode required Boolean DataFrame columns in a specified feature order.
/// @param data Borrowed encoded frame.
/// @param names Nonempty unique predictor names.
/// @param response Required Boolean response name, distinct from predictors.
/// @param labels Output Boolean labels, replaced on success.
/// @return Owned rectangular Boolean rows.
/// @throws std::invalid_argument If names, numeric schema or Boolean cells are invalid.
std::vector<std::vector<int>> winnow_decode(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& names,
    const std::string&              response,
    std::vector<int>&               labels) {
  const std::set<std::string> unique(names.begin(), names.end());
  if (names.empty() || unique.size() != names.size() || unique.count(response))
    throw std::invalid_argument("winnow: invalid predictor-name mapping");
  const auto y = winnow_numeric_column(data, response);
  labels.clear();
  labels.reserve(y.size());
  for (const auto value : y) {
    if (value != 0 && value != 1)
      throw std::invalid_argument("winnow: target must be Boolean");
    labels.push_back(static_cast<int>(value));
  }
  std::vector<std::vector<int>> X(data.nrows(), std::vector<int>(names.size()));
  for (std::size_t j = 0; j < names.size(); ++j) {
    const auto values = winnow_numeric_column(data, names[j]);
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (values[i] != 0 && values[i] != 1)
        throw std::invalid_argument("winnow: predictors must be Boolean");
      X[i][j] = static_cast<int>(values[i]);
    }
  }
  return X;
}
} // namespace

namespace detail {
WinnowModel winnow_train_impl(
    const std::vector<std::vector<int>>& features,
    const std::vector<int>&              labels,
    double                               alpha,
    double                               threshold,
    std::size_t                          max_epochs,
    std::vector<WinnowEpoch>*            history) {
  if (history)
    history->clear();
  if (!std::isfinite(alpha) || alpha <= 1 || !std::isfinite(threshold))
    throw std::invalid_argument("winnow: alpha must be finite and >1; threshold must be finite");
  const auto  dimension = winnow_validate_rows(features, labels);
  WinnowModel model;
  model.alpha     = alpha;
  model.threshold = threshold < 0 ? static_cast<double>(dimension) : threshold;
  model.weights.assign(dimension, 1.0);
  if (features.empty())
    return model;
  for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
    WinnowEpoch record;
    record.epoch = epoch + 1;
    for (std::size_t i = 0; i < features.size(); ++i) {
      const bool predicted = winnow_dot(model.weights, features[i]) > model.threshold;
      if (predicted == bool(labels[i]))
        continue;
      const bool promote = labels[i] == 1;
      for (std::size_t j = 0; j < dimension; ++j)
        if (features[i][j]) {
          const double next = promote ? model.weights[j] * alpha : model.weights[j] / alpha;
          if (!std::isfinite(next) || next <= 0)
            throw std::overflow_error("winnow: positive weight range exhausted");
          model.weights[j] = next;
        }
      if (model.total_mistakes == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("winnow: mistake counter overflow");
      ++model.total_mistakes;
      if (promote)
        ++record.false_negatives;
      else
        ++record.false_positives;
    }
    model.epochs_run    = epoch + 1;
    record.total_weight = std::accumulate(model.weights.begin(), model.weights.end(), 0.0);
    if (!std::isfinite(record.total_weight))
      throw std::overflow_error("winnow: total weight overflow");
    record.cumulative_mistakes = model.total_mistakes;
    const auto mistakes        = record.false_positives + record.false_negatives;
    if (history) {
      std::size_t correct = 0;
      for (std::size_t i = 0; i < features.size(); ++i)
        correct += (winnow_dot(model.weights, features[i]) > model.threshold) == bool(labels[i]);
      record.accuracy = static_cast<double>(correct) / features.size();
      history->push_back(record);
    }
    if (mistakes == 0) {
      model.converged = true;
      break;
    }
  }
  return model;
}
} // namespace detail

double winnow_score(const WinnowModel& model, const std::vector<int>& x) {
  if (x.size() != model.weights.size() || !std::isfinite(model.threshold) || model.threshold < 0)
    throw std::invalid_argument("winnow_score: invalid dimension or threshold");
  for (std::size_t j = 0; j < x.size(); ++j)
    if ((x[j] != 0 && x[j] != 1) || !std::isfinite(model.weights[j]) || model.weights[j] <= 0)
      throw std::invalid_argument("winnow_score: invalid Boolean row or positive weight");
  return winnow_dot(model.weights, x);
}

dstruct::DataFrame winnow_disjunction_data(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& predictors,
    const std::vector<std::string>& relevant,
    double                          cutoff,
    const std::string&              response) {
  const std::set<std::string> columns(predictors.begin(), predictors.end()),
      targets(relevant.begin(), relevant.end());
  if (predictors.empty() || columns.size() != predictors.size() ||
      targets.size() != relevant.size() || response.empty() || columns.count(response) ||
      !std::isfinite(cutoff))
    throw std::invalid_argument("winnow_disjunction_data: invalid names or cutoff");
  for (const auto& name : relevant)
    if (!columns.count(name))
      throw std::invalid_argument("winnow_disjunction_data: relevant feature not selected");
  dstruct::DataFrame  out;
  std::vector<double> y(data.nrows(), 0);
  for (const auto& name : predictors) {
    auto values = winnow_numeric_column(data, name);
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = values[i] > cutoff ? 1 : 0;
      if (targets.count(name) && values[i])
        y[i] = 1;
    }
    out.add_column(name, values);
  }
  out.add_column(response, y);
  return out;
}

WinnowFit winnow_fit(
    const dstruct::DataFrame& data,
    const std::string&        response,
    double                    alpha,
    double                    threshold,
    std::size_t               max_epochs) {
  WinnowFit result;
  result.predictors = data.columns();
  result.predictors.erase(
      std::remove(result.predictors.begin(), result.predictors.end(), response),
      result.predictors.end());
  std::vector<int>                 labels;
  const auto                       X = winnow_decode(data, result.predictors, response, labels);
  std::vector<detail::WinnowEpoch> records;
  result.model = detail::winnow_train_impl(X, labels, alpha, threshold, max_epochs, &records);
  if (X.empty()) {
    result.model.weights.assign(result.predictors.size(), 1);
    result.model.threshold =
        threshold < 0 ? static_cast<double>(result.predictors.size()) : threshold;
  }
  std::vector<double> epoch, mistakes, fp, fn, cumulative, accuracy, total;
  for (const auto& r : records) {
    epoch.push_back(r.epoch);
    mistakes.push_back(r.false_positives + r.false_negatives);
    fp.push_back(r.false_positives);
    fn.push_back(r.false_negatives);
    cumulative.push_back(r.cumulative_mistakes);
    accuracy.push_back(r.accuracy);
    total.push_back(r.total_weight);
  }
  result.history.add_column("epoch", epoch);
  result.history.add_column("mistakes", mistakes);
  result.history.add_column("false_positives", fp);
  result.history.add_column("false_negatives", fn);
  result.history.add_column("cumulative_mistakes", cumulative);
  result.history.add_column("accuracy", accuracy);
  result.history.add_column("total_weight", total);
  std::vector<double> index(result.predictors.size()), logs;
  std::iota(index.begin(), index.end(), 0);
  for (const auto weight : result.model.weights)
    logs.push_back(std::log2(weight));
  result.weights.add_column("feature", result.predictors);
  result.weights.add_column("index", index);
  result.weights.add_column("weight", result.model.weights);
  result.weights.add_column("log2_weight", logs);
  return result;
}

dstruct::DataFrame
winnow_evaluate(const WinnowFit& fit, const dstruct::DataFrame& data, const std::string& response) {
  std::vector<int> labels;
  const auto       X = winnow_decode(data, fit.predictors, response, labels);
  (void)winnow_score(fit.model, std::vector<int>(fit.predictors.size(), 0));
  double positives = 0, predicted_positives = 0, fp = 0, fn = 0;
  for (std::size_t i = 0; i < X.size(); ++i) {
    const int predicted = winnow_predict(fit.model, X[i]);
    positives += labels[i];
    predicted_positives += predicted;
    fp += predicted == 1 && labels[i] == 0;
    fn += predicted == 0 && labels[i] == 1;
  }
  dstruct::DataFrame out;
  out.add_column("rows", std::vector<double>{double(X.size())});
  out.add_column("positives", std::vector<double>{positives});
  out.add_column("predicted_positives", std::vector<double>{predicted_positives});
  out.add_column("false_positives", std::vector<double>{fp});
  out.add_column("false_negatives", std::vector<double>{fn});
  out.add_column("mistakes", std::vector<double>{fp + fn});
  out.add_column(
      "accuracy",
      std::vector<double>{
          X.empty() ? std::numeric_limits<double>::quiet_NaN() : 1 - (fp + fn) / X.size()});
  return out;
}
dstruct::DataFrame winnow_disjunction_probes(
    const WinnowFit&                fit,
    const std::vector<std::string>& relevant) {
  const std::set<std::string> names(fit.predictors.begin(), fit.predictors.end()),
      targets(relevant.begin(), relevant.end());
  if (names.empty() || names.size() != fit.predictors.size() || targets.size() != relevant.size())
    throw std::invalid_argument("winnow_disjunction_probes: invalid predictor mapping");
  for (const auto& name : relevant)
    if (!names.count(name))
      throw std::invalid_argument("winnow_disjunction_probes: unknown relevant name");
  std::vector<std::string> labels;
  std::vector<double>      scores, expected, predicted, correct;
  /// @brief Record one fixed Boolean probe without changing the fitted rule.
  /// @param name Descriptive probe label.
  /// @param row Boolean features in fitted order.
  /// @param truth Expected OR response.
  const auto append = [&](const std::string& name, const std::vector<int>& row, int truth) {
    const double score      = winnow_score(fit.model, row);
    const int    prediction = score > fit.model.threshold;
    labels.push_back(name);
    scores.push_back(score);
    expected.push_back(truth);
    predicted.push_back(prediction);
    correct.push_back(prediction == truth);
  };
  const std::size_t d = fit.predictors.size();
  std::vector<int>  row(d, 0);
  append("zero", row, 0);
  for (const auto& name : relevant) {
    std::fill(row.begin(), row.end(), 0);
    row[std::find(fit.predictors.begin(), fit.predictors.end(), name) - fit.predictors.begin()] = 1;
    append("only " + name, row, 1);
  }
  for (std::size_t j = 0; j < d; ++j)
    row[j] = targets.count(fit.predictors[j]) ? 0 : 1;
  append("other features on", row, 0);
  for (std::size_t j = 0; j < d; ++j)
    row[j] = targets.count(fit.predictors[j]) ? 1 : 0;
  append("relevant features on", row, relevant.empty() ? 0 : 1);
  dstruct::DataFrame out;
  out.add_column("probe", labels);
  out.add_column("score", scores);
  out.add_column("expected", expected);
  out.add_column("predicted", predicted);
  out.add_column("correct", correct);
  return out;
}

} // namespace datamunge::algorithms
