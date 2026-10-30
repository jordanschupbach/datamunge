/// @file xgboost_diagnostics.cpp
/// @brief Select regularisation and stage count by fixed-model validation loss paths.
#include <cmath>
#include <datamunge/stats/xgboost_diagnostics.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {
XGBoostValidationResult xgboost_validation_search(
    const dstruct::DataFrame&  train,
    const dstruct::DataFrame&  validation,
    const std::string&         formula,
    XGBoostClassifierOptions   options,
    const std::vector<double>& lambdas) {
  if (lambdas.empty())
    throw std::invalid_argument("xgboost_validation_search: empty lambda grid");
  for (const auto value : lambdas)
    if (!std::isfinite(value) || value < 0)
      throw std::invalid_argument("xgboost_validation_search: invalid lambda");
  XGBoostValidationResult result;
  double                  best = std::numeric_limits<double>::infinity();
  std::vector<double>     lambda_values, rounds, rows, accuracy, loss, brier;
  for (const double lambda : lambdas) {
    auto current   = options;
    current.lambda = lambda;
    const XGBoostClassifier model(train, formula, current);
    const auto              curve    = model.staged_evaluation(validation);
    std::size_t             selected = 1;
    for (std::size_t m = 2; m < curve.nrows(); ++m)
      if (curve.double_at("mean_log_loss", m) < curve.double_at("mean_log_loss", selected))
        selected = m;
    const double score = curve.double_at("mean_log_loss", selected);
    lambda_values.push_back(lambda);
    rounds.push_back(selected);
    rows.push_back(curve.double_at("evaluated_rows", selected));
    accuracy.push_back(curve.double_at("accuracy", selected));
    loss.push_back(score);
    brier.push_back(curve.double_at("brier", selected));
    if (score < best) {
      best                            = score;
      result.selected_options         = current;
      result.selected_options.n_trees = selected;
    }
  }
  result.candidates.add_column("lambda", lambda_values);
  result.candidates.add_column("round", rounds);
  result.candidates.add_column("evaluated_rows", rows);
  result.candidates.add_column("accuracy", accuracy);
  result.candidates.add_column("mean_log_loss", loss);
  result.candidates.add_column("brier", brier);
  return result;
}
} // namespace datamunge::stats
