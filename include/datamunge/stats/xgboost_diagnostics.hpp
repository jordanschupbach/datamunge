#pragma once

/// @file xgboost_diagnostics.hpp
/// @brief Validation-only selection of regularisation and boosting round count.
#include <datamunge/stats/xgboost_classifier.hpp>

namespace datamunge::stats {

/// @brief Selected XGBoost-style options and one validation summary per supplied lambda.
///
/// candidates has lambda, round, evaluated_rows, accuracy, mean_log_loss and
/// brier. Each row is that lambda's best nonzero stage. selected_options copies
/// the requested options except for the selected lambda and n_trees. No fitted
/// model is retained; callers refit on their development data before a final test.
/// @par Example
/// @code
/// const auto parts=dm::stats::train_test_split(dm::datasets::iris(),0.2,42,"Species");
/// dm::stats::XGBoostClassifierOptions options; options.n_trees=5;
/// const auto search=dm::stats::xgboost_validation_search(
///     parts.first,parts.second,"Species ~ Petal.Length",options,{1.0});
/// const auto candidates=search.candidates.nrows(); // 1
/// @endcode
/// @see xgboost_validation_search()
struct XGBoostValidationResult {
  XGBoostClassifierOptions
      selected_options; ///< Copy of requested options with selected lambda/round count.
  dstruct::DataFrame
      candidates; ///< Owned best-stage validation summaries, in supplied lambda order.
};

/// @brief Select lambda and round count using only a separate validation frame's mean log loss.
///
/// Fits one full-budget model per lambda on train, then scores its fixed path
/// on validation. Selects stages 1..options.n_trees; round-zero priors are excluded.
/// Strictly smaller mean_log_loss wins; exact ties retain the first lambda and
/// earliest round. Frames are borrowed and unchanged. The caller must ensure
/// train/validation are disjoint and reserve any final test set; this function
/// neither checks record identity nor accesses a test frame. Validation scores
/// used for selection are optimistic performance estimates. To evaluate the
/// selected procedure, refit on development data and use an untouched test set.
/// Cost is one training plus staged evaluation per lambda, not one refit per stage.
/// Result storage is O(lambdas.size()); temporary storage follows the fitted model.
/// @param train Training rows with a string response and at least two complete-case classes.
/// @param validation Separate labelled rows; complete-case response classes must be present in train.
/// @param formula Formula resolved against train; validation follows that design.
/// @param options Hyperparameters; n_trees is the maximum budget per lambda, default 100.
/// @param lambdas Nonempty ordered finite nonnegative values; defaults {0.01,0.1,1,10,100}.
/// @return Owned selected options and per-lambda best-stage validation table.
/// @throws std::invalid_argument If lambdas/options/schema/classes/transformed values are invalid.
/// @throws std::runtime_error If no complete rows remain or predictor levels are unseen.
/// @throws std::out_of_range If a required predictor column is missing.
/// @throws std::domain_error If a formula transform is outside its mathematical domain.
/// @par Example
/// @code
/// const auto parts=dm::stats::train_test_split(dm::datasets::iris(),0.2,42,"Species");
/// dm::stats::XGBoostClassifierOptions options; options.n_trees=10; options.max_depth=2;
/// const auto search=dm::stats::xgboost_validation_search(
///     parts.first,parts.second,"Species ~ Petal.Length",options,{0.1,1.0});
/// const bool within_budget=search.selected_options.n_trees<=10; // true
/// @endcode
/// @par References
/// - T. Chen and C. Guestrin (2016), XGBoost: A Scalable Tree Boosting System,
///   https://doi.org/10.1145/2939672.2939785
/// @see XGBoostClassifier::staged_evaluation(), train_test_split()
XGBoostValidationResult xgboost_validation_search(
    const dstruct::DataFrame&  train,
    const dstruct::DataFrame&  validation,
    const std::string&         formula,
    XGBoostClassifierOptions   options = {},
    const std::vector<double>& lambdas = {0.01, 0.1, 1, 10, 100});

} // namespace datamunge::stats
