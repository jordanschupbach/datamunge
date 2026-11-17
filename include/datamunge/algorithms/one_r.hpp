#pragma once
/// @file one_r.hpp
/// @brief Checked nominal-category OneR with actual bucket decisions and fixed-model evaluation.
#include <cstddef>
#include <datamunge/dstruct/dataframe.hpp>
#include <map>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned one-feature nominal rule, preserving original public field order.
/// @par Example
/// @code
/// auto m=dm::algorithms::one_r_train({{0},{1},{1}},{0,1,1});std::print("{}\n",m.errors); // 0
/// @endcode
/// @see one_r_train_trace()
struct OneRModel {
  std::size_t feature{
      0}; ///< Chosen original zero-based feature index;0 placeholder for empty model.
  std::map<int, int>
              value_to_class;   ///< Owned selected value-to-majority-label map, ascending values.
  int         default_class{0}; ///< Global training majority; smallest integer label wins ties.
  std::size_t errors{0};        ///< Selected actual training errors,0 for empty model.
  std::size_t total{0};         ///< Original training rows,0 denotes empty/untrained.

  /// @brief Return selected-rule training accuracy, not held-out generalization.
  /// @return 1-errors/total for trained model, historical0 when total=0.
  /// @par Example
  /// @code
  /// dm::algorithms::OneRModel m;std::print("{}\n",m.accuracy()); // 0
  /// @endcode
  /// @see one_r_evaluate()
  double accuracy() const { return total ? 1. - static_cast<double>(errors) / total : 0.; }
};
/// @brief Actual sorted class tallies and majority decision for one feature value.
/// @par Example
/// @code
/// dm::algorithms::OneRBucket b;std::print("{}\n",b.errors); // 0
/// @endcode
/// @see one_r_train_trace()
struct OneRBucket {
  int value{0}; ///< Actual nominal feature value, including negative integers.
  std::map<int, std::size_t> class_counts; ///< Actual label counts in ascending class order.
  int         prediction{0}; ///< Actual most frequent label; smallest class wins ties.
  std::size_t total{0};      ///< Actual training rows in this value bucket.
  std::size_t errors{0};     ///< Actual bucket total minus largest class count.
};
/// @brief Actual candidate feature errors and value buckets, produced once in training.
/// @par Example
/// @code
/// dm::algorithms::OneRFeature f;std::print("{}\n",f.feature); // 0
/// @endcode
/// @see one_r_train_trace()
struct OneRFeature {
  std::size_t             feature{0}; ///< Original zero-based candidate feature.
  std::size_t             errors{0};  ///< Actual sum of bucket errors for this candidate.
  std::vector<OneRBucket> buckets; ///< Actual ascending-value buckets with original label tallies.
};
/// @brief Owned original training data, actual candidate bucket counts and selected model.
/// @par Example
/// @code
/// auto t=dm::algorithms::one_r_train_trace({{0},{1}},{0,1});std::print("{}\n",t.candidates.size()); // 1
/// @endcode
/// @see one_r_train_trace()
struct OneRTrace {
  std::vector<std::vector<int>> features; ///< Owned complete original row-major categorical matrix.
  std::vector<int>              labels;   ///< Owned complete original integer labels.
  std::vector<OneRFeature>      candidates; ///< Actual original feature-order candidate decisions.
  OneRModel                     model;      ///< Actual selected same-producer one-feature model.
  std::size_t                   captured_entries{
      0}; ///< Actual numeric/record units: inputsND+N, final map2R, candidates1, buckets1+2K.
};
/// @brief Owned categorical integer matrix, integer labels and optional display feature names.
/// @par Example
/// @code
/// auto d=dm::algorithms::one_r_weather_data();std::print("{}\n",d.labels.size()); // 14
/// @endcode
/// @see one_r_train_trace()
struct OneRData {
  std::vector<std::vector<int>> features; ///< Owned complete row-major integer categories.
  std::vector<int>              labels;   ///< Owned matching nominal integer response.
  std::vector<std::string>      names;    ///< Owned display names in feature-column order.
};
/// @brief Owned actual fixed-model predictions, raw confusion and selected counts.
/// @par Example
/// @code
/// auto m=dm::algorithms::one_r_train({{0},{1}},{0,1});auto e=dm::algorithms::one_r_evaluate(m,{{2}},{1});std::print("{}\n",e.fallbacks); // 1
/// @endcode
/// @see one_r_train_trace()
struct OneREvaluation {
  std::size_t rows{0};      ///< Actual evaluated rows,0 allowed.
  std::size_t correct{0};   ///< Actual prediction-equals-label rows.
  std::size_t fallbacks{0}; ///< Actual unseen selected-feature values using frozen default class.
  std::vector<int> predictions; ///< Owned actual fixed-model prediction per input row.
  std::vector<int> classes;     ///< Ascending union of actual labels and actual predicted labels.
  std::vector<std::vector<std::size_t>>
      confusion; ///< Actual raw counts indexed[true class position][predicted class position].
};
/// @brief Fit the canonical best single categorical attribute rule.
/// Validate complete row/label counts and rectangular shape before counting.
/// Nonempty training requires at least one feature; all int categories/labels,
/// including negative/extreme values, are nominal and accepted without ordering.
/// Empty matrices/labels return the normalized empty model(total0/errors0).
/// For each attribute, count labels within every value bucket and choose its
/// most frequent class. Smallest integer class wins ties. Select the first
/// original attribute with minimum training errors. Fallback is the smallest
/// globally tied majority class. No fitting on evaluation rows is performed.
/// No input mutation, retained references, callbacks or shared mutable state.
/// O(D*N*log(N)) map work, O(N) candidate working state plus selected rule map;
/// owned full input/history adds O(D*N) elements/counts, not a peak-memory claim.
/// @param features Complete original row-major nominal integer matrix.
/// @param labels Complete matching nominal integer classes.
/// @return Owned actual OneRModel.
/// @throws std::invalid_argument If complete row/label/shape validation fails.
/// @throws std::length_error If complete/capture geometry or selected quota fails.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// auto m=dm::algorithms::one_r_train({{0},{0},{1}},{1,0,1});std::print("{} {}\n",m.errors,m.default_class); // 1 1
/// @endcode
/// @par References
/// - R. C. Holte(1993), Very Simple Classification Rules Perform Well on Most Commonly Used Datasets, doi:10.1023/A:1022631118932. This native variant handles nominal integer categories without learned numeric discretization.
/// @see OneRModel, one_r_evaluate()
OneRModel one_r_train(
    const std::vector<std::vector<int>>& features,
    const std::vector<int>&              labels);
/// @brief Capture actual candidate counts and majority decisions from canonical OneR fitting.
/// Validate complete row/label counts and rectangular shape before counting.
/// Nonempty training requires at least one feature; all int categories/labels,
/// including negative/extreme values, are nominal and accepted without ordering.
/// Empty matrices/labels return the normalized empty model(total0/errors0).
/// For each attribute, count labels within every value bucket and choose its
/// most frequent class. Smallest integer class wins ties. Select the first
/// original attribute with minimum training errors. Fallback is the smallest
/// globally tied majority class. No fitting on evaluation rows is performed.
/// No input mutation, retained references, callbacks or shared mutable state.
/// O(D*N*log(N)) map work, O(N) candidate working state plus selected rule map;
/// owned full input/history adds O(D*N) elements/counts, not a peak-memory claim.
/// Quota counts original numeric entriesND+N, selected rule map2R, each
/// candidate1 and each bucket1+2K actual class-count entries. Fixed metadata,
/// map nodes and vector capacities excluded. Default1000000,0 disables.
/// Admission precedes each owned copy/append; no partial trace escapes failure.
/// @param features Complete original row-major nominal integer matrix.
/// @param labels Complete matching nominal integer classes.
/// @param max_entries Actual logical numeric/record quota, default1000000,0 disables.
/// @return Owned actual OneRTrace.
/// @throws std::invalid_argument If complete row/label/shape validation fails.
/// @throws std::length_error If complete/capture geometry or selected quota fails.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::one_r_train_trace({{0},{0},{1}},{1,0,1});std::print("{}\n",t.candidates[0].buckets[0].prediction); // 0
/// @endcode
/// @par References
/// - R. C. Holte(1993), Very Simple Classification Rules Perform Well on Most Commonly Used Datasets, doi:10.1023/A:1022631118932. This native variant handles nominal integer categories without learned numeric discretization.
/// @see OneRModel, one_r_evaluate()
OneRTrace one_r_train_trace(
    const std::vector<std::vector<int>>& features,
    const std::vector<int>&              labels,
    std::size_t                          max_entries = 1000000);
/// @brief Predict with one fixed trained nominal rule, falling back for unseen values.
/// Requires total>0, errors<=total, a nonempty rule and selected index<x.size().
/// Other feature values/columns are ignored. O(log V), no mutation or fitting.
/// @param m Complete trained native model, unchanged.
/// @param x Feature row containing the selected index, unchanged.
/// @return Frozen value rule prediction, or frozen global majority for unseen value.
/// @throws std::invalid_argument If model is empty/inconsistent or selected slot absent.
/// @par Example
/// @code
/// auto m=dm::algorithms::one_r_train({{0},{1}},{0,1});std::print("{}\n",dm::algorithms::one_r_predict(m,{2})); // 0
/// @endcode
/// @par References
/// - R. C. Holte(1993), Very Simple Classification Rules Perform Well on Most Commonly Used Datasets, doi:10.1023/A:1022631118932. This native variant handles nominal integer categories without learned numeric discretization.
/// @see one_r_evaluate()
int one_r_predict(const OneRModel& m, const std::vector<int>& x);
/// @brief Evaluate one frozen model without refitting any majority/rule on evaluation rows.
/// Complete matching row/label and rectangular shape are validated first. Nonempty
/// rows require the selected model slot; empty evaluation returns empty counts.
/// Raw confusion uses the sorted union of actual truth and predicted labels.
/// O(N log(V)+N log(K)+K squared) work/output; complete K squared geometry checked.
/// No input/model mutation, retained references or shared mutable state.
/// @param model Complete trained model,total>0,errors<=total,nonempty rule.
/// @param features Complete evaluation rows in the same feature-column order as fitting.
/// @param labels Complete true nominal classes, used only for evaluation counts.
/// @return Owned actual predictions, unseen-value fallback count and raw confusion.
/// @throws std::invalid_argument If complete model/data validation fails.
/// @throws std::length_error If complete confusion/output count geometry fails.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// auto m=dm::algorithms::one_r_train({{0},{1}},{0,1});auto e=dm::algorithms::one_r_evaluate(m,{{0},{2}},{0,1});std::print("{} {}\n",e.correct,e.fallbacks); // 1 1
/// @endcode
/// @par References
/// - R. C. Holte(1993), Very Simple Classification Rules Perform Well on Most Commonly Used Datasets, doi:10.1023/A:1022631118932. This native variant handles nominal integer categories without learned numeric discretization.
/// @see one_r_train(), one_r_dataframe()
OneREvaluation one_r_evaluate(
    const OneRModel&                     model,
    const std::vector<std::vector<int>>& features,
    const std::vector<int>&              labels);
/// @brief Copy complete numeric integer-coded DataFrame predictors/response into nominal rows.
/// Every selected cell must be finite, integral and within int range; no null
/// filtering, rounding, category fitting or data-dependent discretization. Feature
/// names must be nonempty/distinct and exclude response, which prevents target leakage.
/// Named columns are read in caller order. O(rows*columns) work and owned output;
/// frame is unchanged and no reference remains. Use identical column order on test.
/// @param frame Original native DataFrame, empty rows accepted.
/// @param predictors Nonempty distinct numeric nominal/count category columns.
/// @param response Numeric integer-coded class column, different from every predictor.
/// @return Owned categorical matrix, labels and predictor display names.
/// @throws std::invalid_argument If complete names/types/cells fail validation.
/// @throws std::out_of_range If a requested column is absent.
/// @throws std::length_error If complete matrix count geometry is unrepresentable.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// auto d=dm::algorithms::one_r_dataframe(dm::datasets::titanic(),{"Pclass","SibSp","Parch"},"Survived");std::print("{}\n",d.labels.size()); // 891
/// @endcode
/// @see one_r_train(), one_r_evaluate()
OneRData one_r_dataframe(
    const dstruct::DataFrame&       frame,
    const std::vector<std::string>& predictors,
    const std::string&              response);
/// @brief Return the original fourteen-row categorical play-tennis illustration.
/// This is a small didactic fixture, not an independently sampled held-out corpus.
/// Names outlook/temperature/humidity/windy match the original integer columns.
/// @return Fresh owned original categorical rows/labels/display names.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// auto d=dm::algorithms::one_r_weather_data();std::print("{} {}\n",d.features.size(),d.names.size()); // 14 4
/// @endcode
/// @see one_r_train_trace()
OneRData one_r_weather_data();
} // namespace datamunge::algorithms
