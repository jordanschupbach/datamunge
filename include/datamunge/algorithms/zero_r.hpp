#pragma once

/// @file zero_r.hpp
/// @brief DataFrame training, class counts and held-out evaluation for the integer-label ZeroR baseline.
#include <datamunge/algorithms/rule_learners.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <string>

namespace datamunge::algorithms {

/// @brief Train the constant majority-class predictor directly from a numeric response column.
///
/// Converts labels to int without rounding; nulls, non-finite values and labels
/// outside int's range are errors. Predictors are neither inspected nor required,
/// so missing features never drop rows. Ties select the smallest integer label.
/// Uses zero_r_train(vector<int>); O(n log K) time and O(n+K) temporary storage.
/// Empty input returns the same empty model as the vector overload.
/// @param frame Borrowed training dataset, unchanged.
/// @param response Numeric integer-coded class column.
/// @return A value model with prediction, majority_count and total; no reference to frame.
/// @throws std::invalid_argument If labels have wrong type, nulls or invalid values.
/// @throws std::out_of_range If response does not exist.
/// @par Example
/// @code
/// const auto model = dm::algorithms::zero_r_train(dm::datasets::titanic(), "Survived");
/// const auto prediction = model.prediction; // 0
/// @endcode
/// @par References
/// - Weka, ZeroR classifier (majority-class rule),
///   https://weka.sourceforge.io/doc.stable/weka/classifiers/rules/ZeroR.html
/// @see zero_r_train(const std::vector<int>&), zero_r_evaluate(), zero_r_counts()
ZeroRModel zero_r_train(const dstruct::DataFrame& frame, const std::string& response);

/// @brief Tabulate observed integer-coded classes, counts and relative frequencies.
/// @param frame Borrowed data, unchanged.
/// @param response Numeric integer-coded class column, same constraints as zero_r_train().
/// @return Owned columns class, count, proportion, sorted by class; empty input gives no rows.
/// @throws std::invalid_argument If any label/type is invalid.
/// @throws std::out_of_range If response is absent.
/// @par Example
/// @code
/// const auto counts = dm::algorithms::zero_r_counts(dm::datasets::titanic(), "Survived");
/// const auto classes = counts.nrows(); // 2
/// @endcode
/// @see zero_r_train(), zero_r_confusion()
dstruct::DataFrame zero_r_counts(const dstruct::DataFrame& frame, const std::string& response);

/// @brief Per-class confusion summaries for a fixed ZeroR decision on evaluation data.
///
/// Rows contain class, support, predicted, true_positive, recall and precision.
/// The classes are the sorted union of observed labels and model.prediction.
/// Zero-denominator precision and recall are zero. Data are read once; the
/// model is never retrained, even if the evaluation majority differs.
/// @param model Trained model with total>0.
/// @param frame Nonempty evaluation dataset, borrowed and unchanged.
/// @param response Numeric integer-coded labels, no nulls.
/// @return Owned class-level summary; predictors need not be present.
/// @throws std::invalid_argument If model/evaluation data are empty or labels/type are invalid.
/// @throws std::out_of_range If response is absent.
/// @par Example
/// @code
/// const auto data = dm::datasets::titanic();
/// const auto model = dm::algorithms::zero_r_train(data, "Survived");
/// const auto confusion = dm::algorithms::zero_r_confusion(model, data, "Survived");
/// const double positive_recall = confusion.double_at("recall", 1); // 0
/// @endcode
/// @see zero_r_evaluate(), zero_r_counts()
dstruct::DataFrame zero_r_confusion(const ZeroRModel& model, const dstruct::DataFrame& frame,
                                    const std::string& response);

/// @brief Evaluate a fixed ZeroR prediction with ordinary and balanced accuracy.
///
/// Balanced accuracy averages recalls only over classes observed in evaluation,
/// including every minority class. It is 1/K when the predicted class is among
/// K observed classes, and zero if that class is absent. Training accuracy is
/// reported separately from evaluation accuracy; class proportions may change.
/// @param model Trained model with total>0, unchanged.
/// @param frame Nonempty evaluation dataset, unchanged.
/// @param response Numeric integer-coded labels, no nulls.
/// @return One owned row: prediction, training_rows, training_accuracy,
///         evaluated_rows, observed_classes, accuracy, balanced_accuracy.
/// @throws std::invalid_argument If model/data are empty or labels/type are invalid.
/// @throws std::out_of_range If response is absent.
/// @par Example
/// @code
/// const auto data = dm::datasets::titanic();
/// const auto model = dm::algorithms::zero_r_train(data, "Survived");
/// const auto scores = dm::algorithms::zero_r_evaluate(model, data, "Survived");
/// const double balanced = scores.double_at("balanced_accuracy", 0); // 0.5
/// @endcode
/// @par References
/// - K. H. Brodersen et al. (2010), The Balanced Accuracy and Its Posterior Distribution,
///   https://doi.org/10.1109/ICPR.2010.764
/// @see zero_r_confusion(), zero_r_train()
dstruct::DataFrame zero_r_evaluate(const ZeroRModel& model, const dstruct::DataFrame& frame,
                                   const std::string& response);
} // namespace datamunge::algorithms
