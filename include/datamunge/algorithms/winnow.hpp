#pragma once

/// @file winnow.hpp
/// @brief Boolean DataFrame adapters and a shared-loop training trace for Winnow2.
#include <datamunge/algorithms/online_linear_classifiers.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <string>

namespace datamunge::algorithms {

/// @brief Fitted Winnow model, feature order, pass history and final weight table.
///
/// All fields own their contents. history has epoch, mistakes, false_positives,
/// false_negatives, cumulative_mistakes, accuracy and total_weight. weights has
/// feature, index, weight and log2_weight, in predictor order. Accuracy in history
/// evaluates the post-pass weights; mistake counts record online decisions made
/// while weights were changing. They measure different things.
/// @par Example
/// @code
/// const auto binary=dm::algorithms::winnow_disjunction_data(dm::datasets::digits(),
///     {"pixel_27","pixel_28"},{"pixel_27","pixel_28"});
/// const auto fit=dm::algorithms::winnow_fit(binary);
/// const auto dimensions=fit.predictors.size(); // 2
/// @endcode
/// @see winnow_fit(), winnow_evaluate()
struct WinnowFit {
  WinnowModel model; ///< Owned final weights and training counters.
  std::vector<std::string> predictors; ///< Ordered names used to encode new rows.
  dstruct::DataFrame history; ///< One owned record per completed pass, including a zero-mistake final pass.
  dstruct::DataFrame weights; ///< Owned final named weight table, without filtering by relevance.
};

/// @brief Construct a known sparse Boolean OR target from thresholded numeric measurements.
///
/// Each selected feature becomes one exactly when its measurement exceeds cutoff.
/// The response is the OR of the selected relevant features. This is an engineered
/// concept for studying learning, not the source dataset's original classification
/// task. Other measurements can be correlated with relevant features; there is no
/// independent-noise assumption. No parameters are estimated from the frame.
/// Inputs are borrowed and unchanged; O(n*d) time and result storage.
/// @param data Numeric measurements; selected cells must be non-null and finite.
/// @param predictors Nonempty ordered unique column names to threshold.
/// @param relevant Unique subset of predictors defining the OR; empty gives all-zero labels.
/// @param cutoff Finite common cutoff, default 8; equality encodes zero.
/// @param response Output target name, default "target"; nonempty and distinct from predictors.
/// @return Owned Boolean numeric columns followed by the numeric target, in original row order.
/// @throws std::invalid_argument If names, types, selected cells or cutoff are invalid.
/// @par Example
/// @code
/// const auto binary=dm::algorithms::winnow_disjunction_data(dm::datasets::digits(),
///     {"pixel_27","pixel_28"},{"pixel_27","pixel_28"});
/// const auto rows=binary.nrows(); // 1797
/// @endcode
/// @see winnow_fit(), winnow_train()
dstruct::DataFrame winnow_disjunction_data(const dstruct::DataFrame& data,
                                          const std::vector<std::string>& predictors,
                                          const std::vector<std::string>& relevant,
                                          double cutoff=8.0,const std::string& response="target");

/// @brief Fit a Boolean DataFrame with the same update loop as winnow_train(), retaining pass diagnostics.
///
/// Uses every column except response as a predictor, in frame order. Every
/// feature and response must be numeric zero or one; missing/non-finite values
/// are rejected rather than dropped. The trace recomputes post-pass accuracy
/// without updating the weights. O(epochs*n*d) time, O(n*d+epochs+d) storage.
/// No shared mutable state; caller frames are borrowed and unchanged.
/// @param data Encoded Boolean frame with at least one predictor.
/// @param response Numeric target column, default "target".
/// @param alpha Finite update factor greater than one, default two.
/// @param threshold Finite threshold; negative selects dimension, default -1.
/// @param max_epochs Maximum passes, default 100; zero preserves initial weights.
/// @return Owned model, names, complete per-pass history and final weights.
/// @throws std::invalid_argument If schema, Boolean values or options are invalid.
/// @throws std::overflow_error If score/weight arithmetic leaves the positive finite range.
/// @par Example
/// @code
/// const auto binary=dm::algorithms::winnow_disjunction_data(dm::datasets::digits(),
///     {"pixel_27","pixel_28"},{"pixel_27","pixel_28"});
/// const auto fit=dm::algorithms::winnow_fit(binary);
/// const bool converged=fit.model.converged; // true
/// @endcode
/// @par References
/// - N. Littlestone (1988), https://ai.stanford.edu/~pabbeel/depth_qual/littlestone1988.pdf
/// @see winnow_train(), winnow_evaluate()
WinnowFit winnow_fit(const dstruct::DataFrame& data,const std::string& response="target",
                     double alpha=2.0,double threshold=-1.0,std::size_t max_epochs=100);

/// @brief Evaluate a fixed fitted Winnow rule on labelled Boolean rows without updating it.
///
/// Uses fit.predictors in their fitted order; unrelated columns are ignored.
/// Boolean/schema validation matches winnow_fit(). Empty evaluation returns
/// zero counts and NaN accuracy. O(n*d) time and temporary storage.
/// @param fit Read-only fitted model and predictor-name mapping.
/// @param data Encoded Boolean evaluation frame, borrowed and unchanged.
/// @param response Numeric target name, default "target".
/// @return One owned row: rows, positives, predicted_positives, false_positives,
/// false_negatives, mistakes and accuracy (a fraction).
/// @throws std::invalid_argument If schema, Boolean values, dimensions or model state are invalid.
/// @throws std::overflow_error If score arithmetic overflows.
/// @par Example
/// @code
/// const auto binary=dm::algorithms::winnow_disjunction_data(dm::datasets::digits(),
///     {"pixel_27","pixel_28"},{"pixel_27","pixel_28"});
/// const auto fit=dm::algorithms::winnow_fit(binary);
/// const auto result=dm::algorithms::winnow_evaluate(fit,binary);
/// const auto mistakes=result.double_at("mistakes",0); // 0
/// @endcode
/// @see winnow_fit(), winnow_predict()
dstruct::DataFrame winnow_evaluate(const WinnowFit& fit,const dstruct::DataFrame& data,
                                   const std::string& response="target");

/// @brief Probe whether a frozen fitted rule matches a specified OR on simple Boolean counterfactuals.
///
/// Tests all-zero input, each relevant feature alone, all other features on,
/// and all relevant features on. These are valid Boolean vectors but may be
/// outside the source dataset's support. No model update occurs. Passing this
/// finite suite does not prove equivalence on every Boolean vector.
/// O((k+3)*d) time and O(k+d) temporary/result storage for k relevant features.
/// @param fit Read-only valid model and nonempty unique predictor-name mapping.
/// @param relevant Unique subset specifying the assumed monotone OR; empty is allowed.
/// @return Owned probe, score, expected, predicted and correct columns; expected labels follow that OR.
/// @throws std::invalid_argument If the model, mapping or relevant names are invalid.
/// @throws std::overflow_error If score arithmetic overflows.
/// @par Example
/// @code
/// const auto binary=dm::algorithms::winnow_disjunction_data(dm::datasets::digits(),
///     {"pixel_27","pixel_28"},{"pixel_27","pixel_28"});
/// const auto fit=dm::algorithms::winnow_fit(binary);
/// const auto probes=dm::algorithms::winnow_disjunction_probes(fit,{"pixel_27","pixel_28"});
/// const auto rows=probes.nrows(); // 5
/// @endcode
/// @see winnow_evaluate(), winnow_predict()
dstruct::DataFrame winnow_disjunction_probes(const WinnowFit& fit,
                                             const std::vector<std::string>& relevant);

} // namespace datamunge::algorithms
