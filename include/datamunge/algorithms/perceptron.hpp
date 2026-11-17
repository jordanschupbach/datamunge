#pragma once
/// @file perceptron.hpp
/// @brief Checked ordered binary perceptron with actual online visit history.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief A trained Perceptron: a linear decision rule \f$\operatorname{sign}(w\cdot x + b)\f$.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::perceptron_train({{1}},{1},0).epochs_run); // 0
/// @endcode
struct PerceptronModel {
  std::vector<double> weights;                ///< Weight vector \f$w\f$.
  double              bias           = 0.0;   ///< Bias \f$b\f$.
  std::size_t         epochs_run     = 0;     ///< Passes over the data actually made.
  bool                converged      = false; ///< True if an entire epoch made no mistake.
  std::size_t         total_mistakes = 0;     ///< Total updates across all epochs.
};

/// @brief One actual online visit, including zero-margin updates.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::perceptron_trace({{1}},{1},1).visits[0].updated); // true
/// @endcode
struct PerceptronVisit {
  std::size_t         epoch;   ///< One-based pass number.
  std::size_t         row;     ///< Original training row id.
  double              score;   ///< Actual pre-update dot product plus bias.
  bool                updated; ///< True exactly when label*score <= 0.
  std::vector<double> weights; ///< Complete post-visit weights.
  double              bias;    ///< Post-visit bias.
};
/// @brief Owned model and actual ordered visits from the producing fit.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::perceptron_trace({{1}},{1},1).model.total_mistakes); // 1
/// @endcode
struct PerceptronTrace {
  PerceptronModel              model;  ///< Actual returned model, including certification pass.
  std::vector<PerceptronVisit> visits; ///< Actual visits; not a refit or replay.
};
/// @brief Train in fixed row order, updating on negative or zero signed margin.
/// Starts w,b at zero; a complete zero-update pass marks converged. At a finite
/// epoch cap returns last weights, without optimality or generalisation claim.
/// The classical bound applies to augmented vectors (x,1) and their radius/margin,
/// since this implementation also updates the bias. Arithmetic keeps original
/// double operation order; nonfinite intermediate values throw. O(epochs*n*d)
/// time, O(d) model space. Input is borrowed unchanged; no shared mutable state.
/// @param features Rectangular finite rows; empty returns an untrained empty model;
/// zero features are valid and fit only a bias.
/// @param labels Exactly one -1 or +1 per row.
/// @param max_epochs Maximum complete passes, default 100; zero returns zero weights.
/// @param learning_rate Positive finite additive step, default one.
/// @return Owned fitted model; converged requires an actual mistake-free pass.
/// @throws std::invalid_argument If shapes, labels, inputs or rate are invalid.
/// @throws std::overflow_error If score/update arithmetic or counters overflow.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::perceptron_train({{1},{-1}},{1,-1}).converged); // true
/// @endcode
/// @par References
/// - F. Rosenblatt (1958), The perceptron, Psychological Review 65, 386-408,
///   doi:10.1037/h0042519.
PerceptronModel perceptron_train(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    std::size_t                             max_epochs    = 100,
    double                                  learning_rate = 1);
/// @brief Retain each actual online visit from perceptron_train's shared kernel.
/// O(visits*(d+1)) additional history. No second fit or reconstruction is used.
/// @param features Complete finite rectangular rows, unchanged.
/// @param labels Aligned -1/+1 labels.
/// @param max_epochs Maximum passes.
/// @param learning_rate Positive finite additive step.
/// @param max_entries Logical retained entries: one visit record plus d weights
/// per actual visit; default one million, zero disables policy. Excludes model.
/// @return Owned model and every actual pre-score/post-state visit.
/// @throws std::invalid_argument If training arguments are invalid.
/// @throws std::overflow_error If arithmetic or counters overflow.
/// @throws std::length_error If retained-history policy is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::perceptron_trace({{1}},{1},1).visits.size()); // 1
/// @endcode
PerceptronTrace perceptron_trace(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    std::size_t                             max_epochs    = 100,
    double                                  learning_rate = 1,
    std::size_t                             max_entries   = 1000000);
/// @brief Predict by a checked finite dot product plus bias; score zero predicts -1.
/// @param model Finite weights and bias; other training metadata is unused.
/// @param x Exactly model.weights.size() finite features.
/// @return +1 above zero, -1 at or below zero; O(d), no mutation or retained references.
/// @throws std::invalid_argument If dimensions or model/input values are invalid.
/// @throws std::overflow_error If score arithmetic overflows.
/// @par Example
/// @code
/// dm::algorithms::PerceptronModel m;std::print("{}\n",dm::algorithms::perceptron_predict(m,{})); // -1
/// @endcode
int perceptron_predict(const PerceptronModel& model, const std::vector<double>& x);
} // namespace datamunge::algorithms
