#pragma once
/// @file structured_svm.hpp
/// @brief Checked actual Crammer-Singer SGD, objective diagnostics and subgradient states.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the StructuredSVMModel contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMModel r{};
/// (void)r;
/// @endcode
/// @see structured_svm_run()
struct StructuredSVMModel {
  std::vector<std::vector<double>>
      weights; ///< Actual class rows of dim+1 coefficients, final coefficient a regularized bias.
  std::size_t
      num_classes{}; ///< Complete class count; zero denotes the legacy empty/unfitted model.
  std::size_t dim{}; ///< Complete feature width, not counting the unit bias coordinate.

  /// @brief Score one class through a validated finite augmented model.
  /// Requires model rows exactly num_classes by dim+1 and finite weights, class in range, finite feature vector of width dim. Empty/unfitted models reject. O(model values+dim) validation/scoring, borrowed values only and no mutation/shared state.
  /// @param c Actual class index.
  /// @param x Complete borrowed feature vector of width dim.
  /// @return Finite actual bias-augmented linear score.
  /// @throws std::invalid_argument Malformed model/features or class index.
  /// @throws std::overflow_error Actual score not representable.
  /// @par Example
  /// @code
  /// dm::algorithms::StructuredSVMModel m{{{2,1}},1,1};
  /// std::print("{}\n",m.score(0,{3})); // 7
  /// @endcode
  /// @see structured_svm_predict()
  double score(std::size_t c, const std::vector<double>& x) const;
};
/// @brief Own the StructuredSVMParameters contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMParameters r{};
/// (void)r;
/// @endcode
/// @see structured_svm_run()
struct StructuredSVMParameters {
  double lambda{.01}; ///< Finite positive regularization strength, including bias coefficients.
  std::size_t epochs{
      50}; ///< Actual n sampled-with-replacement updates per epoch, not a shuffled pass.
  std::uint64_t seed{}; ///< Local mt19937_64 seed.
};
/// @brief Own the StructuredSVMOptions contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMOptions r{};
/// (void)r;
/// @endcode
/// @see structured_svm_run()
struct StructuredSVMOptions {
  std::size_t max_model_values{1000000}; ///< Complete K*(dim+1) model geometry; zero disables.
  std::size_t max_updates{20000000};     ///< Complete sampled update count; zero disables.
  std::size_t max_selected_operations{
      200000000}; ///< Actual score terms, shrink/hinge assignments and objective norm terms; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual epoch/step records plus copied step-weight entries; zero disables.
};
/// @brief Own the StructuredSVMObjective contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMObjective r{};
/// (void)r;
/// @endcode
/// @see structured_svm_run()
struct StructuredSVMObjective {
  double regularizer{}; ///< Actual lambda/2 times all coefficient squared norms, including biases.
  double mean_hinge{};  ///< Actual mean max_c(score_c+[c!=y])-score_y, zero for empty diagnostics.
  double value{};       ///< Actual regularizer plus mean hinge.
  std::size_t
      correct{}; ///< Actual highest-score training predictions matching labels; lowest-index ties.
};
/// @brief Own the StructuredSVMEpoch contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMEpoch r{};
/// (void)r;
/// @endcode
/// @see structured_svm_run()
struct StructuredSVMEpoch {
  std::size_t epoch{};   ///< Actual completed epoch count; initial zero model is epoch0.
  std::size_t updates{}; ///< Actual completed sampled updates at this point.
  StructuredSVMObjective
      objective; ///< Actual current full-data objective/accuracy diagnostics, no refit.
};
/// @brief Own the StructuredSVMStep contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMStep r{};
/// (void)r;
/// @endcode
/// @see structured_svm_run()
struct StructuredSVMStep {
  std::size_t iteration{}; ///< Actual one-based learning-rate index t.
  std::size_t example{};   ///< Actual locally sampled training row.
  std::size_t truth{};     ///< Actual supplied class label.
  std::size_t violator{};  ///< Actual lowest-index loss-augmented maximizing class before shrink.
  double      rate{};      ///< Actual1/(lambda*t).
  double      shrink{};    ///< Actual1-rate*lambda.
  double      hinge_before{}; ///< Actual selected pre-update multiclass hinge value.
  std::vector<std::vector<double>>
      after; ///< Optional actual complete weights after this shrink/subgradient step.
};
/// @brief Own the StructuredSVMRun contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMRun r{};
/// (void)r;
/// @endcode
/// @see structured_svm_run()
struct StructuredSVMRun {
  StructuredSVMModel model;     ///< Actual final owned linear class weights.
  std::size_t        updates{}; ///< Actual sampled SGD steps completed.
  std::size_t
      score_terms{}; ///< Actual bias starts and feature-products in training and epoch diagnostics.
  std::size_t shrink_assignments{}; ///< Actual coefficient shrink multiplications.
  std::size_t
      hinge_assignments{}; ///< Actual signed feature/bias additions when violator differs from truth.
  std::size_t objective_norm_terms{}; ///< Actual coefficient-square terms in epoch diagnostics.
  std::vector<StructuredSVMEpoch>
      epochs; ///< Actual initial and post-epoch complete diagnostics; empty on empty legacy training input.
  std::vector<StructuredSVMStep> steps; ///< Optional actual sampled subgradient states, not replay.
};
/// @brief Train the actual regularized multiclass hinge model.
/// Crammer-Singer special case, not a general sequence/tree structured-output solver. Scores use an augmented unit bias coordinate, which is regularized and shrunk with the other weights. Each epoch makes n independent sampled-with-replacement updates using local mt19937_64; it is not one visit to every row. At t=1,2,... choose the lowest-index maximum of score_c+[c!=truth] before shrink; apply eta=1/(lambda*t), shrink every coefficient by1-eta*lambda, then add/subtract eta*x_aug to truth/violator if different. No Pegasos projection step is implemented. Zero epochs returns zero weights for nonempty valid data; empty matching inputs preserve the legacy empty/unfitted model. Complete finite rectangular inputs, labels and model/update geometry checked before fitting. Full-data initial/post-epoch diagnostics evaluate actual current weights; SGD objectives need not decrease every step/epoch. O(updates*K*(d+1)+epochs*n*K*(d+1)) selected work, owned model and optional model-copy histories; no shared state.
/// @param features Complete borrowed finite rectangular design matrix.
/// @param labels Complete matching class labels in0..K-1.
/// @param num_classes Positive class count at most int max for nonempty training.
/// @param params Finite positive lambda, actual epochs and local seed; defaults.01,50,0.
/// @param options Complete model/update and actual selected-work/capture policies; zero disables policies.
/// @return Owned actual final model, executed work, complete epoch diagnostics and optional sampled states.
/// @throws std::invalid_argument Malformed inputs, labels, class count or lambda.
/// @throws std::length_error Complete geometry/update, actual work/history or counter/container capacity exceeded.
/// @throws std::overflow_error Actual score/rate/coefficient/objective not representable.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMParameters p;p.epochs=0;
/// auto r=dm::algorithms::structured_svm_run({{1}},{0},1,p);
/// std::print("{} {}\n",r.model.num_classes,r.updates); // 1 0
/// @endcode
/// @par References
/// - K. Crammer and Y. Singer, On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines, JMLR2,2001,265-292. This implementation is linear regularized stochastic subgradient descent without projection.
/// @see structured_svm_train()
StructuredSVMRun structured_svm_run(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    std::size_t                             num_classes,
    const StructuredSVMParameters&          params  = {},
    const StructuredSVMOptions&             options = {});
/// @brief Train the actual regularized multiclass hinge model with sampled subgradient states.
/// Crammer-Singer special case, not a general sequence/tree structured-output solver. Scores use an augmented unit bias coordinate, which is regularized and shrunk with the other weights. Each epoch makes n independent sampled-with-replacement updates using local mt19937_64; it is not one visit to every row. At t=1,2,... choose the lowest-index maximum of score_c+[c!=truth] before shrink; apply eta=1/(lambda*t), shrink every coefficient by1-eta*lambda, then add/subtract eta*x_aug to truth/violator if different. No Pegasos projection step is implemented. Zero epochs returns zero weights for nonempty valid data; empty matching inputs preserve the legacy empty/unfitted model. Complete finite rectangular inputs, labels and model/update geometry checked before fitting. Full-data initial/post-epoch diagnostics evaluate actual current weights; SGD objectives need not decrease every step/epoch. O(updates*K*(d+1)+epochs*n*K*(d+1)) selected work, owned model and optional model-copy histories; no shared state.
/// @param features Complete borrowed finite rectangular design matrix.
/// @param labels Complete matching class labels in0..K-1.
/// @param num_classes Positive class count at most int max for nonempty training.
/// @param params Finite positive lambda, actual epochs and local seed; defaults.01,50,0.
/// @param options Complete model/update and actual selected-work/capture policies; zero disables policies.
/// @return Owned actual final model, executed work, complete epoch diagnostics and optional sampled states.
/// @throws std::invalid_argument Malformed inputs, labels, class count or lambda.
/// @throws std::length_error Complete geometry/update, actual work/history or counter/container capacity exceeded.
/// @throws std::overflow_error Actual score/rate/coefficient/objective not representable.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMParameters p;p.epochs=0;
/// auto r=dm::algorithms::structured_svm_trace({{1}},{0},1,p);
/// std::print("{} {}\n",r.model.num_classes,r.updates); // 1 0
/// @endcode
/// @par References
/// - K. Crammer and Y. Singer, On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines, JMLR2,2001,265-292. This implementation is linear regularized stochastic subgradient descent without projection.
/// @see structured_svm_train()
StructuredSVMRun structured_svm_trace(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    std::size_t                             num_classes,
    const StructuredSVMParameters&          params  = {},
    const StructuredSVMOptions&             options = {});
/// @brief Return the preserved checked multiclass SVM training model.
/// Same actual sampled shrink/subgradient producer with default policies. Bias is regularized; no kernel, projection or generic structured loss-augmented inference is implemented.
/// @param features Complete finite rectangular inputs.
/// @param labels Matching valid class labels.
/// @param num_classes Positive class count for nonempty inputs.
/// @param params Positive lambda, sampled epochs and local seed.
/// @return Owned actual fitted model, or legacy empty model for empty matching input.
/// @throws std::invalid_argument Malformed input/controls.
/// @throws std::length_error Default geometry/work/capacity exceeded.
/// @throws std::overflow_error Actual arithmetic not representable.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMParameters p;p.epochs=0;
/// std::print("{}\n",dm::algorithms::structured_svm_train({{1}},{0},1,p).num_classes); // 1
/// @endcode
/// @see structured_svm_run()
StructuredSVMModel structured_svm_train(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    std::size_t                             num_classes,
    const StructuredSVMParameters&          params);
/// @brief Predict the actual lowest-index highest-scoring class.
/// Validates complete finite model shape and feature width. Empty/unfitted models reject instead of returning an invented class0. Borrowed input/model, constant local state and no mutation.
/// @param model Complete finite fitted model.
/// @param x Complete finite feature vector of model.dim.
/// @return Actual maximum-score class index.
/// @throws std::invalid_argument Malformed/unfitted model or features.
/// @throws std::overflow_error Actual score not representable.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMModel m{{{1,0},{-1,0}},2,1};
/// std::print("{}\n",dm::algorithms::structured_svm_predict(m,{2})); // 0
/// @endcode
/// @see structured_svm_objective()
std::size_t structured_svm_predict(const StructuredSVMModel& model, const std::vector<double>& x);
/// @brief Evaluate the actual regularized full-data multiclass hinge objective.
/// No fitting or parameter mutation. Validates finite model and complete matching data/labels; an empty evaluation set has mean hinge0 and correct0 but retains model regularizer. Default selected-work bounds apply. Bias weights are included in the squared norm.
/// @param model Complete finite fitted model.
/// @param features Complete finite rectangular evaluation rows of model.dim.
/// @param labels Matching labels in model class range.
/// @param lambda Finite positive regularization strength.
/// @return Actual regularizer, mean hinge, total objective and correct prediction count.
/// @throws std::invalid_argument Malformed model/data/lambda.
/// @throws std::length_error Default selected-work/capacity exceeded.
/// @throws std::overflow_error Actual score/objective not representable.
/// @par Example
/// @code
/// dm::algorithms::StructuredSVMModel m{{{0,0},{0,0}},2,1};
/// std::print("{}\n",dm::algorithms::structured_svm_objective(m,{{1}},{1},.01).mean_hinge); // 1
/// @endcode
/// @see structured_svm_run()
StructuredSVMObjective structured_svm_objective(
    const StructuredSVMModel&               model,
    const std::vector<std::vector<double>>& features,
    const std::vector<int>&                 labels,
    double                                  lambda);
} // namespace datamunge::algorithms
