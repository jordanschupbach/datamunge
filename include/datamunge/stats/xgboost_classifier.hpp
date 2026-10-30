#pragma once

/// @file xgboost_classifier.hpp
/// @brief Formula-driven regularised softmax tree boosting and held-out evaluation.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/detail/xgboost_tree.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Hyperparameters for the local XGBoost-style multiclass classifier.
///
/// Numeric settings are validated on model construction. This implementation
/// uses diagonal p*(1-p) Hessians floored at 1e-6; XGBoost 3.1.0's multiclass objective
/// uses a different curvature bound, so identical settings need not yield
/// identical fitted models. No automatic early stopping occurs.
/// @par Example
/// @code
/// dm::stats::XGBoostClassifierOptions options;
/// options.n_trees=20; options.learning_rate=0.1; options.max_depth=2;
/// @endcode
/// @par References
/// - T. Chen and C. Guestrin (2016), XGBoost: A Scalable Tree Boosting System,
///   https://doi.org/10.1145/2939672.2939785
/// @see XGBoostClassifier
struct XGBoostClassifierOptions {
    std::size_t n_trees{100}; ///< Positive boosting rounds; each fits one tree per class.
    double learning_rate{0.3}; ///< Finite positive multiplier of each round's leaf weights.
    std::size_t max_depth{6}; ///< Maximum tree depth; zero allows only the root leaf.
    double lambda{1.0}; ///< Finite nonnegative L2 leaf penalty.
    double alpha{0.0}; ///< Finite nonnegative L1 leaf penalty.
    double gamma{0.0}; ///< Finite nonnegative split-gain penalty.
    double min_child_weight{1.0}; ///< Finite nonnegative minimum summed Hessian per child.
    std::size_t min_samples_leaf{1}; ///< Positive minimum rows per leaf; at most SIZE_MAX/2.
    double subsample{1.0}; ///< Finite row fraction in (0,1], without replacement per round.
    double colsample_bytree{1.0}; ///< Finite feature fraction in (0,1], selected per class tree.
    std::uint64_t seed{42}; ///< Deterministic row/column subsampling seed.
};

/// @brief Owned class labels and probabilities aligned with the original prediction rows.
///
/// Probability columns follow XGBoostClassifier::classes(), in sorted label
/// order. Rows with null predictors have an empty label and probability vector.
/// @par Example
/// @code
/// const auto iris=dm::datasets::iris();
/// const dm::stats::XGBoostClassifier model(iris,"Species ~ Petal.Length + Petal.Width");
/// const auto result=model.predict_detail(iris.head(2));
/// const auto rows=result.probability.size(); // 2
/// @endcode
/// @see XGBoostClassifier::predict_detail()
struct XGBoostClassifierPrediction {
    std::vector<std::string>         class_label; ///< One label per original row; empty for missing predictors.
    std::vector<std::vector<double>> probability; ///< [original row][sorted class], or empty for missing predictors.
};

/// @brief Local regularised second-order tree boosting with softmax class probabilities.
///
/// Fits one tree per class per round to gradients p-y and diagonal Hessians
/// max(p*(1-p),1e-6), starting from log training-class proportions. Scores are
/// updated simultaneously by class within each round; this is a diagonal
/// Newton heuristic, not an exact full-Hessian Newton step or the complete
/// upstream XGBoost system. Formula resolution drops rows with null predictors
/// or response; categorical responses are string labels. Finite transformed
/// predictors are required. Training summaries are in-sample quantities.
/// Models own their fitted design and trees; concurrent const predictions are
/// safe when the input frame is not mutated. The exact split search rescans
/// candidate thresholds: at most O(M*K*(p+1)*n^2*(D+1)) training work for depth D,
/// with O(M*K*2^D) tree storage (also bounded by the available training rows).
/// @par Example
/// @code
/// const dm::stats::XGBoostClassifier model(dm::datasets::iris(),"Species ~ Petal.Length + Petal.Width");
/// const auto classes=model.classes().size(); // 3
/// @endcode
/// @par References
/// - T. Chen and C. Guestrin (2016), https://doi.org/10.1145/2939672.2939785
/// @see staged_evaluation(), evaluation(), XGBoostClassifierOptions
class XGBoostClassifier {
 public:
    /// @brief Fit all configured rounds using the supplied training frame and formula.
    /// @param data Borrowed training frame, unchanged; response must contain at least two string classes.
    /// @param formula Formula naming response and predictors; null complete-case rows are omitted.
    /// @param options Validated hyperparameters; see XGBoostClassifierOptions for defaults and limits.
    /// @throws std::invalid_argument If options, transformed predictors, accumulated scores, formula or response classes are invalid.
    /// @throws std::runtime_error If no complete training rows remain.
    /// @throws std::out_of_range If a required predictor column is missing.
    /// @throws std::domain_error If a formula transform is outside its mathematical domain.
    /// @par Example
    /// @code
    /// dm::stats::XGBoostClassifierOptions options; options.n_trees=10;
    /// const dm::stats::XGBoostClassifier model(dm::datasets::iris(),"Species ~ Petal.Length",options);
    /// const auto rounds=model.n_trees(); // 10
    /// @endcode
    /// @par References
    /// - T. Chen and C. Guestrin (2016), https://doi.org/10.1145/2939672.2939785
    /// @see staged_evaluation(), predict_detail()
    XGBoostClassifier(const dstruct::DataFrame& data, const std::string& formula,
                      XGBoostClassifierOptions options = {});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `XGBoostClassifier("y ~ x", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument As the primary constructor.
    /// @throws std::runtime_error If no complete rows remain.
    /// @throws std::out_of_range If a required predictor column is missing.
    /// @throws std::domain_error If a formula transform is outside its mathematical domain.
    /// @par Example
    /// @code
    /// const dm::stats::XGBoostClassifier model(dm::formula("Species ~ Petal.Length"),dm::datasets::iris());
    /// const auto classes=model.classes().size(); // 3
    /// @endcode
    /// @see XGBoostClassifier(const dstruct::DataFrame&, const std::string&, XGBoostClassifierOptions)
    XGBoostClassifier(const Formula& formula, const dstruct::DataFrame& data, XGBoostClassifierOptions options = {})
        : XGBoostClassifier(data, formula.text(), std::move(options)) {}

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] std::size_t                     n_trees() const { return trees_.size(); }

    // Gain-based importance summed across all trees in all rounds, normalized to sum to 1.
    [[nodiscard]] std::vector<double> feature_importance() const;

    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    [[nodiscard]] double                          training_accuracy() const;
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;
    [[nodiscard]] const std::vector<double>&  training_deviance() const { return training_deviance_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    /// @brief Predict sorted-class argmax labels, preserving original row alignment.
    /// @param newdata Borrowed prediction frame; the response is not needed.
    /// @return Owned label per row, empty for null predictors; ties use the first sorted class.
    /// @throws std::invalid_argument If a used transformed predictor or score is non-finite.
    /// @throws std::runtime_error If categorical predictor levels are unseen.
    /// @throws std::out_of_range If a required predictor column is missing.
    /// @throws std::domain_error If a formula transform is outside its mathematical domain.
    /// @par Example
    /// @code
    /// const dm::stats::XGBoostClassifier model(dm::datasets::iris(),"Species ~ Petal.Length");
    /// const auto labels=model.predict(dm::datasets::iris().head(1));
    /// const auto label=labels[0]; // "setosa"
    /// @endcode
    /// @see predict_detail()
    [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& newdata) const;
    /// @brief Return full-model softmax probabilities and class labels in original row order.
    ///
    /// A null predictor leaves that row's label and probability vector empty.
    /// The response is not required. Ties choose the first sorted class. Class
    /// probabilities use max-subtracted softmax; they are not calibrated posteriors.
    /// O(n*M*K*D) tree traversal, with O(n*K) temporary/result storage.
    /// @param newdata Borrowed frame containing the fitted predictor schema.
    /// @return Owned labels and probabilities; see XGBoostClassifierPrediction.
    /// @throws std::invalid_argument If a used transformed predictor or accumulated score is non-finite.
    /// @throws std::runtime_error If a categorical predictor level was unseen at training.
    /// @throws std::out_of_range If a required predictor column is missing.
    /// @throws std::domain_error If a formula transform is outside its mathematical domain.
    /// @par Example
    /// @code
    /// const dm::stats::XGBoostClassifier model(dm::datasets::iris(),"Species ~ Petal.Length");
    /// const auto result=model.predict_detail(dm::datasets::iris().head(1));
    /// const auto columns=result.probability[0].size(); // 3
    /// @endcode
    /// @see classes(), staged_evaluation()
    [[nodiscard]] XGBoostClassifierPrediction predict_detail(const dstruct::DataFrame& newdata) const;

    /// @brief Evaluate a fixed fitted ensemble after every round, including its round-zero priors.
    ///
    /// Resolves the supplied frame with the training design, omits null response
    /// or predictor rows, and traverses each fitted tree once per usable row.
    /// No refitting or state mutation occurs. mean_log_loss is -mean(log(p_true))
    /// with p_true floored at 1e-15, matching training_deviance(); brier is
    /// mean(sum_k (p_k-y_k)^2), without division by class count. Accuracy is
    /// a fraction. The same complete-case rows are used at every round.
    /// O(n*M*K*D) traversal plus O(n*M*K) scoring; O(n*K+M) working/result storage.
    /// @param data Borrowed labelled evaluation data with the training predictor schema.
    /// @return Owned round, input_rows, evaluated_rows, skipped_rows, accuracy,
    /// mean_log_loss and brier columns for rounds 0 through n_trees().
    /// @throws std::invalid_argument If a complete-case response class is absent from training, the response
    /// schema is invalid, or a used transformed predictor or accumulated score is non-finite.
    /// @throws std::runtime_error If no complete evaluation rows remain or predictor levels are unseen.
    /// @throws std::out_of_range If a required predictor column is missing.
    /// @throws std::domain_error If a formula transform is outside its mathematical domain.
    /// @par Example
    /// @code
    /// dm::stats::XGBoostClassifierOptions options; options.n_trees=5;
    /// const dm::stats::XGBoostClassifier model(dm::datasets::iris(),"Species ~ Petal.Length",options);
    /// const auto curve=model.staged_evaluation(dm::datasets::iris());
    /// const auto rows=curve.nrows(); // 6, including round zero
    /// @endcode
    /// @par References
    /// - T. Chen and C. Guestrin (2016), https://doi.org/10.1145/2939672.2939785
    /// - G. W. Brier (1950), Verification of Forecasts Expressed in Terms of Probability,
    ///   https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2
    /// @see evaluation(), training_deviance()
    [[nodiscard]] dstruct::DataFrame staged_evaluation(const dstruct::DataFrame& data) const;

    /// @brief Return full-model held-out metrics using the same complete-case rules as staged_evaluation().
    ///
    /// Computes the staged path and returns its last row; never refits the model.
    /// Time/storage and score definitions are those of staged_evaluation().
    /// @param data Borrowed labelled evaluation frame, unchanged.
    /// @return One owned row with the final round's counts, accuracy, mean_log_loss and brier.
    /// @throws std::invalid_argument If schema, classes or finite-value constraints are violated.
    /// @throws std::runtime_error If no complete rows remain or categorical predictor levels are unseen.
    /// @throws std::out_of_range If a required predictor column is missing.
    /// @throws std::domain_error If a formula transform is outside its mathematical domain.
    /// @par Example
    /// @code
    /// const dm::stats::XGBoostClassifier model(dm::datasets::iris(),"Species ~ Petal.Length");
    /// const auto scores=model.evaluation(dm::datasets::iris());
    /// const auto rows=scores.double_at("evaluated_rows",0); // 150
    /// @endcode
    /// @see staged_evaluation(), confusion()
    [[nodiscard]] dstruct::DataFrame evaluation(const dstruct::DataFrame& data) const;

    /// @brief Count actual versus predicted labels for a fixed model on complete evaluation rows.
    /// @param data Borrowed labelled evaluation frame, with staged_evaluation()'s validity rules.
    /// @return Owned actual column and pred_<class> count columns, in classes() order, including zero rows.
    /// @throws std::invalid_argument If classes, transformed predictors or response schema are invalid.
    /// @throws std::runtime_error If no complete rows remain or predictor levels are unseen.
    /// @throws std::out_of_range If a required predictor column is missing.
    /// @throws std::domain_error If a formula transform is outside its mathematical domain.
    /// @par Example
    /// @code
    /// const dm::stats::XGBoostClassifier model(dm::datasets::iris(),"Species ~ Petal.Length");
    /// const auto counts=model.confusion(dm::datasets::iris());
    /// const auto rows=counts.nrows(); // 3
    /// @endcode
    /// @see evaluation(), confusion_matrix()
    [[nodiscard]] dstruct::DataFrame confusion(const dstruct::DataFrame& data) const;

    [[nodiscard]] graphics::RPlot plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                         const std::string& y_feature) const;
    [[nodiscard]] graphics::RPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                          std::size_t grid_resolution = 60) const;
    [[nodiscard]] graphics::RPlot plot_training_deviance() const;

 private:
    void fit(const dstruct::DataFrame& data);
    linalg::DenseMatrix<double> score_matrix(const linalg::DenseMatrix<double>& X, std::size_t n_rows) const;

    Formula                    formula_;
    DesignInfo                 design_;
    XGBoostClassifierOptions   options_;

    std::vector<std::string> classes_;
    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_min_;
    std::vector<double>      feature_max_;
    std::vector<double>      base_scores_; // per-class log-prior, aligned with classes_

    // trees_[round][class]
    std::vector<std::vector<detail::XGBoostTree>> trees_;

    std::vector<std::string> training_labels_;
    std::vector<std::string> fitted_classes_;
    std::vector<double>      training_deviance_;
    std::size_t               observations_{0};
};

} // namespace datamunge::stats
