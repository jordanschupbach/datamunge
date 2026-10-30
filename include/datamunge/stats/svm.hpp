#pragma once

/// @file svm.hpp
/// @brief Formula-based soft-margin SVM with pairwise classification and optimality diagnostics.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/formula.hpp>
#include <datamunge/plot/plot.hpp>

#include <array>
#include <optional>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Kernel used for all one-vs-one binary models; see SVM for definitions.
enum class SVMKernel {
    Linear, ///< Dot product.
    Polynomial, ///< (gamma * dot product + coef0)^degree.
    Radial, ///< exp(-gamma * squared distance), positive-semidefinite for gamma > 0.
    Sigmoid ///< tanh(gamma * dot product + coef0); may be indefinite.
};

/// @brief SVM kernel, penalty and deterministic simplified-SMO settings.
struct SVMOptions {
    SVMKernel   kernel{SVMKernel::Radial}; ///< Kernel, radial by default.
    double      cost{1.0};           ///< C: box constraint on the dual alphas
    double      gamma{-1.0};         ///< <= 0 means "auto" = 1 / number of predictors
    double      coef0{0.0};          ///< used by Polynomial and Sigmoid kernels
    int         degree{3};           ///< used by the Polynomial kernel
    double      tolerance{1e-3};     ///< SMO KKT-violation tolerance
    std::size_t max_passes{50};      ///< no-change sweep limit; also capped at max(200, 20*max_passes) total sweeps
    bool        scale{true};         ///< standardize numeric predictors (zero mean, unit variance)
};

/// @brief Predicted labels and uncalibrated one-vs-one vote counts.
struct SVMPrediction {
    std::vector<std::string>         class_label; ///< predicted class per row ("" if predictors were missing)
    std::vector<std::vector<double>> votes;        ///< [row][class] one-vs-one vote counts
};

/// @brief Soft-margin multi-class SVM, similar to R's e1071::svm, with deterministic simplified SMO.
///
/// Trains one binary classifier per class pair. The sorted first class has sign +1;
/// non-negative pair decisions vote for it, and final vote ties use sorted class order.
/// Missing response/predictor rows are dropped for fitting. Formula intercept terms are
/// ignored. Numeric design columns are scaled by training means and sample SDs by default.
/// Prediction uses those fixed statistics. Votes are not calibrated probabilities.
/// The dense pair Gram matrix costs O(m^2) space; each coordinate sweep costs O(m^2).
/// Fits stop on unchanged-sweep or total-sweep budgets; inspect diagnostics() for accuracy.
/// Const methods do not mutate fitted state; returned tables/matrices are owned copies.
///
/// @par Example
/// @code
/// const auto iris = dm::datasets::iris();
/// const dm::stats::SVM model(iris, "Species ~ .");
/// std::print("{:.4f}\n", model.training_accuracy()); // 0.9733
/// @endcode
///
/// @par References
/// - C. Cortes, V. Vapnik (1995). Support-vector networks. Machine Learning 20, 273-297.
///   https://doi.org/10.1007/BF00994018
/// - J. C. Platt (1998). Sequential Minimal Optimization: A Fast Algorithm for Training
///   Support Vector Machines. Microsoft Research, MSR-TR-98-14.
/// @see SVMOptions, diagnostics(), training_details()
class SVM {
 public:
    /// @brief Fits a soft-margin classifier using training-only preprocessing.
    /// @param data Training frame; rows with null predictors/response are omitted.
    /// @param formula R-style formula with a string response and at least one design predictor.
    /// @param options Kernel, positive finite cost and tolerance, finite gamma/coef0,
    ///        degree >= 1, max_passes in [1, SIZE_MAX/20]; defaults as SVMOptions.
    /// @throws std::invalid_argument For invalid options/formula/types, fewer than two
    ///         classes, no predictors, or non-finite design/kernel values.
    /// @throws std::runtime_error If no complete training rows remain.
    /// @throws std::domain_error If a formula transformation is outside its domain.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::SVM model(dm::datasets::iris(), "Species ~ .");
    /// const auto predictions = model.predict(dm::datasets::iris().head(3));
    /// @endcode
    /// @see SVM, diagnostics()
    SVM(const dstruct::DataFrame& data, const std::string& formula, SVMOptions options = {});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `SVM("y ~ x", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument As the `(data, formula, ...)` constructor.
    /// @throws std::runtime_error If no complete training rows remain.
    /// @throws std::domain_error If a formula transformation is outside its domain.
    /// @par Example
    /// See the `(data, formula, ...)` constructor, with the first two arguments reversed.
    /// @see SVM(const dstruct::DataFrame&, const std::string&, SVMOptions)
    SVM(const Formula& formula, const dstruct::DataFrame& data, SVMOptions options = {})
        : SVM(data, formula.text(), std::move(options)) {}

    /// @brief Original formula text.
    /// @return Borrowed text, valid for the model lifetime.
    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    /// @brief Sorted response classes.
    /// @return Borrowed labels, valid for the model lifetime.
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    /// @brief Expanded design-column names.
    /// @return Borrowed names in design order, valid for the model lifetime.
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    /// @brief Retained complete training rows.
    /// @return Observation count.
    [[nodiscard]] std::size_t                     observations() const { return observations_; }

    /// @brief Counts unique training observations with alpha > 1e-8 in any pair.
    /// @return Distinct support count; see training_details() for coefficients.
    [[nodiscard]] std::size_t                num_support_vectors() const;
    /// @brief Counts distinct support observations by observed class.
    /// @return Owned counts aligned with classes().
    [[nodiscard]] std::vector<std::size_t>    support_vectors_per_class() const; // aligned with classes()

    // Training-set fit diagnostics.
    /// @brief Fitted training labels, excluding dropped rows.
    /// @return Borrowed predictions, valid for the model lifetime.
    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    /// @brief Resubstitution accuracy, evaluated on the training rows.
    /// @return Fraction correctly classified; see error() for held-out evaluation.
    [[nodiscard]] double                          training_accuracy() const;
    // Rows = actual class, columns = predicted class, both ordered as classes().
    /// @brief Training confusion counts: actual rows, predicted columns, classes() order.
    /// @return Owned k by k count matrix.
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;

    /// @brief Text summary of settings and training diagnostics.
    /// @return Owned summary, as print_summary() writes.
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to a stream.
    /// @param os Destination stream.
    void                      print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    void                      print_summary() const;

    /// @brief Predicts by pairwise majority vote; see decision_function() for an example.
    /// @param newdata Predictor frame; response is optional.
    /// @return Labels in input order; missing-predictor rows are empty strings.
    /// @throws std::invalid_argument If required columns are missing or incompatible.
    /// @throws std::runtime_error For unseen categorical levels or no complete rows.
    /// @throws std::domain_error For out-of-domain formula transformations.
    [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& newdata) const;
    /// @brief Predicts labels and uncalibrated votes; see decision_function() for an example.
    /// @param newdata Predictor frame; response optional.
    /// @return Owned labels and vote counts in input order; skipped rows have empty votes.
    /// @throws std::invalid_argument If required columns are missing or incompatible.
    /// @throws std::runtime_error For unseen categorical levels or no complete rows.
    /// @throws std::domain_error For out-of-domain formula transformations.
    [[nodiscard]] SVMPrediction            predict_detail(const dstruct::DataFrame& newdata) const;

    /// @brief Raw one-vs-one decision values, as in an SVM's pairwise decision_function.
    ///
    /// Columns enumerate (classes()[i], classes()[j]) with i < j, i outermost.
    /// A non-negative value votes for class i; a negative value votes for class j.
    /// Values are functional margins, not probabilities or geometric distances.
    /// Scaling uses training means and sample standard deviations. No model state is changed.
    /// @param newdata Predictor rows; response optional. Missing-predictor rows produce NaNs.
    /// @return Owned matrix with newdata.nrows() rows and k*(k-1)/2 columns.
    /// @throws std::invalid_argument If required columns are absent or types are incompatible.
    /// @throws std::runtime_error For unseen categorical levels or no complete predictor rows.
    /// @throws std::domain_error If a formula transformation is outside its domain.
    ///
    /// @par Example
    /// @code
    /// const auto iris = dm::datasets::iris();
    /// const dm::stats::SVM model(iris, "Species ~ Petal.Length + Petal.Width");
    /// const auto values = model.decision_function(iris.head(3)); // 3 rows, 3 columns
    /// @endcode
    ///
    /// @par References
    /// - C. Cortes, V. Vapnik (1995). Support-vector networks. Machine Learning 20, 273-297.
    ///   https://doi.org/10.1007/BF00994018
    /// @see predict_detail(), diagnostics()
    [[nodiscard]] linalg::DenseMatrix<double> decision_function(const dstruct::DataFrame& newdata) const;

    /// @brief Per-pair primal/dual objectives and numerical optimality checks.
    ///
    /// One row per pair in decision_function() order: positive_class, negative_class,
    /// observations, support_vectors, bias, norm_squared, hinge_loss, primal, dual,
    /// gap, equality_residual, max_kkt_violation, sweeps. Norm is in kernel feature space;
    /// primal = norm_squared/2 + C*hinge_loss; gap = primal-dual is not clipped.
    /// Equality residual is abs(sum(alpha*y)). KKT residual is max of max(0,1-y*f)
    /// where alpha<C-epsilon and max(0,y*f-1) where alpha>epsilon, with
    /// epsilon=min(1e-8,C*1e-6) to ignore rounding-sized distances from a box face.
    /// Checks describe the returned iterate; the sweep limit does not certify convergence.
    /// Convexity, weak duality and gap interpretation require a positive-semidefinite kernel;
    /// arbitrary sigmoid parameters do not guarantee this. Tables are owned copies; O(k^2).
    /// @return Owned diagnostic table, one row per binary model.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::SVM model(dm::datasets::iris(), "Species ~ .");
    /// std::print("{}\n", model.diagnostics().to_md(4));
    /// @endcode
    ///
    /// @par References
    /// - J. C. Platt (1998). Sequential Minimal Optimization: A Fast Algorithm for Training
    ///   Support Vector Machines. Microsoft Research, MSR-TR-98-14.
    ///   https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/
    /// @see training_details(), decision_function()
    [[nodiscard]] dstruct::DataFrame diagnostics() const;

    /// @brief Training coefficients and margins, including non-support points, for every pair.
    ///
    /// Rows are grouped by pair then retained training order. Columns: positive_class,
    /// negative_class, row (zero-based ORIGINAL input row), label, y (+1 for first class),
    /// alpha, decision, margin (y*decision), slack (max(0,1-margin)). Alpha is the returned
    /// SMO iterate; prediction drops coefficients <=1e-8, so differences at that scale
    /// may occur. Returns owned copies; O(sum of pair training sizes) time/storage.
    /// @return Table of coefficients and functional margins in training feature space.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::SVM model(dm::datasets::iris(), "Species ~ .");
    /// std::print("{}\n", model.training_details().head(5).to_md(4));
    /// @endcode
    ///
    /// @par References
    /// - C. Cortes, V. Vapnik (1995). Support-vector networks. https://doi.org/10.1007/BF00994018
    /// @see diagnostics()
    [[nodiscard]] dstruct::DataFrame training_details() const;

    /// @brief Misclassification rate on new labelled rows, like a classifier's score complement.
    ///
    /// Skips rows with a null response or missing predictors; unknown non-null response labels
    /// count as errors. Uses training preprocessing; does not refit. O(prediction cost).
    /// @param data Frame containing the original response column and all predictors.
    /// @return Fraction of evaluable rows predicted incorrectly, in [0,1].
    /// @throws std::invalid_argument If the response is absent or no evaluable rows remain,
    ///         or predictor columns/types are incompatible.
    /// @throws std::runtime_error For unseen categorical levels or no complete predictor rows.
    /// @throws std::domain_error If a formula transformation is outside its domain.
    ///
    /// @par Example
    /// @code
    /// const auto [train, test] = dm::stats::train_test_split(dm::datasets::iris(), 0.3, 42, "Species");
    /// const dm::stats::SVM model(train, "Species ~ .");
    /// const double test_error = model.error(test);
    /// @endcode
    /// @see training_accuracy(), predict()
    [[nodiscard]] double error(const dstruct::DataFrame& data) const;

    /// @brief Shaded decision regions and original training points for a two-numeric-predictor SVM.
    ///
    /// Evaluates at cell centres and merges runs of equal predicted class into rectangles.
    /// Limits default to training ranges padded by 5% horizontally and 10% vertically.
    /// The result is a sampled approximation, not an exact contour; O(grid_resolution^2)
    /// predictions. Training points are coloured by observed class. Model is unchanged.
    /// @param x_feature Numeric predictor for the horizontal axis.
    /// @param y_feature Other numeric predictor for the vertical axis.
    /// @param grid_resolution Cells per axis, default 100, clamped to at least 2.
    /// @param limits Optional finite increasing {xmin,xmax,ymin,ymax} box.
    /// @return Owned plot with axes in unscaled design-column units.
    /// @throws std::invalid_argument If the model is not on exactly these two numeric
    ///         predictors, or limits are non-finite or not increasing.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::SVM model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    /// model.plot_decision_regions("Petal.Length", "Petal.Width").save_svg("svm.svg");
    /// @endcode
    /// @see decision_function(), predict()
    [[nodiscard]] graphics::RPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
        std::size_t grid_resolution = 100,
        std::optional<std::array<double, 4>> limits = std::nullopt) const;

 private:
    /// @brief Sparse prediction coefficients and full training diagnostics for one class pair.
    struct PairModel {
        std::size_t               class_i{0}; ///< Class with positive decision values.
        std::size_t               class_j{0}; ///< votes for class_j when the decision value is negative
        std::vector<std::size_t>  support_indices; ///< row indices into scaled_training_X_
        std::vector<double>       support_alpha_y;  ///< alpha_k * y_k for each support vector
        std::vector<std::size_t> members; ///< Retained training indices for every pair row.
        std::vector<double> alpha; ///< All dual coefficients, including zeros.
        std::vector<double> decisions; ///< Final unpruned training decision values.
        double norm_squared{0.0}; ///< Squared feature-space normal norm.
        double hinge_loss{0.0}; ///< Sum of hinge losses for this pair.
        double dual{0.0}; ///< Dual objective at the returned iterate.
        double equality_residual{0.0}; ///< Absolute dual equality constraint residual.
        double max_kkt_violation{0.0}; ///< Largest box-aware functional-margin residual.
        std::size_t sweeps{0}; ///< Total SMO sweeps, including no-change sweeps.
        double                    bias{0.0}; ///< Binary intercept in scaled feature space.
    };

    /// @brief Evaluate one binary model at one scaled design row.
    /// @param pair Binary model with sparse support coefficients.
    /// @param x Scaled predictor vector in design order.
    /// @return Functional margin, positive for pair.class_i.
    double pair_decision(const PairModel& pair, const std::vector<double>& x) const;

    /// @brief Resolves the formula, scales the design and fits each pair.
    /// @param data Training frame.
    /// @param options Kernel and solver options.
    void           fit(const dstruct::DataFrame& data, SVMOptions options);
    /// @brief Evaluates the configured kernel.
    /// @param a First scaled design row.
    /// @param b Second scaled design row.
    /// @return Kernel value.
    double         kernel(const std::vector<double>& a, const std::vector<double>& b) const;
    /// @brief Votes over binary models.
    /// @param X Scaled design matrix.
    /// @return Owned labels and votes.
    SVMPrediction  classify(const linalg::DenseMatrix<double>& X) const;
    /// @brief Applies training preprocessing to one row.
    /// @param raw Unscaled design row.
    /// @return Scaled row.
    std::vector<double> scale_row(const std::vector<double>& raw) const;

    Formula    formula_; ///< Original formula.
    DesignInfo design_; ///< Resolved columns and training categorical levels.
    SVMOptions options_; ///< Fitting settings.
    double     effective_gamma_{1.0}; ///< Positive gamma after automatic selection.

    std::vector<std::string> classes_; ///< Sorted response labels.
    std::vector<std::string> predictor_names_; ///< Expanded design column names.
    std::vector<double>      feature_mean_; ///< Training design means, zero if scaling disabled.
    std::vector<double>      feature_scale_; ///< Training sample SDs, one for constant columns or disabled scaling.

    linalg::DenseMatrix<double> scaled_training_X_; ///< Retained preprocessed training design.
    std::vector<std::string>    training_labels_; ///< Response of each retained row.
    std::vector<std::size_t> training_rows_; ///< Original input row of each retained design row.
    std::vector<PairModel>      pair_models_; ///< Binary models in lexicographic class-pair order.

    std::vector<std::string> fitted_classes_; ///< Retained-row resubstitution predictions.
    std::size_t               observations_{0}; ///< Number of complete training rows.
};

} // namespace datamunge::stats
