#pragma once

/// @file naive_bayes_classifier.hpp
/// @brief Naive Bayes classification with Gaussian numeric and smoothed categorical
///        predictors, conditionally independent given the class, with exact leave-one-out
///        evaluation.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <array>
#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Settings for NaiveBayesClassifier.
struct NaiveBayesClassifierOptions {
    /// @brief Additive (Laplace for 1, Lidstone otherwise) smoothing \f$\alpha\f$ of the categorical
    ///        frequency tables: \f$P(x_j=\ell\mid c)=(n_{c\ell}+\alpha)/(n_c+\alpha L_j)\f$. Must be \f$\ge0\f$.
    double laplace_smoothing{1.0};
    /// @brief Relative variance floor for the per-class Gaussians: every per-class variance gets
    ///        `var_smoothing` times the largest overall predictor variance added, as in
    ///        scikit-learn's `GaussianNB` (same default).
    double var_smoothing{1e-9};
    /// @brief Class priors in classes() order (sorted labels); empty (the default) uses the
    ///        class frequencies. Must sum to 1 when given.
    std::vector<double> priors{};
};

/// @brief Predicted classes with their posterior probabilities.
struct NaiveBayesClassifierPrediction {
    /// @brief Predicted class per row of the new data; "" where a predictor was missing.
    std::vector<std::string> class_label;
    /// @brief Posterior probability per class, in classes() order; empty where a predictor was missing.
    std::vector<std::vector<double>> probability;
};

/// @brief Naive Bayes classifier from a formula `class ~ x1 + x2 + ...` mixing numeric and
///        categorical predictors.
///
/// The predictors are modelled as independent given the class,
/// \f$P(c\mid x)\propto\pi_c\prod_jp_j(x_j\mid c)\f$. A numeric predictor has a Gaussian density per
/// class (maximum-likelihood mean and variance, divisor \f$n_c\f$, plus a small floor). A
/// categorical predictor has one smoothed frequency table per class over its levels, modelled
/// as one variable rather than as separate dummy columns. Formula terms must be plain column
/// references. Rows with a missing predictor or class are dropped. Posterior probabilities are
/// computed in log space and normalised with the log-sum-exp shift.
///
/// training_accuracy() and confusion_matrix() are resubstitution figures (the model is fitted
/// and evaluated on the same rows). loo_fitted_classes() and loo_accuracy() are exact
/// leave-one-out figures: the fitted counts, means and variances are downdated without each row
/// in turn, which equals refitting \f$n\f$ times (with the categorical levels kept as those of the
/// full data).
///
/// @par Example
/// @code
/// const auto penguins = dm::datasets::penguins();
/// const dm::stats::NaiveBayesClassifier nb(penguins, "species ~ bill_length_mm + bill_depth_mm + island");
/// std::print("{:.4f} {:.4f}\n", nb.training_accuracy(), nb.loo_accuracy());
/// @endcode
///
/// @par References
/// - D. D. Lewis (1998). "Naive (Bayes) at forty: the independence assumption in information
///   retrieval". *ECML-98*, LNCS 1398, 4–15.
/// - P. Domingos and M. Pazzani (1997). "On the optimality of the simple Bayesian classifier under
///   zero-one loss". *Machine Learning* 29, 103–130.
/// - scikit-learn `GaussianNB` and `CategoricalNB`: https://scikit-learn.org/stable/modules/naive_bayes.html
class NaiveBayesClassifier {
 public:
    /// @brief Fits the classifier.
    /// @param data     Data frame holding the class (a string column) and the predictors.
    /// @param formula  `class ~ predictors`, plain column references only.
    /// @param options  Smoothing and priors.
    /// @throws std::invalid_argument  If a term is not a plain column, the response has fewer than
    ///         2 classes, the smoothing is negative, or the priors are malformed.
    /// @throws std::runtime_error     If no complete rows remain.
    NaiveBayesClassifier(const dstruct::DataFrame& data, const std::string& formula,
                         NaiveBayesClassifierOptions options = {});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `NaiveBayesClassifier("y ~ x", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument  As the `(data, formula, ...)` constructor.
    NaiveBayesClassifier(const Formula& formula, const dstruct::DataFrame& data, NaiveBayesClassifierOptions options = {})
        : NaiveBayesClassifier(data, formula.text(), std::move(options)) {}

    /// @brief The formula the model was fitted with.
    /// @return Formula text.
    [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
    /// @brief The class labels, sorted; every per-class vector follows this order.
    /// @return Sorted distinct labels.
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    /// @brief The predictor columns, in formula order.
    /// @return Column names.
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    /// @brief Whether each predictor is categorical (true) or numeric (false).
    /// @return One flag per predictor.
    [[nodiscard]] const std::vector<bool>& predictor_is_categorical() const { return predictor_is_categorical_; }
    /// @brief Number of training rows used.
    /// @return \f$n\f$.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief The class priors \f$\pi_c\f$.
    /// @return One prior per class.
    [[nodiscard]] const std::vector<double>& class_priors() const { return class_priors_; }
    /// @brief The variance floor added to every per-class variance.
    /// @return `var_smoothing` times the largest overall predictor variance (0 without numeric predictors).
    [[nodiscard]] double variance_floor() const { return variance_floor_; }

    /// @brief Per-class Gaussian parameters of the numeric predictors.
    /// @return Columns `predictor`, `class`, `n`, `mean`, `sd` (the standard deviation used,
    ///         floor included).
    [[nodiscard]] dstruct::DataFrame gaussian_frame() const;
    /// @brief Smoothed per-class frequency tables of the categorical predictors.
    /// @return Columns `predictor`, `class`, `level`, `count`, `probability`.
    [[nodiscard]] dstruct::DataFrame categorical_frame() const;
    /// @brief The terms of the log posterior for one row: the log prior, one log likelihood per
    ///        predictor, their sum, and the normalised posterior.
    /// @param newdata  Data holding the row.
    /// @param row      Row index.
    /// @return Column `term` and one column per class.
    /// @throws std::invalid_argument  If the row has a missing predictor or an unseen level.
    [[nodiscard]] dstruct::DataFrame posterior_terms(const dstruct::DataFrame& newdata, std::size_t row) const;

    /// @brief Resubstitution prediction of each training row.
    /// @return \f$n\f$ labels.
    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    /// @brief Observed class of each training row.
    /// @return \f$n\f$ labels.
    [[nodiscard]] const std::vector<std::string>& training_labels() const { return training_labels_; }
    /// @brief Resubstitution accuracy.
    /// @return Share of training rows predicted correctly by the model fitted on all of them.
    [[nodiscard]] double training_accuracy() const;
    /// @brief Resubstitution confusion matrix.
    /// @return Counts; rows observed, columns predicted, both in classes() order.
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;
    /// @brief Exact leave-one-out prediction of each training row.
    /// @return \f$n\f$ labels.
    [[nodiscard]] const std::vector<std::string>& loo_fitted_classes() const { return loo_fitted_classes_; }
    /// @brief Exact leave-one-out accuracy.
    /// @return Share of rows predicted correctly by the model fitted without them.
    [[nodiscard]] double loo_accuracy() const;

    /// @brief A text summary: priors, per-class parameters and accuracy.
    /// @return Summary text.
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to @p os.
    /// @param os  Output stream.
    void print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    void print_summary() const;

    /// @brief Predicted classes for new rows.
    /// @param newdata  Rows to classify.
    /// @return One label per row ("" where a predictor is missing).
    /// @throws std::invalid_argument  If a categorical predictor has a level unseen in training.
    [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& newdata) const;
    /// @brief Predicted classes with posterior probabilities.
    /// @param newdata  Rows to classify.
    /// @return Labels and probabilities.
    /// @throws std::invalid_argument  If a categorical predictor has a level unseen in training.
    [[nodiscard]] NaiveBayesClassifierPrediction predict_detail(const dstruct::DataFrame& newdata) const;

    /// @brief Scatter of @p data coloured by observed class, misclassified rows (by predict()) marked.
    /// @param data       Rows to draw.
    /// @param x_feature  Column on the x axis.
    /// @param y_feature  Column on the y axis.
    /// @return The plot.
    [[nodiscard]] graphics::RPlot plot_classification(const dstruct::DataFrame& data, const std::string& x_feature,
                                                      const std::string& y_feature) const;
    /// @brief The predicted class over the plane of two numeric predictors, shaded in pale class
    ///        colours, with the training points on top.
    /// @param x_feature        Predictor on the x axis.
    /// @param y_feature        Predictor on the y axis.
    /// @param grid_resolution  Cells per axis.
    /// @param limits           Optional box {x_min, x_max, y_min, y_max}; default the training
    ///                         range padded by 5% in x and 10% in y.
    /// @return The plot, with its axis limits set to the box.
    /// @throws std::invalid_argument  If the model does not have exactly these two numeric predictors.
    [[nodiscard]] graphics::RPlot plot_decision_regions(const std::string& x_feature, const std::string& y_feature,
                                                        std::size_t grid_resolution = 150,
                                                        std::optional<std::array<double, 4>> limits = std::nullopt) const;

 private:
    /// @brief Builds the model and the resubstitution and leave-one-out predictions.
    /// @param data  Training data.
    void fit(const dstruct::DataFrame& data);
    /// @brief Rows of @p frame with every predictor (and, if asked, the class) present.
    /// @param frame             Data.
    /// @param require_response  Also require the class.
    /// @return Row indices.
    std::vector<std::size_t> filter_complete_rows(const dstruct::DataFrame& frame, bool require_response) const;
    /// @brief Per-predictor coded values of one row: numeric value, or level index as a double.
    /// @param frame  Data.
    /// @param row    Row index.
    /// @return One value per predictor.
    std::vector<double> encode_row(const dstruct::DataFrame& frame, std::size_t row) const;
    /// @brief Log prior plus log likelihoods of an encoded row, per class.
    /// @param x  Encoded row.
    /// @return Unnormalised log posteriors.
    std::vector<double> log_posteriors(const std::vector<double>& x) const;
    /// @brief Unnormalised log posteriors of training row @p i under the model fitted without it.
    /// @param i  Training row.
    /// @return Log posteriors (\f$-\infty\f$ for a class left empty).
    std::vector<double> loo_log_posteriors(std::size_t i) const;

    Formula                     formula_;  ///< The model formula.
    DesignInfo                  design_;   ///< Resolved predictor columns.
    NaiveBayesClassifierOptions options_;  ///< Fitting options.

    std::vector<std::string> predictor_names_;           ///< Predictor columns.
    std::vector<bool>        predictor_is_categorical_;  ///< Categorical flag per predictor.
    std::vector<int>         numeric_slot_;               ///< Per predictor: index into the numeric arrays, or -1.
    std::vector<int>         categorical_slot_;           ///< Per predictor: index into the categorical arrays, or -1.

    std::vector<std::string> classes_;       ///< Sorted class labels.
    std::vector<double>      class_priors_;  ///< \f$\pi_c\f$.
    std::vector<double>      class_count_;   ///< \f$n_c\f$.
    double                   variance_floor_{0.0};  ///< Variance added to every per-class variance.

    std::vector<std::vector<double>> numeric_mean_;      ///< [slot][class] mean.
    std::vector<std::vector<double>> numeric_ss_;        ///< [slot][class] centred sum of squares.
    std::vector<std::vector<double>> numeric_variance_;  ///< [slot][class] variance used (floor included).
    std::vector<double>              numeric_all_mean_;  ///< [slot] overall mean.
    std::vector<double>              numeric_all_ss_;    ///< [slot] overall centred sum of squares.
    std::vector<double>              numeric_min_;       ///< [slot] minimum, for plot_decision_regions().
    std::vector<double>              numeric_max_;       ///< [slot] maximum.

    std::vector<std::vector<std::string>>         categorical_levels_;  ///< [slot] sorted levels.
    std::vector<std::vector<std::vector<double>>> categorical_count_;   ///< [slot][class][level] counts.
    std::vector<std::vector<std::vector<double>>> categorical_log_prob_;///< [slot][class][level] log probabilities.

    std::vector<std::vector<double>> training_x_;        ///< Encoded training rows.
    std::vector<std::size_t>         training_class_;    ///< Class index of each training row.
    std::vector<std::string>         training_labels_;   ///< Class of each training row.
    std::vector<std::string>         fitted_classes_;    ///< Resubstitution predictions.
    std::vector<std::string>         loo_fitted_classes_;///< Leave-one-out predictions.
    std::size_t                      observations_{0};   ///< Number of training rows.
};

} // namespace datamunge::stats
