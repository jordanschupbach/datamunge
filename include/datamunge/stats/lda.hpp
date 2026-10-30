#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/figure.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

struct LDAOptions {
    // Class prior probabilities, in the same order as LDA::classes() once
    // sorted alphabetically. Must be positive and sum to ~1. Defaults to
    // the observed class proportions when unset.
    std::optional<std::vector<double>> priors;
};

struct LDAPrediction {
    std::vector<std::string>         class_label;   // predicted class per row ("" if predictors were missing)
    std::vector<std::vector<double>> posterior;      // [row][class] posterior probabilities
    std::vector<std::vector<double>> discriminants;  // [row][LD] projected discriminant scores
};

// Classic multi-class linear discriminant analysis (Fisher/Bayes LDA,
// matching R's MASS::lda), fit from a DataFrame and an R-style formula
// whose left-hand side is a categorical (string) column, e.g.:
//
//   LDA model(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// Assumes a common (pooled) within-class covariance across all classes.
// Rows with a null predictor or null class label are dropped before
// fitting. The right-hand side supports the same term syntax as LM
// (interactions, log/sqrt/poly/I(...), categorical predictors), but any
// intercept term is ignored — classification designs never include one.
class LDA {
 public:
    LDA(const dstruct::DataFrame& data, const std::string& formula, LDAOptions options = {});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `LDA("y ~ x", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument  As the `(data, formula, ...)` constructor.
    LDA(const Formula& formula, const dstruct::DataFrame& data, LDAOptions options = {})
        : LDA(data, formula.text(), std::move(options)) {}

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] std::size_t                     num_discriminants() const { return scaling_.cols(); }

    [[nodiscard]] const std::vector<double>&              priors() const { return priors_; }
    [[nodiscard]] const std::vector<std::vector<double>>& group_means() const { return group_means_; }
    [[nodiscard]] const linalg::DenseMatrix<double>&      scaling() const { return scaling_; }
    [[nodiscard]] const std::vector<double>&              proportion_of_trace() const { return proportion_of_trace_; }

    // Training-set fit diagnostics.
    [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
    [[nodiscard]] double                          training_accuracy() const;
    // Rows = actual class, columns = predicted class, both ordered as classes().
    [[nodiscard]] linalg::DenseMatrix<double> confusion_matrix() const;

    /// @brief Group means as a frame, like `MASS::lda()$means`: a "class" column, then one
    ///        numeric column per predictor, rows in classes() order.
    /// @return  A new frame with classes().size() rows.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    /// std::print("{}", model.group_means_frame().to_md());   // | setosa | 1.462 | 0.246 | ...
    /// @endcode
    /// @see group_means(), to_md()
    [[nodiscard]] dstruct::DataFrame group_means_frame() const;

    /// @brief Discriminant coefficients as a frame, like `MASS::lda()$scaling`: a "predictor"
    ///        column, then one numeric column per discriminant (LD1, LD2, ...).
    /// @return  A new frame with predictor_names().size() rows.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    /// std::print("{}\n", model.coefficients_frame().columns());   // ["predictor", "LD1", "LD2"]
    /// @endcode
    /// @see scaling(), to_md()
    [[nodiscard]] dstruct::DataFrame coefficients_frame() const;

    /// @brief Training confusion table as a frame, like R's `table(actual, predicted)`: an
    ///        "actual" column, then one count column per predicted class, both in classes()
    ///        order.
    /// @return  A new frame with classes().size() rows.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    /// std::print("{}", model.confusion_frame() | dm::rename("actual", "truth") | dm::to_md());
    /// @endcode
    /// @see confusion_matrix(), training_accuracy()
    [[nodiscard]] dstruct::DataFrame confusion_frame() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    /// @brief The fit as Markdown, for reports: the same content as summary() (R's
    ///        `print(MASS::lda(...))`) as a sentence naming the model and its training
    ///        accuracy, then tables of the priors, group means, discriminant coefficients and
    ///        proportion of trace, each after a one-line label.
    /// @param digits  Maximum decimals in the tables (default 4); see DataFrame::to_md().
    /// @return  Markdown text ending in a newline.
    /// @throws std::invalid_argument  if @p digits is negative.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    /// std::print("{}", model.to_md());   // ... Proportion of trace: | LD1 | LD2 | ...
    /// @endcode
    /// @see summary(), DataFrame::to_md()
    [[nodiscard]] std::string to_md(int digits = 4) const;

    [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] LDAPrediction            predict_detail(const dstruct::DataFrame& newdata) const;

    /// @brief Scatter of the training data's first two discriminant scores (LD1 vs LD2; LD1
    ///        against 0 if there is only one), colored by true class.
    ///
    /// With @p decision_regions, the background is shaded by the class the model predicts
    /// there and the boundaries between classes are drawn. LDA picks the class whose mean is
    /// nearest in discriminant space (after a log-prior adjustment), so each region is a
    /// convex polygon, computed exactly by clipping the plot rectangle with one half-plane per
    /// competing class. With up to three classes the two discriminants carry the whole model,
    /// so the regions are the model's own predictions; with more classes they are the
    /// predictions of the model restricted to LD1 and LD2 (MASS's `predict(dimen = 2)`).
    ///
    /// @param decision_regions  Shade class regions and draw boundaries (default true).
    /// @return  The plot, ready to save or adjust.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::LDA model(dm::datasets::iris(),
    ///     "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    /// model.plot_discriminants().save("iris_lda.svg");
    /// model.plot_discriminants(false).save("iris_lda_points.svg");   // points only
    /// @endcode
    ///
    /// @par References
    /// - T. Hastie, R. Tibshirani, J. Friedman (2009). The Elements of Statistical Learning,
    ///   2nd ed., Section 4.3.3 (reduced-rank LDA as nearest centroid). Springer.
    ///   doi:10.1007/978-0-387-84858-7
    /// - W. N. Venables, B. D. Ripley (2002). Modern Applied Statistics with S, 4th ed.,
    ///   Section 12.1. Springer. doi:10.1007/978-0-387-21706-2
    ///
    /// @see plot_decision_regions(), predict_detail()
    [[nodiscard]] graphics::RPlot plot_discriminants(bool decision_regions = true) const;
    void                            save_discriminant_plot(const std::string& path) const;

    /// @brief Training data and decision regions in the space of the two predictors, for a
    ///        model with exactly two (e.g. `"Species ~ Petal.Length + Petal.Width"`).
    ///
    /// Each class region is the convex polygon where that class's discriminant function is
    /// largest, so boundaries are exact straight lines.
    /// @return  The plot, with the predictors as axis labels.
    /// @throws std::logic_error  if the model does not have exactly two predictor columns.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::LDA petals(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
    /// petals.plot_decision_regions().save("petal_regions.svg");
    /// @endcode
    /// @see plot_discriminants()
    [[nodiscard]] graphics::RPlot plot_decision_regions() const;

 private:
    void          fit(const dstruct::DataFrame& data, LDAOptions options);
    LDAPrediction classify(const linalg::DenseMatrix<double>& X) const;

    Formula    formula_;
    DesignInfo design_;

    std::vector<std::string>         classes_;
    std::vector<std::string>         predictor_names_;
    std::vector<double>              priors_;
    std::vector<std::vector<double>> group_means_;
    std::vector<double>              overall_mean_;

    // Cached per-class terms of the Bayes discriminant function
    // delta_k(x) = x . precision_means_[k] - 0.5*mahalanobis_offset_[k] + log(priors_[k]).
    std::vector<std::vector<double>> precision_means_;   // Sigma_w^-1 * mu_k
    std::vector<double>              mahalanobis_offset_; // mu_k^T Sigma_w^-1 mu_k

    linalg::DenseMatrix<double> scaling_;             // p x num_discriminants
    std::vector<double>         proportion_of_trace_;

    std::vector<std::string>         fitted_classes_;
    std::vector<std::string>         training_labels_;
    std::vector<std::vector<double>> training_discriminants_; // [row][LD], for plot_discriminants()
    std::vector<std::vector<double>> training_predictors_;    // [row][predictor], for plot_decision_regions()
    std::size_t                      observations_{0};
};

} // namespace datamunge::stats

namespace datamunge {

/// @brief Default picture of a fitted LDA: the training data in discriminant space over the
///        decision regions, plus the regions in predictor space when the model has exactly two
///        predictors.
///
/// A two-predictor model gives a 1 x 2 grid, 1200 by 520 pixels: LDA::plot_discriminants()
/// on the left, LDA::plot_decision_regions() on the right. Any other model gives the
/// discriminant plot alone, 960 by 640. Either way the result is a graphics::Figure, so title,
/// size, and output work the same.
///
/// @param model  The fitted model; the figure copies what it needs.
/// @return The figure.
///
/// @par Example
/// @code
/// const dm::stats::LDA petals(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
/// dm::plot(petals).title("LDA on the petals").size(1200, 520).save_svg("petals.svg");
/// @endcode
/// @see stats::LDA::plot_discriminants(), stats::LDA::plot_decision_regions(), graphics::Figure
[[nodiscard]] graphics::Figure plot(const stats::LDA& model);

} // namespace datamunge
