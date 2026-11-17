#pragma once
/// @file lda.hpp
/// @brief Pooled-covariance categorical linear discriminant analysis, fitted diagnostics and native regions.

#include <cstddef>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/figure.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Optional positive class priors in alphabetically sorted class order; absent means observed proportions.
/// Priors must have one value per observed class and sum to1 within1e-6.
/// @par Example
/// @code
/// dm::stats::LDAOptions options;
/// options.priors=std::vector<double>{.2,.3,.5};
/// @endcode
/// @par References
/// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
/// @see LDA
struct LDAOptions {
  // Class prior probabilities, in the same order as LDA::classes() once
  // sorted alphabetically. Must be positive and sum to ~1. Defaults to
  // the observed class proportions when unset.
  std::optional<std::vector<double>>
      priors; ///< Owned optional strictly positive class probabilities.
};

/// @brief Owned predictions aligned to every original input row.
/// Missing-predictor rows have empty labels and empty posterior/discriminant rows.
/// @par Example
/// @code
/// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
/// const auto prediction=model.predict_detail(dm::datasets::iris().head(1));
/// std::print("{}\n",prediction.class_label[0]); // setosa
/// @endcode
/// @par References
/// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
/// @see LDA::predict_detail()
struct LDAPrediction {
  std::vector<std::string>
      class_label; ///< Predicted class per original row; empty if its predictors were missing.
  std::vector<std::vector<double>> posterior; ///< Owned row-by-class posteriors, classes() order.
  std::vector<std::vector<double>>
      discriminants; ///< Owned row-by-discriminant scores, centered by the overall training mean.
};

/// @brief Fit pooled within-class covariance linear discriminant analysis from a categorical formula.
/// Owns fitted design, means, precision terms and training observations; no borrowed
/// dataframe lifetime. Class order is alphabetical. Null predictors/response rows
/// are dropped while fitting; finite usable numerical designs and positive-definite
/// within-class scatter are required, with at least2 classes, at least2 rows per
/// class and n>classes+predictors. Formula transforms/interactions are supported;
/// intercept is ignored. No regularization or posterior calibration is performed.
/// Fitting uses scatter W and Sigma=W/(n-classes), not the maximum-likelihood W/n.
/// Bayes scores use the full design; discriminant plots use at most two axes.
/// Const inference borrows no mutable global state; mutable fitting is constructor-only.
/// @par Example
/// @code
/// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
/// std::print("{} {}\n",model.observations(),model.num_discriminants()); // 150 2
/// @endcode
/// @par References
/// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
/// @see LDAOptions, LDAPrediction, predict_detail()
class LDA {
 public:
  /// @brief Fit a complete owned LDA model from a data frame and categorical response formula.
  /// @param data Borrowed training frame; null design/response rows are dropped.
  /// @param formula Categorical response and nonintercept predictor design formula.
  /// @param options Optional positive priors, default observed class proportions.
  /// @throws std::invalid_argument On invalid formula/columns/classes/priors.
  /// @throws std::runtime_error On insufficient rows or nonpositive-definite scatter.
  /// @throws std::bad_alloc On fitting allocations.
  /// @par Example
  /// @code
  /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  /// std::print("{}\n",model.observations()); // 150
  /// @endcode
  /// @par References
  /// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
  /// @see LDA, predict_detail()
  LDA(const dstruct::DataFrame& data, const std::string& formula, LDAOptions options = {});

  /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
  ///
  /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
  /// converts to Formula implicitly, so `LDA("y ~ x", data)` also works.
  /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
  /// @param data  Data frame holding every column the formula names.
  /// @param options  Same as in the `(data, formula, ...)` constructor.
  /// @throws std::invalid_argument  As the `(data, formula, ...)` constructor.
  /// @throws std::invalid_argument On invalid formula/columns/classes/priors.
  /// @throws std::runtime_error On insufficient rows or nonpositive-definite scatter.
  /// @par Example
  /// See the primary data-first constructor example; argument order is reversed.
  /// @par References
  /// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
  /// @see LDA()
  LDA(const Formula& formula, const dstruct::DataFrame& data, LDAOptions options = {})
      : LDA(data, formula.text(), std::move(options)) {}

  /// @brief Borrow the owned original formula text.
  /// @return Owned-lifetime borrowed const reference, invalid after model destruction.
  /// @see LDA
  [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
  /// @brief Borrow alphabetically sorted observed class labels.
  /// @return Owned-lifetime borrowed const reference, invalid after model destruction.
  /// @see LDA
  [[nodiscard]] const std::vector<std::string>& classes() const { return classes_; }
  /// @brief Borrow resolved nonintercept design column names.
  /// @return Owned-lifetime borrowed const reference, invalid after model destruction.
  /// @see LDA
  [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
  /// @brief Return usable fitted row count after null exclusions.
  /// @return Actual fitted count.
  /// @see LDA
  [[nodiscard]] std::size_t observations() const { return observations_; }
  /// @brief Return min(predictors,classes minus1) fitted discriminant axes.
  /// @return Actual fitted count.
  /// @see LDA
  [[nodiscard]] std::size_t num_discriminants() const { return scaling_.cols(); }

  /// @brief Borrow positive fitted priors in classes() order.
  /// @return Owned-lifetime borrowed const reference, invalid after model destruction.
  /// @see LDA
  [[nodiscard]] const std::vector<double>& priors() const { return priors_; }
  /// @brief Borrow class-by-predictor means in class/design order.
  /// @return Owned-lifetime borrowed const reference, invalid after model destruction.
  /// @see LDA
  [[nodiscard]] const std::vector<std::vector<double>>& group_means() const { return group_means_; }
  /// @brief Borrow predictor-by-discriminant within-covariance-normalized axes.
  /// @return Owned-lifetime borrowed const reference, invalid after model destruction.
  /// @see LDA
  [[nodiscard]] const linalg::DenseMatrix<double>& scaling() const { return scaling_; }
  /// @brief Borrow positive between/within eigenvalue shares, not predictive accuracy.
  /// @return Owned-lifetime borrowed const reference, invalid after model destruction.
  /// @see LDA
  [[nodiscard]] const std::vector<double>& proportion_of_trace() const {
    return proportion_of_trace_;
  }

  // Training-set fit diagnostics.
  /// @brief Borrow alphabetically sorted observed class labels.
  /// @return Owned-lifetime borrowed const reference, invalid after model destruction.
  /// @see LDA
  [[nodiscard]] const std::vector<std::string>& fitted_classes() const { return fitted_classes_; }
  /// @brief Return resubstitution accuracy on retained training rows.
  /// @return Actual fraction correct; an optimistic fit diagnostic, not held-out accuracy.
  /// @par Example
  /// @code
  /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  /// std::print("{}\n",model.training_accuracy()>0); // true
  /// @endcode
  /// @par References
  /// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
  /// @see LDA
  [[nodiscard]] double training_accuracy() const;
  // Rows = actual class, columns = predicted class, both ordered as classes().
  /// @brief Return owned raw training confusion counts, actual rows and predicted columns.
  /// @return Square class-order count matrix.
  /// @par Example
  /// @code
  /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  /// std::print("{}\n",model.confusion_matrix().rows()); // 3
  /// @endcode
  /// @par References
  /// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
  /// @see LDA
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

  /// @brief Format fitted priors, means, axes and training accuracy as plain text.
  /// @return Owned formatted summary.
  /// @par Example
  /// @code
  /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  /// const auto text=model.summary();
  /// @endcode
  /// @par References
  /// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
  /// @see LDA
  [[nodiscard]] std::string summary() const;
  /// @brief Write the plain-text summary to a borrowed stream.
  /// @param os Mutable output stream.
  /// @return No value.
  /// @par Example
  /// @code
  /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  /// model.print_summary(std::cout);
  /// @endcode
  /// @par References
  /// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
  /// @see LDA
  void print_summary(std::ostream& os) const;
  /// @brief Write the plain-text fitted summary to standard output.
  /// @return No value.
  /// @par Example
  /// @code
  /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  /// model.print_summary();
  /// @endcode
  /// @par References
  /// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
  /// @see LDA
  void print_summary() const;

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

  /// @brief Classify new rows from the fitted predictor design without response labels.
  /// Null predictor rows are omitted from compact computation and restored as
  /// empty output entries; finite design values are required. Largest Bayes
  /// score wins, alphabetical-first on exact ties; softmax gives model posteriors,
  /// not a calibration guarantee. No refit or mutation.
  /// @throws std::invalid_argument On missing columns or unseen categorical levels.
  /// @throws std::bad_alloc On design/output allocations.
  /// @param newdata Borrowed rows with training predictor columns/levels; response not required.
  /// @return Owned labels, one per original row; empty for missing predictors.
  /// @par Example
  /// @code
  /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  /// std::print("{}\n",model.predict(dm::datasets::iris().head(1))[0]); // setosa
  /// @endcode
  /// @par References
  /// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
  /// @see LDA
  [[nodiscard]] std::vector<std::string> predict(const dstruct::DataFrame& newdata) const;
  /// @brief Classify new rows from the fitted predictor design without response labels.
  /// Null predictor rows are omitted from compact computation and restored as
  /// empty output entries; finite design values are required. Largest Bayes
  /// score wins, alphabetical-first on exact ties; softmax gives model posteriors,
  /// not a calibration guarantee. No refit or mutation.
  /// @throws std::invalid_argument On missing columns or unseen categorical levels.
  /// @throws std::bad_alloc On design/output allocations.
  /// @param newdata Borrowed rows with training predictor columns/levels; response not required.
  /// @return Owned labels, posteriors and centered discriminants aligned to original rows.
  /// @par Example
  /// @code
  /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  /// std::print("{}\n",model.predict_detail(dm::datasets::iris().head(1)).posterior[0].size()); // 3
  /// @endcode
  /// @par References
  /// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
  /// @see LDA
  [[nodiscard]] LDAPrediction predict_detail(const dstruct::DataFrame& newdata) const;

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
  /// @brief Save the native training discriminant plot with fitted decision regions.
  /// @throws std::runtime_error On output failure.
  /// @param path Destination SVG or supported native plot filename; parent must exist.
  /// @return No value.
  /// @par Example
  /// @code
  /// const dm::stats::LDA model(dm::datasets::iris(), "Species ~ Petal.Length + Petal.Width");
  /// model.save_discriminant_plot("lda.svg");
  /// @endcode
  /// @par References
  /// - Fisher(1936), The Use of Multiple Measurements in Taxonomic Problems, https://doi.org/10.1111/j.1469-1809.1936.tb02137.x.
  /// @see LDA
  void save_discriminant_plot(const std::string& path) const;

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
  /// @brief Resolve the classification design, fit pooled scatter/axes and cache actual training results.
  /// @param data Borrowed training frame.
  /// @param options Owned prior settings.
  /// @throws std::invalid_argument On design/class/prior errors.
  /// @throws std::runtime_error On sample-count/scatter failures.
  void fit(const dstruct::DataFrame& data, LDAOptions options);
  /// @brief Classify a complete finite resolved design and project its centered discriminants.
  /// @param X Borrowed compact row-by-predictor design.
  /// @return Owned compact row-aligned predictions/posteriors/discriminants.
  LDAPrediction classify(const linalg::DenseMatrix<double>& X) const;

  Formula    formula_; ///< Owned original formula.
  DesignInfo design_;  ///< Owned fitted formula design/levels.

  std::vector<std::string>         classes_;         ///< Alphabetical observed classes.
  std::vector<std::string>         predictor_names_; ///< Resolved nonintercept design names.
  std::vector<double>              priors_;          ///< Fitted class probabilities.
  std::vector<std::vector<double>> group_means_;     ///< Class-by-predictor means.
  std::vector<double>              overall_mean_;    ///< Observation-weighted predictor mean.

  // Cached per-class terms of the Bayes discriminant function
  // delta_k(x) = x . precision_means_[k] - 0.5*mahalanobis_offset_[k] + log(priors_[k]).
  std::vector<std::vector<double>> precision_means_;    ///< Class-wise Sigma inverse times mean.
  std::vector<double>              mahalanobis_offset_; ///< Class quadratic mean terms.

  linalg::DenseMatrix<double> scaling_;             ///< Predictor-by-discriminant axes.
  std::vector<double>         proportion_of_trace_; ///< Positive generalized eigenvalue shares.

  std::vector<std::string>         fitted_classes_;         ///< Actual retained-row training predictions.
  std::vector<std::string>         training_labels_;        ///< Retained true training labels.
  std::vector<std::vector<double>> training_discriminants_; ///< Owned retained row-by-discriminant training scores.
  std::vector<std::vector<double>>
              training_predictors_; ///< Owned retained row-by-predictor training design.
  std::size_t observations_{0};     ///< Number of retained fitted rows.
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
