#pragma once
/// @file kernel_regression.hpp
/// @brief Nadaraya-Watson local means, inspectable smoothing weights and coverage-aware CV.
#include <cstddef>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {
/// @brief Radial kernel of Euclidean distance in the resolved predictor design.
/// Scalar normalizing constants cancel from the local mean. These are radial kernels,
/// not a product of independent univariate compact kernels.
/// @par Example
/// @code
/// const auto k=dm::stats::KernelRegressionKernel::Uniform;
/// std::print("{}\n",k==dm::stats::KernelRegressionKernel::Uniform); // true
/// @endcode
/// @par References
/// - Nadaraya (1964), On Estimating Regression. https://doi.org/10.1137/1109020
/// @see KernelRegressionOptions
enum class KernelRegressionKernel {
  Gaussian,     ///< exp(-u^2/2); all points have mathematical support.
  Epanechnikov, ///< .75*(1-u^2) for u<1; zero at/outside the boundary.
  Uniform,      ///< One for u<=1, zero outside; the boundary is included.
  Triangular    ///< 1-u for u<1, zero at/outside the boundary.
};
/// @brief Local-mean geometry and conditional leave-one-out bandwidth search.
/// @par Example
/// @code
/// dm::stats::KernelRegressionOptions o;o.bandwidth=1;o.standardize=false;
/// dm::dstruct::DataFrame
/// d;d.add_column("x",std::vector<double>{0,1,2});d.add_column("y",std::vector<double>{0,1,2});
/// const dm::stats::KernelRegression m(d,"y ~ x",o);
/// std::print("{}\n",m.bandwidth()); // 1
/// @endcode
/// @see KernelRegression
struct KernelRegressionOptions {
  KernelRegressionKernel kernel{KernelRegressionKernel::Gaussian}; ///< Valid radial kernel.
  double bandwidth{-1}; ///< Positive finite fixed h; finite negative requests CV; zero invalid.
  std::size_t n_bandwidth{50}; ///< Positive default log-grid size; one uses its lower endpoint.
  bool standardize{true}; ///< Training design mean/population SD; constant columns use scale one.
  std::vector<double> bandwidth_candidates; ///< Optional positive finite CV grid, caller order;
                                            ///< first minimum wins. Ignored for fixed h.
};
/// @brief Nadaraya-Watson regression: nonnegative local averages on a frozen formula design.
///
/// Resolves numeric responses and a no-intercept predictor design with Formula, omitting
/// training rows with null required values jointly. At least two finite complete rows and
/// one design column are required. Standardization uses training population SD, not n-1.
/// Scalar h is in standardized design units when enabled, otherwise raw design units.
/// Categoricals/transforms follow Formula; dummy encoding is a metric choice, not a
/// separate categorical kernel. Gaussian weights are normalized after subtracting a common
/// log weight, avoiding underflow of every original weight at distant queries/tiny h.
/// Compact kernels return NaN when no strictly positive weight remains.
///
/// fitted_values(), rmse() and r_squared() use direct leave-one-out deletions at fixed h
/// and fixed full-training design/scaling. They do not refit preprocessing for each fold.
/// Auto search requires every deleted prediction to be finite: incomplete candidates get
/// objective infinity and a wholly ineligible grid throws. Selected CV loss is a tuning
/// criterion, not an unbiased evaluation of the selected procedure; use held-out data.
///
/// Fixed fitting costs O(n^2*p), default B-candidate search O(B*n^2*p), storage O(n*p+n+B).
/// Explicit smoothing matrices add O(n^2) storage. Models own their data/diagnostics;
/// borrowed accessors live with the model. Const methods use no mutable shared caches.
/// @par Example
/// @code
/// const dm::stats::KernelRegression m(dm::datasets::iris(),"Petal.Length ~ Petal.Width");
/// std::print("{}\n",m.observations()); // 150
/// @endcode
/// @par References
/// - E. A. Nadaraya (1964), On Estimating Regression. https://doi.org/10.1137/1109020
/// - G. S. Watson (1964), Smooth Regression Analysis, Sankhya A 26(4), 359-372.
///   https://www.jstor.org/stable/25049340
/// @see KernelRegressionOptions, bandwidth_diagnostics(), score()
class KernelRegression {
 public:
  /// @brief Fits immediately to an owned copy of the complete resolved design and responses.
  /// @param data Training frame; required nulls are omitted jointly, retained nonfinite values
  /// rejected.
  /// @param formula Numeric response and predictor formula; intercept is suppressed.
  /// @param options Kernel, metric scaling and bandwidth controls, with defaults above.
  /// @throws std::invalid_argument Invalid options, no predictors, too few retained rows or
  /// nonfinite geometry.
  /// @throws std::runtime_error Formula schema/levels or no complete rows; no eligible CV
  /// bandwidth.
  /// @par Example
  /// See the class example.
  KernelRegression(
      const dstruct::DataFrame& data,
      const std::string&        formula,
      KernelRegressionOptions   options = {});
  /// @brief Formula-first constructor, matching R model argument order.
  /// @param formula Parsed formula (a string converts implicitly).
  /// @param data Training frame, with the primary constructor's omission rules.
  /// @param options Same fitting controls as the primary constructor.
  /// @throws std::invalid_argument As the primary constructor.
  /// @throws std::runtime_error As the primary constructor.
  /// @par Example
  /// @code
  /// const auto f=dm::formula("Petal.Length ~ Petal.Width");
  /// const dm::stats::KernelRegression m(f,dm::datasets::iris());
  /// std::print("{}\n",m.observations()); // 150
  /// @endcode
  KernelRegression(
      const Formula&            formula,
      const dstruct::DataFrame& data,
      KernelRegressionOptions   options = {})
      : KernelRegression(data, formula.text(), std::move(options)) {}
  /// @brief Original model formula.
  /// @return Borrowed owned text.
  [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
  /// @brief Resolved no-intercept design column names (possibly transformed/dummy columns).
  /// @return Borrowed names in metric order.
  [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
  /// @brief Complete training observation count.
  /// @return Retained row count.
  [[nodiscard]] std::size_t observations() const { return observations_; }
  /// @brief Original zero-based training row indices, aligned to fitted_values().
  /// @return Borrowed increasing indices.
  [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_rows_; }
  /// @brief Radial kernel used in the fit.
  /// @return Kernel enumerator.
  [[nodiscard]] KernelRegressionKernel kernel() const { return options_.kernel; }
  /// @brief Selected or caller-fixed bandwidth in the resolved metric units.
  /// @return Positive finite h.
  [[nodiscard]] double bandwidth() const { return bandwidth_used_; }
  /// @brief Whether bandwidth was selected from a CV grid.
  /// @return True for negative bandwidth option.
  [[nodiscard]] bool bandwidth_was_selected() const { return bandwidth_was_selected_; }
  /// @brief CV candidate order; empty for a fixed h.
  /// @return Borrowed grid, preserving explicit candidate order if supplied.
  [[nodiscard]] const std::vector<double>& bandwidth_grid() const { return bandwidth_grid_; }
  /// @brief Complete-coverage CV objectives, infinity for incomplete/unrepresentable candidates.
  /// @return Borrowed MSE vector, aligned to bandwidth_grid(); empty for fixed h.
  [[nodiscard]] const std::vector<double>& cv_mean_squared_error() const { return cv_mse_; }
  /// @brief Direct leave-one-out predictions with fixed full-training preprocessing and selected h.
  /// @return Borrowed retained-row predictions, NaN for a compact kernel's unsupported deletions.
  [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
  /// @brief Ordinary full-sample predictions at the training design, without exclusions.
  /// @return Borrowed retained-row local means; distinct from fitted_values().
  [[nodiscard]] const std::vector<double>& full_fitted_values() const { return full_fitted_; }
  /// @brief Conditional deletion R-squared, on rows with finite LOO predictions.
  /// @return 1-SSE/TSS using the covered subset's actual mean; NaN for no coverage or zero TSS.
  [[nodiscard]] double r_squared() const;
  /// @brief Conditional deletion RMSE; always read LOO coverage alongside it.
  /// @return Root mean squared LOO error on covered rows, NaN when none are covered.
  [[nodiscard]] double rmse() const;
  /// @brief Frozen design centring values (zeros if standardization is disabled).
  /// @return Borrowed vector in predictor_names() order.
  [[nodiscard]] const std::vector<double>& feature_means() const { return feature_mean_; }
  /// @brief Frozen design population SDs, with scale one for exact constant columns/disabled
  /// scaling.
  /// @return Borrowed positive scales in predictor_names() order.
  [[nodiscard]] const std::vector<double>& feature_scales() const { return feature_scale_; }

  /// @brief Owned row-aligned local means on new predictors, with no training exclusions.
  /// @param newdata Formula predictor columns; response is unnecessary. Empty input gives empty
  /// output.
  /// @return One prediction per original query row; null/nonfinite predictors or no compact support
  /// give NaN.
  /// @throws std::invalid_argument Missing/invalid schema or unrepresentable normalized-weight
  /// geometry.
  /// @throws std::runtime_error Formula evaluation or unseen categorical levels.
  /// @par Example
  /// @code
  /// const dm::stats::KernelRegression m(dm::datasets::iris(),"Petal.Length ~ Petal.Width");
  /// dm::dstruct::DataFrame q;q.add_column("Petal.Width",std::vector<double>{.2,1.5});
  /// std::print("{}\n",m.predict(q).size()); // 2
  /// @endcode
  [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;
  /// @brief Query coverage, effective sample concentration and nearest design distance.
  /// Effective_n is 1/sum(w_i^2), not a residual df or an uncertainty interval. Gaussian
  /// far-field predictions can have effective_n near one; nonzero support does not imply data
  /// proximity.
  /// @param newdata Same query schema/null semantics as predict().
  /// @return query_row/prediction/positive_weights/effective_n/nearest_distance/status;
  /// status is ok, missing_predictor, nonfinite_predictor or no_support. Counts reflect
  /// positive numerically represented weights; unsupported effective_n is NaN.
  /// @throws std::invalid_argument As predict().
  /// @throws std::runtime_error As predict().
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::KernelRegression m(d,"Petal.Length ~ Petal.Width");
  /// std::print("{}\n",m.prediction_details(d.head(3)).nrows()); // 3
  /// @endcode
  [[nodiscard]] dstruct::DataFrame prediction_details(const dstruct::DataFrame& newdata) const;
  /// @brief Normalized query-by-training weights, in original query/retained training order.
  /// @param newdata Same schema and null semantics as predict().
  /// @return Owned m-by-n matrix; valid rows sum to one, invalid/no-support rows are all NaN.
  /// @throws std::invalid_argument As predict().
  /// @throws std::runtime_error As predict().
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::KernelRegression m(d,"Petal.Length ~ Petal.Width");
  /// std::print("{}\n",m.query_weights(d.head(2)).cols()); // 150
  /// @endcode
  [[nodiscard]] linalg::DenseMatrix<double> query_weights(const dstruct::DataFrame& newdata) const;
  /// @brief Full-sample smoothing matrix S, conditional on the selected h/design.
  /// @return Owned n-by-n row-stochastic matrix; S*y equals full_fitted_values(), not LOO fits.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::KernelRegression m(d,"Petal.Length ~ Petal.Width");
  /// std::print("{}\n",m.smoothing_matrix().rows()); // 150
  /// @endcode
  [[nodiscard]] linalg::DenseMatrix<double> smoothing_matrix() const;
  /// @brief Training full/LOO fits, residuals, leverage and local effective sample sizes.
  /// @return Owned row/actual/full_fit/loo_fit/loo_residual/leverage/effective_n/loo_effective_n
  /// frame.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::KernelRegression m(d,"Petal.Length ~ Petal.Width");
  /// std::print("{}\n",m.training_details().nrows()); // 150
  /// @endcode
  [[nodiscard]] dstruct::DataFrame training_details() const;
  /// @brief Every candidate's LOO coverage, conditional loss and complete-coverage objective.
  /// @return Owned bandwidth/loo_n/coverage/conditional_mse/objective_mse/eligible/selected frame;
  /// a fixed-bandwidth model has one row. Missing deletions give objective infinity even when
  /// conditional_mse is small; all-missing conditional_mse is NaN. Selection uses first minimum.
  /// @par Example
  /// @code
  /// const dm::stats::KernelRegression m(dm::datasets::iris(),"Petal.Length ~ Petal.Width");
  /// std::print("{}\n",m.bandwidth_diagnostics().nrows()); // 50
  /// @endcode
  [[nodiscard]] dstruct::DataFrame bandwidth_diagnostics() const;
  /// @brief Coverage, deletion metrics, apparent error and trace(S) at the chosen h.
  /// @return Owned n/p/bandwidth/loo_n/loo_coverage/loo_rmse/loo_r_squared/apparent_rmse/
  /// effective_df/cv_selected/search_boundary row. trace(S) is conditional smoother df,
  /// not the complexity cost of selecting h. Boundary means the minimum or maximum grid h.
  /// @par Example
  /// @code
  /// const dm::stats::KernelRegression m(dm::datasets::iris(),"Petal.Length ~ Petal.Width");
  /// std::print("{:.0f}\n",m.model_summary().double_at("n",0)); // 150
  /// @endcode
  [[nodiscard]] dstruct::DataFrame model_summary() const;
  /// @brief Scores external responses, including a training-mean prediction baseline.
  /// Null responses are excluded before query evaluation; retained nonfinite responses
  /// throw. Errors/R-squared/baseline use only finite predictions, so always compare coverage.
  /// R-squared uses the covered evaluation subset's mean; the baseline uses the training
  /// response mean without evaluation-label fitting. No refitting or bandwidth choice occurs.
  /// @param data Query predictors and numeric response column named by the training formula.
  /// @return n/n_response/n_predicted/coverage/rmse/mae/r_squared/mean_baseline_rmse row;
  /// no coverage gives NaN metrics, zero evaluation TSS gives NaN R-squared.
  /// @throws std::invalid_argument Invalid response schema/nonfinite response or predict() errors.
  /// @throws std::runtime_error As predict().
  /// @par Example
  /// @code
  /// const auto [train,test]=dm::stats::train_test_split(dm::datasets::iris(),.2,42,"Species");
  /// const dm::stats::KernelRegression m(train,"Petal.Length ~ Petal.Width");
  /// std::print("{:.0f}\n",m.score(test).double_at("n",0)); // 30
  /// @endcode
  [[nodiscard]] dstruct::DataFrame score(const dstruct::DataFrame& data) const;
  /// @brief Formats kernel, bandwidth, LOO coverage and conditional metrics.
  /// @return Owned text; no output is written.
  /// @par Example
  /// See print_summary().
  [[nodiscard]] std::string summary() const;
  /// @brief Writes summary() to a borrowed stream.
  /// @param os Stream to mutate by insertion.
  /// @par Example
  /// See the standard-output overload.
  void print_summary(std::ostream& os) const;
  /// @brief Writes summary() to standard output.
  /// @par Example
  /// @code
  /// const dm::stats::KernelRegression m(dm::datasets::iris(),"Petal.Length ~ Petal.Width");
  /// std::ostringstream out;m.print_summary(out);std::print("{}\n",out.str()==m.summary()); // true
  /// @endcode
  void print_summary() const;
  /// @brief Observations and ordinary full-sample curve for one direct numeric predictor.
  /// @param data Frame with the original predictor/response; null/nonfinite displayed pairs
  /// omitted.
  /// @param grid_resolution At least two equally spaced query points, default 200.
  /// @return Owned editable plot; gaps in compact support remain gaps, not connecting lines.
  /// @throws std::invalid_argument Not one direct numeric predictor, invalid grid or schema.
  /// @throws std::runtime_error As predict().
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();
  /// const dm::stats::KernelRegression m(d,"Petal.Length ~ Petal.Width");
  /// m.plot_fit(d).save_svg("kernel_fit.svg");
  /// @endcode
  [[nodiscard]] graphics::RPlot plot_fit(
      const dstruct::DataFrame& data,
      std::size_t               grid_resolution = 200) const;
  /// @brief Finite LOO predictions against actual retained responses.
  /// @return Owned plot; an all-unsupported fit gives an invalid_argument instead of an empty plot.
  /// @throws std::invalid_argument No finite deletion predictions.
  /// @par Example
  /// @code
  /// const dm::stats::KernelRegression m(dm::datasets::iris(),"Petal.Length ~ Petal.Width");
  /// m.plot_predicted_vs_actual().save_svg("kernel_actual.svg");
  /// @endcode
  [[nodiscard]] graphics::RPlot plot_predicted_vs_actual() const;
  /// @brief Finite LOO residuals against deletion predictions.
  /// @return Owned editable plot with zero reference line.
  /// @throws std::invalid_argument No finite deletion predictions.
  /// @par Example
  /// @code
  /// const dm::stats::KernelRegression m(dm::datasets::iris(),"Petal.Length ~ Petal.Width");
  /// m.plot_residuals_vs_fitted().save_svg("kernel_residuals.svg");
  /// @endcode
  [[nodiscard]] graphics::RPlot plot_residuals_vs_fitted() const;
  /// @brief Finite complete-coverage CV losses against log10 bandwidth, with selected h marked.
  /// @return Owned editable plot; ineligible candidates are omitted, not given zero loss.
  /// @throws std::invalid_argument Fixed-bandwidth model has no selection curve.
  /// @par Example
  /// @code
  /// const dm::stats::KernelRegression m(dm::datasets::iris(),"Petal.Length ~ Petal.Width");
  /// m.plot_cv_curve().save_svg("kernel_cv.svg");
  /// @endcode
  [[nodiscard]] graphics::RPlot plot_cv_curve() const;

 private:
  /// @brief One query's owned normalized weights and resulting diagnostics.
  struct LocalFit {
    std::vector<double> weights;     ///< Normalized retained-row weights, NaN if unsupported.
    double              prediction;  ///< Local mean or NaN.
    double              effective_n; ///< 1/sum(w^2), NaN if unsupported.
    double      nearest_distance;    ///< Nearest distance in design units, before division by h.
    std::size_t positive_weights;    ///< Positive numerically represented kernel weights.
  };
  /// @brief Fits geometry, searches h and caches chosen-h training diagnostics.
  /// @param data Training frame.
  void fit(const dstruct::DataFrame& data);
  /// @brief Computes relative kernel weights and a local mean without response-dependent geometry.
  /// @param query Finite standardized query coordinates, length p.
  /// @param h Positive finite bandwidth.
  /// @param exclude Training row omitted; SIZE_MAX omits none.
  /// @return Owned local result; an unsupported compact window has NaN weights/prediction.
  LocalFit local_fit(const std::vector<double>& query, double h, std::size_t exclude = SIZE_MAX)
      const;
  /// @brief Evaluates one original query row using frozen design and scaling.
  /// @param data Query frame.
  /// @param row Original query index.
  /// @param status Receives ok/missing_predictor/nonfinite_predictor.
  /// @return Standardized coordinates, or empty for invalid row.
  std::vector<double>
             query_row(const dstruct::DataFrame& data, std::size_t row, std::string& status) const;
  Formula    formula_; ///< Owned parsed formula.
  DesignInfo design_;  ///< Frozen transforms/levels and no-intercept design schema.
  KernelRegressionOptions     options_;           ///< Owned validated controls.
  std::vector<std::string>    predictor_names_;   ///< Resolved design names.
  std::vector<std::size_t>    kept_rows_;         ///< Original training indices.
  std::vector<double>         feature_mean_;      ///< Frozen design centring values.
  std::vector<double>         feature_scale_;     ///< Frozen positive design scales.
  std::vector<double>         feature_min_;       ///< Training design minima.
  std::vector<double>         feature_max_;       ///< Training design maxima.
  linalg::DenseMatrix<double> training_X_std_;    ///< Owned finite n-by-p design.
  std::vector<double>         training_y_;        ///< Retained finite responses.
  double                      training_mean_{0};  ///< Training-only response baseline.
  double                      bandwidth_used_{0}; ///< Positive chosen bandwidth.
  bool bandwidth_was_selected_{false};            ///< Negative input bandwidth triggered search.
  std::vector<double> bandwidth_grid_;            ///< Candidate bandwidths in caller/default order.
  std::vector<double> cv_mse_;                    ///< Complete-coverage candidate objectives.
  std::vector<double> cv_conditional_mse_;        ///< Candidate losses on finite deletion rows.
  std::vector<double> cv_counts_;                 ///< Covered deletion row counts per candidate.
  std::vector<double> fitted_;                    ///< Chosen-bandwidth deletion predictions.
  std::vector<double> full_fitted_;               ///< Ordinary full-sample training predictions.
  std::vector<double> leverage_;                  ///< Chosen-bandwidth smoother diagonal.
  std::vector<double> effective_n_;               ///< Full-sample weight effective sizes.
  std::vector<double> loo_effective_n_;           ///< Deletion weight effective sizes.
  std::size_t         observations_{0};           ///< Retained training count.
};
} // namespace datamunge::stats
