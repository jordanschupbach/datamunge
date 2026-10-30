#pragma once
/// @file elastic_net.hpp
/// @brief Formula-based penalized least squares, fold-local CV and KKT/path diagnostics.
#include <cstddef>
#include <cstdint>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::stats {
/// @brief Gaussian elastic-net controls in glmnet's lambda/alpha parameterization.
/// The objective is RSS/(2*n)+lambda*(alpha*L1+(1-alpha)*L2_squared/2), applied to
/// preprocessed slope coefficients. The intercept is unpenalized; response is not scaled.
/// CV-only path/fold controls are ignored for a fixed nonnegative lambda.
/// @par Example
/// @code
/// const dm::stats::ElasticNetOptions o{.5,.5};
/// std::print("{:.1f}\n",o.alpha); // 0.5
/// @endcode
///
/// @par References
/// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
/// @see ElasticNet
struct ElasticNetOptions {
  double      alpha{.5};  ///< L1 proportion in [0,1]; zero ridge, one lasso.
  double      lambda{-1}; ///< Finite >=0 fixes the penalty; any finite negative value requests CV.
  std::size_t n_lambda{100};   ///< Positive automatic path length, including a possible singleton.
  std::size_t cv_folds{5};     ///< Automatic CV needs 2<=folds<=complete rows.
  std::size_t max_iter{10000}; ///< Positive cyclic sweep limit for every path/fold fit.
  double      tol{1e-7}; ///< Positive absolute coefficient-change and scale-relative KKT tolerance.
  std::uint64_t seed{42};          ///< Local SplitMix64 seed for balanced shuffled fold assignment.
  bool          standardize{true}; ///< Scale predictors by population RMS about the fitted origin.
  double        lambda_min_ratio{
      0}; ///< 0 uses .01 if n<p, otherwise .0001; positive <=1 overrides path endpoint ratio.
};
/// @brief Elastic-net penalized least squares via cyclic exact-coordinate descent.
/// Accepts Formula numeric/categorical main effects and interactions. Rows with a null
/// formula value are dropped; retained design entries and responses must be finite.
/// With an intercept, centre X/y; without one, use a zero origin. Optional predictor
/// scaling uses sqrt(sum((x-mean)^2)/n), including for indicator/interaction columns;
/// exact zero scales become one. Response is never standardized. Published slopes and
/// intercept are restored to raw formula-design units; penalties apply to scaled slopes.
/// Automatic paths share a grid derived from the full training set. Each validation
/// fold fits its own predictor/response offsets and predictor scales on training rows,
/// using the fixed full-data Formula schema, and pools held-out squared errors by row.
/// Selection minimizes this MSE, retaining the first (largest lambda) on exact ties.
/// The selected full-data path fit supplies coefficients; exhausted fits are retained
/// and flagged, not silently removed from selection. Inspect fold and path status.
/// Cyclic descent uses actual column norms, including zero/constant columns. Stopping
/// requires max_change<=tol and KKT<=tol*max(1,max_j(abs(X_j^T*y/n))) in preprocessed
/// coordinates. A finite iteration limit returns a partial fit, with converged=false.
/// Positive lambda*(1-alpha) gives a strictly convex slope objective; lasso/OLS can have
/// nonunique coefficients on dependent designs. No covariance, p-values or confidence
/// intervals are inferred from selected coefficients. Unlike the rescaled 2005 elastic
/// net estimator, this implements the direct glmnet-style penalized objective.
/// Owns fitted state, leaves input unchanged; const methods have no mutable caches.
/// One t-sweep fit costs O(t*n*p); a length-L, F-fold path costs O((F+1)*L*t*n*p),
/// plus Formula expansion, with numerical storage O(n*p+L*p+F*L).
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;
/// d.add_column("x",std::vector<double>{-1,-1,1,1});
/// d.add_column("y",std::vector<double>{1,1,5,5});
/// const dm::stats::ElasticNet m(d,"y ~ x",{.5,.5});
/// std::print("{:.1f} {:.0f}\n",m.coefficients()[0],m.intercept()); // 1.4 3
/// @endcode
///
/// @par References
/// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
/// @see Ridge, Lasso, diagnostics(), cv_fold_results()
class ElasticNet {
 public:
  /// @brief Fits a numeric formula, with a fixed penalty or automatic CV path.
  /// @param data Borrowed frame supplying formula variables, unchanged.
  /// @param formula Numeric response formula, e.g. "y ~ x1 + x2 + x1:x2".
  /// @param options Penalty, preprocessing and solver controls; see ElasticNetOptions.
  /// @throws std::invalid_argument Invalid formula/schema/options, nonfinite retained
  /// values or unrepresentable preprocessing, optimization, CV, prediction or path values.
  /// @throws std::runtime_error Formula expansion leaves no complete observations.
  /// @par Example
  /// See the class example.
  /// @see diagnostics(), predict()
  ElasticNet(
      const dstruct::DataFrame& data,
      const std::string&        formula,
      ElasticNetOptions         options = {});
  /// @brief Delegates to the frame-first constructor with R-style argument order.
  /// @param formula Formula object; strings convert implicitly through Formula.
  /// @param data Borrowed formula variables.
  /// @param options Same controls as the primary constructor.
  /// @throws std::invalid_argument Same as the primary constructor.
  /// @throws std::runtime_error No complete observations.
  /// @par Example
  /// See the class example, substituting ElasticNet(dm::formula("y ~ x"),d,{.5,.5}).
  /// @see ElasticNet(const dstruct::DataFrame&,const std::string&,ElasticNetOptions)
  ElasticNet(const Formula& formula, const dstruct::DataFrame& data, ElasticNetOptions options = {})
      : ElasticNet(data, formula.text(), std::move(options)) {}
  /// @brief Original formula text.
  /// @return Borrowed text, valid for model lifetime.
  [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
  /// @brief Whether Formula expansion includes an unpenalized intercept.
  /// @return Intercept flag.
  [[nodiscard]] bool has_intercept() const { return design_.has_intercept; }
  /// @brief Expanded design-column names, excluding the intercept.
  /// @return Borrowed p names, matching coefficients and preprocessing vectors.
  [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
  /// @brief Complete training-row count after Formula null dropping.
  /// @return Positive n.
  [[nodiscard]] std::size_t observations() const { return observations_; }
  /// @brief Maps fitted values/residuals back to input rows.
  /// @return Borrowed original zero-based row indices in complete-row order.
  [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const {
    return kept_row_indices_;
  }
  /// @brief Full-training predictor offsets used for penalty coordinates.
  /// @return Borrowed p means; all zero if the formula has no intercept.
  [[nodiscard]] const std::vector<double>& feature_means() const { return feature_mean_; }
  /// @brief Full-training predictor divisors, including expanded interaction columns.
  /// @return Borrowed p positive scales; ones if disabled or a column has zero RMS.
  [[nodiscard]] const std::vector<double>& feature_scales() const { return feature_scale_; }
  /// @brief Actual L1 mixing proportion, fixed by Ridge/Lasso wrappers when used.
  /// @return Value in [0,1].
  [[nodiscard]] double alpha() const { return options_.alpha; }
  /// @brief Actual nonnegative penalty used by coefficients().
  /// @return Fixed requested strength or the selected path value.
  [[nodiscard]] double lambda() const { return lambda_used_; }
  /// @brief Whether a negative options.lambda requested CV.
  /// @return True for automatic selection, even for a singleton path.
  [[nodiscard]] bool lambda_was_selected() const { return lambda_was_selected_; }
  /// @brief Common positive descending grid derived from the full training set.
  /// @return Borrowed L values, or empty for fixed lambda. For alpha>0 the maximum
  /// is max(abs(X_j^T*y/n))/alpha, with a fallback one at zero correlation; ridge uses
  /// a heuristic divisor .001 and has no finite all-zero cutoff.
  [[nodiscard]] const std::vector<double>& lambda_path() const { return lambda_path_; }
  /// @brief Pooled out-of-fold squared error divided by complete-row count.
  /// @return Borrowed L finite MSEs aligned with lambda_path(), or empty for fixed lambda.
  [[nodiscard]] const std::vector<double>& cv_mean_squared_error() const { return cv_mse_path_; }
  /// @brief Original-design-unit coefficients at every full-data path fit.
  /// @return Borrowed L-by-p matrix, excluding intercepts; empty for fixed lambda.
  [[nodiscard]] const linalg::DenseMatrix<double>& path_coefficients() const {
    return path_coefficients_;
  }
  /// @brief Balanced shuffled held-out fold IDs for complete training rows.
  /// @return Borrowed n zero-based fold IDs aligned with kept_row_indices(); empty for fixed
  /// lambda.
  [[nodiscard]] const std::vector<std::size_t>& fold_ids() const { return fold_ids_; }
  /// @brief Final slopes in original formula-design units, including exact zeros.
  /// @return Borrowed p values; the intercept is separate.
  [[nodiscard]] const std::vector<double>& coefficients() const { return coefficients_; }
  /// @brief Final unpenalized raw-unit intercept.
  /// @return Response mean minus feature-means dot slopes, or zero without intercept.
  [[nodiscard]] double intercept() const { return intercept_; }
  /// @brief Training predictions in complete-row order.
  /// @return Borrowed n values, aligned through kept_row_indices().
  [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
  /// @brief Training observed-minus-fitted residuals.
  /// @return Borrowed n values, excluding dropped rows.
  [[nodiscard]] const std::vector<double>& residuals() const { return residuals_; }
  /// @brief Completed sweeps for the selected full-data fit.
  /// @return Count <=max_iter; zero for an intercept-only model.
  [[nodiscard]] std::size_t iterations() const { return iterations_; }
  /// @brief Whether the selected full-data fit met both change and KKT conditions.
  /// @return Numerical status; fold/path status is separate and does not imply predictive validity.
  [[nodiscard]] bool converged() const { return converged_; }
  /// @brief Selected fit objective and numerical optimality evidence.
  /// @return Borrowed one-row frame: alpha, lambda, iterations, converged (0/1), max_change,
  /// max_kkt_violation, loss (RSS/2n), l1_norm, l2_squared, objective; penalty coordinates.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,.5});
  /// std::print("{}\n",m.diagnostics().nrows()); // 1
  /// @endcode
  ///
  /// @par References
  /// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
  ///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
  /// @see fold_ids(), kept_row_indices()
  [[nodiscard]] const dstruct::DataFrame& diagnostics() const { return diagnostics_; }
  /// @brief Full-data path fits and pooled CV error, without filtering candidates.
  /// @return Borrowed L-row frame: lambda, cv_mse, iterations, converged, max_kkt_violation,
  /// objective. Empty without columns for fixed lambda.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,-1,5,2});
  /// std::print("{}\n",m.path_diagnostics().nrows()); // 5
  /// @endcode
  ///
  /// @par References
  /// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
  ///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
  /// @see fold_ids(), kept_row_indices()
  [[nodiscard]] const dstruct::DataFrame& path_diagnostics() const { return path_diagnostics_; }
  /// @brief Every fold/candidate result with its own preprocessing and solver status.
  /// @return Borrowed F*L rows in fold-major/path order: fold, lambda, n_train, n_test, mse,
  /// iterations, converged, max_kkt_violation. Empty without columns for fixed lambda.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,-1,5,2});
  /// std::print("{}\n",m.cv_fold_results().nrows()); // 10
  /// @endcode
  ///
  /// @par References
  /// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
  ///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
  /// @see fold_ids(), kept_row_indices()
  [[nodiscard]] const dstruct::DataFrame& cv_fold_results() const { return cv_fold_results_; }
  /// @brief Counts original-unit slopes whose absolute value exceeds 1e-10.
  /// @return Thresholded count in 0..p, not a rank or degrees-of-freedom estimate.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,.5});
  /// std::print("{}\n",m.non_zero_coefficients()); // 1
  /// @endcode
  ///
  /// @par References
  /// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
  ///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
  /// @see diagnostics()
  [[nodiscard]] std::size_t non_zero_coefficients() const;
  /// @brief Centred training R-squared, even for a formula without an intercept.
  /// @return 1-RSS/TSS; constant response returns one for exact fit, otherwise NaN.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,.5});
  /// std::print("{:.2f}\n",m.r_squared()); // 0.91
  /// @endcode
  ///
  /// @par References
  /// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
  ///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
  /// @see diagnostics()
  [[nodiscard]] double r_squared() const;
  /// @brief Training root mean squared prediction residual.
  /// @return sqrt(RSS/n) in response units, not an out-of-sample estimate.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,.5});
  /// std::print("{:.1f}\n",m.rmse()); // 0.6
  /// @endcode
  ///
  /// @par References
  /// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
  ///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
  /// @see diagnostics()
  [[nodiscard]] double rmse() const;
  /// @brief Presents original and penalized coordinates for every expanded predictor.
  /// @return Owned p rows: predictor, coefficient, penalty_coefficient, center, scale;
  /// penalty_coefficient=scale*coefficient. The intercept is reported separately.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,.5});
  /// std::print("{}\n",m.coefficient_table().nrows()); // 1
  /// @endcode
  ///
  /// @par References
  /// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
  ///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
  /// @see diagnostics()
  [[nodiscard]] dstruct::DataFrame coefficient_table() const;
  /// @brief Formats coefficients, training metrics, selected lambda and solver status.
  /// @return Owned plain text; finite exhaustion is labelled as nonconvergence.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,.5});
  /// std::print("{}\n",!m.summary().empty()); // true
  /// @endcode
  ///
  /// @par References
  /// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
  ///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
  /// @see diagnostics()
  [[nodiscard]] std::string summary() const;
  /// @brief Writes summary() to a supplied stream.
  /// @param os Output stream; model is unchanged.
  /// @par Example
  /// See summary().
  /// @see summary()
  void print_summary(std::ostream& os) const;
  /// @brief Writes summary() to standard output.
  /// @par Example
  /// See summary().
  /// @see print_summary(std::ostream&)
  void print_summary() const;
  /// @brief Predicts using the frozen Formula schema and original-unit coefficients.
  /// No query centring/scaling is fitted. A row has the same prediction alone or batched.
  /// @param newdata Borrowed predictors; response is not required, extra columns ignored.
  /// @return Owned vector in original row order, NaN for null-predictor rows. Empty input
  /// returns empty before schema validation. If every nonempty row is incomplete, throws.
  /// @throws std::invalid_argument Missing/mistyped columns, unseen categorical levels,
  /// retained nonfinite design entries, or unrepresentable predictions.
  /// @throws std::runtime_error No complete query rows remain.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,.5});
  /// std::print("{:.1f}\n",m.predict(d)[0]); // 1.6
  /// @endcode
  /// @see fitted_values(), kept_row_indices()
  [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;
  /// @brief Scores complete labelled query rows using the frozen model.
  /// @param newdata Borrowed response/predictor columns; null formula rows are omitted.
  /// @return Owned row: n_input, n_scored, rmse, mae, r_squared, mean_baseline_rmse.
  /// Metrics use retained query rows; the baseline is the complete training response mean,
  /// including for no-intercept models. Constant query response gives R-squared one only
  /// for exact prediction, otherwise NaN. No query preprocessing is fitted.
  /// @throws std::invalid_argument Bad schema/levels, retained nonfinite design/response
  /// or unrepresentable metric arithmetic.
  /// @throws std::runtime_error No complete query observations.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,.5});
  /// std::print("{:.1f}\n",m.score(d).double_at("rmse",0)); // 0.6
  /// @endcode
  /// @see predict(), rmse()
  [[nodiscard]] dstruct::DataFrame score(const dstruct::DataFrame& newdata) const;
  /// @brief Original-unit slope traces against log10(lambda).
  /// @return Owned plot of every candidate, including nonconverged fits; inspect
  /// path_diagnostics().
  /// @throws std::invalid_argument Fixed-lambda fit has no automatic path.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,-1,5,2});
  /// std::print("{}\n",!m.plot_coefficient_path().series().empty()); // true
  /// @endcode
  /// @see path_diagnostics(), diagnostics()
  [[nodiscard]] graphics::RPlot plot_coefficient_path() const;
  /// @brief All pooled CV MSEs against log10(lambda), highlighting the selected candidate.
  /// @return Owned plot with every finite candidate; no large-error filtering.
  /// @throws std::invalid_argument Fixed-lambda fit has no automatic path.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,-1,5,2});
  /// std::print("{}\n",!m.plot_cv_curve().series().empty()); // true
  /// @endcode
  /// @see path_diagnostics(), diagnostics()
  [[nodiscard]] graphics::RPlot plot_cv_curve() const;
  /// @brief Training fitted values against actual complete-row responses.
  /// @return Owned scatterplot; not a held-out performance estimate.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,.5});
  /// std::print("{}\n",!m.plot_predicted_vs_actual().series().empty()); // true
  /// @endcode
  /// @see path_diagnostics(), diagnostics()
  [[nodiscard]] graphics::RPlot plot_predicted_vs_actual() const;
  /// @brief Training residuals against fitted values.
  /// @return Owned scatterplot in response units.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x",std::vector<double>{-1,-1,1,1});
  /// d.add_column("y",std::vector<double>{1,1,5,5});
  /// const dm::stats::ElasticNet m(d,"y ~ x",{.5,.5});
  /// std::print("{}\n",!m.plot_residuals_vs_fitted().series().empty()); // true
  /// @endcode
  /// @see path_diagnostics(), diagnostics()
  [[nodiscard]] graphics::RPlot plot_residuals_vs_fitted() const;

 protected:
  /// @brief Shared constructor used by Ridge and Lasso to replace options.alpha.
  /// @param data Borrowed frame.
  /// @param formula Numeric response formula.
  /// @param fixed_alpha Forced mixing proportion.
  /// @param options Other controls, forwarded unchanged.
  /// @throws std::invalid_argument Same as the public constructor.
  /// @throws std::runtime_error No complete training rows.
  /// @par Example
  /// See Ridge and Lasso.
  /// @see ElasticNet
  ElasticNet(
      const dstruct::DataFrame& data,
      const std::string&        formula,
      double                    fixed_alpha,
      ElasticNetOptions         options);
  /// @brief Delegates forced-alpha fitting in formula-first argument order.
  /// @param formula Numeric response formula object.
  /// @param data Borrowed frame.
  /// @param fixed_alpha Forced mixing proportion.
  /// @param options Other fitting controls.
  /// @throws std::invalid_argument Same as the public constructor.
  /// @throws std::runtime_error No complete training rows.
  /// @par Example
  /// See Ridge and Lasso.
  /// @see ElasticNet
  ElasticNet(
      const Formula&            formula,
      const dstruct::DataFrame& data,
      double                    fixed_alpha,
      ElasticNetOptions         options)
      : ElasticNet(data, formula.text(), fixed_alpha, std::move(options)) {}

 private:
  /// @brief Resolves Formula, preprocesses, fits/CVs and restores raw-unit predictions.
  /// @param data Borrowed constructor frame.
  /// @throws std::invalid_argument Invalid options/data or unrepresentable arithmetic.
  /// @throws std::runtime_error No complete formula rows.
  void                        fit(const dstruct::DataFrame& data);
  Formula                     formula_; ///< Original parsed model formula.
  DesignInfo                  design_;  ///< Frozen expanded model schema and categorical levels.
  ElasticNetOptions           options_; ///< Validated controls with wrapper alpha override.
  std::vector<std::string>    predictor_names_; ///< Expanded non-intercept design names.
  std::vector<double>         feature_mean_; ///< Full-training offsets, or zeros without intercept.
  std::vector<double>         feature_scale_; ///< Full-training positive predictor divisors.
  double                      y_mean_{0}; ///< Training response offset, or zero without intercept.
  std::vector<double>         coefficients_;   ///< Original-design-unit slopes.
  double                      intercept_{0};   ///< Unpenalized raw-unit intercept.
  double                      lambda_used_{0}; ///< Fixed or selected finite nonnegative penalty.
  bool                        lambda_was_selected_{false}; ///< Automatic CV branch was used.
  std::vector<double>         lambda_path_;                ///< Descending common candidate grid.
  std::vector<double>         cv_mse_path_;                ///< Pooled row-weighted held-out errors.
  std::size_t                 best_lambda_index_{0}; ///< Selected candidate index when automatic.
  linalg::DenseMatrix<double> path_coefficients_;    ///< Full-data path slopes in original units.
  std::vector<std::size_t>    fold_ids_;      ///< Complete-row fold assignments, empty when fixed.
  std::vector<std::size_t> kept_row_indices_; ///< Original rows used by the fitted Formula matrix.
  dstruct::DataFrame       diagnostics_;      ///< Selected full-data solution evidence.
  dstruct::DataFrame       path_diagnostics_; ///< Every full-data path candidate and CV loss.
  dstruct::DataFrame       cv_fold_results_;  ///< Every fold/candidate numerical result.
  std::size_t              iterations_{0};    ///< Selected full-data cyclic sweep count.
  bool                     converged_{false}; ///< Selected full-data change/KKT conditions passed.
  std::vector<double>      fitted_;           ///< Complete-row raw predictions.
  std::vector<double>      residuals_;        ///< Complete-row observed-minus-predicted residuals.
  std::vector<double>      training_y_;       ///< Retained numeric responses.
  std::size_t              observations_{0};  ///< Complete training-row count.
};
/// @brief Ridge controls forwarded to ElasticNet, fixing alpha=0.
/// Same defaults/constraints as ElasticNetOptions; no alpha field.
/// @par Example
/// @code
/// const dm::stats::RidgeOptions o;
/// std::print("{}\n",o.standardize); // true
/// @endcode
/// @see ElasticNetOptions
struct RidgeOptions {
  double      lambda{-1}; ///< Finite >=0 fixes the penalty; any finite negative value requests CV.
  std::size_t n_lambda{100};   ///< Positive automatic path length, including a possible singleton.
  std::size_t cv_folds{5};     ///< Automatic CV needs 2<=folds<=complete rows.
  std::size_t max_iter{10000}; ///< Positive cyclic sweep limit for every path/fold fit.
  double      tol{1e-7}; ///< Positive absolute coefficient-change and scale-relative KKT tolerance.
  std::uint64_t seed{42};          ///< Local SplitMix64 seed for balanced shuffled fold assignment.
  bool          standardize{true}; ///< Scale predictors by population RMS about the fitted origin.
  double        lambda_min_ratio{
      0}; ///< 0 uses .01 if n<p, otherwise .0001; positive <=1 overrides path endpoint ratio.
};
/// @brief Ridge convenience model using the shared coordinate-descent/CV engine.
/// Fixes alpha=0; all ElasticNet contracts apply. L2 shrinkage does not impose a sparsity
/// threshold, but exact zero coefficients are possible for zero/orthogonal predictors.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;
/// d.add_column("x",std::vector<double>{-1,-1,1,1});
/// d.add_column("y",std::vector<double>{1,1,5,5});
/// const dm::stats::Ridge m(d,"y ~ x",{.5});
/// std::print("{:.0f}\n",m.alpha()); // 0
/// @endcode
///
/// @par References
/// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
/// @see ElasticNet
class Ridge : public ElasticNet {
 public:
  /// @brief Fits the fixed-mixing wrapper immediately.
  /// @param data Borrowed formula variables.
  /// @param formula Numeric response formula.
  /// @param options Shared penalty/preprocessing/solver controls.
  /// @throws std::invalid_argument Same as ElasticNet.
  /// @throws std::runtime_error No complete observations.
  /// @par Example
  /// See the class example.
  /// @see ElasticNet
  Ridge(const dstruct::DataFrame& data, const std::string& formula, RidgeOptions options = {});
  /// @brief Fits the wrapper in formula-first argument order.
  /// @param formula Formula object, accepting implicit string conversion.
  /// @param data Borrowed frame.
  /// @param options Shared controls.
  /// @throws std::invalid_argument Same as ElasticNet.
  /// @throws std::runtime_error No complete observations.
  /// @par Example
  /// See the class example with arguments reordered as (dm::formula("y ~ x"),d,{.5}).
  /// @see ElasticNet
  Ridge(const Formula& formula, const dstruct::DataFrame& data, RidgeOptions options = {})
      : Ridge(data, formula.text(), std::move(options)) {}
};
/// @brief Lasso controls forwarded to ElasticNet, fixing alpha=1.
/// Same defaults/constraints as ElasticNetOptions; no alpha field.
/// @par Example
/// @code
/// const dm::stats::LassoOptions o;
/// std::print("{}\n",o.standardize); // true
/// @endcode
/// @see ElasticNetOptions
struct LassoOptions {
  double      lambda{-1}; ///< Finite >=0 fixes the penalty; any finite negative value requests CV.
  std::size_t n_lambda{100};   ///< Positive automatic path length, including a possible singleton.
  std::size_t cv_folds{5};     ///< Automatic CV needs 2<=folds<=complete rows.
  std::size_t max_iter{10000}; ///< Positive cyclic sweep limit for every path/fold fit.
  double      tol{1e-7}; ///< Positive absolute coefficient-change and scale-relative KKT tolerance.
  std::uint64_t seed{42};          ///< Local SplitMix64 seed for balanced shuffled fold assignment.
  bool          standardize{true}; ///< Scale predictors by population RMS about the fitted origin.
  double        lambda_min_ratio{
      0}; ///< 0 uses .01 if n<p, otherwise .0001; positive <=1 overrides path endpoint ratio.
};
/// @brief Lasso convenience model using the shared coordinate-descent/CV engine.
/// Fixes alpha=1; all ElasticNet contracts apply. L1 thresholding permits exact zero slopes.
/// Dependent designs can have nonunique coefficients; predictions may still be unique.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;
/// d.add_column("x",std::vector<double>{-1,-1,1,1});
/// d.add_column("y",std::vector<double>{1,1,5,5});
/// const dm::stats::Lasso m(d,"y ~ x",{.5});
/// std::print("{:.0f}\n",m.alpha()); // 1
/// @endcode
///
/// @par References
/// - Friedman, Hastie and Tibshirani (2010), Regularization Paths for Generalized Linear
///   Models via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
/// @see ElasticNet
class Lasso : public ElasticNet {
 public:
  /// @brief Fits the fixed-mixing wrapper immediately.
  /// @param data Borrowed formula variables.
  /// @param formula Numeric response formula.
  /// @param options Shared penalty/preprocessing/solver controls.
  /// @throws std::invalid_argument Same as ElasticNet.
  /// @throws std::runtime_error No complete observations.
  /// @par Example
  /// See the class example.
  /// @see ElasticNet
  Lasso(const dstruct::DataFrame& data, const std::string& formula, LassoOptions options = {});
  /// @brief Fits the wrapper in formula-first argument order.
  /// @param formula Formula object, accepting implicit string conversion.
  /// @param data Borrowed frame.
  /// @param options Shared controls.
  /// @throws std::invalid_argument Same as ElasticNet.
  /// @throws std::runtime_error No complete observations.
  /// @par Example
  /// See the class example with arguments reordered as (dm::formula("y ~ x"),d,{.5}).
  /// @see ElasticNet
  Lasso(const Formula& formula, const dstruct::DataFrame& data, LassoOptions options = {})
      : Lasso(data, formula.text(), std::move(options)) {}
};
} // namespace datamunge::stats
