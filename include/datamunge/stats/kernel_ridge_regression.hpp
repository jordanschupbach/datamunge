#pragma once
/// @file kernel_ridge_regression.hpp
/// @brief Free-intercept kernel ridge regression, fixed-metric LOO and fold-refitted validation.
#include <cstddef>
#include <cstdint>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>
#include <iosfwd>
#include <string>
#include <vector>
namespace datamunge::stats {
/// @brief Kernel, penalty, training metric and finite candidate grids for KRR.
/// Lambda multiplies the RKHS penalty beside a sum of squared errors, not their
/// mean: an objective mean loss + rho*norm^2 corresponds to lambda=n*rho.
/// Gamma is inverse squared length in standardized design units when enabled.
/// @par Example
/// @code
/// dm::stats::KernelRidgeRegressionOptions o;
/// o.lambda = .5;
/// o.gamma = std::log(2.);
/// o.standardize = false;
/// @endcode
/// @see KernelRidgeRegression
struct KernelRidgeRegressionOptions {
  std::string kernel{"rbf"}; ///< rbf, linear, or polynomial, case-sensitive.
  double lambda{-1}; ///< Finite nonnegative fixed penalty; negative requests finite-grid search.
  double gamma{-1};  ///< Positive fixed RBF/polynomial scale; negative requests search, zero
                     ///< invalid; ignored for linear.
  double      degree{3}; ///< Polynomial positive integer exponent; ignored by other kernels.
  double      coef0{1};  ///< Polynomial finite nonnegative offset, preserving a PSD kernel.
  std::size_t n_lambda_grid{40}; ///< Positive default log-grid count from 1e-4 to 1e3.
  std::size_t n_gamma_grid{
      12}; ///< Positive default grid count from .03/d_med^2 to 30/d_med^2; one uses 1/d_med^2.
  bool standardize{
      true}; ///< Learn design means/population SDs from fitting rows; constant columns use one.
  bool fit_intercept{
      true}; ///< Free unpenalized intercept, implemented by kernel/response centering.
  std::vector<double> lambda_candidates; ///< Optional nonnegative auto-search values in caller
                                         ///< order; fixed lambda ignores them.
  std::vector<double> gamma_candidates;  ///< Optional positive auto-search values in caller order;
                                         ///< fixed/linear gamma ignores them.
};
/// @brief Dense RKHS least squares with a genuine optional unpenalized intercept.
/// Minimizes sum_i(y_i-b-f(x_i))^2+lambda*||f||_H^2. With a free intercept,
/// center both K and y: alpha=(C*K*C+lambda*I)^-1*C*y and predict with the
/// consistently centered query kernel plus mean(y). This enforces sum(alpha)=0;
/// the equivalent raw-kernel intercept is b=mean(y)-mean_kernel'alpha.
/// Without an intercept, neither K nor y is centered and b=0. Formula intercept
/// markers do not add kernel features; fit_intercept controls b explicitly.
///
/// Numeric Formula design supports expanded transforms/categoricals. Required
/// null fitting rows are omitted jointly; nonfinite retained cells are rejected.
/// Training-only predictor scaling is frozen at query time. The fast LOO identity
/// refits the intercept/kernel centering at each deletion but holds this metric
/// and design schema fixed. It does not refit scaling or hyperparameter selection.
/// Use kernel_ridge_cross_validate() for fold-local preprocessing, and a separate
/// holdout to assess the selected procedure. fitted_values()/rmse()/r_squared()
/// retain their historical LOO meaning; training_predictions() is resubstitution.
///
/// Lambda zero uses a relative-rank spectral pseudoinverse and explicit fixed-metric
/// deletions; it is a numerical minimum-RKHS-norm least-squares limit, not guaranteed
/// interpolation for duplicate/incompatible rows. Positive lambda never uses an
/// arbitrary denominator or leverage floor. Tiny lambdas can still be ill-conditioned.
///
/// G gamma candidates and L penalties cost O(G*(n^2*p+n^3+L*n^2)); zero-penalty
/// deletion diagnostics can add O(G*n^4). Dense fitted storage is O(n^2+n*p).
/// Models own their arrays; const calls use no mutable caches and do not mutate inputs.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;
/// d.add_column("x", std::vector<double>{0, 1});
/// d.add_column("y", std::vector<double>{0, 2});
/// dm::stats::KernelRidgeRegressionOptions o;
/// o.lambda = .5;
/// o.gamma = std::log(2.);
/// o.standardize = false;
/// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
/// std::print("{:.1f} {:.1f}\n", m.training_predictions()[0],
///            m.training_predictions()[1]); //
///                                          // Expected: 0.5 1.5
/// @endcode
/// @par References
/// - Kimeldorf and Wahba (1971), Some results on Tchebycheffian spline functions.
///   https://doi.org/10.1016/0022-247X(71)90184-3
/// - Saunders, Gammerman and Vovk (1998), Ridge Regression Learning Algorithm in Dual Variables.
///   https://eprints.soton.ac.uk/258942/1/Dualrr_ICML98.pdf
/// @see KernelRidgeRegressionOptions, kernel_ridge_cross_validate()
class KernelRidgeRegression {
 public:
  /// @brief Fit the complete resolved rows immediately, selecting finite grids when requested.
  /// @param data Borrowed fitting frame; required null rows omitted jointly.
  /// @param formula Numeric response/predictor Formula text, with its explicit intercept
  /// suppressed.
  /// @param options Kernel, fixed-metric LOO search and preprocessing choices.
  /// @throws std::invalid_argument Invalid controls, nonfinite values, no predictors or fewer than
  /// two rows.
  /// @throws std::runtime_error Formula failure, no usable candidate or unrepresentable
  /// kernel/solve.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::print("{}\n", m.observations()); // 2
  /// @endcode
  /// @see KernelRidgeRegression
  KernelRidgeRegression(
      const dstruct::DataFrame&    data,
      const std::string&           formula,
      KernelRidgeRegressionOptions options = {});
  /// @brief Formula-first constructor, matching R's argument order.
  /// @param formula Parsed numeric-response formula; strings convert implicitly.
  /// @param data Frame with the primary constructor's omission rules.
  /// @param options Same controls as the primary constructor.
  /// @throws std::invalid_argument As the primary constructor.
  /// @throws std::runtime_error As the primary constructor.
  /// @par Example
  /// See the class example, passing Formula(formula) before data.
  /// @see KernelRidgeRegression
  KernelRidgeRegression(
      const Formula&               formula,
      const dstruct::DataFrame&    data,
      KernelRidgeRegressionOptions options = {})
      : KernelRidgeRegression(data, formula.text(), std::move(options)) {}

  /// @brief Fitted Formula text, borrowed.
  /// @return Fitted Formula text, borrowed.
  [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }

  /// @brief Expanded numeric design names, borrowed.
  /// @return Expanded numeric design names, borrowed.
  [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }

  /// @brief Complete retained fitting row count.
  /// @return Complete retained fitting row count.
  [[nodiscard]] std::size_t observations() const { return observations_; }

  /// @brief Kernel family name, borrowed.
  /// @return Kernel family name, borrowed.
  [[nodiscard]] const std::string& kernel() const { return options_.kernel; }

  /// @brief Selected or supplied sum-loss penalty.
  /// @return Selected or supplied sum-loss penalty.
  [[nodiscard]] double lambda() const { return lambda_used_; }

  /// @brief Selected or supplied kernel scale; one for linear.
  /// @return Selected or supplied kernel scale; one for linear.
  [[nodiscard]] double gamma() const { return gamma_used_; }

  /// @brief Whether finite negative lambda requested a grid search.
  /// @return Whether finite negative lambda requested a grid search.
  [[nodiscard]] bool lambda_was_selected() const { return lambda_was_selected_; }

  /// @brief Whether a nonlinear gamma grid was requested.
  /// @return Whether a nonlinear gamma grid was requested.
  [[nodiscard]] bool gamma_was_selected() const { return gamma_was_selected_; }

  /// @brief Owned-fit dual coefficients on the resolved row order, borrowed.
  /// @return Owned-fit dual coefficients on the resolved row order, borrowed.
  [[nodiscard]] const std::vector<double>& dual_coefficients() const { return alpha_; }

  /// @brief Trace of full smoother, including one for a free intercept.
  /// @return Trace of full smoother, including one for a free intercept.
  [[nodiscard]] double effective_degrees_of_freedom() const { return effective_dof_; }

  /// @brief Lambda search grid at selected gamma, empty for fixed lambda.
  /// @return Lambda search grid at selected gamma, empty for fixed lambda.
  [[nodiscard]] const std::vector<double>& lambda_grid() const { return lambda_grid_; }

  /// @brief Fixed-metric LOO loss at selected gamma, aligned with lambda_grid().
  /// @return Fixed-metric LOO loss at selected gamma, aligned with lambda_grid().
  [[nodiscard]] const std::vector<double>& cv_mean_squared_error() const { return cv_mse_; }

  /// @brief Fixed-metric LOO predictions, in retained-row order, borrowed.
  /// @return Fixed-metric LOO predictions, in retained-row order, borrowed.
  [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }

  /// @brief Resubstitution predictions from the full fit, borrowed.
  /// @return Resubstitution predictions from the full fit, borrowed.
  [[nodiscard]] const std::vector<double>& training_predictions() const {
    return training_predictions_;
  }

  /// @brief Original frame row indices retained jointly, borrowed.
  /// @return Original frame row indices retained jointly, borrowed.
  [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_rows_; }

  /// @brief Fitted design offsets, zero when standardize=false, borrowed.
  /// @return Fitted design offsets, zero when standardize=false, borrowed.
  [[nodiscard]] const std::vector<double>& feature_mean() const { return feature_mean_; }

  /// @brief Fitted population-SD divisors, one for constant/disabled scaling, borrowed.
  /// @return Fitted population-SD divisors, one for constant/disabled scaling, borrowed.
  [[nodiscard]] const std::vector<double>& feature_scale() const { return feature_scale_; }

  /// @brief Fitted standardized design rows, borrowed.
  /// @return Fitted standardized design rows, borrowed.
  [[nodiscard]] const linalg::DenseMatrix<double>& training_design() const {
    return training_X_std_;
  }

  /// @brief Complete retained responses, borrowed.
  /// @return Complete retained responses, borrowed.
  [[nodiscard]] const std::vector<double>& response() const { return training_y_; }

  /// @brief Raw selected-kernel training Gram matrix, borrowed.
  /// @return Raw selected-kernel training Gram matrix, borrowed.
  [[nodiscard]] const linalg::DenseMatrix<double>& gram_matrix() const { return gram_; }

  /// @brief Training Gram after optional double centering, borrowed.
  /// @return Training Gram after optional double centering, borrowed.
  [[nodiscard]] const linalg::DenseMatrix<double>& centered_gram_matrix() const {
    return centered_gram_;
  }

  /// @brief Raw-kernel representation intercept; zero if disabled.
  /// @return Raw-kernel representation intercept; zero if disabled.
  [[nodiscard]] double intercept() const { return intercept_; }

  /// @brief Training response mean, used as a descriptive holdout baseline.
  /// @return Training response mean, used as a descriptive holdout baseline.
  [[nodiscard]] double response_mean() const { return y_mean_; }

  /// @brief Fixed-metric LOO R squared; returns zero for a constant response.
  /// @return 1-SSE_LOO/SST about the fitting response mean; can be negative.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::print("{:.0f}\n", m.r_squared()); // -3
  /// @endcode
  /// @see KernelRidgeRegression
  double r_squared() const;

  /// @brief Fixed-metric LOO RMSE, rather than training error.
  /// @return Square root of mean squared LOO residual.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::print("{:.0f}\n", m.rmse()); // 2
  /// @endcode
  /// @see KernelRidgeRegression
  double rmse() const;

  /// @brief Predict with frozen design/scaling and correctly centered query kernels.
  /// @param newdata Matching fitted predictor schema; response ignored, null/nonfinite predictor
  /// rows yield NaN; empty returns empty.
  /// @return Owned predictions aligned with all input rows.
  /// @throws std::runtime_error Unseen fixed categorical levels or unrepresentable kernel;
  /// underlying schema errors propagate.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::print("{}\n", m.predict(d).size()); // 2
  /// @endcode
  /// @see KernelRidgeRegression
  std::vector<double> predict(const dstruct::DataFrame& newdata) const;

  /// @brief Linear response weights; can be signed and sum to one with a free intercept.
  /// @param newdata Same query contract as predict(); invalid rows become all NaN.
  /// @return Owned queries-by-training-rows matrix W such that predict=W*y; work O(m*n^2).
  /// @throws std::runtime_error As predict().
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// const auto w = m.prediction_weights(d);
  /// std::print("{:.2f} {:.2f}\n", w(0, 0), w(0, 1)); // 0.75 0.25
  /// @endcode
  /// @see KernelRidgeRegression
  linalg::DenseMatrix<double> prediction_weights(const dstruct::DataFrame& newdata) const;

  /// @brief Full training linear smoother, including any unpenalized intercept.
  /// @return Owned symmetric n-by-n S; O(n^3) work and O(n^2) output.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::print("{:.2f}\n", m.smoothing_matrix()(0, 0)); // 0.75
  /// @endcode
  /// @see KernelRidgeRegression
  linalg::DenseMatrix<double> smoothing_matrix() const;

  /// @brief Per-row predictions, dual coefficients, leverage and deletion residuals.
  /// @return
  /// row/response/training_prediction/loo_prediction/leverage/dual/training_residual/loo_residual.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::print("{}\n", m.training_diagnostics().nrows()); // 2
  /// @endcode
  /// @see KernelRidgeRegression
  dstruct::DataFrame training_diagnostics() const;

  /// @brief Selected Gram eigenvalues and their penalty shrinkage, excluding the free intercept.
  /// @return mode/eigenvalue/shrinkage/response_projection; numerical zero eigenvalues use zero
  /// shrinkage.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::print("{}\n", m.spectrum().nrows()); // 2
  /// @endcode
  /// @see KernelRidgeRegression
  dstruct::DataFrame spectrum() const;

  /// @brief Objective, numerical solve checks, rank and training/conditional-LOO errors.
  /// system_residual is the infinity norm of (Kc+lambda*I)*alpha-yc. At zero
  /// penalty with incompatible labels it need not vanish: the pseudoinverse
  /// fits only the numerical kernel range. The objective uses sum squared loss.
  /// @return One row:
  /// observations/lambda/gamma/intercept/rank/df/training_rmse/loo_rmse/rkhs_norm_squared/objective/system_residual/alpha_sum.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::print("{:.1f}\n", m.diagnostics().double_at("df", 0)); // 1.5
  /// @endcode
  /// @see KernelRidgeRegression
  dstruct::DataFrame diagnostics() const;

  /// @brief Every internal fixed-metric LOO candidate, preserving caller grid order.
  /// @return candidate/gamma/lambda/conditional_loo_mse/df/selected; first strict minimum wins.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::print("{}\n", m.selection_table().nrows()); // 1
  /// @endcode
  /// @see KernelRidgeRegression
  dstruct::DataFrame selection_table() const;

  /// @brief Evaluate a labelled frame without refitting, using the fitting mean baseline.
  /// @param data Nonempty labelled query frame; all outcomes and predictions must be finite.
  /// @return One-row rows/rmse/r_squared/baseline_rmse; evaluation R squared uses evaluation mean.
  /// @throws std::invalid_argument Missing/nonfinite labels/predictions or no rows; query schema
  /// errors propagate.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::print("{:.1f}\n", m.score(d).double_at("rmse", 0)); // 0.5
  /// @endcode
  /// @see KernelRidgeRegression
  dstruct::DataFrame score(const dstruct::DataFrame& data) const;

  /// @brief Human-readable fitted options and fixed-metric LOO diagnostics.
  /// @return Owned text; does not mutate the fit.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::print("{}\n", !m.summary().empty()); // true
  /// @endcode
  /// @see KernelRidgeRegression
  std::string summary() const;

  /// @brief Write summary text to a stream; modifies its formatting.
  /// @param os Borrowed output stream.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// std::ostringstream out;
  /// m.print_summary(out);
  /// std::print("{}\n", !out.str().empty()); // true
  /// @endcode
  /// @see KernelRidgeRegression
  void print_summary(std::ostream& os) const;

  /// @brief Print the same summary to standard output.
  /// @par Example
  /// See print_summary(std::ostream&); this overload uses std::cout.
  /// @see summary()
  void print_summary() const;

  /// @brief Plot a plain numeric one-predictor fit and supplied observed rows.
  /// @param data Frame holding the raw named predictor and response; nonfinite/null scatter rows
  /// omitted.
  /// @param grid_resolution At least two curve locations, default 200.
  /// @return Owned editable curve/scatter plot; use save_svg() for local bytes.
  /// @throws std::invalid_argument Expanded/multiple/categorical predictor, missing columns or
  /// grid_resolution<2.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// m.plot_fit(d).save_svg("krr_fit.svg");
  /// @endcode
  /// @see KernelRidgeRegression
  graphics::RPlot plot_fit(const dstruct::DataFrame& data, std::size_t grid_resolution = 200) const;

  /// @brief Plot retained responses against fixed-metric LOO predictions.
  /// @return Owned scatter plot, not resubstitution.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// m.plot_predicted_vs_actual().save_svg("krr_loo.svg");
  /// @endcode
  /// @see KernelRidgeRegression
  graphics::RPlot plot_predicted_vs_actual() const;

  /// @brief Plot fixed-metric LOO residuals against the LOO predictions.
  /// @return Owned scatter plot.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// m.plot_residuals_vs_fitted().save_svg("krr_residuals.svg");
  /// @endcode
  /// @see KernelRidgeRegression
  graphics::RPlot plot_residuals_vs_fitted() const;

  /// @brief Plot the selected-gamma conditional-LOO loss against log10 lambda.
  /// @return Owned curve with chosen lambda marked; zero candidate displayed one log decade below
  /// positives.
  /// @throws std::invalid_argument Lambda was fixed.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;
  /// d.add_column("x", std::vector<double>{0, 1});
  /// d.add_column("y", std::vector<double>{0, 2});
  /// dm::stats::KernelRidgeRegressionOptions o;
  /// o.lambda = .5;
  /// o.gamma = std::log(2.);
  /// o.standardize = false;
  /// const dm::stats::KernelRidgeRegression m(d, "y ~ x", o);
  /// dm::stats::KernelRidgeRegressionOptions a = o;
  /// a.lambda = -1;
  /// a.lambda_candidates = {.1, .5, 2};
  /// const dm::stats::KernelRidgeRegression selected(d, "y ~ x", a);
  /// selected.plot_cv_curve().save_svg("krr_cv.svg");
  /// @endcode
  /// @see KernelRidgeRegression
  graphics::RPlot plot_cv_curve() const;

 private:
  /// @brief Resolve/standardize fitting rows and run the finite kernel/penalty search.
  /// @param data Borrowed fitting frame.
  /// @throws std::invalid_argument Invalid options/design/finite values.
  /// @throws std::runtime_error No eligible candidate or numerical failure.
  void fit(const dstruct::DataFrame& data);
  /// @brief Evaluate the configured PSD kernel in standardized design coordinates.
  /// @param a First coordinate vector.
  /// @param b Equal-width second coordinate vector.
  /// @param gamma Positive kernel scale, ignored by linear.
  /// @return Finite kernel value.
  /// @throws std::runtime_error Kernel arithmetic overflows.
  double kernel_scalar(const std::vector<double>& a, const std::vector<double>& b, double gamma)
      const;
  /// @brief Evaluate and optionally center kernels against every fitting row.
  /// @param query_std Standardized query coordinates.
  /// @return Owned kernel row on the fitted centered/raw convention.
  /// @throws std::runtime_error Unrepresentable kernel value.
  std::vector<double> kernel_vector(const std::vector<double>& query_std) const;
  /// @brief Evaluate query rows on the frozen Formula design.
  /// @param newdata Matching frame, possibly empty or with null/nonfinite rows.
  /// @return Owned m-by-p standardized design, with NaN rows when invalid.
  linalg::DenseMatrix<double>  query_design(const dstruct::DataFrame& newdata) const;
  Formula                      formula_;         ///< Frozen formula text.
  DesignInfo                   design_;          ///< Fitted predictor levels and expanded schema.
  KernelRidgeRegressionOptions options_;         ///< Validated fitting controls.
  std::vector<std::string>     predictor_names_; ///< Expanded predictor labels.
  std::vector<double>          feature_mean_;    ///< Training design offsets.
  std::vector<double>          feature_scale_;   ///< Training population-SD divisors.
  std::vector<double>          feature_min_;     ///< Retained raw design minima.
  std::vector<double>          feature_max_;     ///< Retained raw design maxima.
  linalg::DenseMatrix<double>  training_X_std_;  ///< Frozen standardized fitting design.
  std::vector<double>          training_y_;      ///< Retained responses.
  std::vector<std::size_t>     kept_rows_;       ///< Original complete-row indices.
  double                       y_mean_{0};       ///< Full training response mean.
  std::vector<double>         alpha_; ///< Selected dual solution, with zero sum for free intercept.
  double                      lambda_used_{0};   ///< Chosen penalty.
  double                      gamma_used_{1};    ///< Chosen kernel scale.
  double                      effective_dof_{0}; ///< Full smoother trace, including free intercept.
  bool                        lambda_was_selected_{false}; ///< Internal lambda search requested.
  bool                        gamma_was_selected_{false};  ///< Internal gamma search requested.
  std::vector<double>         lambda_grid_;                ///< Lambda candidates at selected gamma.
  std::vector<double>         cv_mse_;                     ///< Associated conditional-LOO losses.
  std::vector<double>         fitted_;                     ///< Fixed-metric deletion predictions.
  std::vector<double>         training_predictions_;       ///< Full-fit resubstitution predictions.
  std::vector<double>         leverage_;                   ///< Full smoother diagonal.
  std::size_t                 observations_{0};            ///< Complete row count.
  linalg::DenseMatrix<double> gram_;                       ///< Selected raw kernel Gram.
  linalg::DenseMatrix<double> centered_gram_;              ///< Optional doubly centered Gram.
  linalg::DenseMatrix<double> vectors_;      ///< Center-projected orthogonal kernel eigenvectors.
  std::vector<double>         eigenvalues_;  ///< Nonnegative numerical kernel spectrum.
  std::vector<double>         projections_;  ///< Response projections on selected eigenvectors.
  std::vector<double>         kernel_means_; ///< Raw Gram column means (zero without intercept).
  double             kernel_grand_mean_{0};  ///< Raw Gram grand mean (zero without intercept).
  double             intercept_{0};          ///< Equivalent raw-kernel intercept.
  std::size_t        rank_{0};           ///< Relative-threshold kernel rank, excluding intercept.
  double             rank_tolerance_{0}; ///< Relative numerical spectral cutoff.
  dstruct::DataFrame candidates_;        ///< Every internal grid evaluation.
};
/// @brief Complete-case, fold-refitted validation for explicitly fixed KRR candidates.
/// Each fold fits its own Formula schema, predictor offsets/scales, response offset
/// and kernel centering. No internal auto selection is allowed. Candidate/fold order
/// is retained; pooled validation SSE divided by all retained rows determines the
/// first strict minimum. Scores tune the candidates and do not estimate selected-model
/// performance without another holdout. No shared mutation; dense repeated fits.
/// Labelled input rows incomplete under the full Formula are omitted jointly.
/// The formula has a numeric response and supported predictors.
/// Candidates use fixed lambda>=0 and positive nonlinear gamma.
/// Fold IDs align with original rows; labels on omitted rows are ignored. Nonnegative
/// contiguous 0..F-1 IDs, at least two folds; every fold needs at least two fitting rows.
/// Results name the candidate to refit on all supplied fitting rows.
/// Invalid folds/options or missing finite validation rows cause std::invalid_argument.
/// Formula, level and numerical failures within a fold propagate to the caller.
/// @par Example
/// @code
/// const auto d = dm::datasets::iris();
/// dm::stats::KernelRidgeRegressionOptions o;
/// o.lambda = .1;
/// o.gamma = 1;
/// std::vector<std::size_t> folds(d.nrows());
/// for (std::size_t i = 0; i < folds.size(); ++i)
///   folds[i] = i % 3;
/// const auto r =
///     dm::stats::kernel_ridge_cross_validate(d, "Petal.Length ~ Petal.Width", {o}, folds);
/// std::print("{}\n", r.best_candidate); // 0
/// @endcode
/// @see KernelRidgeRegression
struct KernelRidgeCVResult {
  std::size_t best_candidate{0}; ///< First candidate attaining minimum pooled validation MSE.
  dstruct::DataFrame candidates; ///< candidate/kernel/lambda/gamma/mse/rmse/selected.
  dstruct::DataFrame
      predictions; ///< candidate/fold/original row/observed/prediction for validation rows.
  dstruct::DataFrame
      folds; ///< candidate/fold/training_rows/validation_rows/mse and fitted feature means/scales.
};
/// @brief Fit explicit candidate options with fold-local preprocessing; see KernelRidgeCVResult.
/// @param data Labelled original rows.
/// @param formula Numeric response/predictor formula.
/// @param candidates Explicit fixed candidate options.
/// @param fold_ids Original-row fold IDs, contiguous on retained rows.
/// @return Owned fold/candidate tables and best index.
/// @throws std::invalid_argument Bad candidates/folds or nonfinite validation values.
/// @throws std::runtime_error Formula or fitting failure within a fold.
/// @par Example
/// @code
/// const auto d = dm::datasets::iris();
/// dm::stats::KernelRidgeRegressionOptions o;
/// o.lambda = .1;
/// o.gamma = 1;
/// std::vector<std::size_t> folds(d.nrows());
/// for (std::size_t i = 0; i < folds.size(); ++i)
///   folds[i] = i % 3;
/// const auto r =
///     dm::stats::kernel_ridge_cross_validate(d, "Petal.Length ~ Petal.Width", {o}, folds);
/// std::print("{}\n", r.best_candidate); // 0
/// @endcode
/// @see KernelRidgeCVResult
KernelRidgeCVResult kernel_ridge_cross_validate(
    const dstruct::DataFrame&                        data,
    const std::string&                               formula,
    const std::vector<KernelRidgeRegressionOptions>& candidates,
    const std::vector<std::size_t>&                  fold_ids);
/// @brief Reproducible noisy sine observations for checking KRR against a known generating mean.
/// Draws uniform x on [0,1], then normal noise with the given SD, using a local mt19937_64.
/// Standard distribution implementations can differ across C++ standard libraries.
/// @param n At least two rows; default 120.
/// @param seed Local generator seed, default 20240815.
/// @param noise_sd Finite positive noise SD, default .15.
/// @return Owned x/y/truth table; truth=sin(2*pi*x), not an estimated mean.
/// @throws std::invalid_argument Too few rows or invalid noise SD.
/// @par Example
/// @code
/// std::print("{}\n", dm::stats::kernel_ridge_wave().nrows()); // 120
/// @endcode
/// @see KernelRidgeRegression
dstruct::DataFrame
kernel_ridge_wave(std::size_t n = 120, std::uint64_t seed = 20240815, double noise_sd = .15);
} // namespace datamunge::stats
