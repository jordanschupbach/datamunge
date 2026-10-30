#pragma once

/// @file quantile_regression.hpp
/// @brief Linear quantile regression (Koenker and Bassett, 1978): check-loss fitting by ADMM or by
///        an exact simplex, the linear-programming dual, standard errors and the quantile process.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief The algorithm QuantileRegression uses to minimise the check loss.
enum class QuantileRegressionMethod {
    /// Exact vertex-to-vertex descent on the linear program (in the spirit of Barrodale and
    /// Roberts), started from the rows nearest the least-squares fit: the fit interpolates exactly
    /// p rows and is certified optimal. The default: exact, and usually a handful of pivots.
    Simplex,
    /// Alternating direction method of multipliers: one least-squares solve and one asymmetric
    /// soft threshold per iteration. Accurate only to its tolerances and far slower here; kept for
    /// comparison and as a template for penalised variants.
    ADMM,
};

/// @brief How QuantileRegression::coefficient_table() estimates the standard errors.
enum class QuantileStandardErrors {
    /// Errors independent of the predictors: \f$\tau(1-\tau)s^2(X^\top WX)^{-1}\f$ with the sparsity
    /// \f$s=1/f(F^{-1}(\tau))\f$ estimated by a difference quotient of residual quantiles.
    IID,
    /// Hendricks and Koenker's sandwich, which lets the error density vary with the predictors:
    /// each row's density is estimated from the refits at \f$\tau\pm h\f$.
    NID,
    /// Pairs bootstrap: the standard deviation of the coefficients refitted on resampled rows.
    Bootstrap,
};

/// @brief Fitting options for QuantileRegression.
///
/// Designated initialisers keep calls short, e.g. `{.quantile = 0.9}`. The ADMM fields (rho,
/// tolerances, adaptive_rho) are ignored by the default simplex.
struct QuantileRegressionOptions {
    /// The quantile level \f$\tau\f$, strictly between 0 and 1; 0.5 is median regression.
    double quantile{0.5};
    /// The ADMM penalty \f$\varrho>0\f$. It changes the path, not the answer; match it to the scale of
    /// the response (the soft-threshold widths are \f$w_i\tau/\varrho\f$ and \f$w_i(1-\tau)/\varrho\f$).
    double rho{1.0};
    /// Iteration cap for either method; converged() is false when it is reached.
    std::size_t max_iterations{10000};
    /// ADMM absolute tolerance on the primal and dual residual norms.
    double absolute_tolerance{1e-6};
    /// ADMM relative tolerance on the primal and dual residual norms.
    double relative_tolerance{1e-5};
    /// If true, ADMM doubles or halves \f$\varrho\f$ every 10 iterations to balance the two residuals.
    bool adaptive_rho{false};
    /// Optional column of positive case weights \f$w_i\f$; unweighted when empty.
    std::optional<std::string> weights_column;
    /// The solver; the exact simplex by default.
    QuantileRegressionMethod method{QuantileRegressionMethod::Simplex};
};

/// @brief Coefficients with standard errors, t statistics, p-values and confidence intervals,
///        from QuantileRegression::coefficient_table().
///
/// The t statistics are referred to Student's t with \f$n-p\f$ degrees of freedom, as R's
/// `quantreg::summary.rq` does; the interval is estimate \f$\pm t_{n-p,(1+\text{level})/2}\f$ times
/// the standard error.
struct QuantileCoefficientTable {
    /// Coefficient names, "(Intercept)" first when present.
    std::vector<std::string> names;
    /// The point estimates \f$\hat\beta_\tau\f$.
    std::vector<double> estimate;
    /// Estimated standard errors.
    std::vector<double> std_error;
    /// estimate / std_error.
    std::vector<double> t_value;
    /// Two-sided p-values for \f$\beta_j=0\f$.
    std::vector<double> p_value;
    /// Lower confidence limits.
    std::vector<double> lower;
    /// Upper confidence limits.
    std::vector<double> upper;
    /// The method that produced the standard errors.
    QuantileStandardErrors method{QuantileStandardErrors::NID};
    /// The confidence level of lower and upper.
    double level{0.95};
    /// The bandwidth \f$h\f$ for IID and NID (Hall and Sheather's, shrunk if \f$\tau\pm h\f$ leaves (0, 1)); 0 for the bootstrap.
    double bandwidth{0.0};
    /// Bootstrap replications used; 0 for IID and NID.
    std::size_t replications{0};
    /// Residual degrees of freedom \f$n-p\f$.
    std::size_t degrees_of_freedom{0};
};

/// @brief Coefficients of one formula fitted over a grid of quantile levels, from quantile_process().
///
/// Row k of each matrix-like member belongs to quantiles[k]; column j to names[j].
struct QuantileProcess {
    /// The quantile levels, in the order given.
    std::vector<double> quantiles;
    /// Coefficient names.
    std::vector<std::string> names;
    /// coefficients[k][j] is \f$\hat\beta_j(\tau_k)\f$.
    std::vector<std::vector<double>> coefficients;
    /// Standard errors, same shape.
    std::vector<std::vector<double>> std_error;
    /// Lower confidence limits, same shape.
    std::vector<std::vector<double>> lower;
    /// Upper confidence limits, same shape.
    std::vector<std::vector<double>> upper;
    /// The confidence level of the limits.
    double level{0.95};
};

/// @brief The check (pinball) loss \f$\rho_\tau(r)=r(\tau-\mathbf 1\{r<0\})\f$.
/// @param residual  The residual \f$r=y-\hat y\f$.
/// @param tau  The quantile level in (0, 1).
/// @return \f$\tau r\f$ for \f$r\ge0\f$ and \f$(\tau-1)r\f$ otherwise; never negative.
[[nodiscard]] double check_loss(double residual, double tau);

/// @brief Hall and Sheather's (1988) bandwidth for the sparsity difference quotient.
///
/// \f$h=n^{-1/3}z_{1-\alpha/2}^{2/3}\bigl(1.5\,\phi(z_\tau)^2/(2z_\tau^2+1)\bigr)^{1/3}\f$ with \f$z_\tau=\Phi^{-1}(\tau)\f$, the
/// default of R's `quantreg::bandwidth.rq`.
/// @param tau  Quantile level in (0, 1).
/// @param n  Sample size, positive.
/// @param alpha  The intended test size, in (0, 1).
/// @return The bandwidth h, in quantile-level units.
/// @throws std::invalid_argument  If an argument is out of range.
[[nodiscard]] double hall_sheather_bandwidth(double tau, std::size_t n, double alpha = 0.05);

/// @brief Bofinger's (1975) bandwidth \f$h=n^{-1/5}\bigl(4.5\,\phi(z_\tau)^4/(2z_\tau^2+1)^2\bigr)^{1/5}\f$, wider than
///        Hall and Sheather's; R's `bandwidth.rq(hs = FALSE)`.
/// @param tau  Quantile level in (0, 1).
/// @param n  Sample size, positive.
/// @return The bandwidth h.
/// @throws std::invalid_argument  If an argument is out of range.
[[nodiscard]] double bofinger_bandwidth(double tau, std::size_t n);

/// @brief Linear quantile regression: \f$\hat\beta_\tau=\arg\min_\beta\sum_iw_i\rho_\tau(y_i-x_i^\top\beta)\f$.
///
/// The conditional \f$\tau\f$-quantile of the response is modelled as linear in the design built
/// from an R-style formula. Formula parsing, null-row omission, categorical encoding and
/// prediction follow the same conventions as stats::LM. The fit is a linear program: the simplex
/// (default) solves it exactly, ADMM to a tolerance, and both expose the dual solution
/// (Koenker's regression rank scores less \f$1-\tau\f$). coefficient_table() adds standard errors.
///
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;
/// d.add_column("x", std::vector<double>{1, 2, 3, 4, 5});
/// d.add_column("y", std::vector<double>{1, 3, 2, 5, 4});
/// const dm::stats::QuantileRegression median(d, "y ~ x");
/// // median.coefficients() == {0.25, 0.75}: the line through the first and last rows,
/// // median.objective() == 1.75, median.basis() == {0, 4}
/// @endcode
///
/// @par References
/// - Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica 46, 33-50.
/// - Koenker, R. (2005). Quantile Regression. Cambridge University Press.
class QuantileRegression {
 public:
    /// @brief Fits the model.
    /// @param data  Data frame holding every column the formula names (and the weights column).
    /// @param formula  R-style formula, e.g. `"y ~ x1 + x2"`; `"y ~ 1"` gives the sample quantile.
    /// @param options  Quantile level, method, tolerances and weights.
    /// @throws std::invalid_argument  On an option out of range, an empty design, fewer rows than
    ///         coefficients, a missing weights column or a non-positive weight.
    /// @throws std::runtime_error  If the design matrix is rank-deficient.
    QuantileRegression(const dstruct::DataFrame& data, const std::string& formula,
                       QuantileRegressionOptions options = {});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `QuantileRegression("y ~ x", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument  As the `(data, formula, ...)` constructor.
    QuantileRegression(const Formula& formula, const dstruct::DataFrame& data, QuantileRegressionOptions options = {})
        : QuantileRegression(data, formula.text(), std::move(options)) {}

    /// @brief The formula as written.
    /// @return The formula text.
    [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
    /// @brief The quantile level \f$\tau\f$.
    /// @return options().quantile.
    [[nodiscard]] double quantile() const { return options_.quantile; }
    /// @brief The options the model was fitted with.
    /// @return The options.
    [[nodiscard]] const QuantileRegressionOptions& options() const { return options_; }
    /// @brief Whether the design has an intercept column.
    /// @return True unless the formula removed it.
    [[nodiscard]] bool has_intercept() const { return design_.has_intercept; }
    /// @brief The number of rows fitted (after dropping rows with nulls).
    /// @return n.
    [[nodiscard]] std::size_t observations() const { return observations_; }

    /// @brief The estimated coefficients \f$\hat\beta_\tau\f$.
    /// @return One value per coefficient_names() entry.
    [[nodiscard]] const std::vector<double>& coefficients() const { return coefficients_; }
    /// @brief The coefficient names.
    /// @return Names, "(Intercept)" first when present.
    [[nodiscard]] const std::vector<std::string>& coefficient_names() const {
        return design_.coefficient_names;
    }
    /// @brief The fitted conditional quantiles \f$x_i^\top\hat\beta_\tau\f$ of the fitted rows.
    /// @return One value per fitted row.
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    /// @brief The residuals \f$y_i-x_i^\top\hat\beta_\tau\f$.
    /// @return One value per fitted row.
    [[nodiscard]] const std::vector<double>& residuals() const { return residuals_; }
    /// @brief The rows of the source data frame that were fitted, in order.
    /// @return Row indices into the constructor's data frame.
    [[nodiscard]] const std::vector<std::size_t>& used_row_indices() const { return used_rows_; }

    /// @brief The minimised check loss \f$\sum_iw_i\rho_\tau(y_i-x_i^\top\hat\beta_\tau)\f$.
    /// @return The objective value.
    [[nodiscard]] double objective() const { return objective_; }
    /// @brief Koenker and Machado's \f$R^1(\tau)=1-V(\hat\beta_\tau)/V(\hat q_\tau)\f$ against the intercept-only fit.
    /// @return A value in [0, 1] for a model with an intercept.
    [[nodiscard]] double pseudo_r_squared() const { return pseudo_r_squared_; }
    /// @brief Iterations taken: ADMM iterations or simplex pivots.
    /// @return The count.
    [[nodiscard]] std::size_t iterations() const { return iterations_; }
    /// @brief Whether the solver met its stopping rule.
    ///
    /// ADMM: both residual norms are below their tolerances. Simplex: no edge from the final
    /// vertex lowers the loss and every dual value lies in \f$[\tau-1,\tau]\f$ (to 1e-9), which proves
    /// optimality.
    /// @return True on convergence.
    [[nodiscard]] bool converged() const { return converged_; }
    /// @brief ADMM's final primal residual norm \f$\|y-X\beta-r\|_2\f$; 0 for the simplex.
    /// @return The norm.
    [[nodiscard]] double primal_residual_norm() const { return primal_residual_norm_; }
    /// @brief ADMM's final dual residual norm \f$\varrho\|X^\top(r^k-r^{k-1})\|_2\f$; 0 for the simplex.
    /// @return The norm.
    [[nodiscard]] double dual_residual_norm() const { return dual_residual_norm_; }

    /// @brief The rows the fit interpolates (zero residual).
    ///
    /// Simplex: exactly the p rows of the final vertex, whose submatrix \f$X_h\f$ gives
    /// \f$\hat\beta=X_h^{-1}y_h\f$. ADMM: the rows whose split residual the soft threshold set to exactly
    /// zero, usually p of them once converged.
    /// @return Indices into the fitted rows (0 .. observations() - 1), ascending.
    [[nodiscard]] const std::vector<std::size_t>& basis() const { return basis_; }

    /// @brief The solution \f$\hat a\f$ of the dual linear program
    ///        \f$\max_a\sum_iw_ia_iy_i\f$ subject to \f$X^\top Wa=0\f$, \f$a_i\in[\tau-1,\tau]\f$.
    ///
    /// \f$a_i=\tau\f$ above the fit and \f$\tau-1\f$ below it; interpolated rows take values inside the
    /// interval. At the optimum \f$\sum_iw_ia_iy_i\f$ equals objective(), and \f$a+1-\tau\f$ are Koenker's
    /// regression rank scores. Simplex: solved exactly from the basis; ADMM: the scaled
    /// multiplier \f$\varrho u_i/w_i\f$.
    /// @return One value per fitted row.
    [[nodiscard]] const std::vector<double>& dual_solution() const { return dual_; }

    /// @brief Standard errors, t statistics, p-values and confidence intervals.
    ///
    /// IID needs no refit; NID refits at \f$\tau\pm h\f$ with Hall and Sheather's bandwidth and clamps
    /// each row's density estimate \f$2h/x_i^\top(\hat\beta_{\tau+h}-\hat\beta_{\tau-h})\f$ at zero, as quantreg does.
    /// The bootstrap refits on \f$n\f$ rows resampled with replacement (weights travel with
    /// their rows) and skips resamples whose design is rank-deficient. Refits use options().
    /// @param method  IID, NID (default) or Bootstrap.
    /// @param level  Confidence level in (0, 1).
    /// @param replications  Bootstrap resamples (ignored otherwise), at least 2.
    /// @param seed  Bootstrap seed.
    /// @return The table.
    /// @throws std::invalid_argument  On a level out of range, fewer than 2 replications, or
    ///         \f$n\le p\f$.
    [[nodiscard]] QuantileCoefficientTable coefficient_table(
        QuantileStandardErrors method = QuantileStandardErrors::NID, double level = 0.95,
        std::size_t replications = 200, std::uint64_t seed = 1) const;

    /// @brief Predicted conditional quantiles for new rows.
    /// @param newdata  Data frame with the predictor columns (the response is not needed).
    /// @return \f$x^\top\hat\beta_\tau\f$ for each row of newdata.
    /// @throws std::invalid_argument  If a categorical predictor has an unseen level.
    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    /// @brief An R-style text summary: call, coefficients, loss, pseudo-\f$R^2\f$ and solver diagnostics.
    /// @return The summary text.
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to a stream.
    /// @param os  The stream.
    void print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    void print_summary() const;

 private:
    /// @brief Resolves the design, then runs the chosen solver and fills every result member.
    /// @param data  The constructor's data frame.
    void fit(const dstruct::DataFrame& data);
    /// @brief Rejects unseen categorical levels in prediction data.
    /// @param newdata  The prediction data frame.
    void validate_categorical_levels(const dstruct::DataFrame& newdata) const;

    Formula                   formula_;      ///< The parsed formula.
    DesignInfo                design_;       ///< The resolved design (columns, names, levels).
    QuantileRegressionOptions options_;      ///< The fitting options.

    linalg::DenseMatrix<double> x_;          ///< The fitted design matrix (n x p).
    std::vector<double>         y_;          ///< The fitted responses.
    std::vector<std::size_t>    used_rows_;  ///< Source rows that were fitted.
    std::vector<double> coefficients_;       ///< The estimates.
    std::vector<double> fitted_;             ///< Fitted values.
    std::vector<double> residuals_;          ///< Residuals.
    std::vector<double> weights_;            ///< Case weights (all 1 when unweighted).
    std::vector<std::size_t> basis_;         ///< Interpolated rows.
    std::vector<double> dual_;               ///< Dual solution.

    double      objective_{0.0};             ///< Minimised check loss.
    double      pseudo_r_squared_{0.0};      ///< Koenker-Machado R1.
    std::size_t observations_{0};            ///< Rows fitted.
    std::size_t iterations_{0};              ///< Iterations or pivots.
    bool        converged_{false};           ///< Stopping rule met.
    double      primal_residual_norm_{0.0};  ///< ADMM primal residual.
    double      dual_residual_norm_{0.0};    ///< ADMM dual residual.
};

/// @brief Fits one formula at each quantile level and collects the coefficients with
///        confidence limits, the quantile process \f$\tau\mapsto\hat\beta(\tau)\f$.
/// @param data  The data frame.
/// @param formula  R-style formula.
/// @param quantiles  Quantile levels, each in (0, 1).
/// @param options  Options for every fit; its quantile field is overwritten.
/// @param method  Standard-error method for the limits.
/// @param level  Confidence level.
/// @return One row per quantile level.
/// @throws std::invalid_argument  If quantiles is empty, or as QuantileRegression and
///         coefficient_table().
[[nodiscard]] QuantileProcess quantile_process(const dstruct::DataFrame& data, const std::string& formula,
                                               const std::vector<double>& quantiles,
                                               QuantileRegressionOptions options = {},
                                               QuantileStandardErrors method = QuantileStandardErrors::NID,
                                               double level = 0.95);

} // namespace datamunge::stats
