#pragma once

/// @file lm.hpp
/// @brief R-`lm()`-style linear models: ordinary and weighted least squares fit from a
///        DataFrame and a formula, with inference, ANOVA, prediction intervals, influence
///        diagnostics and R's four diagnostic plots.
///
/// The model is \f$y = X\beta + \varepsilon\f$ with \f$\varepsilon\sim N(0,\sigma^2 W^{-1})\f$,
/// where the formula builds the design matrix \f$X\f$ (factor columns become treatment-coded
/// dummies) and \f$W\f$ is the identity unless a weights column is given. The estimate is
/// \f$\hat\beta=(X^\top WX)^{-1}X^\top Wy\f$; every other quantity (standard errors, t and F
/// tests, intervals, leverage, Cook's distance) follows R's `summary.lm`, `anova.lm`,
/// `predict.lm` and `influence.measures`, so results can be checked against R directly.
///
/// @par References
/// - J. M. Chambers (1992). "Linear models". Chapter 4 of *Statistical Models in S*,
///   J. M. Chambers and T. J. Hastie (eds.), Wadsworth & Brooks/Cole.
/// - R `lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lm.html

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Fitting options for LM, like the extra arguments of R's `lm()`.
struct LmOptions {
    /// Name of a numeric column of strictly positive case weights; set it for weighted least
    /// squares (R's `weights =`). Unset (the default) fits ordinary least squares.
    std::optional<std::string> weights_column;
};

/// @brief Which interval LM::predict() adds around each fitted value, like R's
///        `predict.lm(interval = ...)`.
enum class PredictionInterval {
    None,       ///< Point predictions only.
    Confidence, ///< Interval for the mean response \f$x_0^\top\beta\f$.
    Prediction  ///< Interval for a new observation \f$x_0^\top\beta+\varepsilon\f$ (wider).
};

/// @brief Predictions from LM::predict(), one entry per row of the new data.
///
/// Rows with a null predictor get NaN in every vector.
struct LmPrediction {
    std::vector<double> fit;    ///< Predicted mean \f$x_0^\top\hat\beta\f$.
    std::vector<double> se_fit; ///< Standard error used for the interval (empty for PredictionInterval::None).
    std::vector<double> lower;  ///< Lower interval bound (empty for PredictionInterval::None).
    std::vector<double> upper;  ///< Upper interval bound (empty for PredictionInterval::None).
};

/// @brief One row of a sequential (type I) ANOVA table from LM::anova().
///
/// The final row is "Residuals", whose f_value and p_value are 0 (R prints them as blank).
struct AnovaRow {
    std::string term;                  ///< Term label, or "Residuals".
    std::size_t degrees_of_freedom{0}; ///< Columns the term adds (residual df on the last row).
    double      sum_sq{0.0};           ///< Reduction in residual sum of squares when the term is added.
    double      mean_sq{0.0};          ///< sum_sq / degrees_of_freedom.
    double      f_value{0.0};          ///< mean_sq / \f$\hat\sigma^2\f$ of the full model.
    double      p_value{0.0};          ///< Upper tail of the F distribution at f_value.
};

/// @brief Result of the nested-model F-test `anova(reduced, full)`, like the two-row table of
///        R's `anova(fit0, fit1)`.
struct NestedAnova {
    std::size_t reduced_df{0};  ///< Residual degrees of freedom of the smaller model.
    std::size_t full_df{0};     ///< Residual degrees of freedom of the larger model.
    double      reduced_rss{0}; ///< Residual sum of squares of the smaller model.
    double      full_rss{0};    ///< Residual sum of squares of the larger model.
    std::size_t df{0};          ///< reduced_df - full_df: the number of constraints tested.
    double      sum_sq{0};      ///< reduced_rss - full_rss.
    double      f_value{0};     ///< (sum_sq / df) / (full_rss / full_df).
    double      p_value{0};     ///< Upper tail of \f$F_{df,\,full\_df}\f$ at f_value.
};

/// @brief An R-`lm()`-style linear model fit by ordinary or weighted least squares from a
///        DataFrame and a formula.
///
/// The constructor resolves the formula against the data (numeric columns enter as they are,
/// string columns as treatment-coded factors with the first level sorted alphabetically as the
/// baseline), drops every row with a null value in a column the formula uses (R's `na.omit`),
/// fits, and computes every summary and diagnostic once. All accessors are then cheap and
/// the object is immutable, so it can be shared between threads.
///
/// Per-observation vectors (fitted values, residuals, leverage, ...) have one entry per
/// *fitted* row, in data order, skipping dropped rows.
///
/// @par Example
/// @code
/// const auto model = dm::stats::LM(dm::formula("len ~ dose + supp"), dm::datasets::tooth_growth());
/// std::print("{}\n", model.coefficient_names());  // ["(Intercept)", "dose", "suppVC"]
/// std::print("{:.4f} {:.4f}\n", model.coefficients()[1], model.r_squared());  // 9.7636 0.7038
/// @endcode
///
/// @par References
/// - J. M. Chambers (1992). "Linear models". Chapter 4 of *Statistical Models in S*,
///   Wadsworth & Brooks/Cole.
/// - R `lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lm.html
///
/// @see Formula, GLM, LMM, anova(const LM&, const LM&)
class LM {
 public:
    /// @brief Fits `formula` to `data`, like R's `lm(formula, data, weights)`.
    ///
    /// @param data  Data frame holding every column the formula names (and the weights column).
    /// @param formula  Model formula such as `"y ~ x1 + x2 + group"`; `- 1` removes the intercept.
    /// @param options  Fitting options; set LmOptions::weights_column for weighted least squares.
    /// @throws std::invalid_argument  If the formula names a missing column, the weights column
    ///         is missing, null in a fitted row, or not strictly positive.
    /// @throws std::runtime_error  If the design matrix is rank deficient.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const dm::stats::LM model(data, "y ~ x");
    /// std::print("{:.2f} {:.2f}\n", model.coefficients()[0], model.coefficients()[1]);  // 0.60 0.80
    /// @endcode
    ///
    /// @par References
    /// - R `lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lm.html
    ///
    /// @see LM(const Formula&, const dstruct::DataFrame&, LmOptions)
    LM(const dstruct::DataFrame& data, const std::string& formula, LmOptions options = {});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `LM("y ~ x", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + x2")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument  As the `(data, formula, ...)` constructor.
    /// @throws std::runtime_error  As the `(data, formula, ...)` constructor.
    ///
    /// @par Example
    /// See LM.
    LM(const Formula& formula, const dstruct::DataFrame& data, LmOptions options = {})
        : LM(data, formula.text(), std::move(options)) {}

    /// @brief The formula as written.
    /// @return The formula text.
    [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
    /// @brief Whether the model has an intercept column.
    /// @return False only when the formula removed it with `- 1` or `+ 0`.
    [[nodiscard]] bool has_intercept() const { return design_.has_intercept; }
    /// @brief Number of fitted rows \f$n\f$ (after dropping rows with nulls).
    /// @return \f$n\f$.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief Number of coefficients \f$p\f$ (columns of the design matrix).
    /// @return \f$p\f$.
    [[nodiscard]] std::size_t rank() const { return coefficients_.size(); }
    /// @brief Residual degrees of freedom \f$n-p\f$.
    /// @return \f$n-p\f$.
    [[nodiscard]] std::size_t degrees_of_freedom() const { return degrees_of_freedom_; }

    /// @brief Data-frame row (0-based) of each fitted row, so per-observation vectors can be
    ///        matched back to the data; rows with nulls are absent.
    /// @return One row index per fitted row, increasing.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("body_mass_g ~ flipper_length_mm"), dm::datasets::penguins());
    /// std::print("{} {}\n", model.observations(), model.fitted_rows()[3]);  // 342 4
    /// @endcode
    [[nodiscard]] const std::vector<std::size_t>& fitted_rows() const { return fitted_rows_; }

    /// @brief Estimated coefficients \f$\hat\beta\f$, in the order of coefficient_names().
    /// @return One estimate per design column.
    [[nodiscard]] const std::vector<double>& coefficients() const { return coefficients_; }
    /// @brief Coefficient names as R prints them: `"(Intercept)"`, numeric column names, and
    ///        `column` + `level` for each non-baseline factor level (e.g. `"speciesGentoo"`).
    /// @return One name per coefficient.
    [[nodiscard]] const std::vector<std::string>& coefficient_names() const { return design_.coefficient_names; }
    /// @brief Fitted values \f$\hat y = X\hat\beta\f$, one per fitted row.
    /// @return The fitted values.
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    /// @brief Raw residuals \f$y-\hat y\f$, one per fitted row (unweighted even for WLS, as in R).
    /// @return The residuals.
    [[nodiscard]] const std::vector<double>& residuals() const { return residuals_; }
    /// @brief Standard errors \f$\hat\sigma\sqrt{[(X^\top WX)^{-1}]_{jj}}\f$ of the coefficients.
    /// @return One standard error per coefficient.
    [[nodiscard]] const std::vector<double>& standard_errors() const { return standard_errors_; }
    /// @brief t statistics \f$\hat\beta_j/\mathrm{SE}_j\f$ for \f$H_0:\beta_j=0\f$.
    /// @return One t value per coefficient.
    [[nodiscard]] const std::vector<double>& t_values() const { return t_values_; }
    /// @brief Two-sided p-values of the t statistics on \f$n-p\f$ degrees of freedom.
    /// @return One p-value per coefficient.
    [[nodiscard]] const std::vector<double>& p_values() const { return p_values_; }
    /// @brief Estimated covariance matrix \f$\hat\sigma^2(X^\top WX)^{-1}\f$ of \f$\hat\beta\f$
    ///        (R's `vcov()`).
    /// @return The \f$p\times p\f$ covariance matrix.
    [[nodiscard]] const linalg::DenseMatrix<double>& covariance() const { return covariance_; }

    /// @brief Coefficient of determination \f$R^2=1-\mathrm{RSS}/\mathrm{TSS}\f$ (weighted for
    ///        WLS; TSS about 0 for a model without intercept, as in R).
    /// @return \f$R^2\f$.
    [[nodiscard]] double r_squared() const { return r_squared_; }
    /// @brief Adjusted \f$R^2 = 1-(1-R^2)(n-1)/(n-p)\f$.
    /// @return Adjusted \f$R^2\f$.
    [[nodiscard]] double adjusted_r_squared() const { return adjusted_r_squared_; }
    /// @brief Residual standard error \f$\hat\sigma=\sqrt{\mathrm{RSS}/(n-p)}\f$.
    /// @return \f$\hat\sigma\f$.
    [[nodiscard]] double sigma() const { return sigma_; }
    /// @brief Overall F statistic for "every coefficient except the intercept is zero".
    /// @return The F statistic, or NaN when the model has no slope or no residual df.
    [[nodiscard]] double f_statistic() const { return f_statistic_; }
    /// @brief p-value of f_statistic().
    /// @return The p-value, or NaN as f_statistic().
    [[nodiscard]] double f_p_value() const { return f_p_value_; }

    /// @brief Residual sum of squares \f$\sum_i w_i(y_i-\hat y_i)^2\f$ (R's `deviance()`).
    ///
    /// @return The (weighted) RSS.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// std::print("{:.1f}\n", dm::stats::LM(data, "y ~ x").rss());  // 3.6
    /// @endcode
    ///
    /// @par References
    /// - R `deviance()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/deviance.html
    ///
    /// @see sigma(), log_likelihood()
    [[nodiscard]] double rss() const { return sigma2_ * static_cast<double>(degrees_of_freedom_); }

    /// @brief Maximised Gaussian log-likelihood, as R's `logLik.lm`.
    ///
    /// \f$\ell=\tfrac12\big(\sum_i\log w_i-n(\log 2\pi+1-\log n+\log\mathrm{RSS})\big)\f$, using
    /// the maximum-likelihood variance \f$\mathrm{RSS}/n\f$ (not \f$\hat\sigma^2\f$).
    ///
    /// @return \f$\ell\f$.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const dm::stats::LM model(data, "y ~ x");
    /// std::print("{:.4f} {:.4f} {:.4f}\n", model.log_likelihood(), model.aic(), model.bic());
    /// // -6.2734 18.5469 17.3752
    /// @endcode
    ///
    /// @par References
    /// - R `logLik()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/logLik.html
    ///
    /// @see aic(), bic()
    [[nodiscard]] double log_likelihood() const;

    /// @brief Akaike information criterion \f$-2\ell+2k\f$ with \f$k=p+1\f$ parameters
    ///        (the coefficients and \f$\sigma\f$), as R's `AIC()`.
    ///
    /// Lower is better; only differences between models fit to the same rows are meaningful.
    ///
    /// @return The AIC.
    ///
    /// @par Example
    /// See log_likelihood().
    ///
    /// @par References
    /// - H. Akaike (1974). "A new look at the statistical model identification". IEEE
    ///   Transactions on Automatic Control 19(6), 716-723. doi:10.1109/TAC.1974.1100705
    ///
    /// @see bic(), log_likelihood()
    [[nodiscard]] double aic() const;

    /// @brief Bayesian (Schwarz) information criterion \f$-2\ell+k\log n\f$ with \f$k=p+1\f$,
    ///        as R's `BIC()`.
    ///
    /// @return The BIC.
    ///
    /// @par Example
    /// See log_likelihood().
    ///
    /// @par References
    /// - G. Schwarz (1978). "Estimating the dimension of a model". The Annals of Statistics
    ///   6(2), 461-464. doi:10.1214/aos/1176344136
    ///
    /// @see aic(), log_likelihood()
    [[nodiscard]] double bic() const;

    /// @brief A two-sided confidence interval for one coefficient.
    struct Interval {
        double lower; ///< Lower bound.
        double upper; ///< Upper bound.
    };

    /// @brief Confidence intervals \f$\hat\beta_j\pm t_{n-p,\,1-\alpha/2}\,\mathrm{SE}_j\f$ for
    ///        every coefficient, like R's `confint()`.
    ///
    /// @param level  Confidence level in (0, 1) (default 0.95).
    /// @return One interval per coefficient.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const auto ci = dm::stats::LM(data, "y ~ x").confidence_intervals()[1];
    /// std::print("{:.4f} {:.4f}\n", ci.lower, ci.upper);  // -0.3024 1.9024
    /// @endcode
    ///
    /// @par References
    /// - R `confint()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/confint.html
    ///
    /// @see coefficient_frame()
    [[nodiscard]] std::vector<Interval> confidence_intervals(double level = 0.95) const;

    /// @brief The coefficient table of R's `summary(fit)` plus `confint(fit)`, as a DataFrame.
    ///
    /// Columns: `term` (string), `estimate`, `std_error`, `t_value`, `p_value`, `lower`,
    /// `upper` (numeric), one row per coefficient. Handy for `to_md()`.
    ///
    /// @param level  Confidence level of the `lower`/`upper` columns (default 0.95).
    /// @return The table.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const auto table = dm::stats::LM(data, "y ~ x").coefficient_frame();
    /// std::print("{} {}\n", table.nrows(), table.columns());
    /// // 2 ["term", "estimate", "std_error", "t_value", "p_value", "lower", "upper"]
    /// @endcode
    ///
    /// @par References
    /// - R `summary.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.lm.html
    ///
    /// @see confidence_intervals(), summary()
    [[nodiscard]] dstruct::DataFrame coefficient_frame(double level = 0.95) const;

    /// @brief Leverages \f$h_{ii}\f$: the diagonal of the hat matrix
    ///        \f$H=W^{1/2}X(X^\top WX)^{-1}X^\top W^{1/2}\f$ (R's `hatvalues()`).
    ///
    /// Each lies in [0, 1) and they sum to \f$p\f$.
    /// @return One leverage per fitted row.
    [[nodiscard]] const std::vector<double>& leverage() const { return leverage_; }
    /// @brief Standardized residuals \f$\sqrt{w_i}\,e_i/(\hat\sigma\sqrt{1-h_{ii}})\f$
    ///        (R's `rstandard()`).
    /// @return One standardized residual per fitted row.
    [[nodiscard]] const std::vector<double>& standardized_residuals() const { return standardized_residuals_; }
    /// @brief Externally studentized residuals, each scaled by \f$\hat\sigma_{(i)}\f$ from the
    ///        fit without row \f$i\f$ (R's `rstudent()`); \f$t_{n-p-1}\f$ under the model.
    /// @return One studentized residual per fitted row.
    [[nodiscard]] const std::vector<double>& studentized_residuals() const { return studentized_residuals_; }
    /// @brief Cook's distances \f$D_i=\frac{r_i^2}{p}\frac{h_{ii}}{1-h_{ii}}\f$ for standardized
    ///        residual \f$r_i\f$ (R's `cooks.distance()`).
    /// @return One distance per fitted row.
    [[nodiscard]] const std::vector<double>& cooks_distance() const { return cooks_distance_; }

    /// @brief Sequential (type I) analysis of variance, like R's `anova(fit)`.
    ///
    /// Terms are added in formula order; each row's sum of squares is the drop in RSS when its
    /// columns join the model, tested against the full model's \f$\hat\sigma^2\f$. The rows'
    /// sums of squares add up to the total (about the mean when there is an intercept). The
    /// last row is "Residuals". Refits one prefix model per term.
    ///
    /// @return One row per formula term, then the "Residuals" row; empty for a model with no terms.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose + supp"), dm::datasets::tooth_growth());
    /// for (const auto& row : model.anova())
    ///     std::print("{} {} {:.2f}\n", row.term, row.degrees_of_freedom, row.sum_sq);
    /// // dose 1 2224.30
    /// // supp 1 205.35
    /// // Residuals 57 1022.56
    /// @endcode
    ///
    /// @par References
    /// - R `anova.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/anova.lm.html
    ///
    /// @see anova_frame(), anova(const LM&, const LM&)
    [[nodiscard]] std::vector<AnovaRow> anova() const;

    /// @brief anova() as a DataFrame with columns `term`, `df`, `sum_sq`, `mean_sq`, `f_value`,
    ///        `p_value`; the Residuals row has null `f_value` and `p_value`, as R prints it.
    ///
    /// @return The table.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose + supp"), dm::datasets::tooth_growth());
    /// std::print("{}\n", model.anova_frame().nrows());  // 3
    /// @endcode
    ///
    /// @par References
    /// - R `anova.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/anova.lm.html
    ///
    /// @see anova()
    [[nodiscard]] dstruct::DataFrame anova_frame() const;

    /// @brief R's `print(summary(fit))` layout: call, residual quantiles, coefficient table with
    ///        significance stars, residual standard error, \f$R^2\f$ and the overall F test.
    ///
    /// @return The summary text.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose"), dm::datasets::tooth_growth());
    /// std::print("{}", model.summary());  // Call: lm(formula = len ~ dose) ...
    /// @endcode
    ///
    /// @par References
    /// - R `summary.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.lm.html
    ///
    /// @see print_summary(), coefficient_frame()
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to @p os.
    /// @param os  Output stream.
    void print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    void print_summary() const;

    /// @brief Point predictions \f$x_0^\top\hat\beta\f$ for each row of @p newdata, like R's
    ///        `predict(fit, newdata)`.
    ///
    /// @p newdata needs every predictor column (not the response); factor columns must use
    /// levels seen in fitting. Rows with a null predictor predict NaN.
    ///
    /// @param newdata  Rows to predict.
    /// @return One prediction per row of @p newdata.
    /// @throws std::invalid_argument  If a column is missing or a factor level is new.
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const DF grid{{"x", DF::numeric_column_type(dm::c(0.0, 6.0))}};
    /// std::print("{::.1f}\n", dm::stats::LM(data, "y ~ x").predict(grid));  // [0.6, 5.4]
    /// @endcode
    ///
    /// @par References
    /// - R `predict.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/predict.lm.html
    ///
    /// @see predict(const dstruct::DataFrame&, PredictionInterval, double), predict_frame()
    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    /// @brief Predictions with a confidence or prediction interval, like R's
    ///        `predict(fit, newdata, interval = ..., level = ...)`.
    ///
    /// With \f$q=x_0^\top(X^\top WX)^{-1}x_0\f$ the confidence interval is
    /// \f$\hat y_0\pm t\,\hat\sigma\sqrt q\f$ and the prediction interval (for a new observation
    /// of unit weight) is \f$\hat y_0\pm t\,\hat\sigma\sqrt{1+q}\f$, with
    /// \f$t=t_{n-p,\,1-\alpha/2}\f$. `se_fit` holds the half-width divided by \f$t\f$ (note R's
    /// `se.fit` is always \f$\hat\sigma\sqrt q\f$).
    ///
    /// @param newdata  Rows to predict.
    /// @param interval  Which interval to add.
    /// @param level  Coverage level in (0, 1) (default 0.95).
    /// @return Fit and, unless @p interval is None, se_fit, lower and upper.
    /// @throws std::invalid_argument  As predict(const dstruct::DataFrame&).
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const DF at3{{"x", DF::numeric_column_type(dm::c(3.0))}};
    /// const auto pi = dm::stats::LM(data, "y ~ x").predict(at3, dm::stats::PredictionInterval::Prediction);
    /// std::print("{:.4f} {:.4f}\n", pi.lower[0], pi.upper[0]);  // -0.8189 6.8189
    /// @endcode
    ///
    /// @par References
    /// - R `predict.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/predict.lm.html
    ///
    /// @see predict_frame()
    [[nodiscard]] LmPrediction predict(const dstruct::DataFrame& newdata, PredictionInterval interval,
                                       double level = 0.95) const;

    /// @brief predict() as a DataFrame with a `fit` column, plus `se_fit`, `lwr` and `upr` when
    ///        an interval is requested; NaN predictions become nulls.
    ///
    /// @param newdata  Rows to predict.
    /// @param interval  Which interval to add (default None).
    /// @param level  Coverage level (default 0.95).
    /// @return One row per row of @p newdata.
    /// @throws std::invalid_argument  As predict(const dstruct::DataFrame&).
    ///
    /// @par Example
    /// @code
    /// using DF = dm::dstruct::DataFrame;
    /// const DF data{{"x", DF::numeric_column_type(dm::c(1.0, 2.0, 3.0, 4.0, 5.0))},
    ///               {"y", DF::numeric_column_type(dm::c(1.0, 3.0, 2.0, 5.0, 4.0))}};
    /// const DF at3{{"x", DF::numeric_column_type(dm::c(3.0))}};
    /// const auto frame = dm::stats::LM(data, "y ~ x").predict_frame(at3, dm::stats::PredictionInterval::Confidence);
    /// std::print("{}\n", frame.columns());  // ["fit", "se_fit", "lwr", "upr"]
    /// @endcode
    ///
    /// @see predict(const dstruct::DataFrame&, PredictionInterval, double)
    [[nodiscard]] dstruct::DataFrame predict_frame(const dstruct::DataFrame& newdata,
                                                   PredictionInterval        interval = PredictionInterval::None,
                                                   double                    level    = 0.95) const;

    /// @brief R's `plot(fit, which = 1)`: residuals against fitted values, with a zero line and
    ///        a lowess smoother; the three largest residuals are labelled with their 1-based data row
    ///        number, as R does.
    ///
    /// A curved smoother suggests a missing nonlinear term; a funnel suggests unequal variance.
    /// @return The plot.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose"), dm::datasets::tooth_growth());
    /// model.plot_residuals_vs_fitted().save("resid_fitted.svg");
    /// @endcode
    ///
    /// @par References
    /// - R `plot.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/plot.lm.html
    ///
    /// @see plot_diagnostics(), lowess()
    [[nodiscard]] graphics::RPlot plot_residuals_vs_fitted() const;

    /// @brief R's `plot(fit, which = 2)`: normal Q-Q plot of the standardized residuals with
    ///        the line through their quartiles.
    ///
    /// Theoretical quantiles use R's `ppoints()` positions. Points bending away at both ends
    /// mean heavier (or lighter) tails than normal.
    /// @return The plot.
    ///
    /// @par Example
    /// See plot_residuals_vs_fitted().
    ///
    /// @par References
    /// - R `plot.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/plot.lm.html
    ///
    /// @see plot_diagnostics()
    [[nodiscard]] graphics::RPlot plot_normal_qq() const;

    /// @brief R's `plot(fit, which = 3)`: \f$\sqrt{|r_i|}\f$ of the standardized residuals
    ///        against fitted values, with a lowess smoother.
    ///
    /// A rising smoother means the spread grows with the mean (heteroscedasticity).
    /// @return The plot.
    ///
    /// @par Example
    /// See plot_residuals_vs_fitted().
    ///
    /// @par References
    /// - R `plot.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/plot.lm.html
    ///
    /// @see plot_diagnostics()
    [[nodiscard]] graphics::RPlot plot_scale_location() const;

    /// @brief R's `plot(fit, which = 5)`: standardized residuals against leverage, with dashed
    ///        Cook's distance contours at 0.5 and 1.
    ///
    /// Points beyond a contour have that much influence on \f$\hat\beta\f$. The contours
    /// \f$r=\pm\sqrt{Dp(1-h)/h}\f$ are drawn only inside the plot area, and the three points
    /// with the largest Cook's distance are labelled with their 1-based data row number.
    /// @return The plot.
    ///
    /// @par Example
    /// See plot_residuals_vs_fitted().
    ///
    /// @par References
    /// - R. D. Cook (1977). "Detection of influential observation in linear regression".
    ///   Technometrics 19(1), 15-18. doi:10.1080/00401706.1977.10489493
    /// - R `plot.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/plot.lm.html
    ///
    /// @see plot_diagnostics(), cooks_distance()
    [[nodiscard]] graphics::RPlot plot_residuals_vs_leverage() const;

    /// @brief R's default `par(mfrow = c(2, 2)); plot(fit)`: the four diagnostic plots in a 2x2
    ///        layout, 1100 x 860 pixels, panels fitted to their cells.
    ///
    /// @return The layout; save it with `.save("diagnostics.svg")`.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose + supp"), dm::datasets::tooth_growth());
    /// const auto fig = model.plot_diagnostics();
    /// std::print("{}\n", fig.panel_count());  // 4
    /// @endcode
    ///
    /// @par References
    /// - R `plot.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/plot.lm.html
    ///
    /// @see save_diagnostic_plots()
    [[nodiscard]] graphics::RLayout plot_diagnostics() const;

    /// @brief Saves the four diagnostic plots as separate SVG files
    ///        `<path_prefix>_residuals_vs_fitted.svg`, `_normal_qq.svg`, `_scale_location.svg`
    ///        and `_residuals_vs_leverage.svg`.
    /// @param path_prefix  Path and file-name stem.
    /// @throws std::runtime_error  If a file cannot be written.
    ///
    /// @par Example
    /// @code
    /// const auto model = dm::stats::LM(dm::formula("len ~ dose"), dm::datasets::tooth_growth());
    /// model.save_diagnostic_plots("tooth");  // tooth_normal_qq.svg, ...
    /// @endcode
    ///
    /// @see plot_diagnostics()
    void save_diagnostic_plots(const std::string& path_prefix) const;

 private:
    /// @brief Resolves the formula, builds the (weighted) design, fits and computes every summary.
    /// @param data  The data frame given to the constructor.
    /// @param options  The options given to the constructor.
    void fit(const dstruct::DataFrame& data, LmOptions options);
    /// @brief Fills leverage_, standardized_residuals_, studentized_residuals_ and cooks_distance_.
    void compute_diagnostics();
    /// @brief Throws if @p newdata has a factor level not seen in fitting.
    /// @param newdata  Rows about to be predicted.
    void validate_categorical_levels(const dstruct::DataFrame& newdata) const;

    Formula    formula_; ///< The parsed formula.
    DesignInfo design_;  ///< Terms, coefficient names and factor levels resolved from the data.
    LmOptions  options_; ///< Options given to the constructor.

    linalg::DenseMatrix<double> design_matrix_; ///< Weighted design \f$W^{1/2}X\f$ used to fit.
    std::vector<double>         response_;      ///< Weighted response \f$W^{1/2}y\f$ used to fit.

    std::vector<double>         coefficients_;    ///< \f$\hat\beta\f$.
    std::vector<double>         fitted_;          ///< \f$X\hat\beta\f$ (unweighted).
    std::vector<double>         residuals_;       ///< \f$y-X\hat\beta\f$ (unweighted).
    std::vector<double>         standard_errors_; ///< Coefficient standard errors.
    std::vector<double>         t_values_;        ///< Coefficient t statistics.
    std::vector<double>         p_values_;        ///< Coefficient p-values.
    linalg::DenseMatrix<double> covariance_;      ///< \f$\hat\sigma^2(X^\top WX)^{-1}\f$.
    linalg::DenseMatrix<double> xtx_inverse_;     ///< \f$(X^\top WX)^{-1}\f$, for leverage and prediction.

    double      r_squared_{0.0};          ///< \f$R^2\f$.
    double      adjusted_r_squared_{0.0}; ///< Adjusted \f$R^2\f$.
    double      sigma_{0.0};              ///< \f$\hat\sigma\f$.
    double      sigma2_{0.0};             ///< \f$\hat\sigma^2=\mathrm{RSS}/(n-p)\f$.
    double      f_statistic_{0.0};        ///< Overall F statistic (NaN if undefined).
    double      f_p_value_{0.0};          ///< p-value of the overall F statistic.
    std::size_t observations_{0};         ///< Fitted rows \f$n\f$.
    std::size_t degrees_of_freedom_{0};   ///< \f$n-p\f$.

    std::vector<double> leverage_;               ///< Hat-matrix diagonal.
    std::vector<double> standardized_residuals_; ///< Internally studentized residuals.
    std::vector<double> studentized_residuals_;  ///< Externally studentized residuals.
    std::vector<double> cooks_distance_;         ///< Cook's distances.
    std::vector<double> weights_;                ///< Case weights (empty for OLS).
    std::vector<std::size_t> fitted_rows_;       ///< Data row of each fitted row.
};

/// @brief F-test of a smaller model nested in a larger one, like R's `anova(reduced, full)`.
///
/// Tests whether the columns the larger model adds are all zero:
/// \f$F=\frac{(\mathrm{RSS}_0-\mathrm{RSS}_1)/(d_0-d_1)}{\mathrm{RSS}_1/d_1}\f$ on
/// \f$(d_0-d_1,\,d_1)\f$ degrees of freedom, with \f$d\f$ the residual degrees of freedom.
/// Nesting is the caller's responsibility (R does not check it either); only the row counts
/// and the degrees of freedom are checked.
///
/// @param reduced  The smaller model.
/// @param full  The larger model, fit to the same rows.
/// @return Both models' residual df and RSS, and the F test.
/// @throws std::invalid_argument  If the models were fit to different numbers of rows, or
///         @p full does not have fewer residual degrees of freedom than @p reduced.
///
/// @par Example
/// @code
/// const auto data = dm::datasets::tooth_growth();
/// const auto small = dm::stats::LM(dm::formula("len ~ dose"), data);
/// const auto large = dm::stats::LM(dm::formula("len ~ dose + supp"), data);
/// const auto test = dm::stats::anova(small, large);
/// std::print("{} {:.3f} {:.5f}\n", test.df, test.f_value, test.p_value);  // 1 11.447 0.00130
/// @endcode
///
/// @par References
/// - R `anova.lm()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/anova.lm.html
///
/// @see LM::anova()
[[nodiscard]] NestedAnova anova(const LM& reduced, const LM& full);

} // namespace datamunge::stats
