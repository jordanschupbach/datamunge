#pragma once

/// @file lmm.hpp
/// @brief Linear mixed models with one grouping factor, fit by REML or maximum likelihood
///        (lme4's `lmer` / nlme's `lme` for a single `(terms | group)` term).

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Options for LMM: the criterion and the settings of the search over \f$\theta\f$.
struct LMMOptions {
    /// @brief REML (the default, as lme4 and nlme) or maximum likelihood (`false`). Use ML to
    ///        compare models with different fixed effects by likelihood ratio.
    bool reml{true};
    /// @brief Population size of the differential-evolution search over \f$\theta\f$.
    std::size_t de_population_size{40};
    /// @brief Generations of the differential-evolution search; Nelder-Mead then polishes the
    ///        best point, so this only needs to find the right basin.
    std::size_t de_max_generations{300};
    /// @brief Box for the differential-evolution search: diagonal entries of \f$\Lambda\f$ in
    ///        \f$[0,b]\f$, off-diagonal in \f$[-b,b]\f$, relative to the residual standard deviation.
    ///        The polish is unconstrained, so an optimum outside the box is still reached if the
    ///        search lands near it.
    double theta_bound{5.0};
    /// @brief Seed of the differential-evolution search.
    std::uint64_t seed{42};
};

/// @brief A linear mixed model \f$y=X\beta+Zb+\varepsilon\f$ with one grouping factor, fit by
///        REML or ML from a DataFrame and an lme4-style formula.
///
/// The formula is a fixed-effects formula plus one random-effects term, e.g.
/// `"score ~ hours + (1 + hours | school)"`. The fixed part supports everything Formula does
/// (factors, interactions, `I(...)`, `poly(...)`). The random part is an intercept and/or plain
/// numeric columns against one grouping column: `(1 | g)`, `(x | g)` (intercept and slope),
/// `(0 + x | g)` (slope only). Each group \f$g\f$ gets a random vector \f$b_g\sim N(0,\Sigma)\f$,
/// independent across groups and of the errors \f$\varepsilon\sim N(0,\sigma^2I)\f$. Rows with a
/// null in any column used are dropped.
///
/// With \f$\Sigma=\sigma^2\Lambda\Lambda^\top\f$ (\f$\Lambda\f$ lower triangular, its entries are
/// \f$\theta\f$), the marginal covariance of group \f$g\f$'s responses is \f$\sigma^2M_g\f$ with
/// \f$M_g=I+Z_g\Lambda\Lambda^\top Z_g^\top\f$. For fixed \f$\theta\f$, \f$\beta\f$ is the generalised least
/// squares estimate and \f$\sigma^2\f$ has a closed form, so the (RE)ML criterion is profiled down
/// to a function of \f$\theta\f$ alone (deviance_at()). It is minimised by differential evolution
/// followed by a Nelder-Mead polish. Fixed effects, standard errors, variance components,
/// log-likelihood and BLUPs agree with R's `nlme::lme` and `lme4::lmer` (standard errors use
/// \f$\sigma^2(X^\top V^{-1}X)^{-1}\f$; tests on them use the normal distribution, so they are
/// z tests, like lme4's t values read against \f$N(0,1)\f$).
///
/// @par Example
/// @code
/// const auto growth = dm::datasets::orthodont();
/// const dm::stats::LMM fit(growth, "distance ~ age + (1 | subject)");
/// std::print("{:.4f} {:.4f}\n", fit.random_effect_std_devs()[0], fit.residual_std_dev());  // 2.1147 1.4316
/// @endcode
///
/// @par References
/// - D. Bates, M. Maechler, B. Bolker and S. Walker (2015). "Fitting linear mixed-effects
///   models using lme4". Journal of Statistical Software 67(1). doi:10.18637/jss.v067.i01
/// - J. C. Pinheiro and D. M. Bates (2000). *Mixed-Effects Models in S and S-PLUS*. Springer.
/// - G. K. Robinson (1991). "That BLUP is a good thing: the estimation of random effects".
///   Statistical Science 6(1), 15-32.
///
/// @see GLMM, INLAMixedModel, LM
class LMM {
public:
    /// @brief Fits the model.
    /// @param data  Data frame holding every column the formula names.
    /// @param formula  Fixed effects plus one `(terms | group)` term.
    /// @param options  Criterion (REML or ML) and search settings.
    /// @throws std::invalid_argument  If the formula has no or several random-effects terms, a
    ///         column is missing or not numeric where required, the grouping column has fewer
    ///         than 2 levels, or too few complete rows remain.
    /// @throws std::runtime_error  If the fixed-effects design is rank deficient.
    LMM(const dstruct::DataFrame& data, const std::string& formula, LMMOptions options = {});

    /// @brief Fits the same model from a formula object, in R's `(formula, data)` argument order.
    ///
    /// Same as the `(data, formula, ...)` constructor; a string literal or std::string
    /// converts to Formula implicitly, so `LMM("y ~ x + (1 | g)", data)` also works.
    /// @param formula  Model formula, e.g. `dm::formula("y ~ x1 + (1 | g)")`.
    /// @param data  Data frame holding every column the formula names.
    /// @param options  Same as in the `(data, formula, ...)` constructor.
    /// @throws std::invalid_argument  As the `(data, formula, ...)` constructor.
    LMM(const Formula& formula, const dstruct::DataFrame& data, LMMOptions options = {})
        : LMM(data, formula.text(), std::move(options)) {}

    /// @brief The formula as given.
    /// @return The formula text.
    [[nodiscard]] const std::string& formula_text() const { return formula_text_; }
    /// @brief Name of the grouping column.
    /// @return The column after `|`.
    [[nodiscard]] const std::string& group_variable() const { return group_variable_; }
    /// @brief Whether each group has a random intercept.
    /// @return `false` for `(0 + x | g)`.
    [[nodiscard]] bool has_random_intercept() const { return random_intercept_; }
    /// @brief Names of the random effects, in the order of random_effects() entries:
    ///        `"(Intercept)"` first if present, then the slope columns.
    /// @return \f$q\f$ names.
    [[nodiscard]] const std::vector<std::string>& random_effect_names() const { return random_effect_names_; }
    /// @brief Whether the model was fit by REML.
    /// @return LMMOptions::reml.
    [[nodiscard]] bool is_reml() const { return options_.reml; }
    /// @brief Number of rows used in the fit.
    /// @return \f$n\f$.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief Number of groups.
    /// @return The number of distinct grouping values among the used rows.
    [[nodiscard]] std::size_t num_groups() const { return group_labels_.size(); }
    /// @brief Number of fixed effects.
    /// @return \f$p\f$, the columns of \f$X\f$.
    [[nodiscard]] std::size_t rank() const { return coefficients_.size(); }

    /// @brief Fixed-effect estimates \f$\hat\beta=(X^\top V^{-1}X)^{-1}X^\top V^{-1}y\f$.
    /// @return \f$p\f$ coefficients, in coefficient_names() order.
    [[nodiscard]] const std::vector<double>& coefficients() const { return coefficients_; }
    /// @brief Names of the fixed effects, as Formula codes them.
    /// @return \f$p\f$ names.
    [[nodiscard]] const std::vector<std::string>& coefficient_names() const { return design_.coefficient_names; }
    /// @brief Standard errors: square roots of the diagonal of \f$\hat\sigma^2(X^\top M^{-1}X)^{-1}\f$.
    /// @return \f$p\f$ standard errors.
    [[nodiscard]] const std::vector<double>& standard_errors() const { return standard_errors_; }
    /// @brief Coefficients divided by their standard errors.
    /// @return \f$p\f$ z statistics.
    [[nodiscard]] const std::vector<double>& z_values() const { return z_values_; }
    /// @brief Two-sided p-values of the z statistics against \f$N(0,1)\f$.
    /// @return \f$p\f$ p-values.
    [[nodiscard]] const std::vector<double>& p_values() const { return p_values_; }
    /// @brief Conditional fitted values \f$x_i^\top\hat\beta+z_i^\top\hat b_{g(i)}\f$.
    /// @return One value per used row.
    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    /// @brief Conditional residuals, response minus fitted_values().
    /// @return One value per used row.
    [[nodiscard]] const std::vector<double>& residuals() const { return residuals_; }
    /// @brief Fixed effects as a table: `term`, `estimate`, `std_error`, `z_value`, `p_value`.
    /// @return One row per fixed effect.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::LMM fit(dm::datasets::orthodont(), "distance ~ age + (1 | subject)");
    /// std::print("{}", fit.coefficient_frame().to_md());
    /// @endcode
    [[nodiscard]] dstruct::DataFrame coefficient_frame() const;

    /// @brief Estimated residual variance \f$\hat\sigma^2\f$.
    /// @return \f$\hat\sigma^2\f$.
    [[nodiscard]] double residual_variance() const { return sigma2_; }
    /// @brief Estimated residual standard deviation \f$\hat\sigma\f$.
    /// @return \f$\hat\sigma\f$.
    [[nodiscard]] double residual_std_dev() const { return sigma_; }
    /// @brief The \f$q\times q\f$ random-effect covariance \f$\hat\Sigma=\hat\sigma^2\Lambda\Lambda^\top\f$.
    /// @return \f$\hat\Sigma\f$, rows and columns in random_effect_names() order.
    [[nodiscard]] const linalg::DenseMatrix<double>& random_effect_covariance() const { return random_effect_covariance_; }
    /// @brief Standard deviations of the random effects, \f$\sqrt{\hat\Sigma_{kk}}\f$.
    /// @return \f$q\f$ standard deviations.
    [[nodiscard]] std::vector<double> random_effect_std_devs() const;
    /// @brief Correlation between random effects @p i and @p j.
    /// @param i  First random effect.
    /// @param j  Second random effect.
    /// @return \f$\hat\Sigma_{ij}/\sqrt{\hat\Sigma_{ii}\hat\Sigma_{jj}}\f$, or 0 if either variance is 0.
    [[nodiscard]] double random_effect_correlation(std::size_t i, std::size_t j) const;
    /// @brief The fitted \f$\theta\f$: the lower-triangular entries of \f$\Lambda\f$, row by row
    ///        (\f$\Lambda_{00},\Lambda_{10},\Lambda_{11},\dots\f$), with a non-negative diagonal.
    /// @return \f$q(q+1)/2\f$ values. For a random intercept, \f$\theta=\hat\tau/\hat\sigma\f$.
    [[nodiscard]] const std::vector<double>& theta() const { return theta_; }

    /// @brief Grouping values, in order of first appearance in the data.
    /// @return One label per group (numbers are formatted as text).
    [[nodiscard]] const std::vector<std::string>& group_labels() const { return group_labels_; }
    /// @brief BLUPs \f$\hat b_g=\Lambda\Lambda^\top Z_g^\top M_g^{-1}(y_g-X_g\hat\beta)\f$, the
    ///        conditional means of the random effects given the data.
    /// @return One vector of \f$q\f$ effects per group, in group_labels() order.
    [[nodiscard]] const std::vector<std::vector<double>>& random_effects() const { return random_effects_; }
    /// @brief BLUPs as a table: the grouping column, `n` (rows in the group) and one column per
    ///        random effect, named as random_effect_names().
    /// @return One row per group.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::LMM fit(dm::datasets::orthodont(), "distance ~ age + (1 | subject)");
    /// std::print("{}", fit.random_effects_frame().head(3).to_md());
    /// @endcode
    [[nodiscard]] dstruct::DataFrame random_effects_frame() const;

    /// @brief Maximised log-likelihood (ML) or restricted log-likelihood (REML).
    /// @return \f$-\tfrac12\f$ deviance().
    [[nodiscard]] double log_likelihood() const { return -0.5 * deviance_; }
    /// @brief The minimised criterion, \f$-2\f$ times the (restricted) log-likelihood.
    /// @return deviance_at(theta()).
    [[nodiscard]] double deviance() const { return deviance_; }
    /// @brief The profiled criterion at any \f$\theta\f$, with \f$\beta\f$ and \f$\sigma^2\f$ at their
    ///        optimal values for that \f$\theta\f$.
    ///
    /// ML: \f$n\log(2\pi\hat\sigma^2_\theta)+n+\sum_g\log|M_g|\f$ with \f$\hat\sigma^2_\theta=r^\top M^{-1}r/n\f$.
    /// REML: \f$(n-p)\log(2\pi\hat\sigma^2_\theta)+(n-p)+\sum_g\log|M_g|+\log|X^\top M^{-1}X|\f$ with
    /// \f$\hat\sigma^2_\theta=r^\top M^{-1}r/(n-p)\f$. Evaluating it on a grid draws the likelihood
    /// profile of the variance components.
    /// @param theta  \f$q(q+1)/2\f$ entries of \f$\Lambda\f$, as theta().
    /// @return The criterion (smaller is better).
    /// @throws std::invalid_argument  If @p theta has the wrong length.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::LMM fit(dm::datasets::orthodont(), "distance ~ age + (1 | subject)");
    /// std::print("{:.3f} {:.3f}\n", fit.deviance_at(fit.theta()), fit.deviance_at({0.0}));  // 447.003 509.169
    /// @endcode
    [[nodiscard]] double deviance_at(const std::vector<double>& theta) const;
    /// @brief Akaike information criterion, deviance() \f$+2k\f$ with \f$k=p+q(q+1)/2+1\f$.
    /// @return The AIC (REML-based for REML fits, as in lme4).
    [[nodiscard]] double aic() const;
    /// @brief Bayesian information criterion, deviance() \f$+k\log n\f$.
    /// @return The BIC.
    [[nodiscard]] double bic() const;

    /// @brief lme4-style summary: criterion, random-effect variances and correlations, residual
    ///        variance, and the fixed-effects table.
    /// @return The summary text.
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to @p os.
    /// @param os  Output stream.
    void print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    void print_summary() const;

    /// @brief Response-scale predictions for new rows.
    ///
    /// Rows whose group was seen while fitting get \f$x^\top\hat\beta+z^\top\hat b_g\f$; rows with an
    /// unseen or null group get the population-level prediction \f$x^\top\hat\beta\f$.
    /// @param newdata  Rows with the fixed-effect columns (and, optionally, the group column).
    /// @return One prediction per row of @p newdata (NaN where a fixed-effect column is null).
    /// @throws std::invalid_argument  If a factor in @p newdata has a level unseen in fitting.
    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

private:
    /// @brief Builds the design, finds \f$\theta\f$ and computes every fitted quantity.
    /// @param data  The data passed to the constructor.
    void fit(const dstruct::DataFrame& data);

    std::string              formula_text_;        ///< Formula as given.
    std::string              group_variable_;      ///< Grouping column.
    bool                     random_intercept_{true}; ///< Whether `(1 | g)` is present.
    std::vector<std::string> random_slope_vars_;   ///< Random-slope columns.
    std::vector<std::string> random_effect_names_; ///< "(Intercept)" + slope names, in Z-column order.

    Formula    fixed_formula_; ///< Fixed-effects part of the formula.
    DesignInfo design_;        ///< Its resolved design.
    LMMOptions options_;       ///< Options used.

    linalg::DenseMatrix<double> design_matrix_; ///< \f$X\f$, rows aligned with response_.
    std::vector<double>         response_;      ///< \f$y\f$.

    std::vector<std::string>              group_labels_;         ///< Distinct groups, first-seen order.
    std::vector<std::size_t>              group_index_;          ///< Group of each row.
    std::vector<std::vector<std::size_t>> group_rows_;           ///< Rows of each group.
    linalg::DenseMatrix<double>           random_effect_design_; ///< \f$Z\f$ (n x q), rows aligned with \f$X\f$.

    std::vector<double>         theta_;  ///< Lower-triangular entries of \f$\Lambda\f$.
    linalg::DenseMatrix<double> lambda_; ///< \f$\Lambda\f$ (q x q).

    std::vector<double>         coefficients_;            ///< \f$\hat\beta\f$.
    std::vector<double>         fitted_;                  ///< Conditional fitted values.
    std::vector<double>         residuals_;               ///< Conditional residuals.
    std::vector<double>         standard_errors_;         ///< SEs of \f$\hat\beta\f$.
    std::vector<double>         z_values_;                ///< z statistics.
    std::vector<double>         p_values_;                ///< Two-sided p-values.
    linalg::DenseMatrix<double> fixed_effect_covariance_; ///< \f$\hat\sigma^2(X^\top M^{-1}X)^{-1}\f$.

    linalg::DenseMatrix<double>      random_effect_covariance_; ///< \f$\hat\Sigma\f$.
    std::vector<std::vector<double>> random_effects_;           ///< BLUPs, one vector per group.

    double      sigma2_{0.0};       ///< \f$\hat\sigma^2\f$.
    double      sigma_{0.0};        ///< \f$\hat\sigma\f$.
    double      deviance_{0.0};     ///< Minimised criterion.
    std::size_t observations_{0};   ///< Rows used.
};

} // namespace datamunge::stats
