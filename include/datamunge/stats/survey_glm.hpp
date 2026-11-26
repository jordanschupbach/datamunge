#pragma once

/// @file survey_glm.hpp
/// @brief Survey-weighted generalized linear models: pseudo-likelihood estimates with design-based sandwich standard errors.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/survey.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Family of a survey-weighted GLM, each with its canonical link.
///
/// Only the estimating equations matter, so binomial and Poisson give the quasi-binomial and
/// quasi-Poisson fits (proportions and non-integer counts are allowed).
/// @par Example
/// @code
/// const auto f = dm::stats::SurveyGlmFamily::binomial;
/// std::println("{}", f == dm::stats::SurveyGlmFamily::poisson); // false
/// @endcode
enum class SurveyGlmFamily {
    gaussian, ///< Identity link: weighted least squares.
    binomial, ///< Logit link; responses in [0, 1].
    poisson   ///< Log link; nonnegative responses.
};

/// @brief Options for fit_survey_glm().
/// @par Example
/// @code
/// const dm::stats::SurveyGlmOptions o{.family = dm::stats::SurveyGlmFamily::poisson};
/// std::println("{}", o.max_iterations); // 50
/// @endcode
struct SurveyGlmOptions {
    SurveyGlmFamily family{SurveyGlmFamily::gaussian}; ///< Family and canonical link.
    std::vector<bool> domain{};                         ///< Units in the analysis domain (empty: all); others keep their place in the design.
    std::size_t max_iterations{50};                     ///< IRLS iteration limit.
    double tolerance{1e-10};                            ///< Relative change in deviance for convergence.
};

/// @brief Survey-weighted GLM fit.
/// @par Example
/// @code
/// const dm::stats::SurveyGlmFit f;
/// std::println("{}", f.converged); // false
/// @endcode
struct SurveyGlmFit {
    std::vector<double> coefficients;           ///< Pseudo-maximum-likelihood estimates.
    std::vector<double> standard_errors;        ///< Design-based (linearization sandwich) standard errors.
    linalg::DenseMatrix<double> covariance;     ///< Sandwich covariance \f$A^{-1}BA^{-1}\f$.
    std::vector<double> t_values;               ///< Coefficient over standard error.
    std::vector<double> p_values;               ///< Two-sided t p-values on df_residual degrees of freedom.
    std::vector<double> lower;                  ///< Lower confidence limits (t quantile, design confidence).
    std::vector<double> upper;                  ///< Upper confidence limits.
    std::vector<double> model_standard_errors;  ///< Standard errors of the weighted fit treated as an iid sample with weights scaled to mean 1 (dispersion 1 for binomial and Poisson), as glm().
    std::vector<double> design_effects;         ///< Squared ratio of design-based to model-based standard errors.
    std::vector<double> fitted;                 ///< Fitted means for every unit (domain or not).
    double df_residual{};                       ///< Domain design degrees of freedom + 1 - number of coefficients, as svyglm.
    std::size_t iterations{};                   ///< IRLS iterations.
    bool converged{false};                      ///< True if the deviance converged.
};

/// @brief Fits a GLM to survey data by weighted pseudo-likelihood with design-based standard errors.
///
/// Solves the census estimating equations \f$\sum_iw_ix_i(y_i-\mu_i)=0\f$ (canonical links) by
/// iteratively reweighted least squares. The covariance is the sandwich \f$A^{-1}BA^{-1}\f$ with
/// \f$A=\sum_iw_iv(\mu_i)x_ix_i^\top\f$ and B the linearized design covariance of the score totals
/// (survey_score_covariance()), as `survey::svyglm`. Inference uses t distributions on
/// \f$\nu+1-p\f$ degrees of freedom, where \f$\nu\f$ is the design degrees of freedom. For a domain
/// fit, units outside the domain contribute zero scores but stay in the design, which gives the
/// correct domain variance (unlike subsetting the data).
/// @param design Sampling design (weights, strata, clusters, finite-population corrections, confidence).
/// @param x Model matrix (include a column of ones for an intercept), one row per unit.
/// @param y Responses.
/// @param options Family, domain and iteration control.
/// @return Coefficients, design-based and model-based standard errors, and t inference.
/// @throws std::invalid_argument For mismatched sizes, responses outside the family's support,
/// an empty domain, or fewer domain units than coefficients.
/// @throws std::runtime_error If the weighted information matrix is singular.
/// @par Example
/// @code
/// const dm::stats::SurveyDesign d{.weights = {1, 2, 1, 2, 1, 2}};
/// const dm::linalg::DenseMatrix<double> x(6, 2, {1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5});
/// const auto f = dm::stats::fit_survey_glm(d, x, {1, 3, 5, 7, 9, 11});
/// std::println("{:.3f} {:.3f}", f.coefficients[0], f.coefficients[1]); // 1.000 2.000
/// @endcode
/// @par References
/// - D. A. Binder (1983). "On the variances of asymptotically normal estimators from complex
///   surveys." International Statistical Review 51(3), 279-292. doi:10.2307/1402588
/// - T. Lumley (2010). Complex Surveys: A Guide to Analysis Using R. Wiley, chapter 5.
/// @see survey_glm_wald_test(), survey_score_covariance()
[[nodiscard]] SurveyGlmFit fit_survey_glm(const SurveyDesign& design, const linalg::DenseMatrix<double>& x,
                                          const std::vector<double>& y, const SurveyGlmOptions& options = {});

/// @brief Design-based Wald F test that several coefficients are zero, as `survey::regTermTest`.
/// @par Example
/// @code
/// const dm::stats::SurveyGlmWald w{.statistic = 2.5, .df1 = 2, .df2 = 30, .p_value = 0.1};
/// std::println("{}", w.df1); // 2
/// @endcode
struct SurveyGlmWald {
    double chi_square{}; ///< \f$\hat\beta_S^\top V_S^{-1}\hat\beta_S\f$.
    double statistic{};  ///< chi_square divided by the number of tested coefficients.
    double df1{};        ///< Number of tested coefficients.
    double df2{};        ///< Residual design degrees of freedom of the fit.
    double p_value{};    ///< Upper-tail F p-value.
};

/// @brief Tests that the selected coefficients of a survey GLM are jointly zero.
/// @param fit Fitted model.
/// @param indices Coefficient indices to test.
/// @return F statistic on (number tested, df_residual) degrees of freedom.
/// @throws std::invalid_argument For an empty, repeated or out-of-range index set, or no residual degrees of freedom.
/// @throws std::runtime_error If the selected covariance block is singular.
/// @par Example
/// @code
/// dm::stats::SurveyGlmFit f;
/// f.coefficients = {1, 0.5};
/// f.covariance = dm::linalg::DenseMatrix<double>(2, 2, {1, 0, 0, 0.25});
/// f.df_residual = 20;
/// std::println("{:.1f}", dm::stats::survey_glm_wald_test(f, {1}).statistic); // 1.0
/// @endcode
/// @see fit_survey_glm()
[[nodiscard]] SurveyGlmWald survey_glm_wald_test(const SurveyGlmFit& fit, const std::vector<std::size_t>& indices);

} // namespace datamunge::stats
