#pragma once

/// @file meta_analysis.hpp
/// @brief Inverse-variance meta-analysis of study effect estimates with known sampling variances.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::stats {

/// @brief A pooled meta-analytic estimate with heterogeneity statistics.
/// @par Example
/// @code
/// dm::stats::MetaAnalysis m{.estimate = 0.25, .standard_error = 0.125};
/// std::println("{}", m.estimate / m.standard_error); // 2
/// @endcode
struct MetaAnalysis {
    std::size_t k{};             ///< Number of studies.
    double estimate{};           ///< Pooled effect.
    double standard_error{};     ///< Standard error of the pooled effect.
    double statistic{};          ///< estimate / standard_error (z, or t under Hartung-Knapp).
    double p_value{};            ///< Two-sided p-value for a zero effect.
    double df{std::numeric_limits<double>::quiet_NaN()}; ///< t degrees of freedom (NaN for normal-based inference).
    double variance_scale{1.0}; ///< Hartung-Knapp factor applied to the squared standard error (1 for Wald).
    double lower{};              ///< Lower confidence limit.
    double upper{};              ///< Upper confidence limit.
    std::vector<double> weights; ///< Normalised study weights (summing to 1).
    double q{};                  ///< Cochran's Q, \f$\sum w_i(y_i-\hat\theta_{CE})^2\f$ with \f$w_i=1/v_i\f$.
    double q_df{};               ///< k - 1.
    double q_p_value{};          ///< Chi-squared p-value of Q (test of homogeneity).
    double i2{};                 ///< I^2 in percent: the share of variability beyond sampling error (see each model).
    double h2{};                 ///< H^2, total over sampling variability (see each model).
    double tau2{};               ///< Between-study variance (0 for the common-effect model).
    double tau{};                ///< sqrt(tau2).
    double tau2_lower{};         ///< Q-profile confidence limits for tau2 (NaN for the common-effect model).
    double tau2_upper{};         ///< Upper Q-profile limit for tau2.
    double prediction_lower{};   ///< Prediction interval for the effect in a new study (equals the CI when tau2 = 0).
    double prediction_upper{};   ///< Upper prediction limit.
    std::size_t iterations{};    ///< Iterations of the tau2 estimator (0 for closed forms).
    bool converged{true};        ///< Whether an iterative tau2 estimator converged.
};

/// @brief Common-effect (fixed-effect, "equal-effects") meta-analysis by inverse-variance weighting.
///
/// The pooled estimate is \f$\hat\theta=\sum w_iy_i/\sum w_i\f$ with \f$w_i=1/v_i\f$ and standard error
/// \f$(\sum w_i)^{-1/2}\f$; the interval is \f$\hat\theta\pm z_{1-\alpha/2}\,\mathrm{SE}\f$. Valid inference about a single
/// true effect assumes all studies estimate the same quantity; under heterogeneity the interval is too
/// narrow for the average effect (see Q, I^2 and the random-effects model). Here
/// \f$I^2=\max(0,(Q-df)/Q)\f$ and \f$H^2=Q/df\f$; tau2 is 0 and the prediction interval equals the
/// confidence interval. Matches `metafor::rma(yi, vi, method = "EE")`.
/// @param effects Study estimates y_i (for example log odds ratios).
/// @param variances Their sampling variances v_i (> 0).
/// @param confidence Interval coverage in (0, 1).
/// @return The pooled estimate, weights and heterogeneity statistics.
/// @throws std::invalid_argument For fewer than one study, mismatched lengths, nonfinite values,
/// nonpositive variances or confidence outside (0, 1).
/// @par Example
/// @code
/// const auto m = dm::stats::common_effect_meta({0.1, 0.3}, {0.01, 0.01});
/// std::println("{:.2f} {:.4f}", m.estimate, m.standard_error); // 0.20 0.0707
/// @endcode
MetaAnalysis common_effect_meta(const std::vector<double>& effects, const std::vector<double>& variances,
                                double confidence = 0.95);

/// @brief Estimators of the between-study variance tau^2.
/// @par Example
/// @code
/// const auto e = dm::stats::TauEstimator::paule_mandel;
/// std::println("{}", e == dm::stats::TauEstimator::reml); // false
/// @endcode
enum class TauEstimator {
    dersimonian_laird, ///< Method of moments from Cochran's Q (closed form; biased downward when tau2 is large).
    paule_mandel,      ///< Generalised Q equal to its expectation k - p (empirical Bayes).
    reml,              ///< Restricted maximum likelihood (Fisher scoring).
    ml,                ///< Maximum likelihood (Fisher scoring; biased downward for few studies).
};

/// @brief Inference for the average effect in random-effects models.
/// @par Example
/// @code
/// const auto i = dm::stats::MetaInference::hartung_knapp;
/// std::println("{}", i == dm::stats::MetaInference::wald); // false
/// @endcode
enum class MetaInference {
    wald,                    ///< Normal quantiles with the model-based standard error.
    hartung_knapp,           ///< Standard error scaled by the generalised Q / (k - p), t with k - p df.
    hartung_knapp_truncated, ///< As hartung_knapp but the scale is at least 1 (never narrower than Wald).
};

/// @brief Options for random_effects_meta().
/// @par Example
/// @code
/// const dm::stats::RandomEffectsOptions o{.estimator = dm::stats::TauEstimator::dersimonian_laird};
/// std::println("{}", o.confidence); // 0.95
/// @endcode
struct RandomEffectsOptions {
    TauEstimator estimator{TauEstimator::reml}; ///< tau2 estimator.
    double confidence{0.95};                    ///< Coverage of the confidence, tau2 and prediction intervals.
    MetaInference inference{MetaInference::wald}; ///< Wald or Hartung-Knapp inference for the average effect.
    bool t_prediction{false};                   ///< Use t with k - 2 df for the prediction interval (Higgins et al. 2009).
    std::size_t max_iterations{1000};           ///< Cap on Fisher-scoring or root-finding iterations.
    double tolerance{1e-12};                    ///< Convergence threshold on the tau2 update.
};

/// @brief Random-effects meta-analysis: \f$y_i\sim N(\mu,v_i+\tau^2)\f$.
///
/// After estimating tau^2, the average effect is the weighted mean with \f$w_i^*=1/(v_i+\tau^2)\f$, standard
/// error \f$(\sum w_i^*)^{-1/2}\f$ and a Wald (normal) interval. The prediction interval
/// \f$\hat\mu\pm z\sqrt{\mathrm{SE}^2+\hat\tau^2}\f$ describes where the true effect of a new study is expected to lie
/// (with `t_prediction`, z is replaced by \f$t_{k-2}\f$). With Hartung-Knapp inference the squared standard
/// error is multiplied by \f$s^2=\sum w_i^*(y_i-\hat\mu)^2/(k-1)\f$ (truncated below at 1 in the truncated
/// variant), and the test, confidence and prediction intervals use \f$t_{k-1}\f$ (`metafor` test "knha" and
/// "adhoc"); this accounts for the uncertainty in tau^2 that the Wald interval ignores. The tau^2 interval is the Q-profile interval, which
/// inverts the generalised Q statistic against \f$\chi^2_{k-1}\f$. \f$I^2=100\hat\tau^2/(\hat\tau^2+s^2)\f$ and
/// \f$H^2=(\hat\tau^2+s^2)/s^2\f$ with the typical within-study variance \f$s^2=(k-1)\sum w_i/[(\sum w_i)^2-\sum w_i^2]\f$.
/// Matches `metafor::rma(yi, vi, method = "DL" | "PM" | "REML" | "ML", test = "z" | "knha" | "adhoc")`,
/// `predict()` and `confint()`.
/// @param effects Study estimates.
/// @param variances Their sampling variances (> 0).
/// @param options Estimator, confidence and iteration control.
/// @return The pooled estimate with tau2, its interval and the prediction interval.
/// @throws std::invalid_argument For fewer than two studies (three with `t_prediction`), mismatched
/// lengths, nonfinite values, nonpositive variances or confidence outside (0, 1).
/// @par Example
/// @code
/// const auto m = dm::stats::random_effects_meta({0.0, 1.0, 0.2, 0.9}, {0.01, 0.01, 0.01, 0.01});
/// std::println("{:.3f} {:.4f}", m.estimate, m.tau2); // 0.525 0.2392
/// @endcode
MetaAnalysis random_effects_meta(const std::vector<double>& effects, const std::vector<double>& variances,
                                 const RandomEffectsOptions& options = {});

/// @brief Leave-one-out influence diagnostics of a meta-regression, as `metafor::influence()`.
/// @par Example
/// @code
/// dm::stats::MetaInfluence inf;
/// std::println("{}", inf.cooks_distance.size()); // 0
/// @endcode
struct MetaInfluence {
    std::vector<double> hat;                      ///< Leverages, diagonal of X (X'W X)^{-1} X' W.
    std::vector<double> weight;                   ///< Normalised model weights (percent).
    std::vector<double> standardized_residuals;   ///< Residual / its model-based standard deviation (internal).
    std::vector<double> studentized_residuals;    ///< Deleted residual / sqrt(v_i + tau2_(-i) + Var(prediction_(-i))).
    std::vector<double> dffits;                   ///< Change in the fitted value on deletion, standardised.
    std::vector<double> cooks_distance;           ///< (b - b_(-i))' Var(b)^{-1} (b - b_(-i)).
    std::vector<double> covariance_ratio;         ///< det Var(b_(-i)) / det Var(b).
    std::vector<double> tau2_deleted;             ///< Residual tau2 without the study.
    std::vector<double> qe_deleted;               ///< QE without the study.
    std::vector<std::vector<double>> dfbetas;     ///< Per study, (b - b_(-i)) over the SE from all k studies weighted
                                                  ///< with tau2_(-i), for each coefficient (as metafor).
    std::vector<bool> influential;                ///< metafor's rule: |DFFITS| > 3 sqrt(p/(k-p)), P(chi2_p < Cook) > 0.5,
                                                  ///< hat > 3p/k or any |DFBETAS| > 1.
};

/// @brief Result of meta_regression().
/// @par Example
/// @code
/// dm::stats::MetaRegression r;
/// std::println("{}", r.coefficients.size()); // 0
/// @endcode
struct MetaRegression {
    std::size_t k{};                       ///< Number of studies.
    std::size_t p{};                       ///< Number of coefficients (intercept plus moderators).
    std::vector<double> coefficients;      ///< Intercept first, then one per moderator column.
    std::vector<double> standard_errors;   ///< Their standard errors (Hartung-Knapp scaled if requested).
    std::vector<double> statistics;        ///< z or t statistics.
    std::vector<double> p_values;          ///< Two-sided p-values.
    std::vector<double> lower;             ///< Lower confidence limits.
    std::vector<double> upper;             ///< Upper confidence limits.
    linalg::DenseMatrix<double> covariance; ///< Covariance matrix of the coefficients.
    double df{std::numeric_limits<double>::quiet_NaN()}; ///< k - p under Hartung-Knapp, NaN otherwise.
    double variance_scale{1.0};            ///< Hartung-Knapp factor (1 for Wald).
    double qm{};                           ///< Omnibus moderator test: chi-squared, or F under Hartung-Knapp.
    double qm_df{};                        ///< Its numerator degrees of freedom (number of moderators).
    double qm_p_value{};                   ///< Its p-value.
    double qe{};                           ///< Residual heterogeneity statistic \f$y'P_0y\f$ with \f$P_0\f$ at tau2 = 0.
    double qe_df{};                        ///< k - p.
    double qe_p_value{};                   ///< Chi-squared p-value of QE.
    double tau2{};                         ///< Residual between-study variance.
    double tau2_lower{};                   ///< Q-profile limits for the residual tau2.
    double tau2_upper{};                   ///< Upper Q-profile limit.
    double i2{};                           ///< Residual I^2 in percent.
    double h2{};                           ///< Residual H^2.
    double r2{};                           ///< Pseudo R^2 in percent: proportional reduction of tau2 from the intercept-only model.
    std::vector<double> fitted;            ///< Fitted values x_i' b.
    std::vector<double> residuals;         ///< y_i - x_i' b.
    std::vector<double> weights;           ///< Normalised weights 1/(v_i + tau2).
    std::size_t iterations{};              ///< tau2 estimator iterations.
    bool converged{true};                  ///< Whether the tau2 estimator converged.
    MetaInfluence influence;               ///< Leave-one-out diagnostics (empty unless requested).
};

/// @brief Options for meta_regression().
/// @par Example
/// @code
/// const dm::stats::MetaRegressionOptions o{.influence = true};
/// std::println("{}", o.model.confidence); // 0.95
/// @endcode
struct MetaRegressionOptions {
    RandomEffectsOptions model{}; ///< tau2 estimator, inference, confidence and iteration control.
    bool influence{false};        ///< Compute leave-one-out influence diagnostics (k refits).
};

/// @brief Mixed-effects meta-regression \f$y_i\sim N(x_i'\beta,v_i+\tau^2)\f$ with an intercept.
///
/// tau^2 is the residual between-study variance not explained by the moderators, estimated as in
/// random_effects_meta() but with k - p residual degrees of freedom; the coefficients are weighted least
/// squares with weights \f$1/(v_i+\hat\tau^2)\f$. QM tests all moderator coefficients jointly (chi-squared, or
/// \f$F_{m,k-p}\f$ under Hartung-Knapp) and QE tests for residual heterogeneity. Pseudo-R^2 compares tau^2
/// with that of the intercept-only model (truncated to [0, 100]). Moderators are centred and scaled
/// internally for numerical stability; results are reported on the original scale. Influence diagnostics
/// refit the model without each study. Matches `metafor::rma(yi, vi, mods = ~ ..., method, test)`, `influence()`,
/// `rstandard()` and `confint()`.
/// @param effects Study estimates.
/// @param variances Their sampling variances.
/// @param moderators k x m moderator matrix (an intercept is added; m >= 1).
/// @param options Model options and whether to compute influence diagnostics.
/// @return The fitted model.
/// @throws std::invalid_argument For invalid studies, a moderator matrix with the wrong number of rows or
/// no columns, a constant moderator, k <= p, or a singular design.
/// @par Example
/// @code
/// dm::linalg::DenseMatrix<double> x(4, 1);
/// for (std::size_t i = 0; i < 4; ++i) x(i, 0) = double(i);
/// const auto r = dm::stats::meta_regression({0.1, 0.3, 0.5, 0.7}, {0.01, 0.01, 0.01, 0.01}, x);
/// std::println("{:.2f} {:.2f}", r.coefficients[0], r.coefficients[1]); // 0.10 0.20
/// @endcode
MetaRegression meta_regression(const std::vector<double>& effects, const std::vector<double>& variances,
                               const linalg::DenseMatrix<double>& moderators,
                               const MetaRegressionOptions& options = {});

/// @brief Predictor of the effect size in tests for small-study effects (funnel-plot asymmetry).
/// @par Example
/// @code
/// const auto p = dm::stats::SmallStudyPredictor::inverse_sample_size;
/// std::println("{}", p == dm::stats::SmallStudyPredictor::standard_error); // false
/// @endcode
enum class SmallStudyPredictor {
    standard_error,           ///< sqrt(v_i): Egger's test.
    variance,                 ///< v_i.
    sample_size,              ///< n_i (the limit estimate then extrapolates to n = 0 and is not meaningful).
    inverse_sample_size,      ///< 1/n_i: Peters' test, for effects whose variance depends on the effect (log odds ratios).
    inverse_sqrt_sample_size, ///< 1/sqrt(n_i).
};

/// @brief Regression model of tests for small-study effects.
/// @par Example
/// @code
/// const auto m = dm::stats::SmallStudyModel::random_effects;
/// std::println("{}", m == dm::stats::SmallStudyModel::weighted_regression); // false
/// @endcode
enum class SmallStudyModel {
    weighted_regression, ///< Weighted least squares with a multiplicative dispersion and t with k - 2 df (classical Egger).
    random_effects,      ///< Mixed-effects meta-regression with additive tau^2 (inference from RandomEffectsOptions).
};

/// @brief Options for egger_test().
/// @par Example
/// @code
/// const dm::stats::EggerOptions o{.model = dm::stats::SmallStudyModel::random_effects};
/// std::println("{}", o.inverse_variance_weights); // true
/// @endcode
struct EggerOptions {
    SmallStudyPredictor predictor{SmallStudyPredictor::standard_error}; ///< Predictor of the effects.
    SmallStudyModel model{SmallStudyModel::weighted_regression};        ///< Regression model.
    std::vector<double> sample_sizes{};    ///< Total study sizes n_i (required by the sample-size predictors).
    bool inverse_variance_weights{true};   ///< Weighted regression only: weights 1/v_i (false gives equal weights).
    RandomEffectsOptions random_effects{}; ///< Estimator, inference and confidence (confidence is used by both models).
};

/// @brief Result of egger_test().
/// @par Example
/// @code
/// dm::stats::EggerTest e{.slope = 1.2, .slope_se = 0.4};
/// std::println("{:.1f}", e.slope / e.slope_se); // 3.0
/// @endcode
struct EggerTest {
    double slope{};          ///< Coefficient of the predictor (the asymmetry); 0 without small-study effects.
    double slope_se{};       ///< Its standard error.
    double statistic{};      ///< slope / slope_se (t for the weighted regression or Hartung-Knapp, else z).
    double df{std::numeric_limits<double>::quiet_NaN()}; ///< t degrees of freedom, NaN for z.
    double p_value{};        ///< Two-sided p-value for no asymmetry.
    double limit_estimate{}; ///< Intercept: the effect predicted for an infinitely precise study.
    double limit_se{};       ///< Its standard error.
    double limit_lower{};    ///< Confidence limits of the limit estimate.
    double limit_upper{};    ///< Upper confidence limit.
    double dispersion{};     ///< Weighted regression: residual variance multiplier s^2; random effects: 1.
    double tau2{};           ///< Random effects: residual tau2; weighted regression: 0.
};

/// @brief Regression test for funnel-plot asymmetry (small-study effects), as `metafor::regtest`.
///
/// Regresses the effects on a precision-related predictor. With the standard error and the weighted
/// regression this is Egger's test: equivalent to regressing \f$y_i/\sqrt{v_i}\f$ on \f$1/\sqrt{v_i}\f$ and
/// testing the intercept. A nonzero slope means that less precise studies report systematically different
/// effects, which publication bias, outcome-reporting bias or genuine heterogeneity related to study size
/// can cause. The intercept is the "limit estimate" for an infinitely precise study (PET for the standard
/// error, PEESE for the variance). For log odds ratios the standard error depends on the effect itself, so
/// Egger's test is anti-conservative; the inverse sample size predictor (Peters' test) avoids this.
/// @param effects Study estimates.
/// @param variances Their sampling variances.
/// @param options Predictor, model, sample sizes and weighting.
/// @return The asymmetry test and limit estimate.
/// @throws std::invalid_argument For fewer than three studies, invalid inputs, missing or nonpositive sample
/// sizes for a sample-size predictor, or a constant predictor.
/// @par Example
/// @code
/// const auto e = dm::stats::egger_test({0.15, 0.25, 0.35, 0.45}, {0.01, 0.04, 0.09, 0.16});
/// std::println("{:.3f} {:.3f}", e.limit_estimate, e.slope); // 0.050 1.000
/// @endcode
EggerTest egger_test(const std::vector<double>& effects, const std::vector<double>& variances,
                     const EggerOptions& options = {});

} // namespace datamunge::stats
