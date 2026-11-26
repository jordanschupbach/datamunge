#pragma once

/// @file aipw.hpp
/// @brief Cross-fitted doubly robust estimators of the average treatment effect: AIPW and TMLE.

#include <datamunge/linalg/dense_matrix.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace datamunge::causal {

/// @brief Learns P(A = 1 | X) on training rows and predicts it for test rows.
using PropensityLearner = std::function<std::vector<double>(
    const linalg::DenseMatrix<double>& train_x, const std::vector<int>& train_treatment,
    const linalg::DenseMatrix<double>& test_x)>;

/// @brief Learns E[Y | X] on training rows of one treatment arm and predicts it for test rows.
using OutcomeLearner = std::function<std::vector<double>(
    const linalg::DenseMatrix<double>& train_x, const std::vector<double>& train_outcome,
    const linalg::DenseMatrix<double>& test_x)>;

/// @brief Main-effects logistic regression propensity learner (the default).
/// @param train_x Training covariates.
/// @param train_treatment Training indicators (0 or 1) with both groups present.
/// @param test_x Covariates to predict for.
/// @return Predicted treatment probabilities for the test rows.
/// @throws std::runtime_error When the logistic fit fails or separates the groups.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6, 1, {1, 2, 3, 4, 5, 6});
/// const auto p = dm::causal::logistic_propensity_learner(x, {0, 1, 0, 1, 0, 1}, x);
/// std::println("{}", p.size()); // 6
/// @endcode
/// @see cross_fitted_aipw()
[[nodiscard]] std::vector<double> logistic_propensity_learner(const linalg::DenseMatrix<double>& train_x,
                                                              const std::vector<int>& train_treatment,
                                                              const linalg::DenseMatrix<double>& test_x);

/// @brief Ordinary least-squares outcome learner with an intercept (the default).
/// @param train_x Training covariates.
/// @param train_outcome Training outcomes.
/// @param test_x Covariates to predict for.
/// @return Linear predictions for the test rows.
/// @throws std::runtime_error For a rank-deficient training design.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(3, 1, {0, 1, 2});
/// const auto m = dm::causal::linear_outcome_learner(x, {1, 3, 5}, dm::linalg::DenseMatrix<double>(1, 1, 4.0));
/// std::println("{:.1f}", m[0]); // 9.0
/// @endcode
/// @see cross_fitted_aipw()
[[nodiscard]] std::vector<double> linear_outcome_learner(const linalg::DenseMatrix<double>& train_x,
                                                         const std::vector<double>& train_outcome,
                                                         const linalg::DenseMatrix<double>& test_x);

/// @brief Options for cross_fitted_aipw().
/// @par Example
/// @code
/// const dm::causal::AipwOptions o{.folds = 10};
/// std::println("{} {}", o.folds, o.propensity_bound); // 10 0.01
/// @endcode
struct AipwOptions {
    std::size_t folds{5};                ///< Number of folds; 1 fits and predicts on the full sample (no cross-fitting).
    std::vector<std::size_t> fold_ids{}; ///< Explicit fold of each unit (0-based); overrides `folds` and `seed`.
    std::uint64_t seed{1};               ///< Seed for the random, treatment-stratified fold assignment.
    double propensity_bound{0.01};       ///< Predicted propensities are clipped to [bound, 1 - bound].
    double confidence{0.95};             ///< Level of the normal confidence intervals.
    PropensityLearner propensity{};      ///< Propensity learner (empty: logistic_propensity_learner()).
    OutcomeLearner outcome{};            ///< Arm-specific outcome learner (empty: linear_outcome_learner()).
};

/// @brief Cross-fitted AIPW estimate of the average treatment effect with influence-function inference.
/// @par Example
/// @code
/// const dm::causal::AipwResult r{.ate = 1.0, .standard_error = 0.25};
/// std::println("{}", r.ate / r.standard_error); // 4
/// @endcode
struct AipwResult {
    double ate{};                          ///< Average treatment effect \f$E[Y(1)-Y(0)]\f$.
    double standard_error{};               ///< Standard deviation of the influence values over \f$\sqrt n\f$.
    double lower{};                        ///< Lower normal confidence limit.
    double upper{};                        ///< Upper normal confidence limit.
    double p_value{};                      ///< Two-sided normal p-value for no effect.
    double treated_mean{};                 ///< AIPW estimate of \f$E[Y(1)]\f$.
    double control_mean{};                 ///< AIPW estimate of \f$E[Y(0)]\f$.
    std::vector<double> influence;         ///< Influence value of each unit for the ATE (mean zero).
    std::vector<double> propensity;        ///< Out-of-fold propensities after clipping.
    std::vector<double> control_prediction; ///< Out-of-fold predictions of E[Y | X, A = 0].
    std::vector<double> treated_prediction; ///< Out-of-fold predictions of E[Y | X, A = 1].
    std::vector<std::size_t> fold_ids;     ///< Fold of each unit.
    std::size_t clipped{};                 ///< Number of propensities moved to the bounds.
};

/// @brief Cross-fitted augmented inverse probability weighted (doubly robust) ATE.
///
/// Splits the units into folds (stratified by treatment). For each fold, the propensity and the two
/// arm-specific outcome regressions are trained on the other folds and predicted on this one, so no
/// unit's nuisance predictions use its own outcome. The estimate averages
/// \f[\psi_i=\hat\mu_1(X_i)-\hat\mu_0(X_i)+\frac{A_i(Y_i-\hat\mu_1(X_i))}{\hat e(X_i)}
/// -\frac{(1-A_i)(Y_i-\hat\mu_0(X_i))}{1-\hat e(X_i)},\f]
/// and the standard error is the sample SD of \f$\psi_i\f$ over \f$\sqrt n\f$. The estimator is consistent
/// if either nuisance model is correct, and asymptotically normal and efficient when both converge
/// fast enough (for example at \f$n^{-1/4}\f$ rates). Cross-fitting removes the overfitting bias of
/// flexible learners without requiring Donsker conditions.
/// @param outcome Finite outcomes.
/// @param treatment Indicators (0 or 1) with both groups represented.
/// @param covariates Finite covariates in columns.
/// @param options Folds, bounds, confidence and nuisance learners.
/// @return Effect, inference, potential-outcome means and the out-of-fold nuisance predictions.
/// @throws std::invalid_argument For invalid inputs, fold ids out of range, a fold whose training
/// complement lacks a treatment group, or a bound outside [0, 0.5).
/// @throws std::runtime_error When a learner fails or returns the wrong number of predictions.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(8, 1, {0, 1, 2, 3, 0, 1, 2, 3});
/// const std::vector<double> y{0, 1, 4, 5, 0, 3, 2, 5}; // y = x + 2a
/// const auto r = dm::causal::cross_fitted_aipw(y, {0, 0, 1, 1, 0, 1, 0, 1}, x, {.folds = 1});
/// std::println("{:.3f}", r.ate); // 2.000
/// @endcode
/// @par References
/// - J. M. Robins, A. Rotnitzky and L. P. Zhao (1994). Estimation of regression coefficients when some
///   regressors are not always observed. JASA 89(427), 846-866. doi:10.1080/01621459.1994.10476818
/// - V. Chernozhukov et al. (2018). Double/debiased machine learning for treatment and structural
///   parameters. The Econometrics Journal 21(1), C1-C68. doi:10.1111/ectj.12097
/// @see binary_ate(), entropy_balance()
[[nodiscard]] AipwResult cross_fitted_aipw(const std::vector<double>& outcome, const std::vector<int>& treatment,
                                           const linalg::DenseMatrix<double>& covariates,
                                           const AipwOptions& options = {});

/// @brief Options for targeted_mle().
/// @par Example
/// @code
/// const dm::causal::TmleOptions o{.folds = 1};
/// std::println("{} {}", o.folds, o.propensity_bound); // 1 0.025
/// @endcode
struct TmleOptions {
    std::size_t folds{5};                ///< Folds for the nuisance predictions (1: no cross-fitting).
    std::vector<std::size_t> fold_ids{}; ///< Explicit fold of each unit (0-based); overrides `folds` and `seed`.
    std::uint64_t seed{1};               ///< Seed for the treatment-stratified fold assignment.
    double propensity_bound{0.025};      ///< Propensities are clipped to [bound, 1 - bound].
    double outcome_bound{0.0005};        ///< Scaled initial predictions are clipped to [bound, 1 - bound] before the logit.
    std::optional<std::pair<double, double>> outcome_range{}; ///< Known outcome bounds (default: sample minimum and maximum).
    double confidence{0.95};             ///< Level of the normal confidence intervals.
    PropensityLearner propensity{};      ///< Propensity learner (empty: logistic_propensity_learner()).
    OutcomeLearner outcome{};            ///< Arm-specific outcome learner (empty: linear_outcome_learner()).
};

/// @brief Targeted maximum likelihood estimate of the ATE.
/// @par Example
/// @code
/// const dm::causal::TmleResult r{.ate = 2.0, .initial_ate = 1.5};
/// std::println("{}", r.ate - r.initial_ate); // 0.5
/// @endcode
struct TmleResult {
    double ate{};                    ///< Targeted ATE on the outcome scale.
    double standard_error{};         ///< Influence-function standard error.
    double lower{};                  ///< Lower normal confidence limit.
    double upper{};                  ///< Upper normal confidence limit.
    double p_value{};                ///< Two-sided normal p-value for no effect.
    double treated_mean{};           ///< Targeted estimate of \f$E[Y(1)]\f$.
    double control_mean{};           ///< Targeted estimate of \f$E[Y(0)]\f$.
    double initial_ate{};            ///< Plug-in ATE from the initial outcome predictions (after bounding).
    std::array<double, 2> epsilon{}; ///< Fluctuation coefficients of the control [0] and treated [1] clever covariates.
    std::size_t iterations{};        ///< Newton iterations of the fluctuation fits (both arms).
    bool converged{false};           ///< True when the fluctuation score equation was solved.
    std::vector<double> influence;   ///< Efficient influence values on the outcome scale.
    std::vector<double> propensity;  ///< Out-of-fold propensities after clipping.
    std::vector<std::size_t> fold_ids; ///< Fold of each unit.
    std::size_t clipped{};           ///< Number of propensities moved to the bounds.
};

/// @brief Targeted maximum likelihood estimation (TMLE) of the average treatment effect.
///
/// Maps the outcome to [0, 1] with the sample (or given) range and obtains out-of-fold initial
/// predictions \f$\bar Q(a,X)\f$ and propensities as cross_fitted_aipw() does. It then fluctuates the
/// initial fit by a logistic regression of \f$Y^*\f$ on the clever covariates \f$H_1=A/\hat e(X)\f$ and
/// \f$H_0=(1-A)/(1-\hat e(X))\f$ with offset \f$\mathrm{logit}\,\bar Q(A,X)\f$, one coefficient per arm,
/// which targets both potential-outcome means. The targeted plug-in
/// \f$\frac1n\sum\{\bar Q^*(1,X_i)-\bar Q^*(0,X_i)\}\f$ solves the efficient influence-function estimating
/// equation, so it is doubly robust and efficient like AIPW, but it is a substitution estimator: it
/// respects the outcome bounds and is often more stable when some propensities are small. This is the
/// one-step logistic fluctuation of the R tmle package with `target.gwt = FALSE`.
/// @param outcome Finite outcomes (binary or bounded continuous).
/// @param treatment Indicators (0 or 1) with both groups represented.
/// @param covariates Finite covariates in columns.
/// @param options Folds, bounds, outcome range, confidence and learners.
/// @return Targeted effect, inference, initial estimate and fluctuation details.
/// @throws std::invalid_argument For invalid inputs, constant outcomes or outcomes outside the given range.
/// @throws std::runtime_error When a learner fails.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(8, 1, {0, 1, 2, 3, 0, 1, 2, 3});
/// const std::vector<double> y{0, 1, 4, 5, 0, 3, 2, 5}; // y = x + 2a
/// const auto r = dm::causal::targeted_mle(y, {0, 0, 1, 1, 0, 1, 0, 1}, x,
///                                         {.folds = 1, .outcome_range = std::pair{-1.0, 6.0}});
/// std::println("{:.3f}", r.ate); // 2.000
/// @endcode
/// @par References
/// - M. J. van der Laan and D. Rubin (2006). Targeted maximum likelihood learning. The International
///   Journal of Biostatistics 2(1). doi:10.2202/1557-4679.1043
/// - S. Gruber and M. J. van der Laan (2012). tmle: An R package for targeted maximum likelihood
///   estimation. Journal of Statistical Software 51(13). doi:10.18637/jss.v051.i13
/// @see cross_fitted_aipw()
[[nodiscard]] TmleResult targeted_mle(const std::vector<double>& outcome, const std::vector<int>& treatment,
                                      const linalg::DenseMatrix<double>& covariates,
                                      const TmleOptions& options = {});

} // namespace datamunge::causal
