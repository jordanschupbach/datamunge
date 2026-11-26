#pragma once

/// @file mediation.hpp
/// @brief Natural direct and indirect effects by parametric g-computation.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::causal {

/// @brief Regression model for the mediator or the outcome.
enum class MediationModel {
    linear,   ///< Normal linear regression (least squares).
    logistic, ///< Logistic regression for a 0/1 variable.
};

/// @brief Options for mediation_analysis().
/// @par Example
/// @code
/// const dm::causal::MediationOptions o{.outcome_model = dm::causal::MediationModel::logistic};
/// std::println("{}", o.interaction); // true
/// @endcode
struct MediationOptions {
    MediationModel mediator_model{MediationModel::linear}; ///< Mediator regression on treatment and covariates.
    MediationModel outcome_model{MediationModel::linear};  ///< Outcome regression on treatment, mediator, covariates.
    bool interaction{true};      ///< Include a treatment-by-mediator term in the outcome model.
    double control_value{0.0};   ///< Treatment level a*.
    double treated_value{1.0};   ///< Treatment level a.
    std::size_t bootstrap{0};    ///< Nonparametric bootstrap resamples for SEs and percentile CIs (0 skips).
    std::uint64_t seed{20};      ///< Seed of the bootstrap resampling.
    double confidence{0.95};     ///< Level of the percentile intervals.
    std::size_t quadrature_points{40}; ///< Gauss-Hermite nodes for a linear mediator with a logistic outcome.
};

/// @brief One mediation quantity with bootstrap uncertainty (NaN without bootstrap).
/// @par Example
/// @code
/// const dm::causal::MediationEffect e{.estimate = 0.05};
/// std::println("{}", e.estimate); // 0.05
/// @endcode
struct MediationEffect {
    double estimate{};       ///< Point estimate from the original sample.
    double standard_error{}; ///< Bootstrap standard deviation.
    double lower{};          ///< Lower percentile limit.
    double upper{};          ///< Upper percentile limit.
};

/// @brief Result of mediation_analysis().
///
/// With \f$Q(a,a')=E[Y(a,M(a'))]\f$ averaged over the covariates, `indirect_treated` (natural
/// indirect effect, ACME under treatment) is \f$Q(1,1)-Q(1,0)\f$, `indirect_control` is
/// \f$Q(0,1)-Q(0,0)\f$, `direct_control` (natural direct effect) is \f$Q(1,0)-Q(0,0)\f$,
/// `direct_treated` is \f$Q(1,1)-Q(0,1)\f$ and `total` is \f$Q(1,1)-Q(0,0)\f$, where 1 and 0 stand
/// for the treated and control values. Binary outcomes use the risk-difference scale.
/// @par Example
/// @code
/// dm::causal::MediationResult r;
/// r.total.estimate = 0.2;
/// r.indirect_treated.estimate = 0.05;
/// std::println("{:.2f}", r.indirect_treated.estimate / r.total.estimate); // 0.25
/// @endcode
struct MediationResult {
    MediationEffect indirect_control;    ///< ACME at the control level (delta(0)).
    MediationEffect indirect_treated;    ///< ACME at the treated level (delta(1)); the natural indirect effect.
    MediationEffect direct_control;      ///< ADE at the control mediator (zeta(0)); the natural direct effect.
    MediationEffect direct_treated;      ///< ADE at the treated mediator (zeta(1)).
    MediationEffect total;               ///< Total effect.
    MediationEffect proportion_mediated; ///< indirect_treated / total.
    std::vector<double> mediator_coefficients; ///< Intercept, treatment, covariates.
    std::vector<double> outcome_coefficients;  ///< Intercept, treatment, mediator, [interaction], covariates.
    double mediator_sigma{};   ///< Residual SD of a linear mediator model (n - p denominator).
    std::size_t bootstrap_failures{}; ///< Resamples whose model fits failed (skipped).
};

/// @brief Natural direct and indirect effects by parametric g-computation.
///
/// Fits the mediator model \f$M\mid A,C\f$ and the outcome model \f$Y\mid A,M,C\f$ (with an
/// optional \f$A\times M\f$ term) and evaluates the mediation formula
/// \f$E[Y(a,M(a'))]=n^{-1}\sum_i\int E[Y\mid a,m,c_i]\,dF(m\mid a',c_i)\f$: exactly for a binary
/// mediator, in closed form for a linear mediator with a linear outcome, and by Gauss-Hermite
/// quadrature over the normal mediator otherwise. The effects are causal only under sequential
/// ignorability: no unmeasured treatment-outcome, mediator-outcome or treatment-mediator
/// confounding given C, and no mediator-outcome confounder affected by treatment. They also require
/// correctly specified models. Point estimates equal the exact g-formula, and R's `mediation`
/// where its estimates are exact (linear models without an interaction; elsewhere it simulates).
/// @param outcome Outcome (0/1 for a logistic outcome model).
/// @param treatment Treatment.
/// @param mediator Mediator (0/1 for a logistic mediator model).
/// @param covariates n-by-p pre-treatment covariates, or an empty matrix.
/// @param options Models, treatment levels and bootstrap settings.
/// @return Effects with bootstrap uncertainty, and the fitted coefficients.
/// @throws std::invalid_argument For mismatched sizes, nonfinite values, non-binary variables under a
/// logistic model, equal treatment levels, or bad options.
/// @throws std::runtime_error When a model fit is singular or does not converge.
/// @par Example
/// @code
/// std::vector<double> a, m, y;
/// for (int i = 0; i < 200; ++i) {
///     a.push_back(i % 2);
///     m.push_back(0.5 * a.back() + std::sin(1.7 * i));
///     y.push_back(a.back() + 2 * m.back() + std::cos(2.3 * i));
/// }
/// const auto r = dm::causal::mediation_analysis(y, a, m, {}, {.interaction = false});
/// std::println("{:.2f}", r.indirect_treated.estimate / (r.mediator_coefficients[1] *
///                                                      r.outcome_coefficients[2])); // 1.00
/// @endcode
MediationResult mediation_analysis(const std::vector<double>& outcome, const std::vector<double>& treatment,
                                   const std::vector<double>& mediator,
                                   const linalg::DenseMatrix<double>& covariates = {},
                                   const MediationOptions& options = {});

} // namespace datamunge::causal
