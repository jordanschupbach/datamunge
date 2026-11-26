#pragma once

/// @file weighting.hpp
/// @brief Balancing weights for treatment effect estimation: entropy balancing and weighted contrasts.

#include <datamunge/causal/matching.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace datamunge::causal {

/// @brief Population whose covariate distribution the weights target.
enum class WeightingEstimand {
    att, ///< Treated units: controls are reweighted to the treated moments.
    atc, ///< Control units: treated units are reweighted to the control moments.
    ate, ///< Whole sample: each group is reweighted to the full-sample moments.
};

/// @brief Options for entropy_balance().
/// @par Example
/// @code
/// const dm::causal::EntropyBalanceOptions o{.moments = 2};
/// std::println("{}", o.moments); // 2
/// @endcode
struct EntropyBalanceOptions {
    WeightingEstimand estimand{WeightingEstimand::att}; ///< Target population.
    std::size_t moments{1};            ///< 1 balances means; 2 also balances squares of non-binary covariates.
    std::vector<double> base_weights{}; ///< Positive design or sampling weights (empty: all ones).
    double tolerance{1e-10};           ///< Relative tolerance on the balance constraints.
    std::size_t max_iterations{200};   ///< Newton iteration limit.
};

/// @brief Entropy balancing weights and diagnostics.
/// @par Example
/// @code
/// const dm::causal::EntropyBalanceResult r{.converged = true};
/// std::println("{}", r.converged); // true
/// @endcode
struct EntropyBalanceResult {
    std::vector<double> weights;                  ///< Unit weights; each reweighted group sums to its base-weight total.
    std::array<double, 2> effective_sample_size{}; ///< Kish effective size \f$(\sum w)^2/\sum w^2\f$ of controls [0] and treated [1].
    double max_weight_share{};                    ///< Largest single weight as a share of its group's total.
    CovariateBalance before;                      ///< Base-weighted balance before reweighting.
    CovariateBalance after;                       ///< Balance after reweighting.
    std::size_t iterations{};                     ///< Newton iterations (summed over reweighted groups).
    bool converged{false};                        ///< True when every reweighted group met the tolerance.
};

/// @brief Entropy balancing: the weights closest to the base weights (in Kullback-Leibler divergence)
/// that make the reweighted covariate moments equal the target moments exactly.
///
/// Within each reweighted group the solution has the form \f$w_i=q_i\exp(\lambda^\top c(x_i))\f$,
/// where \f$c\f$ are the constrained moment functions; \f$\lambda\f$ solves the convex dual by damped
/// Newton steps (calibrate_weights() with the raking distance). Covariates are centered on their
/// targets and scaled before solving, which does not change the solution. Exact balance is impossible
/// when a target lies outside the convex hull of the group's moments; then `converged` is false.
/// @param treatment Indicators (0 or 1) with both groups represented.
/// @param covariates Finite covariates in columns.
/// @param options Estimand, number of moments, base weights and solver settings.
/// @return Weights, effective sample sizes and balance before and after.
/// @throws std::invalid_argument For mismatched sizes, invalid indicators, nonpositive base weights,
/// non-finite covariates or moments other than 1 or 2.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6, 1, {1, 2, 3, 2, 4, 6});
/// const auto r = dm::causal::entropy_balance({1, 1, 1, 0, 0, 0}, x);
/// std::println("{:.6f}", r.after.control_mean[0]); // 2.000000
/// @endcode
/// @par References
/// - J. Hainmueller (2012). Entropy balancing for causal effects: A multivariate reweighting method to
///   produce balanced samples in observational studies. Political Analysis 20(1), 25-46. doi:10.1093/pan/mpr025
/// - Q. Zhao and D. Percival (2017). Entropy balancing is doubly robust. Journal of Causal Inference 5(1). doi:10.1515/jci-2016-0010
/// @see weighted_effect(), covariate_balance(), stats::calibrate_weights()
[[nodiscard]] EntropyBalanceResult entropy_balance(const std::vector<int>& treatment,
                                                   const linalg::DenseMatrix<double>& covariates,
                                                   const EntropyBalanceOptions& options = {});

/// @brief Weighted difference in mean outcomes with a heteroskedasticity-robust (HC0) standard error.
///
/// Fits the weighted regression of outcome on an intercept and treatment, as `lm_weightit(...,
/// vcov = "HC0")`. The standard error treats the weights as fixed and so ignores the uncertainty of
/// estimating them; for balancing weights this is usually conservative.
/// @param outcome Finite outcomes.
/// @param treatment Indicators (0 or 1).
/// @param weights Nonnegative weights; units with zero weight are excluded.
/// @param confidence Confidence level of the t interval on matched units minus 2 df.
/// @return Estimate, standard error, df, p-value and interval.
/// @throws std::invalid_argument For mismatched sizes, invalid values or too few weighted units.
/// @par Example
/// @code
/// const auto e = dm::causal::weighted_effect({1, 2, 4, 6}, {0, 0, 1, 1}, {1, 1, 1, 1});
/// std::println("{}", e.estimate); // 3.5
/// @endcode
/// @see entropy_balance(), matched_effect()
[[nodiscard]] MatchedEffect weighted_effect(const std::vector<double>& outcome, const std::vector<int>& treatment,
                                            const std::vector<double>& weights, double confidence = 0.95);

/// @brief Covariate-adjusted weighted effect: the treatment coefficient of the weighted regression of
/// outcome on an intercept, treatment and covariates, with an HC0 sandwich standard error.
///
/// Equivalent to `lm_weightit(y ~ treat + covariates, vcov = "HC0")`. When the weights balance the
/// covariates exactly (entropy balancing of their means) the estimate equals the unadjusted weighted
/// contrast, but the residuals exclude the variation the covariates explain, so the standard error
/// is much closer to the true sampling variability than the unadjusted one. Covariates are centered
/// and scaled internally, which does not change the treatment coefficient.
/// @param outcome Finite outcomes.
/// @param treatment Indicators (0 or 1).
/// @param weights Nonnegative weights; units with zero weight are excluded.
/// @param covariates Finite adjustment covariates in columns, one row per unit.
/// @param confidence Confidence level of the t interval on weighted units minus parameters df.
/// @return Estimate, standard error, df, p-value and interval.
/// @throws std::invalid_argument For mismatched sizes, invalid values or too few weighted units.
/// @throws std::runtime_error For a rank-deficient weighted design.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6, 1, {1, 2, 3, 1, 2, 3});
/// const auto e = dm::causal::weighted_effect({2, 3, 5, 1, 1, 3}, {1, 1, 1, 0, 0, 0},
///                                            {1, 1, 1, 1, 1, 1}, x);
/// std::println("{:.4f}", e.estimate); // 1.6667
/// @endcode
/// @see entropy_balance()
[[nodiscard]] MatchedEffect weighted_effect(const std::vector<double>& outcome, const std::vector<int>& treatment,
                                            const std::vector<double>& weights,
                                            const linalg::DenseMatrix<double>& covariates, double confidence = 0.95);

} // namespace datamunge::causal
