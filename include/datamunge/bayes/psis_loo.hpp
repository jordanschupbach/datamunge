#pragma once

/// @file psis_loo.hpp
/// @brief Pareto-smoothed importance sampling and approximate leave-one-out cross-validation.

#include <cstddef>
#include <vector>

namespace datamunge::bayes {

/// @brief Importance weights after Pareto smoothing of their upper tail.
/// @par Example
/// @code
/// const auto s = dm::bayes::pareto_smoothed_importance_sampling(std::vector<double>(100, 0.0));
/// std::println("{:.0f}", s.effective_sample_size); // 100
/// @endcode
struct ParetoSmoothedWeights {
    std::vector<double> log_weights; ///< Smoothed, truncated log weights normalised to sum to 1.
    double pareto_k{};               ///< Estimated generalized Pareto shape of the tail (infinite if not fitted).
    std::size_t tail_length{};       ///< Number of largest weights replaced by Pareto quantiles.
    double effective_sample_size{};  ///< r_eff / sum of squared normalised weights.
};

/// @brief Pareto-smoothed importance sampling (PSIS) of one vector of log importance ratios.
///
/// The M = ceil(min(0.2 S, 3 sqrt(S / r_eff))) largest ratios are taken as the tail. A
/// generalized Pareto distribution is fitted to their excesses over the next-largest ratio by the
/// Zhang-Stephens empirical Bayes method, its shape is shrunk towards 0.5 by the weakly informative
/// adjustment \f$(Mk+5)/(M+10)\f$, and the tail is replaced by the fitted quantiles at
/// \f$(j-1/2)/M\f$. Weights are then truncated at the largest raw ratio and normalised. With fewer
/// than five tail draws, or a constant tail, no smoothing is done and k is infinite. This
/// reproduces `loo::psis` (2.9).
/// @param log_ratios Log importance ratios for S draws (finite, at least two).
/// @param relative_efficiency MCMC relative efficiency r_eff in (0, 1], or above 1 for antithetic draws.
/// @return Smoothed normalised log weights, k-hat, tail length and effective sample size.
/// @throws std::invalid_argument For fewer than two draws, nonfinite values or a nonpositive r_eff.
/// @par Example
/// @code
/// std::vector<double> lr;
/// for (int s = 0; s < 1000; ++s)
///     lr.push_back(0.5 * std::sin(0.37 * s));
/// const auto w = dm::bayes::pareto_smoothed_importance_sampling(lr);
/// std::println("{} {}", w.tail_length, w.pareto_k < 0.5); // 95 true
/// @endcode
ParetoSmoothedWeights pareto_smoothed_importance_sampling(const std::vector<double>& log_ratios,
                                                          double relative_efficiency = 1.0);

/// @brief Sample-size-dependent reliability threshold for k-hat: min(1 - 1/log10(S), 0.7).
/// @param draws Number of draws S (at least 2).
/// @return The threshold above which PSIS estimates are unreliable.
/// @throws std::invalid_argument For fewer than two draws.
/// @par Example
/// @code
/// std::println("{:.2f}", dm::bayes::pareto_k_threshold(4000)); // 0.70
/// @endcode
double pareto_k_threshold(std::size_t draws);

/// @brief One observation's PSIS-LOO contribution.
/// @par Example
/// @code
/// dm::bayes::PsisLooPoint p{.elpd = -1.2, .pareto_k = 0.3};
/// std::println("{}", p.pareto_k < 0.7); // true
/// @endcode
struct PsisLooPoint {
    double elpd{};                  ///< log p(y_i | y_{-i}) estimated by PSIS.
    double effective_parameters{};  ///< lpd_i - elpd_i.
    double pareto_k{};              ///< k-hat of the observation's importance ratios.
    double effective_sample_size{}; ///< PSIS effective sample size.
};

/// @brief Totals of PSIS-LOO and their observation-level contributions.
/// @par Example
/// @code
/// dm::bayes::PsisLooResult r;
/// r.elpd_loo = -50;
/// std::println("{}", -2 * r.elpd_loo); // 100
/// @endcode
struct PsisLooResult {
    double elpd_loo{};      ///< Sum of the pointwise elpd.
    double se_elpd_loo{};   ///< sqrt(n) times the SD of the pointwise elpd.
    double p_loo{};         ///< Effective number of parameters, lpd - elpd_loo.
    double se_p_loo{};      ///< Standard error of p_loo.
    double looic{};         ///< -2 elpd_loo.
    double se_looic{};      ///< 2 se_elpd_loo.
    double lpd{};           ///< Sum of the in-sample log pointwise predictive densities.
    double k_threshold{};   ///< pareto_k_threshold() for the number of draws.
    std::size_t high_k{};   ///< Observations with k-hat above the threshold.
    std::vector<PsisLooPoint> pointwise; ///< Contributions in observation order.
};

/// @brief Approximate leave-one-out cross-validation by PSIS (Vehtari, Gelman and Gabry 2017).
///
/// For each observation the log importance ratios are \f$-\log p(y_i\mid\theta^{(s)})\f$. After
/// Pareto smoothing, \f$\widehat{\mathrm{elpd}}_i=\log\sum_s w_{is}p(y_i\mid\theta^{(s)})\f$ with
/// normalised weights. Totals and standard errors follow `loo::loo`. Observations with k-hat above
/// the threshold need exact refitting or moment matching.
/// @param log_likelihood Draws by rows and observations by columns (finite, at least two draws).
/// @param relative_efficiency Per-observation r_eff, or empty for 1.
/// @return Totals, standard errors and pointwise diagnostics.
/// @throws std::invalid_argument For too few draws, ragged or empty rows, nonfinite values, or an
/// r_eff vector of the wrong length.
/// @par Example
/// @code
/// std::vector<std::vector<double>> ll(400, std::vector<double>(3));
/// for (int s = 0; s < 400; ++s)
///     for (int i = 0; i < 3; ++i)
///         ll[s][i] = -1 - 0.1 * std::cos(0.7 * s + i);
/// const auto r = dm::bayes::psis_loo(ll);
/// std::println("{} {}", r.pointwise.size(), r.elpd_loo < r.lpd); // 3 true
/// @endcode
PsisLooResult psis_loo(const std::vector<std::vector<double>>& log_likelihood,
                       const std::vector<double>& relative_efficiency = {});

/// @brief Paired comparison of two PSIS-LOO fits on the same observations.
/// @par Example
/// @code
/// dm::bayes::LooComparison c{.elpd_diff = -3, .se_diff = 1.5};
/// std::println("{}", c.elpd_diff / c.se_diff); // -2
/// @endcode
struct LooComparison {
    double elpd_diff{}; ///< elpd_loo(second) - elpd_loo(first).
    double se_diff{};   ///< sqrt(n) times the SD of the pointwise differences.
};

/// @brief Difference in expected log predictive density between two models, as `loo::loo_compare`.
/// @param first PSIS-LOO of the reference model.
/// @param second PSIS-LOO of the other model on the same observations.
/// @return The difference second - first and its standard error.
/// @throws std::invalid_argument When the observation counts differ or are below 2.
/// @par Example
/// @code
/// dm::bayes::PsisLooResult a, b;
/// a.pointwise = {{.elpd = -1}, {.elpd = -2}};
/// b.pointwise = {{.elpd = -1.5}, {.elpd = -2.5}};
/// std::println("{}", dm::bayes::loo_compare(a, b).elpd_diff); // -1
/// @endcode
LooComparison loo_compare(const PsisLooResult& first, const PsisLooResult& second);

} // namespace datamunge::bayes
