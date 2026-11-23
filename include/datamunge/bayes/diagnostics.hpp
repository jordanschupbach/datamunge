#pragma once

/// @file diagnostics.hpp
/// @brief MCMC convergence diagnostics and posterior summaries: effective sample size,
///        split \f$\hat R\f$ and the energy Bayesian fraction of missing information.
///
/// A set of chains is passed as `std::vector<std::vector<double>>`, one inner vector of
/// draws of a single scalar parameter per chain. draws_of() extracts that layout from the
/// row-per-draw `samples` returned by HMC, NUTS and the other samplers.

#include <datamunge/bayes/nuts.hpp>

#include <concepts>
#include <cstddef>
#include <functional>
#include <vector>

namespace datamunge::bayes {

/// @brief Column @p index of row-per-draw samples (one row per draw, one column per parameter).
/// @param samples Draws as returned in NUTSResult::samples and similar.
/// @param index Parameter index.
/// @return The draws of that parameter, in order.
/// @throws std::out_of_range if a row is shorter than @p index + 1.
std::vector<double> draws_of(const std::vector<std::vector<double>>& samples, std::size_t index);

/// @brief Componentwise mean of row-per-draw samples.
/// @param samples Draws, all rows the same length.
/// @return The mean vector.
/// @throws std::invalid_argument for no draws or ragged rows.
std::vector<double> draws_mean(const std::vector<std::vector<double>>& samples);

/// @brief Sample covariance matrix (divisor \f$n-1\f$) of row-per-draw samples, e.g. to build an
///        importance-sampling proposal from a pilot run.
/// @param samples At least 2 draws, all rows the same length.
/// @return The \f$d\times d\f$ covariance as nested vectors.
/// @throws std::invalid_argument for fewer than 2 draws or ragged rows.
std::vector<std::vector<double>> draws_covariance(const std::vector<std::vector<double>>& samples);

/// @brief Draws of parameter @p index from every chain, in the chains layout used below.
/// @param chains Results of NUTS::sample_chains().
/// @param index Parameter index.
/// @return One vector of draws per chain.
std::vector<std::vector<double>> chains_of(const std::vector<NUTSResult>& chains, std::size_t index);

/// @brief Evaluates a scalar derived quantity at every retained NUTS draw.
/// @tparam Transform Callable returning a value convertible to double.
/// @param chains Results of NUTS::sample_chains().
/// @param transform Mapping called once per draw, with a const parameter vector.
/// @return One vector per chain, preserving chain and within-chain order, including empty chains.
/// @throws std::bad_alloc If output allocation fails. Exceptions from the callable propagate.
/// @par Example
/// @code
/// const std::vector<dm::bayes::NUTSResult> chains{{.samples = {{1, 2}, {3, 5}}}};
/// const auto delta = dm::bayes::chains_of(chains, [](const auto& q) { return q[1] - q[0]; });
/// std::print("{}", delta[0][1]); // 2
/// @endcode
/// @see summarize_draws(), draws_of()
template <class Transform>
    requires requires(Transform& transform, const std::vector<double>& draw) {
        { std::invoke(transform, draw) } -> std::convertible_to<double>;
    }
std::vector<std::vector<double>> chains_of(const std::vector<NUTSResult>& chains, Transform&& transform) {
    std::vector<std::vector<double>> result;
    result.reserve(chains.size());
    for (const auto& chain : chains) {
        auto& values = result.emplace_back();
        values.reserve(chain.samples.size());
        for (const auto& draw : chain.samples) {
            values.push_back(std::invoke(transform, draw));
        }
    }
    return result;
}

/// @brief Multi-chain effective sample size (Geyer's initial monotone sequence estimator, as
///        in Stan and Gelman et al. 2013, BDA3 §11.5).
///
/// With \f$M\f$ chains of \f$N\f$ draws, within-chain variance \f$W\f$, and
/// \f$\widehat{\mathrm{var}}^+=\tfrac{N-1}{N}W+B/N\f$, the lag-\f$t\f$ autocorrelation is
/// \f$\hat\rho_t=1-(W-\overline{\gamma_t})/\widehat{\mathrm{var}}^+\f$ where
/// \f$\overline{\gamma_t}\f$ averages the chains' autocovariances. Pair sums
/// \f$P_k=\hat\rho_{2k}+\hat\rho_{2k+1}\f$ are accumulated while positive and forced to be
/// non-increasing, and \f$\hat N_{\text{eff}}=MN/(-1+2\sum_k P_k)\f$.
/// @param chains At least one chain; all chains the same length, at least 4 draws.
/// @return The effective sample size (can exceed MN for antithetic chains).
/// @throws std::invalid_argument for empty, ragged or too-short chains.
double effective_sample_size(const std::vector<std::vector<double>>& chains);

/// @brief Split \f$\hat R\f$ (potential scale reduction, BDA3 §11.4).
///
/// Each chain is cut in half, giving \f$2M\f$ chains of \f$n=\lfloor N/2\rfloor\f$ draws, and
/// \f$\hat R=\sqrt{\widehat{\mathrm{var}}^+/W}\f$ with
/// \f$\widehat{\mathrm{var}}^+=\tfrac{n-1}{n}W+B/n\f$. Values near 1 (below about 1.01) indicate
/// the chains agree with each other and with themselves over time.
/// @param chains At least one chain; all chains the same length, at least 4 draws.
/// @return Split \f$\hat R\f$.
/// @throws std::invalid_argument for empty, ragged or too-short chains.
double split_rhat(const std::vector<std::vector<double>>& chains);

/// @brief Location and scale components of rank-normalized split R-hat.
/// @par Example
/// @code
/// const auto result = dm::bayes::rank_normalized_rhat({{1, 2, 3, 4}, {4, 3, 2, 1}});
/// std::println("{:.3f}", result.maximum); // 1.619
/// @endcode
/// @see rank_normalized_rhat()
struct RankNormalizedRhat {
    double rank_normalized; ///< R-hat of normal scores of pooled split-chain ranks.
    double folded;          ///< R-hat of normal scores of absolute deviations from the pooled median.
    double maximum;         ///< Larger component, or NaN if either component is undefined.
};

/// @brief Diagnoses differences in chain location and scale with split rank-normalized R-hat.
/// Cuts off the middle draw when chain lengths are odd. Ranks pool the retained
/// draws and average ties. The folded component uses absolute deviations from
/// the pooled median of the original chains, then splits and ranks.
/// @param chains One or more equal-length chains of at least four finite draws.
/// @return Location, folded and maximum R-hat. A constant transformed sample gives
/// NaN for that component. Positive between-chain variance with zero within-chain
/// variance gives infinity.
/// @throws std::invalid_argument If the chains are empty, too short, ragged or contain nonfinite draws.
/// @par Example
/// @code
/// const auto result = dm::bayes::rank_normalized_rhat({{0, 0, 0, 0}, {1, 1, 1, 1}});
/// std::println("{}", std::isinf(result.rank_normalized)); // true
/// @endcode
/// @par References
/// - Vehtari et al. (2021), sections 3 and 4, https://doi.org/10.1214/20-BA1221
/// @see split_rhat(), bulk_tail_effective_sample_size()
RankNormalizedRhat rank_normalized_rhat(const std::vector<std::vector<double>>& chains);

/// @brief Rank-normalized bulk ESS and quantile-specific tail ESS.
/// @par Example
/// @code
/// const auto result = dm::bayes::bulk_tail_effective_sample_size({{1, 2, 3, 4, 5, 6}});
/// std::println("{}", result.tail == std::min(result.lower_tail, result.upper_tail)); // true
/// @endcode
/// @see bulk_tail_effective_sample_size()
struct BulkTailEffectiveSampleSize {
    double bulk;       ///< Effective sample size of rank-normalized split chains.
    double lower_tail; ///< Effective sample size for the lower quantile indicator.
    double upper_tail; ///< Effective sample size for the upper quantile indicator.
    double tail;       ///< Smaller tail estimate, or NaN if either tail estimate is undefined.
};

/// @brief Estimates central and tail sampling efficiency using split chains.
/// Bulk ESS uses pooled normal scores of the retained draws. The tail estimates
/// use inclusive indicators below and above the respective type-7 pooled quantiles
/// of the original draws. The estimates use Geyer's monotone paired autocorrelations
/// with an even-lag endpoint adjustment and can exceed the retained draw count.
/// @param chains One or more equal-length chains of at least six finite draws.
/// The middle draw of each odd-length chain is omitted when computing autocorrelations.
/// @param tail_probability Lower tail quantile probability, strictly between zero and one half.
/// The upper quantile is at one minus this probability.
/// @return Bulk and both tail estimates. A constant transformed sample gives NaN.
/// @throws std::invalid_argument Invalid chains or tail probability.
/// @throws std::length_error A chain exceeds the autocovariance FFT size limit.
/// @par Example
/// @code
/// const auto result = dm::bayes::bulk_tail_effective_sample_size({{1, 2, 3, 4, 5, 6}});
/// std::println("{:.3f}", result.bulk); // 4.669
/// @endcode
/// @par References
/// - Vehtari et al. (2021), sections 3 and 4, https://doi.org/10.1214/20-BA1221
/// @see effective_sample_size(), rank_normalized_rhat()
BulkTailEffectiveSampleSize bulk_tail_effective_sample_size(
    const std::vector<std::vector<double>>& chains, double tail_probability = 0.05);

/// @brief Energy Bayesian fraction of missing information (Betancourt 2016):
///        \f$\sum_n(E_n-E_{n-1})^2/\sum_n(E_n-\bar E)^2\f$.
///
/// Small values (below about 0.3) mean momentum resampling moves the chain between energy
/// levels too slowly to explore heavy tails.
/// @param energy Per-draw Hamiltonians, e.g. NUTSResult::energy.
/// @return E-BFMI.
/// @throws std::invalid_argument for fewer than 2 values or constant energy.
double energy_bfmi(const std::vector<double>& energy);

/// One-parameter posterior summary across chains.
struct PosteriorSummary {
    /// @brief Posterior mean.
    double mean{0.0};
    /// @brief Posterior standard deviation.
    double sd{0.0};
    /// @brief 5% quantile.
    double q05{0.0};
    /// @brief Median.
    double q50{0.0};
    /// @brief 95% quantile.
    double q95{0.0};
    /// @brief Monte Carlo standard error of the mean, sd / sqrt(ess).
    double mcse{0.0};
    /// @brief effective_sample_size() of the pooled chains.
    double ess{0.0};
    /// @brief split_rhat() of the chains.
    double rhat{0.0};
};

/// @brief Mean, sd, 5/50/95% quantiles, MCSE, ESS and split \f$\hat R\f$ of one parameter.
/// @param chains Draws of the parameter, one vector per chain (see chains_of()).
/// @return The summary.
/// @throws std::invalid_argument as effective_sample_size().
PosteriorSummary summarize_draws(const std::vector<std::vector<double>>& chains);

/// A histogram scaled as a probability density.
struct DensityHistogram {
    /// @brief Bin centres.
    std::vector<double> centers;
    /// @brief Density in each bin: count / (total draws x bin width).
    std::vector<double> density;
    /// @brief Common bin width.
    double width{0.0};
};

/// @brief Equal-width density histogram of draws over [@p lo, @p hi], for overlaying an exact
///        density curve. Draws outside the range are not binned but still count in the total,
///        so the bars integrate to the fraction of draws inside the range.
/// @param draws Draws (pool chains first if needed).
/// @param lo Left edge.
/// @param hi Right edge (greater than @p lo).
/// @param bins Number of bins (at least 1).
/// @return Centres, densities and width.
/// @throws std::invalid_argument for empty draws, @p hi <= @p lo or zero bins.
DensityHistogram posterior_histogram(const std::vector<double>& draws, double lo, double hi, std::size_t bins);

/// @brief One observation's contribution to the widely applicable information criterion.
/// @par Example
/// @code
/// const auto point = dm::bayes::waic({{-2}, {-2}}).pointwise.front();
/// std::println("{} {}", point.log_predictive_density, point.effective_parameters); // -2 0
/// @endcode
/// @see waic()
struct WaicPoint {
    double log_predictive_density; ///< Log of the posterior mean likelihood for this observation.
    double effective_parameters;   ///< Sample variance of the observation's log likelihood across draws.
    double elpd;                   ///< Log predictive density minus the effective parameter contribution.
};

/// @brief Summed predictive scores and their observation-level contributions.
/// @par Example
/// @code
/// const auto result = dm::bayes::waic({{-1, -3}, {-1, -3}});
/// std::println("{} {}", result.elpd, result.waic); // -4 8
/// @endcode
/// @see datamunge::bayes::waic()
struct WaicResult {
    double log_predictive_density;   ///< Sum of the observation-level log predictive densities.
    double effective_parameters;     ///< Sum of the variance-based effective parameter contributions.
    double elpd;                     ///< Estimated expected log pointwise predictive density.
    double waic;                     ///< Deviance-scale score, equal to minus twice elpd.
    std::vector<WaicPoint> pointwise; ///< Contributions in input observation order.
};

/// @brief Estimates expected log pointwise predictive density by variance-based WAIC.
///
/// Computes a maximum-shifted log mean likelihood and subtracts the sample variance
/// of log likelihoods across posterior draws (divisor S - 1) for each observation.
/// Columns must correspond to the predictive units being scored. The result exposes
/// pointwise variances so callers can identify contributions above 0.4, where the
/// WAIC approximation can be unreliable.
/// @param log_likelihood At least two equally weighted posterior draws by rows and at
/// least one observation by columns. All entries must be finite log likelihoods.
/// @return Owned pointwise scores and totals, including WAIC on the deviance scale.
/// @throws std::invalid_argument Too few draws, empty or ragged rows, or any nonfinite log likelihood.
/// @throws std::overflow_error A score cannot be represented as a finite double.
/// @par Example
/// @code
/// const auto result = dm::bayes::waic({{-1000}, {-1002}});
/// std::println("{:.3f} {:.3f}", result.effective_parameters, result.waic); // 2.000 2005.132
/// @endcode
/// @par References
/// - Vehtari, Gelman and Gabry (2017), section 2.2, https://doi.org/10.1007/s11222-016-9696-4
/// @see WaicResult, WaicPoint
WaicResult waic(const std::vector<std::vector<double>>& log_likelihood);

} // namespace datamunge::bayes
