#pragma once

/// @file diagnostics.hpp
/// @brief MCMC convergence diagnostics and posterior summaries: effective sample size,
///        split \f$\hat R\f$ and the energy Bayesian fraction of missing information.
///
/// A set of chains is passed as `std::vector<std::vector<double>>`, one inner vector of
/// draws of a single scalar parameter per chain. draws_of() extracts that layout from the
/// row-per-draw `samples` returned by HMC, NUTS and the other samplers.

#include <cstddef>
#include <vector>

namespace datamunge::bayes {

struct NUTSResult;

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

} // namespace datamunge::bayes
