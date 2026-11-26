#pragma once

/// @file sbc.hpp
/// @brief Simulation-based calibration of Bayesian inference algorithms.

#include <datamunge/random/random.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::bayes {

/// @brief Draws a parameter vector from the prior.
using SbcPriorDraw = std::function<std::vector<double>(random::SplitMix64&)>;

/// @brief Simulates a dataset (flattened to doubles) given parameters.
using SbcSimulator = std::function<std::vector<double>(const std::vector<double>&, random::SplitMix64&)>;

/// @brief Fits the model to a dataset and returns posterior draws, one parameter vector per draw.
using SbcFit = std::function<std::vector<std::vector<double>>(const std::vector<double>&, random::SplitMix64&)>;

/// @brief Options for simulation_based_calibration().
/// @par Example
/// @code
/// const dm::bayes::SbcOptions o{.replications = 200};
/// std::println("{}", o.bins); // 20
/// @endcode
struct SbcOptions {
    std::size_t replications{1000};   ///< Prior draws N, each with a simulated dataset and a refit.
    std::size_t bins{20};             ///< Histogram bins for the chi-squared test (must divide L + 1 evenly or nearly).
    std::size_t ecdf_simulations{2000}; ///< Uniform-rank simulations used to calibrate the ECDF test.
    std::uint64_t seed{1};            ///< Seed of all randomness.
};

/// @brief Rank diagnostics of one parameter (or derived quantity).
/// @par Example
/// @code
/// dm::bayes::SbcParameter p;
/// p.ranks = {0, 3, 7};
/// std::println("{}", p.ranks.size()); // 3
/// @endcode
struct SbcParameter {
    std::vector<std::size_t> ranks;     ///< Rank of the true value among the L posterior draws (0..L).
    std::vector<std::size_t> histogram; ///< Counts of ranks in `bins` equal-width bins of 0..L.
    double chi_squared{};               ///< Pearson statistic against equal expected counts.
    double chi_squared_p_value{};       ///< Upper chi-squared(bins - 1) p-value.
    double ecdf_max_deviation{};        ///< max_z |F_N(z) - z| over the scaled ranks' ECDF.
    double ecdf_p_value{};              ///< Monte Carlo p-value of the deviation under discrete uniform ranks.
    double mean_rank_fraction{};        ///< Mean of rank / L; 0.5 when calibrated, <0.5 if overestimating.
};

/// @brief Output of simulation_based_calibration().
/// @par Example
/// @code
/// dm::bayes::SbcResult r;
/// r.posterior_draws = 99;
/// std::println("{}", r.posterior_draws + 1); // 100
/// @endcode
struct SbcResult {
    std::vector<SbcParameter> parameters; ///< One entry per parameter.
    std::size_t posterior_draws{};        ///< L, the number of posterior draws per fit.
    std::size_t failures{};               ///< Replications whose fit threw (skipped).
};

/// @brief Simulation-based calibration (Talts et al. 2018).
///
/// For each replication, \f$\tilde\theta\sim p(\theta)\f$, \f$\tilde y\sim p(y\mid\tilde\theta)\f$ and L
/// posterior draws \f$\theta_1,\dots,\theta_L\f$ from the algorithm applied to \f$\tilde y\f$; the
/// rank of \f$\tilde\theta_j\f$ is the number of draws below it (ties split at random). If the
/// algorithm samples the exact posterior, each rank is uniform on {0, ..., L}. A smile-shaped
/// histogram indicates underdispersed posteriors, a frown overdispersed ones, and a slope bias. Two
/// tests are reported per parameter: Pearson's chi-squared over equal-width bins, and the maximum
/// absolute deviation of the rank ECDF from uniform, calibrated by simulating discrete uniform ranks.
/// The posterior draws should be (nearly) independent; thin MCMC output to about its ESS first.
/// @param prior Prior sampler.
/// @param simulate Data simulator.
/// @param fit Inference algorithm under test.
/// @param options Replications, bins and seed.
/// @return Ranks, histograms and uniformity tests per parameter.
/// @throws std::invalid_argument For bad options, or fits returning draws of inconsistent count or
/// dimension.
/// @throws std::runtime_error When every replication's fit fails.
/// @par Example
/// Exact conjugate posterior of a normal mean: ranks are uniform.
/// @code
/// const auto r = dm::bayes::simulation_based_calibration(
///     [](dm::random::SplitMix64& g) { return std::vector<double>{g.normal()}; },
///     [](const std::vector<double>& t, dm::random::SplitMix64& g) {
///         return std::vector<double>{t[0] + g.normal()};
///     },
///     [](const std::vector<double>& y, dm::random::SplitMix64& g) {
///         std::vector<std::vector<double>> d;
///         for (int l = 0; l < 99; ++l)
///             d.push_back({y[0] / 2 + std::sqrt(0.5) * g.normal()});
///         return d;
///     },
///     {.replications = 500});
/// std::println("{}", r.parameters[0].chi_squared_p_value > 0.001); // true
/// @endcode
SbcResult simulation_based_calibration(const SbcPriorDraw& prior, const SbcSimulator& simulate, const SbcFit& fit,
                                       const SbcOptions& options = {});

} // namespace datamunge::bayes
