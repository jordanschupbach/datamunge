#pragma once
/// @file bootstrap_designs.hpp
/// @brief Bootstrap designs that preserve strata, clusters or serial dependence, plus null simulation.

#include <cstddef>
#include <functional>
#include <vector>

#include <datamunge/random/random.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>
#include <datamunge/stats/resample_types.hpp>

namespace datamunge::stats {

/// @brief Draws one bootstrap sample with replacement separately within each stratum.
///
/// Result position i draws a source row with the same stratum as original row i.
/// Each stratum retains its observed size, including interleaved strata.
/// @param strata One label per source row, with at least one row. Labels need not be contiguous.
/// @param rng Caller-owned generator advanced by the draws.
/// @return Zero-based source row indices, one per original row.
/// @throws std::invalid_argument Empty label vector.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto rows = dm::stats::stratified_bootstrap_indices({10, 20, 10}, rng);
/// std::print("{}\n", rows[1]); // 1
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/boot/html/boot.html
/// @see cluster_bootstrap_indices(), bootstrap_summary()
std::vector<std::size_t> stratified_bootstrap_indices(const std::vector<std::size_t>& strata,
                                                        random::RandomGenerator& rng);

/// @brief Bootstrap source rows with separate boundaries for every sampled cluster copy.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto copy = dm::stats::cluster_bootstrap_indices({7, 8, 7}, rng);
/// std::print("{}\n", copy.cluster_offsets.size()); // 3
/// @endcode
/// @see cluster_bootstrap_indices()
struct ClusterBootstrapSample {
  std::vector<std::size_t> indices;         ///< Source rows in cluster draw order.
  std::vector<std::size_t> cluster_offsets; ///< Starts at zero, ends at indices.size(), and separates copies.
};

/// @brief Draws whole clusters with replacement, using the observed number of clusters.
///
/// Every draw samples one cluster uniformly and copies its rows in their
/// original relative order. The total number of rows can vary between draws.
/// The returned offsets distinguish repeated copies of a source cluster.
/// @param clusters One label per source row, with at least one row.
/// @param rng Caller-owned generator advanced by one cluster choice per draw.
/// @return Zero-based source rows and boundaries between sampled cluster copies.
/// @throws std::invalid_argument Empty label vector.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto copy = dm::stats::cluster_bootstrap_indices({7, 7, 7}, rng);
/// std::print("{}\n", copy.indices.size()); // 3
/// @endcode
/// @par References
/// - C. A. Field and A. H. Welsh (2007), https://doi.org/10.1111/j.1467-9868.2007.00593.x
/// @see stratified_bootstrap_indices()
ClusterBootstrapSample cluster_bootstrap_indices(const std::vector<std::size_t>& clusters,
                                                  random::RandomGenerator& rng);

/// @brief Draws a moving-block or circular-block bootstrap sample of the original series length.
///
/// Draws independent starts for full blocks of consecutive rows. The last
/// block contributes only the prefix needed to reach the original length.
/// @param series_size Positive number of ordered source rows.
/// @param block_size Block length from one through series_size.
/// @param rng Caller-owned generator advanced by the block start draws.
/// @param circular Allow blocks to wrap at the end when true, default false.
/// @return Zero-based indices in resampled time order.
/// @throws std::invalid_argument Invalid series or block length.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto rows = dm::stats::block_bootstrap_indices(5, 2, rng, true);
/// std::print("{}\n", rows.size()); // 5
/// @endcode
/// @par References
/// - H. R. Künsch (1989), https://doi.org/10.1214/aos/1176347265
/// - D. N. Politis and J. P. Romano (1992), https://purl.stanford.edu/xh812zd4638
/// @see stationary_bootstrap_indices()
std::vector<std::size_t> block_bootstrap_indices(std::size_t series_size, std::size_t block_size,
                                                   random::RandomGenerator& rng, bool circular = false);

/// @brief Draws a stationary bootstrap sample by geometric runs with circular continuation.
///
/// The first row is uniform. Each subsequent row restarts at a uniform
/// index with probability 1/mean_block_size or continues to the next row,
/// wrapping at the end. A mean block length of one draws independent rows.
/// @param series_size Positive number of ordered source rows.
/// @param mean_block_size Finite expected block length of at least one.
/// @param rng Caller-owned generator advanced by starts and restart decisions.
/// @return Zero-based indices in resampled time order, of length series_size.
/// @throws std::invalid_argument Invalid series size or mean block length.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto rows = dm::stats::stationary_bootstrap_indices(7, 2.5, rng);
/// std::print("{}\n", rows.size()); // 7
/// @endcode
/// @par References
/// - D. N. Politis and J. P. Romano (1994), https://doi.org/10.1080/01621459.1994.10476870
/// @see block_bootstrap_indices()
std::vector<std::size_t> stationary_bootstrap_indices(std::size_t series_size, double mean_block_size,
                                                        random::RandomGenerator& rng);

/// @brief Generates balanced bootstrap replicates from one shuffled pool of row indices.
///
/// Every source row appears exactly replicates times across all draws.
/// Each returned replicate has sample_size rows drawn from the pooled copies.
/// @param sample_size Positive number of source rows and length of each replicate.
/// @param replicates Positive number of bootstrap replicates.
/// @param rng Caller-owned generator advanced by the shuffle.
/// @return Replicates in draw order with zero-based source row indices.
/// @throws std::invalid_argument Zero sample size or replicate count.
/// @throws std::length_error Product of dimensions exceeds the addressable index range.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto rows = dm::stats::balanced_bootstrap_indices(3, 4, rng);
/// std::print("{} {}\n", rows.size(), rows.front().size()); // 4 3
/// @endcode
/// @par References
/// - A. C. Davison, D. V. Hinkley and E. Schechtman (1986), https://doi.org/10.1093/biomet/73.3.555
/// @see stratified_bootstrap_indices(), bootstrap_summary()
std::vector<std::vector<std::size_t>> balanced_bootstrap_indices(std::size_t sample_size,
                                                                  std::size_t replicates,
                                                                  random::RandomGenerator& rng);

/// @brief Calibrates a statistic by simulating from the fitted null and refitting on every draw.
///
/// Calls simulate and refit_statistic once per replicate. The observed and
/// simulated statistics use the same definition. The p-value counts ties
/// as extreme and uses (1 + extreme)/(1 + draws). Two-sided ordering uses
/// absolute statistics and is intended for a signed statistic centered at zero.
/// @param observed Finite statistic evaluated on the observed data.
/// @param draws Positive number of null simulations.
/// @param simulate Generates a nonempty finite dataset from the fitted null using the supplied generator.
/// @param refit_statistic Repeats the original fitting and statistic calculation on a simulated dataset.
/// @param rng Caller-owned generator used by simulate.
/// @param alternative Tail of the simulated statistic, default greater.
/// @return Observed statistic, simulation count in parameter1, extreme count in parameter2, and calibrated p-value.
/// @throws std::invalid_argument Invalid arguments or invalid generated sample.
/// @throws std::domain_error A refitted statistic is nonfinite.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto test = dm::stats::parametric_bootstrap_test(3, 4,
///     [](auto& stream) { return std::vector{stream.uniform01()}; },
///     [](const auto& sample) { return sample.front(); }, rng);
/// std::print("{:.1f}\n", test.p_value); // 0.2
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/boot/html/boot.html
/// @see wild_bootstrap_test()
HypothesisTestResult parametric_bootstrap_test(
    double observed, std::size_t draws,
    const std::function<std::vector<double>(random::RandomGenerator&)>& simulate,
    const SampleStatistic& refit_statistic, random::RandomGenerator& rng,
    Alternative alternative = Alternative::Greater);

/// @brief Mean-zero, unit-variance weights for wild residual bootstrap tests.
/// @par Example
/// @code
/// std::print("{}\n", dm::stats::WildBootstrapWeight::Rademacher !=
///                    dm::stats::WildBootstrapWeight::Mammen); // true
/// @endcode
/// @see wild_bootstrap_test()
enum class WildBootstrapWeight {
  Rademacher, ///< Equally likely weights -1 and +1.
  Mammen     ///< Asymmetric two-point weights with third moment one.
};

/// @brief Calibrates a fixed-design regression statistic by wild residual resampling.
///
/// Replicate responses are fitted_null + weight * residual, with an
/// independent weight per observation. Supply null-restricted fitted values
/// and residuals, with any leverage adjustment already applied. Capture the
/// fixed design and refit the same statistic in the callback. Calibration
/// uses the same tail ordering and finite simulation correction as parametric_bootstrap_test().
/// @param observed Finite observed statistic.
/// @param fitted_null Nonempty finite fitted responses under the null.
/// @param residuals Finite residuals aligned with fitted_null.
/// @param draws Positive number of bootstrap replicates.
/// @param refit_statistic Repeats model fitting and statistic calculation on the generated response.
/// @param rng Caller-owned generator advanced by the replicate weights.
/// @param weight Multiplier distribution, default Rademacher.
/// @param alternative Tail of the signed statistic, default two-sided.
/// @return Observed statistic, simulation and extreme counts, and calibrated p-value.
/// @throws std::invalid_argument Invalid input or generated response.
/// @throws std::domain_error A refitted statistic is nonfinite.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto test = dm::stats::wild_bootstrap_test(1, {0, 0}, {1, -1}, 4,
///     [](const auto& response) { return response.front(); }, rng);
/// std::print("{:.1f}\n", test.p_value); // 1.0
/// @endcode
/// @par References
/// - E. Mammen (1993), https://doi.org/10.1214/aos/1176349025
/// @see parametric_bootstrap_test(), WildBootstrapWeight
HypothesisTestResult wild_bootstrap_test(
    double observed, const std::vector<double>& fitted_null, const std::vector<double>& residuals,
    std::size_t draws, const SampleStatistic& refit_statistic, random::RandomGenerator& rng,
    WildBootstrapWeight weight = WildBootstrapWeight::Rademacher,
    Alternative alternative = Alternative::TwoSided);

/// @brief Draws a scalar functional from Rubin's Bayesian bootstrap posterior.
///
/// Each replicate draws one Dirichlet(1, ..., 1) weight vector with
/// coordinates in observation order and evaluates the supplied functional
/// once. The callable can capture any data aligned to those coordinates.
/// Duplicate data rows still receive distinct weights.
/// @param observations Number of observed rows, at least one.
/// @param draws Number of posterior draws, at least one.
/// @param weighted_functional Callable accepting the normalized row weights and returning a finite scalar.
/// @param rng Caller-owned generator advanced by the weight draws.
/// @return Posterior draws in generation order.
/// @throws std::invalid_argument Empty callable or zero observations or draws.
/// @throws std::domain_error Nonfinite functional value. Exceptions from the callable propagate.
/// @throws std::length_error A requested dimension exceeds container capacity.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto draws = dm::stats::bayesian_bootstrap(1, 3, [](const auto& weights) { return 2 * weights.front(); }, rng);
/// std::print("{} {}\n", draws.size(), draws.front()); // 3 2
/// @endcode
/// @par References
/// - D. B. Rubin (1981), https://doi.org/10.1214/aos/1176345338
/// @see parametric_bootstrap_test(), datamunge::random::dirichlet_sample()
std::vector<double> bayesian_bootstrap(std::size_t observations, std::size_t draws,
                                       const SampleStatistic& weighted_functional, random::RandomGenerator& rng);

} // namespace datamunge::stats
