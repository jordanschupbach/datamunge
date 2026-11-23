#pragma once

/// @file consensus_clustering.hpp
/// @brief Pairwise co-assignment estimates from repeated row subsampling.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/kmeans.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief Controls for fixed-count clustering of subsamples.
/// @par Example
/// @code
/// const dm::stats::ConsensusClusteringOptions options{.subsamples=25};
/// std::println("{}", options.subsamples); // 25
/// @endcode
struct ConsensusClusteringOptions {
    std::size_t clusters{2};    ///< Requested number of clusters in each subsample.
    std::size_t subsamples{100}; ///< Number of independent subsample draws.
    double sample_fraction{.8}; ///< Fraction of rows retained without replacement, rounded down.
    std::uint64_t seed{1};      ///< Seed for the local subsampling stream.
};

/// @brief Pairwise observation and co-assignment counts and their ratio.
/// @par Example
/// @code
/// const dm::stats::ConsensusClusteringResult result;
/// std::println("{}", result.probability.rows()); // 0
/// @endcode
struct ConsensusClusteringResult {
    linalg::DenseMatrix<std::uint64_t> co_sampled; ///< Number of draws containing both observations.
    linalg::DenseMatrix<std::uint64_t> co_assigned; ///< Number of those draws giving them equal labels.
    linalg::DenseMatrix<double> probability;       ///< Co-assignment conditional on drawing both, NaN when never co-sampled.
    std::size_t sample_size{};                    ///< Number of rows drawn per subsample.
};

/// @brief Returns one label per sampled row with the requested number of nonempty clusters.
using ConsensusPartitioner = std::function<std::vector<std::size_t>(
    const linalg::DenseMatrix<double>& observations, std::size_t clusters)>;

/// @brief Estimates consensus co-assignment using k-means on independent subsamples.
/// @param observations Finite observations in rows, at least one feature column.
/// @param options Cluster count, subsampling and random seed settings.
/// @param clustering K-means controls, with its cluster count set by options.clusters.
/// @return Symmetric count and probability matrices in original row order.
/// @throws std::invalid_argument For invalid geometry or controls.
/// @throws std::length_error If the pairwise matrices exceed storage capacity.
/// @throws std::domain_error If k-means produces an empty cluster in a subsample.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,{0,.1,5,5.1});
/// const auto fit=dm::stats::consensus_clustering(x,{.subsamples=3,.sample_fraction=1});
/// std::println("{:.0f}", fit.probability(0,1)); // 1
/// @endcode
/// @par References
/// Monti, Tamayo, Mesirov and Golub (2003), https://doi.org/10.1023/A:1023949509487
[[nodiscard]] ConsensusClusteringResult consensus_clustering(
    const linalg::DenseMatrix<double>& observations, const ConsensusClusteringOptions& options = {},
    const KMeansOptions& clustering = {});

/// @brief Estimates consensus co-assignment for a caller-supplied clustering method.
/// Each subsample contains distinct rows in their original order. Numeric
/// cluster labels can change between draws because only equality within
/// each draw matters. A pair omitted from every draw has no estimate.
/// @param observations Finite observations in rows, at least one feature column.
/// @param partition Function called once per subsample in original coordinates.
/// @param options Cluster count, subsampling and random seed settings.
/// @return Symmetric count and probability matrices in original row order.
/// @throws std::invalid_argument For invalid controls, observations or label vector length.
/// @throws std::length_error If the pairwise matrices exceed storage capacity.
/// @throws std::domain_error If a subsample partition has the wrong number of nonempty clusters.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,{0,.1,5,5.1});
/// const dm::stats::ConsensusPartitioner split=[](const auto&,std::size_t) {
///     return std::vector<std::size_t>{3,3,8,8};
/// };
/// const auto fit=dm::stats::consensus_clustering(x,split,{.subsamples=3,.sample_fraction=1});
/// std::println("{}", fit.co_sampled(0,3)); // 3
/// @endcode
[[nodiscard]] ConsensusClusteringResult consensus_clustering(const linalg::DenseMatrix<double>& observations,
    const ConsensusPartitioner& partition, const ConsensusClusteringOptions& options);

} // namespace datamunge::stats
