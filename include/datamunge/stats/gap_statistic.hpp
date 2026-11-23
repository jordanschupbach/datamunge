#pragma once

/// @file gap_statistic.hpp
/// @brief Simulated reference calibration of within-cluster dispersion.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/kmeans.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief Orientation of the uniform reference box.
enum class GapReference {
    PrincipalComponentBox, ///< Uniform along observed principal axes, with zero-variance axes dropped.
    AxisAlignedBox          ///< Uniform along original feature axes.
};

/// @brief Candidate count, reference simulation size and reproducibility controls.
/// @par Example
/// @code
/// const dm::stats::GapStatisticOptions options{.max_clusters=4, .reference_samples=40};
/// std::println("{}", options.reference_samples); // 40
/// @endcode
struct GapStatisticOptions {
    std::size_t max_clusters{8};         ///< Largest candidate, below the number of observations.
    std::size_t reference_samples{50};   ///< Independent null datasets, at least two.
    std::uint64_t seed{1};              ///< Seed for the uniform reference draws.
    GapReference reference{GapReference::PrincipalComponentBox}; ///< Null bounding box orientation.
};

/// @brief Observed dispersion, simulated log dispersion and gap at one candidate count.
/// @par Example
/// @code
/// const dm::stats::GapStatisticEntry entry;
/// std::println("{}", entry.clusters); // 0
/// @endcode
struct GapStatisticEntry {
    std::size_t clusters{}; ///< Number of clusters in this row.
    double log_dispersion{}; ///< Log of the observed within-cluster sum of squares.
    double reference_mean_log_dispersion{}; ///< Mean log within-cluster sum of squares under the null.
    double gap{}; ///< Null mean minus observed log dispersion.
    double simulation_error{}; ///< Sample standard deviation of null log dispersion times sqrt(1 + 1/B).
};

/// @brief Gap curve and the count chosen by the adjacent one-standard-error rule.
/// @par Example
/// @code
/// const dm::stats::GapStatisticResult result;
/// std::println("{}", result.entries.size()); // 0
/// @endcode
struct GapStatisticResult {
    std::vector<GapStatisticEntry> entries; ///< Rows for consecutive counts starting at one.
    std::size_t selected_clusters{};       ///< First k with gap(k) >= gap(k+1) - simulation_error(k+1), or the largest k.
};

/// @brief Partitions observations in input order using the requested number of nonempty clusters.
/// Labels may be any distinct identifiers. Data passed to the callback use the original feature coordinates.
using GapPartitioner = std::function<std::vector<std::size_t>(
    const linalg::DenseMatrix<double>& observations, std::size_t clusters)>;

/// @brief Calibrates k-means dispersion against simulated uniform reference datasets.
/// @param observations Finite numeric features in rows, at least two observations.
/// @param options Simulation settings and maximum cluster count.
/// @param clustering K-means settings. Counts are set by the candidate being evaluated.
/// @return Observed and simulated gaps and the selected cluster count.
/// @throws std::invalid_argument On invalid observations or controls.
/// @throws std::domain_error If a partition has empty clusters or its dispersion is zero.
/// @throws std::runtime_error If the reference-box decomposition fails.
/// @throws std::overflow_error If a reference observation falls outside the finite value range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(6,1,{-3,-2.9,-2.8,3,3.1,3.2});
/// const auto gap=dm::stats::gap_statistic(x,{.max_clusters=2,.reference_samples=15});
/// std::println("{}", gap.selected_clusters); // 2
/// @endcode
/// @par References
/// Tibshirani, Walther and Hastie (2001), https://doi.org/10.1111/1467-9868.00293
[[nodiscard]] GapStatisticResult gap_statistic(const linalg::DenseMatrix<double>& observations,
    const GapStatisticOptions& options = {}, const KMeansOptions& clustering = {});

/// @brief Computes the gap curve with a caller-supplied partitioning method.
/// Uses squared Euclidean dispersion even when the callback uses a
/// different dissimilarity. The k=1 partition is determined directly.
/// @param observations Observations as in gap_statistic().
/// @param partition Function used on the observations and on each reference dataset.
/// @param options Simulation settings as in gap_statistic().
/// @return Gap curve and count selected with the same rule as gap_statistic().
/// @throws std::invalid_argument On invalid inputs or callback output length.
/// @throws std::domain_error If the callback returns the wrong number of labels or dispersion is zero.
/// @throws std::runtime_error If the reference-box decomposition fails.
/// @throws std::overflow_error If a reference observation falls outside the finite value range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,{-5,-4,4,5});
/// const dm::stats::GapPartitioner split=[](const auto& rows,std::size_t) {
///     std::vector<std::size_t> labels(rows.rows());
///     for (std::size_t i=0; i<rows.rows(); ++i) labels[i]=i/2;
///     return labels;
/// };
/// const auto gap=dm::stats::gap_statistic(x,split,{.max_clusters=2,.reference_samples=8});
/// std::println("{}", gap.entries.size()); // 2
/// @endcode
[[nodiscard]] GapStatisticResult gap_statistic(
    const linalg::DenseMatrix<double>& observations, const GapPartitioner& partition,
    const GapStatisticOptions& options);

} // namespace datamunge::stats
