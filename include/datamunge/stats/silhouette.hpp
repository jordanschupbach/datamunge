#pragma once

/// @file silhouette.hpp
/// @brief Per-observation and per-cluster silhouette widths from dissimilarities.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Assigned and neighboring cluster and the dissimilarities underlying one silhouette width.
/// @par Example
/// @code
/// const dm::stats::SilhouetteObservation row;
/// std::println("{}", row.width); // 0
/// @endcode
struct SilhouetteObservation {
    std::size_t cluster{}; ///< Original input cluster label.
    std::size_t neighbor{}; ///< Original closest other cluster label, first in ascending order for a tie.
    double within_mean{}; ///< Mean dissimilarity to other members of the assigned cluster, zero for a singleton.
    double neighbor_mean{}; ///< Minimum mean dissimilarity to all members of another cluster.
    double width{}; ///< Difference of the means divided by their maximum, zero for a singleton or zero denominator.
};

/// @brief Mean silhouette width within one cluster.
/// @par Example
/// @code
/// const dm::stats::SilhouetteCluster cluster{7,2,.5};
/// std::println("{}", cluster.count); // 2
/// @endcode
struct SilhouetteCluster {
    std::size_t label{}; ///< Original cluster label.
    std::size_t count{}; ///< Number of observations assigned to this cluster.
    double mean_width{}; ///< Average silhouette width of those observations.
};

/// @brief Silhouette widths in observation order and summaries in ascending label order.
/// @par Example
/// @code
/// const dm::stats::SilhouetteResult result;
/// std::println("{}", result.observations.size()); // 0
/// @endcode
struct SilhouetteResult {
    std::vector<SilhouetteObservation> observations; ///< Width and underlying means for each row.
    std::vector<SilhouetteCluster> clusters; ///< Count and mean width per cluster, sorted by label.
    double mean_width{}; ///< Average over individual observations.
};

/// @brief Computes silhouette widths from an observed partition and pairwise dissimilarities.
/// The mean within a cluster excludes the observation itself. A singleton
/// has width zero, including when every cluster is a singleton. Equal zero
/// within- and between-cluster means also give width zero.
/// @param dissimilarities Finite nonnegative square symmetric matrix with a zero diagonal.
/// @param labels Labels in row order, with at least two distinct labels.
/// @return Observation widths, nearest alternative clusters, and mean widths.
/// @throws std::invalid_argument For invalid dissimilarities or labels.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> d(3,3,{0,1,4,1,0,3,4,3,0});
/// const auto result=dm::stats::silhouette_analysis(d,{2,2,8});
/// std::println("{:.2f}", result.observations[0].width); // 0.75
/// @endcode
/// @par References
/// Rousseeuw (1987), https://doi.org/10.1016/0377-0427(87)90125-7
[[nodiscard]] SilhouetteResult silhouette_analysis(
    const linalg::DenseMatrix<double>& dissimilarities, const std::vector<std::size_t>& labels);

} // namespace datamunge::stats
