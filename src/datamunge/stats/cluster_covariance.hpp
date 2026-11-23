#pragma once

/// @file cluster_covariance.hpp
/// @brief Inclusion-exclusion for clustered coefficient-space scores.

#include <cstdint>
#include <vector>

namespace datamunge::stats {

struct ClusterCovarianceResult;

namespace detail {

/// @brief Accumulates the clustered sandwich covariance from scores with the bread already applied.
/// @param scores Nonempty fitted rows with one value per coefficient.
/// @param labels One clustering label per score row in each dimension.
/// @param residual_df Fitted model residual degrees of freedom.
/// @param finite_sample_adjustment Apply a separate group-count correction to each component.
/// @return Covariance and the groups, counts, and multiplier for each cluster subset.
ClusterCovarianceResult cluster_covariance_from_scores(const std::vector<std::vector<long double>>& scores,
    const std::vector<std::vector<std::int64_t>>& labels, std::size_t residual_df, bool finite_sample_adjustment);

} // namespace detail
} // namespace datamunge::stats
