#pragma once

/// @file spectral_clustering.hpp
/// @brief Row-normalized spectral partitioning of symmetric affinity graphs.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/kmeans.hpp>

namespace datamunge::stats {

/// @brief Selected Laplacian eigenvalues, embedding and cluster assignments.
/// @par Example
/// @code
/// const dm::stats::SpectralClusteringFit fit;
/// std::println("{}", fit.labels.size()); // 0
/// @endcode
struct SpectralClusteringFit {
    std::vector<double> eigenvalues; ///< Selected eigenvalues in increasing order, including trivial zero modes.
    linalg::DenseMatrix<double> embedding; ///< Row-normalized selected eigenvectors, vertices by eigenvectors.
    std::vector<std::size_t> labels; ///< Zero-based cluster labels in vertex order, numbered by first occurrence.
    std::size_t connected_components{}; ///< Count of graph components, including isolated vertices.
};

/// @brief Clusters a weighted graph with the symmetric normalized Laplacian.
/// Uses its smallest k eigenvectors, normalizes each row, and runs k-means
/// in the embedding. The number k comes from options.n_clusters. Graph
/// components give an exact assignment when their count equals k. An
/// isolated vertex is a zero mode of the normalized Laplacian.
/// @param affinity Nonempty square, nonnegative finite weights, symmetric with zero diagonal.
/// @param options Requested cluster count and k-means restart and stopping settings.
/// @return Selected eigenvalues, normalized embedding, and assignments.
/// @throws std::invalid_argument For invalid weights, controls or more components than clusters.
/// @throws std::length_error If dense workspaces exceed storage capacity.
/// @throws std::runtime_error If the eigensolver or partition fails.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> w(2,2,{0,1,1,0});
/// const auto fit=dm::stats::fit_spectral_clustering(w,{.n_clusters=1});
/// std::println("{}", fit.labels[1]); // 0
/// @endcode
/// @par References
/// Ng, Jordan and Weiss (2001), https://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm
[[nodiscard]] SpectralClusteringFit fit_spectral_clustering(
    const linalg::DenseMatrix<double>& affinity, const KMeansOptions& options);

/// @brief Builds an unweighted symmetric nearest-neighbor graph and clusters its embedding.
/// Connects two rows when either row is among the other's nearest neighbors,
/// using Euclidean distance and row order to resolve distance ties.
/// @param observations Rows of finite numeric features.
/// @param neighbors Positive neighbor count strictly below the number of rows.
/// @param options Requested cluster count and k-means settings.
/// @return The same outputs as fit_spectral_clustering() on the constructed graph.
/// @throws std::invalid_argument For malformed observations, neighbor count or controls.
/// @throws std::length_error If dense workspaces exceed storage capacity.
/// @throws std::runtime_error If the eigensolver or partition fails.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,{0,1,10,11});
/// const auto fit=dm::stats::fit_spectral_clustering_knn(x,1,{.n_clusters=2});
/// std::println("{} {}", fit.labels[0], fit.labels[3]); // 0 1
/// @endcode
[[nodiscard]] SpectralClusteringFit fit_spectral_clustering_knn(
    const linalg::DenseMatrix<double>& observations, std::size_t neighbors,
    const KMeansOptions& options);

} // namespace datamunge::stats
