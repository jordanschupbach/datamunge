#pragma once

/// @file knn_graph.hpp
/// @brief Symmetric k-nearest-neighbour graphs and their random-walk operator.
///
/// The graph joins points i and j when either is among the other's k nearest neighbours
/// (Euclidean distance), with unit edge weights. Its adjacency matrix \f$W\f$ is symmetric
/// and sparse. The random walk that moves to a uniformly chosen neighbour has transition
/// matrix \f$P=D^{-1}W\f$, where \f$D\f$ holds the degrees. \f$P\f$ is not symmetric, but it
/// is similar to the symmetric \f$S=D^{-1/2}WD^{-1/2}\f$, so its eigenvalues are real and lie
/// in \f$[-1,1]\f$, with 1 repeated once per connected component. Eigenvalues just below 1
/// signal clusters that the walk rarely leaves, which is the basis of spectral clustering.

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// A symmetric, unweighted k-nearest-neighbour graph stored in compressed sparse rows.
class KnnGraph {
public:
    /// @brief Builds the graph by brute-force neighbour search.
    /// @param points  one row per point, all of the same dimension.
    /// @param k       neighbours per point (1 <= k < number of points).
    /// @throws std::invalid_argument for ragged input or an out-of-range k.
    KnnGraph(const std::vector<std::vector<double>>& points, std::size_t k);

    /// @brief Number of vertices.
    [[nodiscard]] std::size_t size() const { return degree_.size(); }
    /// @brief Number of undirected edges.
    [[nodiscard]] std::size_t edges() const { return neighbors_.size() / 2; }
    /// @brief Degree of each vertex (the diagonal of D).
    [[nodiscard]] const std::vector<double>& degrees() const { return degree_; }
    /// @brief Neighbours of vertex @p i, in increasing order.
    [[nodiscard]] std::vector<std::size_t> neighbors(std::size_t i) const;
    /// @brief Number of connected components.
    [[nodiscard]] std::size_t components() const;

    /// @brief Returns \f$Wx\f$.
    [[nodiscard]] std::vector<double> adjacency_apply(const std::vector<double>& x) const;
    /// @brief Returns \f$Px=D^{-1}Wx\f$, the random-walk transition matrix applied to @p x.
    [[nodiscard]] std::vector<double> random_walk_apply(const std::vector<double>& x) const;

    /// @brief All eigenvalues of \f$P\f$, in decreasing order, from a dense symmetric
    ///        eigensolver applied to \f$S=D^{-1/2}WD^{-1/2}\f$. Costs \f$O(n^3)\f$; meant as a
    ///        reference for iterative methods.
    [[nodiscard]] std::vector<double> random_walk_spectrum() const;

private:
    std::vector<std::size_t> row_start_;  ///< CSR row offsets (size n+1).
    std::vector<std::size_t> neighbors_;  ///< CSR column indices.
    std::vector<double>      degree_;     ///< Vertex degrees.
};

}  // namespace datamunge::stats
