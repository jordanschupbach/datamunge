#pragma once

/// @file hdbscan.hpp
/// @brief Hierarchical density-based clustering via mutual reachability and excess of mass.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Density and size controls for Euclidean HDBSCAN.
/// @par Example
/// @code
/// const dm::stats::HdbscanOptions options{.min_cluster_size=4};
/// std::println("{}", options.min_cluster_size); // 4
/// @endcode
struct HdbscanOptions {
    std::size_t min_cluster_size{5}; ///< Smallest retained cluster, at least two.
    std::size_t min_samples{}; ///< Core-distance neighbor rank including the point itself. Zero uses min_cluster_size.
    bool allow_single_cluster{}; ///< Also let excess of mass select the root, whose birth membership is the entire sample.
};

/// @brief One undirected edge in the mutual-reachability minimum spanning tree.
/// @par Example
/// @code
/// const dm::stats::HdbscanMstEdge edge{0,1,.5};
/// std::println("{}", edge.distance); // 0.5
/// @endcode
struct HdbscanMstEdge {
    std::size_t first{}; ///< First point index.
    std::size_t second{}; ///< Second point index, greater than first.
    double distance{}; ///< Maximum of Euclidean distance and both endpoint core distances.
};

/// @brief A split in the condensed hierarchy, expressed on the inverse-distance scale.
/// @par Example
/// @code
/// const dm::stats::HdbscanCondensedEdge edge{7,3,2,1};
/// std::println("{}", edge.child_size); // 1
/// @endcode
struct HdbscanCondensedEdge {
    std::size_t parent{}; ///< Condensed cluster identifier.
    std::size_t child{}; ///< Point index below n, or condensed cluster identifier at or above n.
    long double lambda{}; ///< Inverse mutual-reachability distance of the split, infinity at zero distance.
    std::size_t child_size{}; ///< Number of points descending into the child at the split.
};

/// @brief Birth and persistence of one retained branch of the condensed tree.
/// @par Example
/// @code
/// const dm::stats::HdbscanCluster cluster;
/// std::println("{}", cluster.selected); // false
/// @endcode
struct HdbscanCluster {
    std::size_t id{}; ///< Unique condensed cluster identifier, starting at n for the root.
    std::size_t birth_size{}; ///< Number of points at birth.
    long double birth_lambda{}; ///< Density at which the cluster first appears, zero for the root.
    long double stability{}; ///< Sum of each birth member's persistence beyond the birth density.
    bool selected{}; ///< True if excess of mass retains the cluster in the flat partition.
};

/// @brief Flat assignments, core distances and the mutual-reachability hierarchy.
/// @par Example
/// @code
/// const dm::stats::HdbscanFit result;
/// std::println("{}", result.labels.size()); // 0
/// @endcode
struct HdbscanFit {
    std::vector<std::int64_t> labels; ///< Zero-based labels in input order, -1 for noise.
    std::vector<double> core_distances; ///< Core distance for every point in input order.
    std::vector<HdbscanMstEdge> minimum_spanning_tree; ///< Tree edges sorted by distance and then endpoints.
    std::vector<HdbscanCondensedEdge> condensed_tree; ///< Outgoing splits of each retained cluster.
    std::vector<HdbscanCluster> clusters; ///< Condensed clusters in identifier order, including the root.
};

/// @brief Fits Euclidean HDBSCAN and selects stable clusters by excess of mass.
/// Builds the exact mutual-reachability minimum spanning tree and processes
/// tied edges as a single level. Condensation prunes children smaller than
/// min_cluster_size. Each retained point receives the label of its selected
/// ancestor at cluster birth. Labels follow the selected cluster IDs.
/// @param observations Nonempty-feature matrix of finite points in rows, at least two.
/// @param options Cluster size, core-distance rank and root selection policy.
/// @return Flat labels and the tree and cluster statistics used to select them.
/// @throws std::invalid_argument For invalid geometry, controls or nonfinite observations.
/// @throws std::length_error For observation counts outside the hierarchy storage range.
/// @throws std::overflow_error If a Euclidean distance is outside the finite double range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(5,1,{0,.1,5,5.1,20});
/// const auto fit=dm::stats::fit_hdbscan(x,{.min_cluster_size=2});
/// std::println("{}", fit.labels.back()); // -1
/// @endcode
/// @par References
/// Campello, Moulavi, Zimek and Sander (2015), https://doi.org/10.1145/2733381
[[nodiscard]] HdbscanFit fit_hdbscan(const linalg::DenseMatrix<double>& observations,
    const HdbscanOptions& options = {});

} // namespace datamunge::stats
