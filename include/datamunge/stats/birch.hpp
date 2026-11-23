#pragma once

/// @file birch.hpp
/// @brief Incremental BIRCH cluster-feature trees and count-weighted leaf clustering.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace datamunge::stats {
/// @brief Internal BIRCH cluster-feature tree nodes.
namespace birch_detail {
/// @brief Shared internal CF-tree node.
struct Node;
}

/// @brief Branch capacity and leaf absorption threshold.
/// @par Example
/// @code
/// const dm::stats::BirchOptions options{.threshold=.2, .branching_factor=4};
/// std::println("{}", options.branching_factor); // 4
/// @endcode
struct BirchOptions {
    double threshold{.5}; ///< Maximum RMS distance from the centroid after absorbing a point into a leaf summary.
    std::size_t branching_factor{50}; ///< Maximum number of features or children per node, at least two.
};

/// @brief Mergeable cluster feature in count, centroid and RMS-radius form.
/// @par Example
/// @code
/// const dm::stats::BirchFeature feature{2,{1},.5};
/// std::println("{}", feature.count); // 2
/// @endcode
struct BirchFeature {
    std::size_t count{}; ///< Number of streamed observations.
    std::vector<double> centroid; ///< Mean feature values.
    long double radius{}; ///< Root mean squared Euclidean distance from the centroid.
};

/// @brief An owned snapshot of leaf summaries and their global Ward clusters.
/// @par Example
/// @code
/// const dm::stats::BirchPartition partition;
/// std::println("{}", partition.clusters.size()); // 0
/// @endcode
struct BirchPartition {
    std::vector<BirchFeature> leaves; ///< Leaf summaries in tree traversal order.
    std::vector<std::size_t> leaf_labels; ///< Zero-based global label of each leaf.
    std::vector<BirchFeature> clusters; ///< Count-weighted global summaries, numbered by first member leaf.

    /// @brief Predicts the label of each row from its closest leaf centroid.
    /// @param observations Finite rows with the same width as the leaf centroids.
    /// @return Global labels in input order, resolving equal distances to the first leaf.
    /// @throws std::invalid_argument For invalid input dimensions, values or partition geometry.
    /// @par Example
    /// @code
    /// dm::stats::Birch tree(1,{.threshold=.2});
    /// tree.partial_fit(dm::linalg::DenseMatrix<double>(3,1,{0,.1,4}));
    /// const auto part=tree.cluster_leaves(2);
    /// std::println("{}", part.predict(dm::linalg::DenseMatrix<double>(1,1,{4}))[0]); // 1
    /// @endcode
    [[nodiscard]] std::vector<std::size_t> predict(const linalg::DenseMatrix<double>& observations) const;
};

/// @brief Stores a balanced BIRCH tree of streaming cluster features.
/// Observations are processed in row order without retaining raw rows.
/// A leaf is absorbed only if its combined RMS radius stays within the
/// threshold. Overflowing nodes split around their farthest feature pair.
/// Copies act as snapshots and can accept more rows independently.
/// @par Example
/// @code
/// dm::stats::Birch tree(1,{.threshold=.2});
/// tree.partial_fit(dm::linalg::DenseMatrix<double>(3,1,{0,.1,4}));
/// std::println("{}", tree.leaf_summaries().size()); // 2
/// @endcode
/// @par References
/// Zhang, Ramakrishnan and Livny (1996), https://doi.org/10.1145/233269.233324
class Birch {
public:
    /// @brief Constructs an empty tree with a fixed feature width.
    /// @param dimensions Positive number of features per row.
    /// @param options Nonnegative finite threshold and branching factor of at least two.
    /// @throws std::invalid_argument For invalid dimensions or options.
    /// @par Example
    /// @code
    /// const dm::stats::Birch tree(2);
    /// std::println("{}", tree.height()); // 0
    /// @endcode
    explicit Birch(std::size_t dimensions, const BirchOptions& options = {});

    /// @brief Adds a batch in row order and commits the entire batch together.
    /// @param observations Matrix with the configured number of columns and finite entries.
    /// @throws std::invalid_argument For invalid shape or observations.
    /// @throws std::length_error If the total observation count overflows.
    /// @par Example
    /// @code
    /// dm::stats::Birch tree(1);
    /// tree.partial_fit(dm::linalg::DenseMatrix<double>(2,1,{0,1}));
    /// tree.partial_fit(dm::linalg::DenseMatrix<double>(1,1,{2}));
    /// std::println("{}", tree.observations()); // 3
    /// @endcode
    void partial_fit(const linalg::DenseMatrix<double>& observations);

    /// @brief Number of streamed observations.
    [[nodiscard]] std::size_t observations() const noexcept;

    /// @brief Number of tree levels, zero when empty and one for a single leaf node.
    [[nodiscard]] std::size_t height() const noexcept;

    /// @brief Copies leaf summaries in depth-first tree traversal order.
    /// @return Owned summaries with one entry per leaf feature.
    /// @par Example
    /// @code
    /// dm::stats::Birch tree(1);
    /// tree.partial_fit(dm::linalg::DenseMatrix<double>(2,1,{0,.1}));
    /// std::println("{}", tree.leaf_summaries()[0].count); // 2
    /// @endcode
    [[nodiscard]] std::vector<BirchFeature> leaf_summaries() const;

    /// @brief Clusters the leaf summaries by Ward's increase in within-cluster sum of squares.
    /// The merge criterion uses original observation counts as weights.
    /// @param clusters Target count from one through the number of leaf summaries.
    /// @return Snapshot of leaves, global labels and count-weighted merged features.
    /// @throws std::invalid_argument For an out-of-range cluster count.
    /// @throws std::length_error If the Ward merge queue exceeds storage capacity.
    /// @par Example
    /// @code
    /// dm::stats::Birch tree(1);
    /// tree.partial_fit(dm::linalg::DenseMatrix<double>(3,1,{0,.1,4}));
    /// std::println("{}", tree.cluster_leaves(1).clusters[0].count); // 3
    /// @endcode
    [[nodiscard]] BirchPartition cluster_leaves(std::size_t clusters) const;

private:
    std::size_t dimensions_; ///< Required number of input columns.
    BirchOptions options_; ///< Fixed radius threshold and branching limits.
    std::shared_ptr<const birch_detail::Node> root_; ///< Owned immutable tree version.
};

} // namespace datamunge::stats
