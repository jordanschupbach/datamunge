#pragma once

/// @file best_bin_first.hpp
/// @brief Best Bin First (Beis and Lowe, 1997): budgeted nearest-neighbour search in k-d trees.
///
/// An exact k-d tree search backtracks into every branch that could hold a closer point; in high
/// dimensions that degrades to scanning almost everything. Best Bin First keeps a priority queue
/// of unexplored branches ordered by a lower bound on their distance to the query, always
/// descends into the most promising bin next, and stops after a fixed budget of examined points.
/// It finds the true nearest neighbour most of the time and a close one otherwise, for a fraction
/// of the work; it is the search behind SIFT feature matching. With an unbounded budget it is
/// exact. BestBinFirst is the planar version; BestBinFirstKd works in any dimension, with
/// bucketed leaves, k nearest neighbours, and a choice of queue priority.

#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace datamunge::geometry {

/// @brief Planar Best Bin First over a k-d tree that alternates the x and y axes.
///
/// @par Example
/// @code
/// const dm::geometry::BestBinFirst bbf(points);
/// const auto hit = bbf.nearest_counted({0.5, 0.5}, 20);   // index and points examined
/// @endcode
/// @see BestBinFirstKd
class BestBinFirst {
public:
    /// @brief Builds the tree over @p points (median splits, alternating axes).
    /// @param points The points to index; result indices refer to this order.
    explicit BestBinFirst(std::vector<Point2D> points) : pts_(std::move(points)) {
        idx_.resize(pts_.size());
        for (std::size_t i = 0; i < pts_.size(); ++i) idx_[i] = static_cast<int>(i);
        if (!pts_.empty()) root_ = build(0, static_cast<int>(pts_.size()), 0);
    }

    /// @brief A search result and its cost.
    struct Result {
        int index{-1};   ///< Index of the (approximate) nearest point, -1 if none.
        int examined{0}; ///< How many points were inspected.
    };

    /// @brief (Approximate) nearest point to @p query, examining at most @p max_search points.
    /// @param query The query point.
    /// @param max_search The budget of examined points; a large value gives the exact neighbour.
    /// @return The nearest point found and the number of points inspected.
    Result nearest_counted(const Point2D& query, int max_search) const {
        if (root_ < 0) return {};
        using QItem = std::pair<double, int>; // (boundary distance^2, node)
        std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> pq;
        pq.emplace(0.0, root_);

        double best = std::numeric_limits<double>::infinity();
        int    best_pt = -1, examined = 0;
        while (!pq.empty()) {
            auto [bd, ni] = pq.top();
            pq.pop();
            if (bd > best) break; // nothing left can beat the current best
            // Descend to a leaf, queueing the far branches by boundary distance.
            while (ni >= 0) {
                const Node& nd = nodes_[ni];
                const double d = sq(query, pts_[nd.pt]);
                if (d < best) { best = d; best_pt = nd.pt; }
                if (++examined >= max_search) return {best_pt, examined};
                const double diff = (nd.axis == 0 ? query.x - pts_[nd.pt].x : query.y - pts_[nd.pt].y);
                const int    near = diff < 0 ? nd.left : nd.right;
                const int    far  = diff < 0 ? nd.right : nd.left;
                if (far >= 0) pq.emplace(diff * diff, far);
                ni = near;
            }
        }
        return {best_pt, examined};
    }

    /// @brief Index of the (approximate) nearest point to @p query, examining at most
    ///        @p max_search points.
    /// @param query The query point.
    /// @param max_search The budget of examined points.
    /// @return The index, or -1 for an empty tree.
    int nearest(const Point2D& query, int max_search) const {
        return nearest_counted(query, max_search).index;
    }

    /// @brief Number of indexed points.
    /// @return The size of the point set passed to the constructor.
    [[nodiscard]] std::size_t size() const noexcept { return pts_.size(); }

    /// @brief The indexed points, in their original order (result indices refer to these).
    /// @return The point set passed to the constructor.
    [[nodiscard]] const std::vector<Point2D>& points() const noexcept { return pts_; }

private:
    /// One tree node: the splitting point, its axis, and the two children (-1 if absent).
    struct Node {
        int pt;          ///< Index of the splitting point.
        int axis;        ///< Splitting axis: 0 for x, 1 for y.
        int left{-1};    ///< Left child node, -1 if absent.
        int right{-1};   ///< Right child node, -1 if absent.
    };

    /// Squared Euclidean distance.
    static double sq(const Point2D& a, const Point2D& b) {
        const double dx = a.x - b.x, dy = a.y - b.y;
        return dx * dx + dy * dy;
    }

    /// Builds the subtree over idx_[lo, hi) and returns its node index (-1 if empty).
    int build(int lo, int hi, int depth) {
        if (lo >= hi) return -1;
        const int axis = depth & 1;
        const int mid  = (lo + hi) / 2;
        std::nth_element(idx_.begin() + lo, idx_.begin() + mid, idx_.begin() + hi,
                         [&](int a, int b) { return axis == 0 ? pts_[a].x < pts_[b].x : pts_[a].y < pts_[b].y; });
        const int id = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{idx_[mid], axis, -1, -1});
        const int L = build(lo, mid, depth + 1);
        const int R = build(mid + 1, hi, depth + 1);
        nodes_[id].left  = L;
        nodes_[id].right = R;
        return id;
    }

    std::vector<Point2D> pts_;   ///< The indexed points.
    std::vector<int>     idx_;   ///< Point indices, permuted during the build.
    std::vector<Node>    nodes_; ///< The tree nodes.
    int                  root_{-1}; ///< Root node index, -1 for an empty tree.
};

/// @brief Which lower bound orders the Best Bin First priority queue.
enum class BinPriority {
    /// Squared distance to the splitting hyperplane alone (Beis and Lowe's original key).
    Hyperplane,
    /// Squared distance to the whole cell, maintained incrementally (Arya and Mount, 1993).
    Cell,
};

/// @brief Result of a BestBinFirstKd search.
struct BestBinFirstResult {
    std::vector<int>    indices;   ///< Neighbour indices, nearest first.
    std::vector<double> distances; ///< Their Euclidean distances, ascending.
    int  examined{0};              ///< Points whose distance was computed.
    int  leaves{0};                ///< Leaf buckets visited.
    bool exact{false};             ///< True when the search ended by pruning, not by the budget,
                                   ///< so the answer is the exact k nearest neighbours.
};

/// @brief Best Bin First over a bucketed k-d tree in any dimension.
///
/// The tree splits each node at the median of the coordinate with the largest spread, down to
/// leaves of at most `leaf_size` points. A search descends to the query's leaf, queueing every
/// branch it passes with a lower bound on its distance (see BinPriority), then repeatedly opens
/// the queued branch with the smallest bound. It stops when that bound exceeds the k-th best
/// distance found, which proves the answer exact, or when @p max_examined points have been
/// examined.
///
/// @par Example
/// @code
/// const dm::geometry::BestBinFirstKd tree(rows);            // rows: vector<vector<double>>
/// const auto hit = tree.knn(query, 1, 50);                   // at most 50 distance computations
/// const auto exact = tree.knn(query, 1);                     // unbounded: exact
/// @endcode
/// @see BestBinFirst, knn_brute_force()
class BestBinFirstKd {
public:
    /// @brief Builds the tree.
    /// @param points The points, all of the same dimension; result indices refer to this order.
    /// @param leaf_size Maximum points per leaf bucket, at least 1.
    /// @param priority The queue key; Cell is tighter, Hyperplane is the original.
    /// @throws std::invalid_argument if the points have different dimensions or leaf_size < 1.
    explicit BestBinFirstKd(std::vector<std::vector<double>> points, int leaf_size = 8,
                            BinPriority priority = BinPriority::Cell);

    /// @brief The k (approximate) nearest neighbours of @p query.
    /// @param query A point of the tree's dimension.
    /// @param k How many neighbours, at least 1.
    /// @param max_examined Budget of distance computations; 0 means unbounded (exact search).
    /// @return Neighbours nearest first, with the work done.
    /// @throws std::invalid_argument if the dimension is wrong or k < 1.
    [[nodiscard]] BestBinFirstResult knn(const std::vector<double>& query, int k, int max_examined = 0) const;

    /// @brief Number of indexed points.
    /// @return The size of the point set.
    [[nodiscard]] std::size_t size() const noexcept { return points_.size(); }

    /// @brief Dimension of the points.
    /// @return The common length of the point vectors (0 for an empty tree).
    [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }

    /// @brief Number of leaf buckets in the tree.
    /// @return The leaf count.
    [[nodiscard]] std::size_t leaf_count() const noexcept;

    /// @brief The indexed points.
    /// @return The points passed to the constructor, in order.
    [[nodiscard]] const std::vector<std::vector<double>>& points() const noexcept { return points_; }

private:
    /// One node: a split (axis >= 0) with two children, or a leaf holding order_[begin, end).
    struct Node {
        int    axis{-1};    ///< Splitting coordinate, -1 for a leaf.
        double split{0.0};  ///< Splitting value: left holds coordinates <= split.
        int    left{-1};    ///< Left child node.
        int    right{-1};   ///< Right child node.
        int    begin{0};    ///< First position in order_ (leaves).
        int    end{0};      ///< One past the last position in order_ (leaves).
    };

    /// Builds the subtree over order_[begin, end) and returns its node index.
    int build(int begin, int end);

    std::vector<std::vector<double>> points_;  ///< The indexed points.
    std::vector<int>                 order_;   ///< Point indices grouped by leaf.
    std::vector<Node>                nodes_;   ///< The tree nodes; node 0 is the root.
    std::size_t                      dim_{0};  ///< Point dimension.
    int                              leaf_size_{8};  ///< Maximum bucket size.
    BinPriority                      priority_{BinPriority::Cell};  ///< Queue key.
};

/// @brief Exact k nearest neighbours by scanning every point.
/// @param points The points.
/// @param query The query, of the same dimension.
/// @param k How many neighbours.
/// @return Indices of the k nearest points, nearest first (ties by lower index).
///
/// @par Example
/// @code
/// const auto nn = dm::geometry::knn_brute_force(rows, query, 5);
/// @endcode
/// @see BestBinFirstKd
std::vector<int> knn_brute_force(const std::vector<std::vector<double>>& points, const std::vector<double>& query, int k);

/// @brief Directed k-nearest-neighbour lists by brute force: entry i holds the k points nearest
///        to point i, nearest first, excluding i itself (ties by lower index).
/// @param points The points, all of one dimension.
/// @param k Neighbours per point, 1 <= k < points.size().
/// @return One list of k indices per point.
/// @throws std::invalid_argument for an out-of-range k.
///
/// @par Example
/// @code
/// const auto lists = dm::geometry::knn_lists(rows, 10);   // lists[i][0] is i's nearest neighbour
/// @endcode
/// @see knn_brute_force()
std::vector<std::vector<int>> knn_lists(const std::vector<std::vector<double>>& points, int k);

} // namespace datamunge::geometry
