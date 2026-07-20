#pragma once

#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::geometry {

/// @brief A 2D k-d tree for nearest-neighbor / range queries over a fixed point set, built
///        once (via a median-splitting, alternating-axis partition) at construction --
///        doesn't support incremental insertion/removal after that.
class KDTree2D {
 public:
    explicit KDTree2D(std::vector<Point2D> points) : size_(points.size()) {
        if (!points.empty()) root_ = build(points, 0, points.size(), 0);
    }

    [[nodiscard]] bool empty() const { return size_ == 0; }
    [[nodiscard]] std::size_t size() const { return size_; }

    /// @brief The closest point to @p query. Throws std::logic_error if the tree is empty.
    [[nodiscard]] Point2D nearest(const Point2D& query) const {
        if (!root_) {
            throw std::logic_error("KDTree2D::nearest: tree is empty");
        }
        Point2D best = root_->point;
        double best_dist_sq = squared_distance(query, best);
        nearest_recurse(root_.get(), query, best, best_dist_sq);
        return best;
    }

    /// @brief The @p k closest points to @p query, nearest first. Returns fewer than @p k
    ///        points if the tree itself has fewer than @p k.
    [[nodiscard]] std::vector<Point2D> k_nearest(const Point2D& query, std::size_t k) const {
        if (k == 0 || !root_) {
            return {};
        }
        MaxHeap heap;
        k_nearest_recurse(root_.get(), query, k, heap);

        std::vector<Point2D> result;
        result.reserve(heap.size());
        while (!heap.empty()) {
            result.push_back(heap.top().second);
            heap.pop();
        }
        std::reverse(result.begin(), result.end()); // the heap pops furthest-first; reverse to nearest-first
        return result;
    }

    /// @brief All points within @p radius (inclusive) of @p query, in no particular order.
    [[nodiscard]] std::vector<Point2D> points_in_radius(const Point2D& query, double radius) const {
        std::vector<Point2D> result;
        if (root_) {
            radius_recurse(root_.get(), query, radius * radius, result);
        }
        return result;
    }

 private:
    struct Node {
        Point2D point;
        int axis; // 0 = split on x, 1 = split on y
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };
    // A custom comparator (rather than std::pair's default, lexicographic one) is needed here:
    // the default falls back to comparing Point2D via operator< when two distances tie, and
    // Point2D deliberately has no ordering (it's a coordinate pair, not a sortable key).
    struct DistanceCompare {
        bool operator()(const std::pair<double, Point2D>& a, const std::pair<double, Point2D>& b) const {
            return a.first < b.first;
        }
    };
    using MaxHeap = std::priority_queue<std::pair<double, Point2D>, std::vector<std::pair<double, Point2D>>, DistanceCompare>;

    std::unique_ptr<Node> root_;
    std::size_t size_;

    static std::unique_ptr<Node> build(std::vector<Point2D>& points, std::size_t lo, std::size_t hi, int depth) {
        if (lo >= hi) {
            return nullptr;
        }
        const int axis = depth % 2;
        const std::size_t mid = lo + (hi - lo) / 2;
        std::nth_element(points.begin() + static_cast<std::ptrdiff_t>(lo), points.begin() + static_cast<std::ptrdiff_t>(mid),
                          points.begin() + static_cast<std::ptrdiff_t>(hi),
                          [axis](const Point2D& a, const Point2D& b) { return axis == 0 ? a.x < b.x : a.y < b.y; });

        auto node = std::make_unique<Node>();
        node->point = points[mid];
        node->axis = axis;
        node->left = build(points, lo, mid, depth + 1);
        node->right = build(points, mid + 1, hi, depth + 1);
        return node;
    }

    static double axis_value(const Point2D& p, int axis) { return axis == 0 ? p.x : p.y; }

    static void nearest_recurse(const Node* node, const Point2D& query, Point2D& best, double& best_dist_sq) {
        if (!node) {
            return;
        }
        const double d = squared_distance(query, node->point);
        if (d < best_dist_sq) {
            best_dist_sq = d;
            best = node->point;
        }

        const double diff = axis_value(query, node->axis) - axis_value(node->point, node->axis);
        const Node* near_side = diff < 0.0 ? node->left.get() : node->right.get();
        const Node* far_side = diff < 0.0 ? node->right.get() : node->left.get();

        nearest_recurse(near_side, query, best, best_dist_sq);
        // Only descend into the far side if the splitting plane itself is closer than the
        // best distance found so far -- otherwise nothing on that side can possibly improve it.
        if (diff * diff < best_dist_sq) {
            nearest_recurse(far_side, query, best, best_dist_sq);
        }
    }

    static void k_nearest_recurse(const Node* node, const Point2D& query, std::size_t k, MaxHeap& heap) {
        if (!node) {
            return;
        }
        const double d = squared_distance(query, node->point);
        if (heap.size() < k) {
            heap.emplace(d, node->point);
        } else if (d < heap.top().first) {
            heap.pop();
            heap.emplace(d, node->point);
        }

        const double diff = axis_value(query, node->axis) - axis_value(node->point, node->axis);
        const Node* near_side = diff < 0.0 ? node->left.get() : node->right.get();
        const Node* far_side = diff < 0.0 ? node->right.get() : node->left.get();

        k_nearest_recurse(near_side, query, k, heap);
        if (heap.size() < k || diff * diff < heap.top().first) {
            k_nearest_recurse(far_side, query, k, heap);
        }
    }

    static void radius_recurse(const Node* node, const Point2D& query, double radius_sq, std::vector<Point2D>& result) {
        if (!node) {
            return;
        }
        if (squared_distance(query, node->point) <= radius_sq) {
            result.push_back(node->point);
        }
        const double diff = axis_value(query, node->axis) - axis_value(node->point, node->axis);
        const Node* near_side = diff <= 0.0 ? node->left.get() : node->right.get();
        const Node* far_side = diff <= 0.0 ? node->right.get() : node->left.get();

        radius_recurse(near_side, query, radius_sq, result);
        if (diff * diff <= radius_sq) {
            radius_recurse(far_side, query, radius_sq, result);
        }
    }
};

} // namespace datamunge::geometry
