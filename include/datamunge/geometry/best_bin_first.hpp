#pragma once

// Best Bin First (Beis & Lowe, 1997): approximate nearest-neighbor search in a
// k-d tree. An exact k-d tree search backtracks into every branch that *could*
// hold a closer point; in high dimensions that degrades to scanning almost
// everything. BBF instead keeps a priority queue of unexplored branches ordered
// by the distance from the query to the branch's splitting boundary, always
// descending into the most promising bin next, and *stops after a fixed budget*
// of examined points. It thus finds the true nearest neighbor most of the time
// and a very-close one otherwise, for a fraction of the work -- the search behind
// SIFT feature matching. With an unbounded budget it reduces to exact search.

#include <datamunge/geometry/point2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace datamunge::geometry {

class BestBinFirst {
public:
    explicit BestBinFirst(std::vector<Point2D> points) : pts_(std::move(points)) {
        idx_.resize(pts_.size());
        for (std::size_t i = 0; i < pts_.size(); ++i) idx_[i] = static_cast<int>(i);
        if (!pts_.empty()) root_ = build(0, static_cast<int>(pts_.size()), 0);
    }

    struct Result {
        int index{-1};   // index of the (approximate) nearest point, -1 if none
        int examined{0}; // how many points were inspected
    };

    // (Approximate) nearest point to `query`, examining at most `max_search`
    // points, with the count of points inspected. Large max_search yields the
    // exact nearest neighbor.
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

    // Index of the (approximate) nearest point to `query`, examining at most
    // `max_search` points. Large max_search yields the exact nearest neighbor.
    // Returns -1 for an empty tree.
    int nearest(const Point2D& query, int max_search) const {
        return nearest_counted(query, max_search).index;
    }

private:
    struct Node { int pt; int axis; int left{-1}; int right{-1}; };

    static double sq(const Point2D& a, const Point2D& b) {
        const double dx = a.x - b.x, dy = a.y - b.y;
        return dx * dx + dy * dy;
    }

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

    std::vector<Point2D> pts_;
    std::vector<int>     idx_;
    std::vector<Node>    nodes_;
    int                  root_{-1};
};

} // namespace datamunge::geometry
