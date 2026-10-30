/// @file bentley_ottmann.cpp
/// @brief Bentley-Ottmann sweep with a balanced-tree status, brute-force reference, and
///        parallel-coordinate segments.

#include <datamunge/geometry/bentley_ottmann.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace datamunge::geometry {

namespace detail {

bool proper_intersection(const Point2D& A, const Point2D& B, const Point2D& C, const Point2D& D, Point2D& out) {
    const double d1 = cross(C, D, A), d2 = cross(C, D, B), d3 = cross(A, B, C), d4 = cross(A, B, D);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
        // The crossing divides AB in the ratio |d1| : |d2|.
        const double t = d1 / (d1 - d2);
        out.x = A.x + t * (B.x - A.x);
        out.y = A.y + t * (B.y - A.y);
        return true;
    }
    return false;
}

}  // namespace detail

SweepResult bentley_ottmann_sweep(const std::vector<Segment>& input) {
    std::vector<Segment> segs = input;
    const int            n    = static_cast<int>(segs.size());
    for (auto& s : segs)  // orient each segment left to right
        if (s.b.x < s.a.x || (s.b.x == s.a.x && s.b.y < s.a.y)) std::swap(s.a, s.b);

    struct Event {
        double x, y;
        int    type;  // 0 = left end point, 1 = crossing, 2 = right end point
        int    s, t;
    };
    auto later = [](const Event& A, const Event& B) {
        if (A.x != B.x) return A.x > B.x;
        if (A.type != B.type) return A.type > B.type;
        return A.y > B.y;
    };
    std::priority_queue<Event, std::vector<Event>, decltype(later)> queue(later);
    for (int i = 0; i < n; ++i) {
        queue.push({segs[i].a.x, segs[i].a.y, 0, i, -1});
        queue.push({segs[i].b.x, segs[i].b.y, 2, i, -1});
    }

    // Status: active segments ordered by height at the sweep line; at equal height (just
    // inserted at a crossing) the smaller slope is lower immediately to the right.
    double sweep_x = -std::numeric_limits<double>::infinity();
    // While two segments are reinserted at their crossing, both are pinned to the crossing's
    // height, so that they tie exactly and their slopes decide the order. Recomputing their
    // heights from x would add rounding error, which is large for steep segments.
    std::vector<char> pinned(n, 0);
    double            pin_y = 0.0;
    auto   height  = [&](int i) {
        if (pinned[i]) return pin_y;
        const auto& s = segs[i];
        if (sweep_x == s.a.x) return s.a.y;
        const double t = (sweep_x - s.a.x) / (s.b.x - s.a.x);
        return s.a.y + t * (s.b.y - s.a.y);
    };
    auto slope = [&](int i) { return (segs[i].b.y - segs[i].a.y) / (segs[i].b.x - segs[i].a.x); };
    auto below = [&](int i, int j) {
        if (i == j) return false;
        const double hi = height(i), hj = height(j);
        const double tol = 1e-12 * (1.0 + std::abs(hi) + std::abs(hj));
        if (std::abs(hi - hj) > tol) return hi < hj;
        const double si = slope(i), sj = slope(j);
        if (si != sj) return si < sj;
        return i < j;
    };
    std::set<int, decltype(below)>                  status(below);
    std::vector<std::set<int, decltype(below)>::iterator> where(n, status.end());
    std::set<std::pair<int, int>>                   scheduled, reported;
    SweepResult                                     out;

    auto check = [&](int a, int b) {
        if (a < 0 || b < 0) return;
        const std::pair<int, int> key{std::min(a, b), std::max(a, b)};
        if (scheduled.count(key)) return;
        ++out.intersection_tests;
        Point2D p;
        if (detail::proper_intersection(segs[a].a, segs[a].b, segs[b].a, segs[b].b, p) && p.x >= sweep_x) {
            scheduled.insert(key);
            queue.push({p.x, p.y, 1, key.first, key.second});
        }
    };
    auto prev_of = [&](std::set<int, decltype(below)>::iterator it) {
        return it == status.begin() ? -1 : *std::prev(it);
    };
    auto next_of = [&](std::set<int, decltype(below)>::iterator it) {
        auto nx = std::next(it);
        return nx == status.end() ? -1 : *nx;
    };

    while (!queue.empty()) {
        const Event e = queue.top();
        queue.pop();
        ++out.events;
        sweep_x = e.x;
        if (e.type == 0) {
            where[e.s] = status.insert(e.s).first;
            out.max_status = std::max(out.max_status, status.size());
            check(e.s, prev_of(where[e.s]));
            check(e.s, next_of(where[e.s]));
        } else if (e.type == 2) {
            const auto it = where[e.s];
            const int  lo = prev_of(it), hi = next_of(it);
            status.erase(it);
            where[e.s] = status.end();
            check(lo, hi);
        } else {
            if (reported.count({e.s, e.t})) {  // already handled with others through the same point
                --out.events;
                continue;
            }
            // Every segment through the crossing point: usually just e.s and e.t, but real data
            // can put three or more through one point. They form a block in the status.
            const double tol = 1e-10 * (1.0 + std::abs(e.y));
            auto lo = where[e.s], hi = where[e.s];
            const auto through = [&](std::set<int, decltype(below)>::iterator it) {
                return std::abs(height(*it) - e.y) <= tol || *it == e.s || *it == e.t;
            };
            while (lo != status.begin() && through(std::prev(lo))) --lo;
            while (std::next(hi) != status.end() && through(std::next(hi))) ++hi;
            std::vector<int> block;
            for (auto it = lo;; ++it) {
                block.push_back(*it);
                if (it == hi) break;
            }
            for (std::size_t a = 0; a < block.size(); ++a)
                for (std::size_t b = a + 1; b < block.size(); ++b) {
                    const std::pair<int, int> key{std::min(block[a], block[b]), std::max(block[a], block[b])};
                    Point2D p;
                    if (reported.count(key)) continue;
                    if (key == std::pair<int, int>{e.s, e.t}) {  // the event's own pair: already tested
                        reported.insert(key);
                        out.intersections.push_back({{e.x, e.y}, e.s, e.t});
                        continue;
                    }
                    ++out.intersection_tests;
                    if (!detail::proper_intersection(segs[key.first].a, segs[key.first].b, segs[key.second].a,
                                                     segs[key.second].b, p))
                        continue;
                    reported.insert(key);
                    scheduled.insert(key);
                    out.intersections.push_back({p, key.first, key.second});
                }
            // Remove the block by iterator (no comparisons), then reinsert it pinned to the
            // crossing height, so the segments tie and their slopes reverse their order.
            for (const int b : block) status.erase(where[b]);
            pin_y = e.y;
            for (const int b : block) pinned[b] = 1;
            for (const int b : block) where[b] = status.insert(b).first;
            auto first = where[block.front()], last = where[block.front()];
            for (const int b : block) {
                if (below(b, *first)) first = where[b];
                if (below(*last, b)) last = where[b];
            }
            for (const int b : block) pinned[b] = 0;
            check(*first, prev_of(first));
            check(*last, next_of(last));
        }
    }
    return out;
}

std::vector<Point2D> bentley_ottmann(std::vector<Segment> segs) {
    std::vector<Point2D> points;
    for (const auto& x : bentley_ottmann_sweep(segs).intersections) points.push_back(x.point);
    return points;
}

std::vector<SegmentIntersection> segment_intersections_brute(const std::vector<Segment>& segs) {
    std::vector<SegmentIntersection> out;
    for (int i = 0; i < static_cast<int>(segs.size()); ++i)
        for (int j = i + 1; j < static_cast<int>(segs.size()); ++j) {
            Point2D p;
            if (detail::proper_intersection(segs[i].a, segs[i].b, segs[j].a, segs[j].b, p)) out.push_back({p, i, j});
        }
    return out;
}

std::vector<Segment> parallel_coordinate_segments(const std::vector<double>& left, const std::vector<double>& right,
                                                  double tie_break) {
    if (left.size() != right.size()) throw std::invalid_argument("parallel_coordinate_segments: lengths differ");
    const auto rescale = [](const std::vector<double>& v) {
        const auto [lo, hi] = std::ranges::minmax(v);
        std::vector<double> out(v.size(), 0.5);
        if (hi > lo)
            for (std::size_t i = 0; i < v.size(); ++i) out[i] = (v[i] - lo) / (hi - lo);
        return out;
    };
    const auto u = rescale(left), v = rescale(right);
    std::vector<Segment> segs;
    for (std::size_t i = 0; i < u.size(); ++i) {
        const double offset = tie_break * static_cast<double>(i);
        segs.push_back({{0.0, u[i] + offset}, {1.0, v[i] + offset}});
    }
    return segs;
}

}  // namespace datamunge::geometry
