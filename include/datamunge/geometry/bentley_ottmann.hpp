#pragma once

// Bentley-Ottmann sweep-line algorithm: report all pairwise intersections of a
// set of line segments. A vertical sweep line moves left to right; only segments
// adjacent in the sweep-line order can intersect next, so each of the k
// intersections is found by a local neighbour check. With a balanced-BST status
// this is O((n + k) log n); this implementation keeps the status in a vector
// (O(n) updates) for clarity while preserving the event-driven structure.
//
// Assumes segments in general position: no vertical segments, no shared
// endpoints, and no three segments through a common point.

#include <datamunge/geometry/point2d.hpp>

#include <cmath>
#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace datamunge::geometry {

struct Segment {
    Point2D a, b;
};

namespace detail {

// Proper intersection (strict crossing) of segments AB and CD; sets `out`.
inline bool proper_intersection(const Point2D& A, const Point2D& B, const Point2D& C, const Point2D& D,
                                Point2D& out) {
    const double d1 = cross(C, D, A), d2 = cross(C, D, B), d3 = cross(A, B, C), d4 = cross(A, B, D);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
        const double a1 = B.y - A.y, b1 = A.x - B.x, c1 = a1 * A.x + b1 * A.y;
        const double a2 = D.y - C.y, b2 = C.x - D.x, c2 = a2 * C.x + b2 * C.y;
        const double det = a1 * b2 - a2 * b1;
        if (det == 0) return false;
        out.x = (b2 * c1 - b1 * c2) / det;
        out.y = (a1 * c2 - a2 * c1) / det;
        return true;
    }
    return false;
}

} // namespace detail

inline std::vector<Point2D> bentley_ottmann(std::vector<Segment> segs) {
    const int n = static_cast<int>(segs.size());
    for (auto& s : segs) // orient each segment left-to-right
        if (s.b.x < s.a.x || (s.b.x == s.a.x && s.b.y < s.a.y)) std::swap(s.a, s.b);

    struct Event {
        double x, y;
        int    type; // 0 = left endpoint, 1 = right endpoint, 2 = crossing
        int    s, t;
    };
    auto cmp = [](const Event& A, const Event& B) { return A.x != B.x ? A.x > B.x : A.y > B.y; };
    std::priority_queue<Event, std::vector<Event>, decltype(cmp)> pq(cmp);
    for (int i = 0; i < n; ++i) {
        pq.push({segs[i].a.x, segs[i].a.y, 0, i, -1});
        pq.push({segs[i].b.x, segs[i].b.y, 1, i, -1});
    }

    double sweepx = -1e18;
    auto   yAt    = [&](int i) {
        const auto& s = segs[i];
        if (s.b.x == s.a.x) return s.a.y;
        const double t = (sweepx - s.a.x) / (s.b.x - s.a.x);
        return s.a.y + t * (s.b.y - s.a.y);
    };

    std::vector<int>            status; // active segments, ordered by y at the sweep line
    std::set<std::pair<int, int>> scheduled;
    std::vector<Point2D>        result;

    auto position = [&](int s) {
        for (int i = 0; i < static_cast<int>(status.size()); ++i)
            if (status[i] == s) return i;
        return -1;
    };
    auto try_cross = [&](int a, int b) {
        if (a < 0 || b < 0) return;
        const std::pair<int, int> key{a < b ? a : b, a < b ? b : a};
        if (scheduled.count(key)) return;
        Point2D p;
        if (detail::proper_intersection(segs[a].a, segs[a].b, segs[b].a, segs[b].b, p) && p.x > sweepx + 1e-12) {
            scheduled.insert(key);
            pq.push({p.x, p.y, 2, key.first, key.second});
        }
    };

    while (!pq.empty()) {
        const Event e = pq.top();
        pq.pop();
        sweepx = e.x;
        if (e.type == 0) { // insert segment e.s
            const double yy  = yAt(e.s);
            int          ins = 0;
            while (ins < static_cast<int>(status.size()) && yAt(status[ins]) < yy) ++ins;
            status.insert(status.begin() + ins, e.s);
            const int below = ins - 1 >= 0 ? status[ins - 1] : -1;
            const int above = ins + 1 < static_cast<int>(status.size()) ? status[ins + 1] : -1;
            try_cross(e.s, below);
            try_cross(e.s, above);
        } else if (e.type == 1) { // remove segment e.s
            const int i = position(e.s);
            if (i < 0) continue;
            const int below = i - 1 >= 0 ? status[i - 1] : -1;
            const int above = i + 1 < static_cast<int>(status.size()) ? status[i + 1] : -1;
            status.erase(status.begin() + i);
            try_cross(below, above);
        } else { // crossing of e.s and e.t at (e.x, e.y)
            result.push_back({e.x, e.y}); // a verified proper crossing
            const int i = position(e.s), j = position(e.t);
            if (i < 0 || j < 0) continue;
            if ((i > j ? i - j : j - i) != 1) continue; // not currently adjacent: report only
            std::swap(status[i], status[j]);
            const int lo = i < j ? i : j, hi = i < j ? j : i;
            const int outer_lo = lo - 1 >= 0 ? status[lo - 1] : -1;
            const int outer_hi = hi + 1 < static_cast<int>(status.size()) ? status[hi + 1] : -1;
            try_cross(status[lo], outer_lo);
            try_cross(status[hi], outer_hi);
        }
    }
    return result;
}

} // namespace datamunge::geometry
