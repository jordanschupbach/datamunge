#pragma once

// The Shamos-Hoey algorithm (1976): decide whether *any* two of n line segments
// intersect, in O(n log n) time, using a sweep line. A vertical line sweeps left
// to right; the segments it currently crosses are kept in a balanced structure
// ordered by their height at the sweep position. Two segments can only cross if
// they are ever *adjacent* in that vertical order, so it suffices to test each
// segment against its immediate neighbors as it is inserted (at its left
// endpoint) and as neighbors become adjacent when a segment leaves (at its right
// endpoint). This is the detection-only ancestor of Bentley-Ottmann, which goes
// on to report every intersection.

#include <datamunge/geometry/point2d.hpp>
#include <datamunge/geometry/segment_intersection.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <utility>
#include <vector>

namespace datamunge::geometry {

namespace detail {

struct ShSeg { Point2D p, q; }; // normalized so p.x <= q.x

inline double sh_y_at(const ShSeg& s, double x) {
    if (s.q.x <= s.p.x) return std::min(s.p.y, s.q.y); // vertical: use lower endpoint
    double t = (x - s.p.x) / (s.q.x - s.p.x);
    t = std::max(0.0, std::min(1.0, t));
    return s.p.y + t * (s.q.y - s.p.y);
}

// Comparator ordering segment indices by height at the shared sweep position.
struct ShCmp {
    const std::vector<ShSeg>* segs;
    const double*             sweep_x;
    bool operator()(int a, int b) const {
        const double ya = sh_y_at((*segs)[a], *sweep_x), yb = sh_y_at((*segs)[b], *sweep_x);
        if (std::fabs(ya - yb) > 1e-12) return ya < yb;
        return a < b;
    }
};

} // namespace detail

// Returns true iff at least two of the given segments intersect (endpoints
// included). Each segment is a pair of endpoints.
inline bool any_segments_intersect(const std::vector<std::pair<Point2D, Point2D>>& input) {
    const std::size_t n = input.size();
    std::vector<detail::ShSeg> segs(n);
    for (std::size_t i = 0; i < n; ++i) {
        Point2D a = input[i].first, b = input[i].second;
        if (b.x < a.x) std::swap(a, b);
        segs[i] = {a, b};
    }

    // Events: (x, isLeft, segIndex). Left endpoints processed before right at equal x.
    struct Event { double x; int type; int seg; }; // type 0 = left, 1 = right
    std::vector<Event> events;
    events.reserve(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        events.push_back({segs[i].p.x, 0, static_cast<int>(i)});
        events.push_back({segs[i].q.x, 1, static_cast<int>(i)});
    }
    std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
        if (a.x != b.x) return a.x < b.x;
        return a.type < b.type;
    });

    double            sweep_x = 0.0;
    detail::ShCmp     cmp{&segs, &sweep_x};
    std::set<int, detail::ShCmp> status(cmp);

    auto hits = [&](int i, int j) {
        return segments_intersect(segs[i].p, segs[i].q, segs[j].p, segs[j].q);
    };

    for (const auto& e : events) {
        sweep_x = e.x;
        if (e.type == 0) { // left endpoint: insert and check neighbors
            auto [it, ok] = status.insert(e.seg);
            (void)ok;
            if (it != status.begin()) { auto below = std::prev(it); if (hits(*below, e.seg)) return true; }
            auto above = std::next(it);
            if (above != status.end() && hits(*above, e.seg)) return true;
        } else { // right endpoint: neighbors become adjacent, then remove
            auto it = status.find(e.seg);
            if (it == status.end()) continue;
            auto above = std::next(it);
            if (it != status.begin() && above != status.end()) {
                auto below = std::prev(it);
                if (hits(*below, *above)) return true;
            }
            status.erase(it);
        }
    }
    return false;
}

} // namespace datamunge::geometry
