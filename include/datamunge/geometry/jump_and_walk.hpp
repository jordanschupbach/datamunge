#pragma once

// Jump-and-Walk point location in a triangulation. Given a triangulation
// (e.g. Delaunay) and a query point, find the triangle that contains it in
// expected O(n^{1/3}) time -- far cheaper than scanning all triangles.
//
//   Jump: sample a handful (~ n^{1/3}) of triangles at random and start from
//         the one whose centroid is nearest the query.
//   Walk: from that triangle, repeatedly step across the edge that the query
//         lies on the far side of (an orientation test), moving strictly
//         closer, until a triangle is reached that the query is inside.
//
// The stochastic (Devillers) walk picks a random starting edge at each step,
// which provably terminates without cycling on a Delaunay triangulation.

#include <datamunge/geometry/delaunay.hpp>
#include <datamunge/geometry/point2d.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace datamunge::geometry {

namespace detail {

inline double jw_orient(const Point2D& a, const Point2D& b, const Point2D& p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

// A cheap deterministic PRNG so location is reproducible.
struct JwRng {
    std::uint64_t s;
    explicit JwRng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    std::uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

// Undirected-edge key (u,v) with u < v, packed into 64 bits.
inline std::uint64_t jw_edge_key(std::size_t u, std::size_t v) {
    if (u > v) std::swap(u, v);
    return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint32_t>(v);
}

} // namespace detail

// Precomputed adjacency so repeated queries share the edge->triangle map.
struct TriangulationLocator {
    const std::vector<Point2D>*                     points{nullptr};
    const std::vector<Triangle>*                    tris{nullptr};
    std::unordered_map<std::uint64_t, std::array<int, 2>> across; // edge -> up to two triangle ids

    TriangulationLocator(const std::vector<Point2D>& pts, const std::vector<Triangle>& ts)
        : points(&pts), tris(&ts) {
        auto add = [&](std::size_t u, std::size_t v, int t) {
            const auto key = detail::jw_edge_key(u, v);
            auto       it  = across.find(key);
            if (it == across.end()) across.emplace(key, std::array<int, 2>{t, -1});
            else it->second[1] = t;
        };
        for (int t = 0; t < static_cast<int>(ts.size()); ++t) {
            add(ts[t].a, ts[t].b, t);
            add(ts[t].b, ts[t].c, t);
            add(ts[t].c, ts[t].a, t);
        }
    }

    // The triangle on the other side of undirected edge {u,v} from triangle `from`,
    // or -1 if {u,v} is a hull edge.
    int neighbor(std::size_t u, std::size_t v, int from) const {
        auto it = across.find(detail::jw_edge_key(u, v));
        if (it == across.end()) return -1;
        const auto& pr = it->second;
        return pr[0] == from ? pr[1] : pr[0];
    }

    // Index of the triangle containing `q`, or -1 if `q` lies outside the
    // triangulation. `seed` makes the randomized jump/walk reproducible.
    int locate(const Point2D& q, std::uint64_t seed = 1) const {
        using namespace detail;
        const auto& P = *points;
        const auto& T = *tris;
        const int   n = static_cast<int>(T.size());
        if (n == 0) return -1;
        JwRng rng(seed);

        // Jump: sample ~ n^{1/3} triangles, keep the one nearest the query.
        int    samples = 1;
        while (samples * samples * samples < n) ++samples;
        int    cur   = 0;
        double bestd = 1e300;
        for (int s = 0; s < samples; ++s) {
            const int   t  = static_cast<int>(rng.next() % static_cast<std::uint64_t>(n));
            const auto& tr = T[t];
            const double cx = (P[tr.a].x + P[tr.b].x + P[tr.c].x) / 3.0;
            const double cy = (P[tr.a].y + P[tr.b].y + P[tr.c].y) / 3.0;
            const double d  = (cx - q.x) * (cx - q.x) + (cy - q.y) * (cy - q.y);
            if (d < bestd) { bestd = d; cur = t; }
        }

        // Walk: cross the edge whose far side holds q, choosing a random start
        // edge each step so the walk cannot cycle.
        for (int step = 0; step < 4 * n + 16; ++step) {
            const auto&        tr = T[cur];
            const std::size_t  vs[3] = {tr.a, tr.b, tr.c};
            const int          r     = static_cast<int>(rng.next() % 3);
            int                moved = -1;
            for (int i = 0; i < 3; ++i) {
                const int         k = (r + i) % 3;
                const std::size_t u = vs[k], v = vs[(k + 1) % 3];
                // CCW triangle: q inside means it is left of every directed edge.
                if (jw_orient(P[u], P[v], q) < 0) { // q is on the far side of {u,v}
                    const int nb = neighbor(u, v, cur);
                    if (nb == -1) return -1;        // walked off the convex hull
                    moved = nb;
                    break;
                }
            }
            if (moved == -1) return cur; // inside every edge -> found it
            cur = moved;
        }
        return -1;
    }
};

// Convenience one-shot: locate `q` in the triangulation of `points`/`tris`.
inline int locate_triangle(const std::vector<Point2D>&  points,
                           const std::vector<Triangle>& tris,
                           const Point2D&                q,
                           std::uint64_t                 seed = 1) {
    return TriangulationLocator(points, tris).locate(q, seed);
}

} // namespace datamunge::geometry
