/// @file hull_algorithms.cpp
/// @brief Graham scan, Jarvis march, Chan's algorithm with binary-search tangents, Quickhull,
///        convex layers and random point clouds.

#include <datamunge/geometry/hull_algorithms.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>

namespace datamunge::geometry {

/// @brief Helpers private to this file.
namespace {

/// @brief Sort points lexicographically by (x, y) and drop duplicates.
std::vector<Point2D> sorted_unique(std::vector<Point2D> pts) {
    std::sort(pts.begin(), pts.end(),
              [](const Point2D& a, const Point2D& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    return pts;
}

/// @brief Twice the signed area (shoelace); positive when the polygon is counterclockwise.
double signed_area2(const std::vector<Point2D>& poly) {
    double            a = 0;
    const std::size_t n = poly.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Point2D& p = poly[i];
        const Point2D& q = poly[(i + 1) % n];
        a += p.x * q.y - q.x * p.y;
    }
    return a;
}

/// @brief Reverse a polygon in place if it is clockwise.
void make_ccw(std::vector<Point2D>& poly) {
    if (poly.size() >= 3 && signed_area2(poly) < 0) std::reverse(poly.begin(), poly.end());
}

/// @brief Hash of a point's coordinate bit patterns (with -0 folded into +0).
struct PointHash {
    /// @brief Hash a point.
    std::size_t operator()(const Point2D& p) const noexcept {
        const auto hx = std::bit_cast<std::uint64_t>(p.x + 0.0);
        const auto hy = std::bit_cast<std::uint64_t>(p.y + 0.0);
        std::uint64_t h = hx * 0x9E3779B97F4A7C15ULL ^ (hy + 0x632BE59BD9B4E019ULL + (hx << 6) + (hx >> 2));
        h ^= h >> 29;
        return static_cast<std::size_t>(h * 0xBF58476D1CE4E5B9ULL);
    }
};

/// @brief Remove duplicate points in O(n) expected time, keeping first occurrences in order.
std::vector<Point2D> unique_in_order(const std::vector<Point2D>& points) {
    std::unordered_map<Point2D, char, PointHash> seen;
    seen.reserve(points.size() * 2);
    std::vector<Point2D> out;
    out.reserve(points.size());
    for (const Point2D& p : points)
        if (seen.emplace(p, 0).second) out.push_back(p);
    return out;
}

/// @brief Graham scan on points, adding its orientation tests to @p tests.
std::vector<Point2D> graham_counted(std::vector<Point2D> points, long long& tests) {
    std::vector<Point2D> pts = sorted_unique(std::move(points));
    const int            n   = static_cast<int>(pts.size());
    if (n < 3) return pts;

    int piv = 0;
    for (int i = 1; i < n; ++i)
        if (pts[i].y < pts[piv].y || (pts[i].y == pts[piv].y && pts[i].x < pts[piv].x)) piv = i;
    std::swap(pts[0], pts[piv]);
    const Point2D p0 = pts[0];
    std::sort(pts.begin() + 1, pts.end(), [&](const Point2D& a, const Point2D& b) {
        ++tests;
        const double c = cross(p0, a, b);
        if (c != 0) return c > 0; // a before b when p0->a->b turns left
        return squared_distance(p0, a) < squared_distance(p0, b);
    });

    std::vector<Point2D> hull;
    for (const Point2D& p : pts) {
        while (hull.size() >= 2) {
            ++tests;
            if (cross(hull[hull.size() - 2], hull.back(), p) > 0) break;
            hull.pop_back();
        }
        hull.push_back(p);
    }
    return hull; // counterclockwise
}

/// @brief Jarvis march on points, adding its orientation tests to @p tests.
std::vector<Point2D> jarvis_counted(std::vector<Point2D> points, long long& tests) {
    std::vector<Point2D> pts = sorted_unique(std::move(points));
    const int            n   = static_cast<int>(pts.size());
    if (n < 3) return pts;

    const int            start = 0; // leftmost (lowest on ties) after the sort
    std::vector<Point2D> hull;
    int                  cur = start;
    do {
        hull.push_back(pts[cur]);
        const int first = (cur + 1) % n;
        int       next  = first;
        for (int q = 0; q < n; ++q) {
            if (q == cur || q == first) continue;
            ++tests;
            const double o = cross(pts[cur], pts[next], pts[q]);
            if (o < 0 || (o == 0 && squared_distance(pts[cur], pts[q]) > squared_distance(pts[cur], pts[next])))
                next = q;
        }
        cur = next;
    } while (cur != start && static_cast<int>(hull.size()) <= n + 1);

    make_ccw(hull);
    return hull;
}

/// @brief Whether candidate @p q beats @p best as the next wrap vertex from @p cur: more
///        clockwise, or collinear and farther.
bool better_wrap(const Point2D& cur, const Point2D& best, const Point2D& q, long long& tests) {
    ++tests;
    const double o = cross(cur, best, q);
    return o < 0 || (o == 0 && squared_distance(cur, q) > squared_distance(cur, best));
}

/// @brief Tangent of a small or degenerate polygon by scanning every vertex.
int tangent_scan(const std::vector<Point2D>& poly, const Point2D& p, long long& tests) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(poly.size()); ++i)
        if (better_wrap(p, poly[best], poly[i], tests)) best = i;
    return best;
}

/// @brief Recursively collect the hull vertices strictly left of the directed edge a->b.
void quickhull_rec(const std::vector<Point2D>& pts, const Point2D& a, const Point2D& b, std::vector<Point2D>& out) {
    int    far  = -1;
    double best = 0;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        const double c = cross(a, b, pts[i]);
        if (c > best) {
            best = c;
            far  = static_cast<int>(i);
        }
    }
    if (far < 0) return; // no point strictly left: edge a->b is on the hull
    const Point2D c = pts[far];

    std::vector<Point2D> left_ac, left_cb;
    for (const Point2D& p : pts) {
        if (cross(a, c, p) > 0) left_ac.push_back(p);
        else if (cross(c, b, p) > 0) left_cb.push_back(p);
    }
    quickhull_rec(left_ac, a, c, out);
    out.push_back(c);
    quickhull_rec(left_cb, c, b, out);
}

} // namespace

std::vector<Point2D> graham_scan(std::vector<Point2D> points) {
    long long tests = 0;
    return graham_counted(std::move(points), tests);
}

std::vector<Point2D> graham_scan(std::vector<Point2D> points, long long& orientation_tests) {
    return graham_counted(std::move(points), orientation_tests);
}

std::vector<Point2D> jarvis_march(std::vector<Point2D> points) {
    long long tests = 0;
    return jarvis_counted(std::move(points), tests);
}

std::vector<Point2D> jarvis_march(std::vector<Point2D> points, long long& orientation_tests) {
    return jarvis_counted(std::move(points), orientation_tests);
}

TangentSearch convex_polygon_tangent(const std::vector<Point2D>& polygon, const Point2D& p) {
    TangentSearch out;
    const int     k = static_cast<int>(polygon.size());
    if (k == 0) return out;
    if (k <= 3) {
        for (int i = 0; i < k; ++i) out.probes.push_back(i);
        out.index = tangent_scan(polygon, p, out.orientation_tests);
        return out;
    }
    // Vertex i is "more clockwise" than j (seen from p) when v_i lies right of p->v_j.
    const auto v = [&](int i) -> const Point2D& { return polygon[((i % k) + k) % k]; };
    const auto before = [&](int i, int j) {
        ++out.orientation_tests;
        return cross(p, v(j), v(i)) < 0;
    };
    // Edge i (v_i -> v_{i+1}) turns clockwise as seen from p.
    const auto down = [&](int i) { return before(i + 1, i); };
    const auto is_min = [&](int i) { return !down(i) && down(i - 1); };

    int  a = 0, b = k; // the minimum lies on the chain v_a .. v_b (v_k = v_0)
    int  found = -1;
    out.probes.push_back(0);
    if (is_min(0)) found = 0;
    while (found < 0 && b - a > 2) {
        const int c = (a + b) / 2;
        out.probes.push_back(c);
        if (is_min(c)) {
            found = c;
            break;
        }
        const bool down_a = down(a);
        const bool down_c = down(c);
        if (down_a) {
            if (!down_c) b = c;            // descending at a, rising at c: minimum between
            else if (before(c, a)) a = c;  // same descent, c lower: minimum after c
            else b = c;                    // c on the descent after the maximum
        } else {
            if (down_c) a = c;             // rising at a, falling at c: minimum after c
            else if (before(c, a)) b = c;  // c on a later rise: minimum between
            else a = c;                    // same rise, c higher: minimum after c
        }
    }
    if (found < 0) { // a short chain is left: scan it
        found = a % k;
        for (int i = a + 1; i <= b; ++i) {
            out.probes.push_back(i % k);
            if (better_wrap(p, v(found), v(i), out.orientation_tests)) found = i % k;
        }
    }
    // Of two vertices collinear with p, take the farther one.
    for (const int nb : {found - 1, found + 1})
        if (better_wrap(p, v(found), v(nb), out.orientation_tests)) found = ((nb % k) + k) % k;
    out.index = found;
    return out;
}

ChanResult chans_algorithm_trace(const std::vector<Point2D>& points, const ChanOptions& options) {
    ChanResult           result;
    std::vector<Point2D> pts = unique_in_order(points);
    const long long      n   = static_cast<long long>(pts.size());
    result.distinct_points   = static_cast<int>(n);
    if (n < 3) {
        result.hull = sorted_unique(pts);
        return result;
    }

    // The lowest (then leftmost) point is a hull vertex: the wrap starts there.
    int start = 0;
    for (int i = 1; i < n; ++i)
        if (pts[i].y < pts[start].y || (pts[i].y == pts[start].y && pts[i].x < pts[start].x)) start = i;

    for (long long m = std::clamp<long long>(options.initial_m, 3, n);; m = std::min(n, options.square_guess ? m * m : 2 * m)) {
        ChanRound round;
        round.m      = m;
        round.groups = static_cast<int>((n + m - 1) / m);
        std::vector<std::vector<Point2D>> hulls(round.groups);
        int start_group = -1, start_index = -1;
        for (int g = 0; g < round.groups; ++g) {
            const long long lo = g * m, hi = std::min(n, (g + 1) * m);
            hulls[g] = graham_counted(std::vector<Point2D>(pts.begin() + lo, pts.begin() + hi), round.hull_tests);
            round.group_hull_vertices += static_cast<int>(hulls[g].size());
            if (start >= lo && start < hi)
                for (int i = 0; i < static_cast<int>(hulls[g].size()); ++i)
                    if (hulls[g][i] == pts[start]) start_group = g, start_index = i;
        }

        std::vector<Point2D> path;
        int                  cg = start_group, ci = start_index; // current vertex's group and index
        for (long long step = 0; step < m; ++step) {
            const Point2D cur = hulls[cg][ci];
            path.push_back(cur);
            ++round.wrap_steps;
            // Its own group offers the next vertex of its hull; every other group its tangent.
            int     bg = -1, bi = -1;
            Point2D best;
            for (int g = 0; g < round.groups; ++g) {
                const auto& h = hulls[g];
                int         t = -1;
                if (g == cg) {
                    if (h.size() < 2) continue;
                    t = (ci + 1) % static_cast<int>(h.size());
                } else if (options.binary_tangent) {
                    const TangentSearch ts = convex_polygon_tangent(h, cur);
                    round.wrap_tests += ts.orientation_tests;
                    t = ts.index;
                } else {
                    t = tangent_scan(h, cur, round.wrap_tests);
                }
                if (bg < 0 || better_wrap(cur, best, h[t], round.wrap_tests)) {
                    bg   = g;
                    bi   = t;
                    best = h[t];
                }
            }
            cg = bg;
            ci = bi;
            if (best == pts[start]) {
                round.completed = true;
                break;
            }
        }
        result.orientation_tests += round.hull_tests + round.wrap_tests;
        if (options.record_geometry) {
            round.group_hulls = hulls;
            round.wrap_path   = path;
        }
        const bool done = round.completed;
        result.rounds.push_back(std::move(round));
        if (done || m >= n) { // with m = n there is one group, whose hull closes the wrap
            result.hull = std::move(path);
            make_ccw(result.hull);
            return result;
        }
    }
}

std::vector<Point2D> chans_algorithm(std::vector<Point2D> points) { return chans_algorithm_trace(points).hull; }

std::vector<Point2D> quickhull(std::vector<Point2D> points) {
    std::vector<Point2D> pts = sorted_unique(std::move(points));
    const int            n   = static_cast<int>(pts.size());
    if (n < 3) return pts;

    const Point2D a = pts.front(); // leftmost (sorted by x, then y)
    const Point2D b = pts.back();  // rightmost

    std::vector<Point2D> above, below;
    for (const Point2D& p : pts) {
        const double c = cross(a, b, p);
        if (c > 0) above.push_back(p);
        else if (c < 0) below.push_back(p);
    }

    std::vector<Point2D> hull;
    hull.push_back(a);
    quickhull_rec(above, a, b, hull); // upper chain a -> b
    hull.push_back(b);
    quickhull_rec(below, b, a, hull); // lower chain b -> a
    make_ccw(hull);
    return hull;
}

std::vector<int> convex_layers(const std::vector<Point2D>& points) {
    std::vector<int>     layer(points.size(), -1);
    std::vector<Point2D> rest = unique_in_order(points);
    std::unordered_map<Point2D, int, PointHash> depth;
    for (int d = 0; !rest.empty(); ++d) {
        const std::vector<Point2D> hull = chans_algorithm_trace(rest).hull;
        std::unordered_map<Point2D, char, PointHash> on(hull.size() * 2);
        for (const Point2D& v : hull) {
            on.emplace(v, 0);
            depth.emplace(v, d);
        }
        std::erase_if(rest, [&](const Point2D& p) { return on.contains(p); });
    }
    for (std::size_t i = 0; i < points.size(); ++i) layer[i] = depth.at(points[i]);
    return layer;
}

std::vector<Point2D> random_point_cloud(int n, PointCloud cloud, std::uint64_t seed) {
    std::mt19937_64                        rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double>       normal(0.0, 1.0);
    constexpr double                       grid = 1048576.0; // 2^20
    const auto snap = [&](double v) { return std::round(v * grid) / grid; };
    std::vector<Point2D> pts;
    pts.reserve(static_cast<std::size_t>(std::max(n, 0)));
    if (cloud == PointCloud::Parabola) {
        std::vector<int> xs(static_cast<std::size_t>(std::max(n, 0)));
        std::iota(xs.begin(), xs.end(), -n / 2);
        std::shuffle(xs.begin(), xs.end(), rng);
        for (const int x : xs) pts.push_back({static_cast<double>(x), static_cast<double>(x) * x});
        return pts;
    }
    for (int i = 0; i < n; ++i) {
        switch (cloud) {
        case PointCloud::Square: pts.push_back({snap(unit(rng)), snap(unit(rng))}); break;
        case PointCloud::Disk: {
            const double r = std::sqrt(unit(rng)), t = 2 * std::numbers::pi * unit(rng);
            pts.push_back({snap(r * std::cos(t)), snap(r * std::sin(t))});
            break;
        }
        default: {
            const double x = normal(rng);
            pts.push_back({snap(x), snap(normal(rng))});
            break;
        }
        }
    }
    return pts;
}

} // namespace datamunge::geometry
