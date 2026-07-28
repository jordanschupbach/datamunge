#include <gtest/gtest.h>

#include <datamunge/geometry/bentley_ottmann.hpp>
#include <datamunge/geometry/convex_hull.hpp>
#include <datamunge/geometry/de_boor.hpp>
#include <datamunge/geometry/hull_algorithms.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::geometry;

// ---------------- Quickhull ----------------

namespace {
std::vector<Point2D> canon(std::vector<Point2D> h) {
    if (h.size() < 2) return h;
    std::size_t s = 0;
    for (std::size_t i = 1; i < h.size(); ++i)
        if (h[i].x < h[s].x || (h[i].x == h[s].x && h[i].y < h[s].y)) s = i;
    std::rotate(h.begin(), h.begin() + s, h.end());
    return h;
}
bool eqp(const std::vector<Point2D>& a, const std::vector<Point2D>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}
} // namespace

TEST(Quickhull, AgreesWithMonotoneChain) {
    std::mt19937_64                    rng(1);
    std::uniform_int_distribution<int> coord(-100, 100);
    for (int t = 0; t < 500; ++t) {
        const int            npts = 3 + static_cast<int>(rng() % 60);
        std::vector<Point2D> pts;
        for (int i = 0; i < npts; ++i) pts.push_back({(double)coord(rng), (double)coord(rng)});
        EXPECT_TRUE(eqp(canon(quickhull(pts)), canon(convex_hull(pts)))) << "t=" << t;
    }
    // Degenerate: collinear and edge point excluded.
    EXPECT_TRUE(eqp(canon(quickhull({{0, 0}, {1, 1}, {2, 2}})), canon(convex_hull({{0, 0}, {1, 1}, {2, 2}}))));
    std::vector<Point2D> sq = {{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 0}, {2, 2}};
    EXPECT_EQ(quickhull(sq).size(), 4u);
}

// ---------------- De Boor ----------------

TEST(DeBoor, ConstantControlPoints) {
    // All control points equal => curve is that constant point everywhere.
    const std::vector<Point2D> ctrl(6, {3.0, -2.0});
    const std::vector<double>  knots = {0, 0, 0, 0, 1, 2, 3, 3, 3, 3}; // degree 3, 6 control pts
    for (double t = 0.0; t <= 3.0; t += 0.25) {
        const Point2D p = de_boor(3, knots, ctrl, t);
        EXPECT_NEAR(p.x, 3.0, 1e-9) << "t=" << t;
        EXPECT_NEAR(p.y, -2.0, 1e-9) << "t=" << t;
    }
}

TEST(DeBoor, ClampedEndpointInterpolation) {
    const std::vector<Point2D> ctrl  = {{0, 0}, {1, 3}, {3, 3}, {4, 0}, {6, -2}};
    const std::vector<double>  knots = {0, 0, 0, 0, 1, 2, 2, 2, 2}; // degree 3, 5 control pts
    const Point2D              start = de_boor(3, knots, ctrl, 0.0);
    const Point2D              end   = de_boor(3, knots, ctrl, 2.0 - 1e-12);
    EXPECT_NEAR(start.x, ctrl.front().x, 1e-6);
    EXPECT_NEAR(start.y, ctrl.front().y, 1e-6);
    EXPECT_NEAR(end.x, ctrl.back().x, 1e-6);
    EXPECT_NEAR(end.y, ctrl.back().y, 1e-6);
}

TEST(DeBoor, MatchesBezierForBernsteinKnots) {
    // Degree-3 B-spline with knots [0,0,0,0,1,1,1,1] is the cubic Bezier of its 4 control points.
    const std::vector<Point2D> ctrl  = {{0, 0}, {1, 2}, {3, 2}, {4, 0}};
    const std::vector<double>  knots = {0, 0, 0, 0, 1, 1, 1, 1};
    for (double t = 0.0; t <= 1.0; t += 0.1) {
        const double u = t > 1.0 ? 1.0 : t;
        // Bernstein cubic Bezier
        const double b0 = (1 - u) * (1 - u) * (1 - u), b1 = 3 * u * (1 - u) * (1 - u),
                     b2 = 3 * u * u * (1 - u), b3 = u * u * u;
        const double bx = b0 * ctrl[0].x + b1 * ctrl[1].x + b2 * ctrl[2].x + b3 * ctrl[3].x;
        const double by = b0 * ctrl[0].y + b1 * ctrl[1].y + b2 * ctrl[2].y + b3 * ctrl[3].y;
        const Point2D p = de_boor(3, knots, ctrl, u < 1.0 ? u : 1.0 - 1e-12);
        EXPECT_NEAR(p.x, bx, 1e-6) << "t=" << t;
        EXPECT_NEAR(p.y, by, 1e-6) << "t=" << t;
    }
}

// ---------------- Bentley-Ottmann ----------------

namespace {
std::vector<Point2D> brute_intersections(const std::vector<Segment>& segs) {
    std::vector<Point2D> out;
    for (std::size_t i = 0; i < segs.size(); ++i)
        for (std::size_t j = i + 1; j < segs.size(); ++j) {
            Point2D p;
            if (detail::proper_intersection(segs[i].a, segs[i].b, segs[j].a, segs[j].b, p)) out.push_back(p);
        }
    return out;
}
void sort_pts(std::vector<Point2D>& v) {
    std::sort(v.begin(), v.end(), [](const Point2D& a, const Point2D& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
}
} // namespace

TEST(BentleyOttmann, MatchesBruteForce) {
    std::mt19937_64                        rng(3);
    std::uniform_real_distribution<double> coord(0.0, 1000.0);
    for (int t = 0; t < 300; ++t) {
        const int            m = 3 + static_cast<int>(rng() % 25);
        std::vector<Segment> segs;
        for (int i = 0; i < m; ++i)
            segs.push_back({{coord(rng), coord(rng)}, {coord(rng), coord(rng)}});

        auto bo    = bentley_ottmann(segs);
        auto brute = brute_intersections(segs);
        sort_pts(bo);
        sort_pts(brute);
        ASSERT_EQ(bo.size(), brute.size()) << "t=" << t << " counts differ";
        for (std::size_t i = 0; i < bo.size(); ++i) {
            EXPECT_NEAR(bo[i].x, brute[i].x, 1e-6) << "t=" << t;
            EXPECT_NEAR(bo[i].y, brute[i].y, 1e-6) << "t=" << t;
        }
    }
}

TEST(BentleyOttmann, KnownCrossing) {
    // An X: two segments crossing at (1,1).
    std::vector<Segment> segs = {{{0, 0}, {2, 2}}, {{0, 2}, {2, 0}}};
    auto                 x = bentley_ottmann(segs);
    ASSERT_EQ(x.size(), 1u);
    EXPECT_NEAR(x[0].x, 1.0, 1e-9);
    EXPECT_NEAR(x[0].y, 1.0, 1e-9);
    // Two parallel segments: no intersection.
    EXPECT_TRUE(bentley_ottmann({{{0, 0}, {2, 0}}, {{0, 1}, {2, 1}}}).empty());
}
