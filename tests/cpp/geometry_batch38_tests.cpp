#include <gtest/gtest.h>

#include <datamunge/geometry/delaunay.hpp>
#include <datamunge/geometry/geometric_hashing.hpp>
#include <datamunge/geometry/jump_and_walk.hpp>
#include <datamunge/geometry/laplacian_smoothing.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::geometry;

// ---------------- Laplacian smoothing ----------------

TEST(LaplacianSmoothing, OneStepIsNeighborCentroid) {
    // Path graph 0-1-2; only vertex 1 is free. lambda=1 => it jumps to the midpoint.
    std::vector<Point2D>          v{{0, 0}, {2, 5}, {4, 0}};
    std::vector<std::vector<int>> nb{{1}, {0, 2}, {1}};
    std::vector<char>             fixed{1, 0, 1};
    const auto                    out = laplacian_smooth(v, nb, fixed, 1.0, 1);
    EXPECT_NEAR(out[1].x, 2.0, 1e-12);
    EXPECT_NEAR(out[1].y, 0.0, 1e-12);
    EXPECT_EQ(out[0].x, 0.0); // fixed
    EXPECT_EQ(out[2].x, 4.0);
}

TEST(LaplacianSmoothing, EnergyDecreasesMonotonically) {
    std::mt19937_64                        rng(7);
    std::uniform_real_distribution<double> jitter(-3, 3);
    // A 12-vertex ring; boundary is the first and last (fixed), rest free, jittered.
    const int                     n = 12;
    std::vector<Point2D>          v(n);
    std::vector<std::vector<int>> nb(n);
    std::vector<char>             fixed(n, 0);
    for (int i = 0; i < n; ++i) v[i] = {static_cast<double>(i) + jitter(rng), jitter(rng)};
    for (int i = 0; i + 1 < n; ++i) { nb[i].push_back(i + 1); nb[i + 1].push_back(i); }
    fixed[0] = fixed[n - 1] = 1;
    double prev = dirichlet_energy(v, nb);
    for (int it = 0; it < 30; ++it) {
        v                = laplacian_smooth(v, nb, fixed, 0.5, 1);
        const double cur = dirichlet_energy(v, nb);
        EXPECT_LE(cur, prev + 1e-9) << "iteration " << it;
        prev = cur;
    }
}

// ---------------- Jump-and-Walk point location ----------------

TEST(JumpAndWalk, MatchesBruteForceContainment) {
    std::mt19937_64                        rng(11);
    std::uniform_real_distribution<double> coord(0, 100);
    for (int trial = 0; trial < 40; ++trial) {
        std::vector<Point2D> pts;
        for (int i = 0; i < 40; ++i) pts.push_back({coord(rng), coord(rng)});
        const auto tris = delaunay_triangulation(pts);
        if (tris.empty()) continue;
        TriangulationLocator loc(pts, tris);

        auto inside = [&](int t, const Point2D& q) {
            const auto& tr = tris[t];
            auto        o  = [](const Point2D& a, const Point2D& b, const Point2D& p) {
                return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
            };
            const double d0 = o(pts[tr.a], pts[tr.b], q), d1 = o(pts[tr.b], pts[tr.c], q),
                         d2 = o(pts[tr.c], pts[tr.a], q);
            return d0 >= -1e-9 && d1 >= -1e-9 && d2 >= -1e-9;
        };

        for (int qidx = 0; qidx < 30; ++qidx) {
            const Point2D q{coord(rng), coord(rng)};
            const int     got = loc.locate(q, 1 + qidx);
            // Brute force: does ANY triangle contain q?
            int brute = -1;
            for (int t = 0; t < static_cast<int>(tris.size()); ++t)
                if (inside(t, q)) { brute = t; break; }
            if (brute == -1) {
                EXPECT_EQ(got, -1) << "trial " << trial << " q outside hull";
            } else {
                ASSERT_NE(got, -1) << "trial " << trial << " q=" << q.x << "," << q.y;
                EXPECT_TRUE(inside(got, q)) << "located triangle must contain q";
            }
        }
    }
}

TEST(JumpAndWalk, VertexAndCentroidAreLocated) {
    std::vector<Point2D> pts{{0, 0}, {10, 0}, {10, 10}, {0, 10}, {5, 5}};
    const auto           tris = delaunay_triangulation(pts);
    ASSERT_FALSE(tris.empty());
    TriangulationLocator loc(pts, tris);
    // Centroid of the first triangle must be located inside it.
    const auto&   t0 = tris[0];
    const Point2D c{(pts[t0.a].x + pts[t0.b].x + pts[t0.c].x) / 3,
                    (pts[t0.a].y + pts[t0.b].y + pts[t0.c].y) / 3};
    EXPECT_NE(loc.locate(c, 3), -1);
    // A point well outside the square is reported outside.
    EXPECT_EQ(loc.locate({-50, -50}, 3), -1);
}

// ---------------- Geometric hashing ----------------

namespace {
Point2D apply_sim(const Point2D& p, double s, double ang, double tx, double ty) {
    const double c = s * std::cos(ang), d = s * std::sin(ang);
    return {c * p.x - d * p.y + tx, d * p.x + c * p.y + ty};
}
} // namespace

TEST(GeometricHashing, RecoversKnownSimilarity) {
    std::vector<Point2D> model{{0, 0}, {1, 0}, {0.3, 0.8}, {1.2, 0.9}, {0.6, -0.4}, {-0.2, 0.5}};
    const auto           table = build_geometric_hash(model, 0.05);

    std::mt19937_64                        rng(5);
    std::uniform_real_distribution<double> sd(0.5, 3.0), ad(-3.0, 3.0), td(-20, 20);
    for (int trial = 0; trial < 50; ++trial) {
        const double s = sd(rng), ang = ad(rng), tx = td(rng), ty = td(rng);
        std::vector<Point2D> scene;
        for (const auto& p : model) scene.push_back(apply_sim(p, s, ang, tx, ty));
        // Tolerance scales with the transform.
        const auto match = recognize(table, model, scene, 1e-6 * (1 + s * 5), 2);
        ASSERT_TRUE(match.found) << "trial " << trial;
        EXPECT_EQ(match.inliers, static_cast<int>(model.size())) << "trial " << trial;
        EXPECT_NEAR(match.scale, s, 1e-6 * s) << "trial " << trial;
        // Angles compared modulo 2pi.
        double da = std::fmod(match.angle - ang + 3 * M_PI, 2 * M_PI) - M_PI;
        EXPECT_NEAR(da, 0.0, 1e-6) << "trial " << trial;
    }
}

TEST(GeometricHashing, FindsModelAmidClutter) {
    std::vector<Point2D> model{{0, 0}, {2, 0}, {1, 1.5}, {2.2, 1.1}, {0.5, -0.7}};
    const auto           table = build_geometric_hash(model, 0.05);

    const double s = 1.7, ang = 0.9, tx = 12, ty = -8;
    std::vector<Point2D> scene;
    for (const auto& p : model) scene.push_back(apply_sim(p, s, ang, tx, ty));
    // Add clutter points far from the transformed model.
    std::mt19937_64                        rng(3);
    std::uniform_real_distribution<double> clut(-40, -25);
    for (int i = 0; i < 8; ++i) scene.push_back({clut(rng), clut(rng)});

    const auto match = recognize(table, model, scene, 1e-4, 2);
    ASSERT_TRUE(match.found);
    EXPECT_EQ(match.inliers, static_cast<int>(model.size()));
    EXPECT_NEAR(match.scale, s, 1e-4);
}
