#include <gtest/gtest.h>

#include <datamunge/algorithms/bkm.hpp>
#include <datamunge/geometry/distance_transform.hpp>
#include <datamunge/geometry/gjk.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using namespace datamunge::algorithms;
using namespace datamunge::geometry;

// ---------------- Euclidean distance transform ----------------

TEST(DistanceTransform, MatchesBruteForce) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 300; ++t) {
        const int rows = 1 + static_cast<int>(rng() % 20), cols = 1 + static_cast<int>(rng() % 20);
        std::vector<std::vector<char>> g(rows, std::vector<char>(cols, 0));
        std::vector<std::pair<int, int>> sites;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                if (rng() % 5 == 0) { g[r][c] = 1; sites.emplace_back(r, c); }
        const auto dt = euclidean_distance_transform(g);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                double best = std::numeric_limits<double>::infinity();
                for (auto [sr, sc] : sites)
                    best = std::min(best, std::sqrt((double)((r - sr) * (r - sr) + (c - sc) * (c - sc))));
                if (sites.empty()) EXPECT_TRUE(std::isinf(dt[r][c]));
                else EXPECT_NEAR(dt[r][c], best, 1e-9) << "r=" << r << " c=" << c;
            }
    }
}

// ---------------- GJK distance ----------------

namespace {

// Brute-force convex-polygon distance: 0 if they overlap, else min vertex-edge.
double point_seg_dist(const Point2D& p, const Point2D& a, const Point2D& b) {
    const Point2D ab{b.x - a.x, b.y - a.y};
    const double  den = ab.x * ab.x + ab.y * ab.y;
    double        t   = den > 0 ? ((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / den : 0;
    t                 = std::max(0.0, std::min(1.0, t));
    const double dx = p.x - (a.x + t * ab.x), dy = p.y - (a.y + t * ab.y);
    return std::sqrt(dx * dx + dy * dy);
}
bool point_in_convex(const Point2D& p, const std::vector<Point2D>& poly) {
    int n = poly.size();
    if (n < 3) return false;
    bool pos = false, neg = false;
    for (int i = 0; i < n; ++i) {
        const Point2D& a = poly[i];
        const Point2D& b = poly[(i + 1) % n];
        const double   cr = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if (cr > 1e-9) pos = true;
        if (cr < -1e-9) neg = true;
    }
    return !(pos && neg);
}
bool segs_cross(const Point2D& a, const Point2D& b, const Point2D& c, const Point2D& d) {
    auto cr = [](const Point2D& o, const Point2D& p, const Point2D& q) {
        return (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
    };
    const double d1 = cr(c, d, a), d2 = cr(c, d, b), d3 = cr(a, b, c), d4 = cr(a, b, d);
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
}
double brute_poly_dist(const std::vector<Point2D>& A, const std::vector<Point2D>& B) {
    for (const auto& p : A) if (point_in_convex(p, B)) return 0.0;
    for (const auto& p : B) if (point_in_convex(p, A)) return 0.0;
    for (std::size_t i = 0; i < A.size(); ++i)     // edge-crossing overlap
        for (std::size_t j = 0; j < B.size(); ++j)
            if (segs_cross(A[i], A[(i + 1) % A.size()], B[j], B[(j + 1) % B.size()])) return 0.0;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < A.size(); ++i)
        for (std::size_t j = 0; j < B.size(); ++j) {
            best = std::min(best, point_seg_dist(A[i], B[j], B[(j + 1) % B.size()]));
            best = std::min(best, point_seg_dist(B[j], A[i], A[(i + 1) % A.size()]));
        }
    return best;
}

// A convex polygon (regular n-gon) centred at (cx,cy) with radius r.
std::vector<Point2D> ngon(double cx, double cy, double r, int n, double rot) {
    std::vector<Point2D> p;
    for (int i = 0; i < n; ++i) {
        const double a = rot + 2 * M_PI * i / n;
        p.push_back({cx + r * std::cos(a), cy + r * std::sin(a)});
    }
    return p;
}

} // namespace

TEST(Gjk, DistanceMatchesBruteForce) {
    std::mt19937_64                        rng(2);
    std::uniform_real_distribution<double> pos(-10, 10), rad(0.5, 3.0), rot(0, 6.28);
    std::uniform_int_distribution<int>     sides(3, 6);
    for (int t = 0; t < 2000; ++t) {
        auto A = ngon(pos(rng), pos(rng), rad(rng), sides(rng), rot(rng));
        auto B = ngon(pos(rng), pos(rng), rad(rng), sides(rng), rot(rng));
        EXPECT_NEAR(gjk_distance(A, B), brute_poly_dist(A, B), 1e-6) << "t=" << t;
    }
}

TEST(Gjk, OverlapAndTouch) {
    auto sq = [](double cx, double cy, double h) {
        return std::vector<Point2D>{{cx - h, cy - h}, {cx + h, cy - h}, {cx + h, cy + h}, {cx - h, cy + h}};
    };
    EXPECT_NEAR(gjk_distance(sq(0, 0, 1), sq(5, 0, 1)), 3.0, 1e-9);  // gap 3
    EXPECT_NEAR(gjk_distance(sq(0, 0, 1), sq(0, 0, 1)), 0.0, 1e-9);  // identical (overlap)
    EXPECT_NEAR(gjk_distance(sq(0, 0, 1), sq(1.5, 0, 1)), 0.0, 1e-9); // overlapping
}

// ---------------- BKM ----------------

TEST(Bkm, LnMatchesStdLog) {
    for (double v = 1.0; v <= 4.7; v += 0.01) EXPECT_NEAR(bkm_ln(v), std::log(v), 1e-9) << "v=" << v;
    EXPECT_NEAR(bkm_ln(1.0), 0.0, 1e-12);
    EXPECT_NEAR(bkm_ln(M_E), 1.0, 1e-9);
}

TEST(Bkm, ExpMatchesStdExp) {
    for (double y = 0.0; y <= 1.5; y += 0.005) EXPECT_NEAR(bkm_exp(y), std::exp(y), std::exp(y) * 1e-9) << "y=" << y;
    EXPECT_NEAR(bkm_exp(0.0), 1.0, 1e-12);
    EXPECT_NEAR(bkm_exp(1.0), M_E, 1e-8);
}
