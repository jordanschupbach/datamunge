#include <gtest/gtest.h>

#include <datamunge/algorithms/filtered_back_projection.hpp>
#include <datamunge/algorithms/miser.hpp>
#include <datamunge/geometry/best_bin_first.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace datamunge::algorithms;
using namespace datamunge::geometry;

// ---------------- MISER ----------------

TEST(Miser, IntegratesLinearExactlyInExpectation) {
    // integral of x+y over [0,1]^2 = 1
    auto f = [](const std::vector<double>& x) { return x[0] + x[1]; };
    const auto r = miser_integrate(f, {0, 0}, {1, 1}, 200000, 1);
    EXPECT_NEAR(r.value, 1.0, 5e-3);
    EXPECT_GT(r.variance, 0.0);
}

TEST(Miser, IntegratesPeakedGaussian) {
    // integral exp(-100((x-.5)^2+(y-.5)^2)) over [0,1]^2 ~ pi/100
    auto f = [](const std::vector<double>& x) {
        const double dx = x[0] - 0.5, dy = x[1] - 0.5;
        return std::exp(-100.0 * (dx * dx + dy * dy));
    };
    const double truth = M_PI / 100.0;
    const auto   r     = miser_integrate(f, {0, 0}, {1, 1}, 300000, 7);
    EXPECT_NEAR(r.value, truth, 5e-4);
}

TEST(Miser, BeatsPlainMonteCarloOnPeakedIntegrand) {
    auto f = [](const std::vector<double>& x) {
        const double dx = x[0] - 0.5, dy = x[1] - 0.5;
        return std::exp(-200.0 * (dx * dx + dy * dy));
    };
    const double truth = M_PI / 200.0; // integral over the plane, well inside [0,1]^2
    const long   calls = 40000;
    const int    reps  = 12;

    // Compare RMS error over several seeds -- MISER should reduce variance on this
    // sharply peaked integrand.
    double miser_mse = 0, plain_mse = 0;
    for (int seed = 1; seed <= reps; ++seed) {
        const double mv = miser_integrate(f, {0, 0}, {1, 1}, calls, seed).value;
        miser_mse += (mv - truth) * (mv - truth);

        std::mt19937_64                        rng(1000 + seed);
        std::uniform_real_distribution<double> u(0, 1);
        double                                 s = 0;
        for (long k = 0; k < calls; ++k) {
            const double a = u(rng), b = u(rng), dx = a - 0.5, dy = b - 0.5;
            s += std::exp(-200.0 * (dx * dx + dy * dy));
        }
        const double pv = s / calls; // volume 1
        plain_mse += (pv - truth) * (pv - truth);
    }
    EXPECT_LT(std::sqrt(miser_mse / reps), std::sqrt(plain_mse / reps))
        << "miser RMS " << std::sqrt(miser_mse / reps) << " plain RMS " << std::sqrt(plain_mse / reps);
}

// ---------------- Filtered back-projection ----------------

TEST(FilteredBackProjection, ReconstructsDiskPhantom) {
    const int           N = 64;
    std::vector<double> img(N * N, 0.0);
    const double        cx = (N - 1) / 2.0, cy = (N - 1) / 2.0, R = 14;
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            const double dx = c - cx, dy = r - cy;
            if (dx * dx + dy * dy <= R * R) img[r * N + c] = 1.0;
        }
    const int           nd = static_cast<int>(std::ceil(N * 1.5));
    std::vector<double> ang;
    const int           M = 120;
    for (int i = 0; i < M; ++i) ang.push_back(M_PI * i / M);

    const auto sino = radon_transform(img, N, ang, nd);
    const auto rec  = filtered_back_projection(sino, N, ang, nd);

    // Pearson correlation with the phantom
    const int n = N * N;
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int i = 0; i < n; ++i) { sx += img[i]; sy += rec[i]; sxx += img[i] * img[i]; syy += rec[i] * rec[i]; sxy += img[i] * rec[i]; }
    const double corr = (n * sxy - sx * sy) / std::sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    EXPECT_GT(corr, 0.95);

    // interior recovers ~1, background ~0
    double din = 0, dbg = 0; int cin = 0, cbg = 0;
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            const double dx = c - cx, dy = r - cy, d2 = dx * dx + dy * dy;
            if (d2 <= (R - 3) * (R - 3)) { din += rec[r * N + c]; ++cin; }
            else if (d2 > (R + 5) * (R + 5)) { dbg += rec[r * N + c]; ++cbg; }
        }
    EXPECT_NEAR(din / cin, 1.0, 0.1);
    EXPECT_NEAR(dbg / cbg, 0.0, 0.05);
}

// ---------------- Best Bin First ----------------

TEST(BestBinFirst, UnboundedMatchesBruteForce) {
    std::mt19937_64                        rng(1);
    std::uniform_real_distribution<double> u(-50, 50);
    std::vector<Point2D>                   pts;
    for (int i = 0; i < 1500; ++i) pts.push_back({u(rng), u(rng)});
    BestBinFirst bbf(pts);
    auto brute = [&](const Point2D& q) {
        int bi = 0; double bd = 1e300;
        for (std::size_t i = 0; i < pts.size(); ++i) {
            const double dx = q.x - pts[i].x, dy = q.y - pts[i].y, d = dx * dx + dy * dy;
            if (d < bd) { bd = d; bi = static_cast<int>(i); }
        }
        return bi;
    };
    for (int t = 0; t < 400; ++t) {
        const Point2D q{u(rng), u(rng)};
        EXPECT_EQ(bbf.nearest(q, static_cast<int>(pts.size()) + 1), brute(q)) << "t=" << t;
    }
}

TEST(BestBinFirst, BoundedIsAccurateAndValid) {
    std::mt19937_64                        rng(5);
    std::uniform_real_distribution<double> u(-50, 50);
    std::vector<Point2D>                   pts;
    for (int i = 0; i < 3000; ++i) pts.push_back({u(rng), u(rng)});
    BestBinFirst bbf(pts);
    auto brute = [&](const Point2D& q) {
        int bi = 0; double bd = 1e300;
        for (std::size_t i = 0; i < pts.size(); ++i) {
            const double dx = q.x - pts[i].x, dy = q.y - pts[i].y, d = dx * dx + dy * dy;
            if (d < bd) { bd = d; bi = static_cast<int>(i); }
        }
        return bi;
    };
    int exact = 0; const int total = 400;
    for (int t = 0; t < total; ++t) {
        const Point2D q{u(rng), u(rng)};
        const int     b = brute(q);
        const int     f = bbf.nearest(q, 50); // budget << 3000
        ASSERT_GE(f, 0);
        if (f == b) ++exact;
        const double df = std::hypot(q.x - pts[f].x, q.y - pts[f].y);
        const double db = std::hypot(q.x - pts[b].x, q.y - pts[b].y);
        EXPECT_LE(df, 2.0 * db + 1e-9); // always a close neighbor
    }
    EXPECT_GT(exact, static_cast<int>(0.9 * total)); // exact for the vast majority
}
