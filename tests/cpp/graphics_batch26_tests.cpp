// Tests for the rendering & geometry batch: marching squares, summed-area table, Slerp,
// painter's algorithm, scanline fill, and Newell's normal. Checked against closed-form
// geometry (areas, contour radius, quaternion identities) and brute-force references.

#include <gtest/gtest.h>

#include <datamunge/algorithms/marching_squares.hpp>
#include <datamunge/algorithms/newell_normal.hpp>
#include <datamunge/algorithms/painters_algorithm.hpp>
#include <datamunge/algorithms/scanline_fill.hpp>
#include <datamunge/algorithms/slerp.hpp>
#include <datamunge/algorithms/summed_area_table.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::algorithms;

// ---------------------------------------------------------------------------- summed-area table
TEST(SummedAreaTable, MatchesBruteForce) {
    std::vector<std::vector<double>> g = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    SummedAreaTable sat(g);
    // Brute force every rectangle and compare.
    for (std::size_t r0 = 0; r0 < 4; ++r0)
        for (std::size_t c0 = 0; c0 < 4; ++c0)
            for (std::size_t r1 = r0; r1 < 4; ++r1)
                for (std::size_t c1 = c0; c1 < 4; ++c1) {
                    double brute = 0;
                    for (std::size_t r = r0; r <= r1; ++r)
                        for (std::size_t c = c0; c <= c1; ++c) brute += g[r][c];
                    EXPECT_DOUBLE_EQ(sat.rect_sum(r0, c0, r1, c1), brute);
                }
    EXPECT_DOUBLE_EQ(sat.rect_sum(0, 0, 3, 3), 136.0);  // sum 1..16
    EXPECT_DOUBLE_EQ(sat.rect_mean(0, 0, 1, 1), 3.5);   // mean of {1,2,5,6}
}

// ---------------------------------------------------------------------------- Slerp
TEST(Slerp, Endpoints) {
    Quaternion q0 = {1, 0, 0, 0};                          // identity
    Quaternion q1 = {std::cos(M_PI / 4), 0, 0, std::sin(M_PI / 4)};  // 90 deg about z
    auto a = slerp(q0, q1, 0.0), b = slerp(q0, q1, 1.0);
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(a[i], q0[i], 1e-12);
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(b[i], q1[i], 1e-12);
}

TEST(Slerp, StaysUnitAndConstantVelocity) {
    Quaternion q0 = {1, 0, 0, 0};
    Quaternion q1 = {std::cos(M_PI / 4), 0, 0, std::sin(M_PI / 4)};  // 90 deg about z
    // Halfway should be 45 deg about z: quaternion {cos(22.5deg),0,0,sin(22.5deg)}.
    auto mid = slerp(q0, q1, 0.5);
    EXPECT_NEAR(quat_norm(mid), 1.0, 1e-12);
    EXPECT_NEAR(mid[0], std::cos(M_PI / 8), 1e-9);
    EXPECT_NEAR(mid[3], std::sin(M_PI / 8), 1e-9);
    // Equal parameter steps sweep equal angles (constant angular velocity).
    double a1 = quat_angle_between(slerp(q0, q1, 0.0), slerp(q0, q1, 0.25));
    double a2 = quat_angle_between(slerp(q0, q1, 0.25), slerp(q0, q1, 0.5));
    EXPECT_NEAR(a1, a2, 1e-9);
}

TEST(Slerp, TakesShorterArc) {
    Quaternion q0 = {1, 0, 0, 0};
    Quaternion q1 = {-1, 0, 0, 0};  // same rotation as q0 (identity) up to sign
    auto m = slerp(q0, q1, 0.5);
    // Shorter arc: q1 flipped to +q0, so the whole path stays at the identity.
    EXPECT_NEAR(std::fabs(m[0]), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------- Newell's normal
TEST(NewellNormal, PlanarSquareInZPlane) {
    std::vector<Point3D> square = {{0, 0, 0}, {2, 0, 0}, {2, 2, 0}, {0, 2, 0}};
    auto n = newell_normal(square);
    EXPECT_NEAR(n[0], 0.0, 1e-12);
    EXPECT_NEAR(n[1], 0.0, 1e-12);
    EXPECT_NEAR(n[2], 1.0, 1e-12);  // CCW in xy-plane -> +z
    EXPECT_NEAR(newell_polygon_area(square), 4.0, 1e-12);
}

TEST(NewellNormal, TiltedTriangleMatchesCrossProduct) {
    std::vector<Point3D> tri = {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}};
    auto n = newell_normal(tri);
    // Cross product of edges (1,0,1) and (0,1,1) = (-1, -1, 1), normalize.
    double inv = 1.0 / std::sqrt(3.0);
    EXPECT_NEAR(n[0], -inv, 1e-9);
    EXPECT_NEAR(n[1], -inv, 1e-9);
    EXPECT_NEAR(n[2], inv, 1e-9);
}

// ---------------------------------------------------------------------------- painter's algorithm
TEST(Painters, BackToFrontOrder) {
    std::vector<double> depths = {1.0, 5.0, 3.0};  // object 1 farthest
    auto order = painter_order(depths);
    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], 1);  // depth 5 (farthest) drawn first
    EXPECT_EQ(order[1], 2);  // depth 3
    EXPECT_EQ(order[2], 0);  // depth 1 (nearest) drawn last
}

TEST(Painters, NearerRectOccludesFarther) {
    std::vector<DepthRect> rects = {
        {0, 0, 3, 3, 10.0, 1},  // far, color 1, covers whole 4x4
        {1, 1, 2, 2, 2.0, 2}};  // near, color 2, small center square
    auto raster = painter_render(rects, 4, 4);
    EXPECT_EQ(raster[0][0], 1);  // corner: only far rect
    EXPECT_EQ(raster[1][1], 2);  // center: near rect wins
    EXPECT_EQ(raster[2][2], 2);
}

// ---------------------------------------------------------------------------- scanline fill
TEST(ScanlineFill, FillsSquarePixelCount) {
    // 10x10 axis-aligned square from (0,0) to (10,10): centers at 0.5..9.5 -> 100 pixels.
    std::vector<Point2D> sq = {{0, 0}, {10, 0}, {10, 10}, {0, 10}};
    auto px = scanline_fill(sq);
    EXPECT_EQ(px.size(), 100u);
}

TEST(ScanlineFill, TriangleAreaApproxMatchesPixelCount) {
    // Right triangle with legs 20: area 200; scanline pixel count should be close.
    std::vector<Point2D> tri = {{0, 0}, {20, 0}, {0, 20}};
    auto px = scanline_fill(tri);
    EXPECT_NEAR(static_cast<double>(px.size()), 200.0, 25.0);  // within boundary tolerance
}

// ---------------------------------------------------------------------------- marching squares
TEST(MarchingSquares, CircleContourLiesOnIsoRadius) {
    // Field f(x,y) = radius from center (10,10) on a 21x21 grid; contour at iso=6
    // should trace points about distance 6 from the center.
    const int N = 21;
    std::vector<std::vector<double>> grid(N, std::vector<double>(N));
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            grid[r][c] = std::sqrt((c - 10.0) * (c - 10.0) + (r - 10.0) * (r - 10.0));
    auto segs = marching_squares(grid, 6.0);
    ASSERT_FALSE(segs.empty());
    double max_err = 0.0;
    for (const auto& s : segs) {
        double mx = 0.5 * (s[0] + s[2]) - 10.0, my = 0.5 * (s[1] + s[3]) - 10.0;
        max_err = std::max(max_err, std::fabs(std::sqrt(mx * mx + my * my) - 6.0));
    }
    EXPECT_LT(max_err, 0.7);  // segment midpoints hug the iso-radius circle
}

TEST(MarchingSquares, EmptyWhenAllBelowOrAbove) {
    std::vector<std::vector<double>> lo(5, std::vector<double>(5, 0.0));
    EXPECT_TRUE(marching_squares(lo, 1.0).empty());
    std::vector<std::vector<double>> hi(5, std::vector<double>(5, 2.0));
    EXPECT_TRUE(marching_squares(hi, 1.0).empty());
}
