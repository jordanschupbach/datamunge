#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

#include <random>
#include <stdexcept>

using datamunge::geometry::min_enclosing_circle;
using datamunge::geometry::Point2D;

TEST(MinEnclosingCircle, TwoPointsGiveTheirMidpointAndHalfDistance) {
    const auto c = min_enclosing_circle({{0, 0}, {4, 0}});
    EXPECT_DOUBLE_EQ(c.center.x, 2.0);
    EXPECT_DOUBLE_EQ(c.center.y, 0.0);
    EXPECT_DOUBLE_EQ(c.radius, 2.0);
}

TEST(MinEnclosingCircle, RightTriangleGivesTheHypotenuseCircle) {
    // For a right (or obtuse) triangle, the minimum enclosing circle has the longest side
    // (here the hypotenuse, (4,0)-(0,3), length 5) as its diameter -- a well-known fact,
    // giving an exactly hand-computable expected answer.
    const auto c = min_enclosing_circle({{0, 0}, {4, 0}, {0, 3}});
    EXPECT_NEAR(c.center.x, 2.0, 1e-9);
    EXPECT_NEAR(c.center.y, 1.5, 1e-9);
    EXPECT_NEAR(c.radius, 2.5, 1e-9);
}

TEST(MinEnclosingCircle, InteriorThirdPointDoesNotEnlargeTheTwoPointCircle) {
    // (2, 1) lies strictly inside the circle already determined by (0,0) and (4,0), so the
    // minimum enclosing circle should stay exactly that 2-point circle.
    const auto c = min_enclosing_circle({{0, 0}, {4, 0}, {2, 1}});
    EXPECT_NEAR(c.center.x, 2.0, 1e-9);
    EXPECT_NEAR(c.center.y, 0.0, 1e-9);
    EXPECT_NEAR(c.radius, 2.0, 1e-9);
}

TEST(MinEnclosingCircle, SinglePointGivesAZeroRadiusCircleAtThatPoint) {
    const auto c = min_enclosing_circle({{3, 4}});
    EXPECT_DOUBLE_EQ(c.center.x, 3.0);
    EXPECT_DOUBLE_EQ(c.center.y, 4.0);
    EXPECT_DOUBLE_EQ(c.radius, 0.0);
}

TEST(MinEnclosingCircle, EveryRandomPointLiesWithinTheReturnedCircle) {
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> unif(0.0, 10.0);
    std::vector<Point2D> points;
    for (int i = 0; i < 200; ++i) points.push_back({unif(rng), unif(rng)});

    const auto c = min_enclosing_circle(points);
    for (const auto& p : points) {
        EXPECT_LE(datamunge::geometry::distance(c.center, p), c.radius + 1e-6);
    }
}

TEST(MinEnclosingCircle, CollinearPointsAreHandledWithoutError) {
    const auto c = min_enclosing_circle({{0, 0}, {1, 0}, {2, 0}, {3, 0}});
    EXPECT_NEAR(c.center.x, 1.5, 1e-9);
    EXPECT_NEAR(c.center.y, 0.0, 1e-9);
    EXPECT_NEAR(c.radius, 1.5, 1e-9);
}

TEST(MinEnclosingCircle, RejectsEmptyInput) {
    EXPECT_THROW(static_cast<void>(min_enclosing_circle({})), std::invalid_argument);
}
