#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

#include <cmath>
#include <stdexcept>

using datamunge::geometry::closest_pair;
using datamunge::geometry::Point2D;

TEST(ClosestPair, FindsTheHandComputedClosestPair) {
    // (0,0)-(1,0) are distance 1 apart, the closest of any pair here (others are >= sqrt(2)).
    const std::vector<Point2D> points{{0, 0}, {1, 0}, {5, 5}, {5, 6}, {-3, -3}};
    const auto result = closest_pair(points);

    EXPECT_DOUBLE_EQ(result.distance, 1.0);
    const bool matches_ab = (result.a == Point2D{0, 0} && result.b == Point2D{1, 0}) ||
                             (result.a == Point2D{1, 0} && result.b == Point2D{0, 0});
    EXPECT_TRUE(matches_ab);
}

TEST(ClosestPair, TwoPointsAreTriviallyTheClosestPair) {
    const std::vector<Point2D> points{{0, 0}, {3, 4}};
    const auto result = closest_pair(points);
    EXPECT_DOUBLE_EQ(result.distance, 5.0); // 3-4-5 triangle
}

TEST(ClosestPair, RejectsFewerThanTwoPoints) {
    const std::vector<Point2D> points{{0, 0}};
    EXPECT_THROW(static_cast<void>(closest_pair(points)), std::invalid_argument);
}
