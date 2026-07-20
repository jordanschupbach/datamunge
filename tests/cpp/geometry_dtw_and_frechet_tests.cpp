#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

#include <stdexcept>

using datamunge::geometry::discrete_frechet_distance;
using datamunge::geometry::dynamic_time_warping;
using datamunge::geometry::Point2D;

// ---- Dynamic Time Warping ----

TEST(DynamicTimeWarping, IdenticalSequencesHaveZeroDistanceAndADiagonalPath) {
    const std::vector<double> a{1, 2, 3, 4, 5};
    const auto result = dynamic_time_warping(a, a);

    EXPECT_DOUBLE_EQ(result.distance, 0.0);
    ASSERT_EQ(result.path_a.size(), 5u);
    ASSERT_EQ(result.path_b.size(), 5u);
    for (int i = 0; i < static_cast<int>(result.path_a.size()); ++i) {
        EXPECT_EQ(result.path_a[static_cast<std::size_t>(i)], i);
        EXPECT_EQ(result.path_b[static_cast<std::size_t>(i)], i);
    }
}

TEST(DynamicTimeWarping, SingleElementSequencesGiveTheirAbsoluteDifference) {
    const auto result = dynamic_time_warping({5.0}, {8.0});
    EXPECT_DOUBLE_EQ(result.distance, 3.0);
    EXPECT_EQ(result.path_a, (std::vector<int>{0}));
    EXPECT_EQ(result.path_b, (std::vector<int>{0}));
}

TEST(DynamicTimeWarping, PathAlwaysStartsAtOriginAndEndsAtFinalIndices) {
    const std::vector<double> a{0, 1, 2, 1, 0};
    const std::vector<double> b{0, 2, 1};
    const auto result = dynamic_time_warping(a, b);

    ASSERT_FALSE(result.path_a.empty());
    EXPECT_EQ(result.path_a.front(), 0);
    EXPECT_EQ(result.path_b.front(), 0);
    EXPECT_EQ(result.path_a.back(), static_cast<int>(a.size() - 1));
    EXPECT_EQ(result.path_b.back(), static_cast<int>(b.size() - 1));
    // Every step must be monotone non-decreasing in both coordinates.
    for (std::size_t i = 1; i < result.path_a.size(); ++i) {
        EXPECT_GE(result.path_a[i], result.path_a[i - 1]);
        EXPECT_GE(result.path_b[i], result.path_b[i - 1]);
    }
}

TEST(DynamicTimeWarping, MonotoneEndpointCostsLowerBoundTheDistance) {
    // Whatever the optimal warping does in between, the path must start by pairing a[0] with
    // b[0] and end by pairing a's last element with b's last element, so their costs are an
    // unavoidable lower bound on the total.
    const std::vector<double> a{1, 2, 3};
    const std::vector<double> b{2, 3, 4};
    const auto result = dynamic_time_warping(a, b);
    EXPECT_GE(result.distance, std::abs(a.front() - b.front()) + std::abs(a.back() - b.back()));
}

TEST(DynamicTimeWarping, RejectsEmptySequences) {
    EXPECT_THROW(static_cast<void>(dynamic_time_warping({}, {1.0})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(dynamic_time_warping({1.0}, {})), std::invalid_argument);
}

// ---- Discrete Frechet distance ----

TEST(DiscreteFrechetDistance, IdenticalCurvesHaveZeroDistance) {
    const std::vector<Point2D> p{{0, 0}, {1, 1}, {2, 2}};
    EXPECT_DOUBLE_EQ(discrete_frechet_distance(p, p), 0.0);
}

TEST(DiscreteFrechetDistance, ParallelLinesGiveTheirSeparationDistance) {
    const std::vector<Point2D> p{{0, 0}, {1, 0}, {2, 0}};
    const std::vector<Point2D> q{{0, 1}, {1, 1}, {2, 1}};
    EXPECT_DOUBLE_EQ(discrete_frechet_distance(p, q), 1.0);
}

TEST(DiscreteFrechetDistance, SinglePointCurvesGiveTheirDistance) {
    EXPECT_DOUBLE_EQ(discrete_frechet_distance({{0, 0}}, {{3, 4}}), 5.0); // 3-4-5 triangle
}

TEST(DiscreteFrechetDistance, IsAtLeastTheDtwStyleEndpointDistances) {
    // Both measures share the same mandatory endpoint pairings (curve starts matched, curve
    // ends matched), so Frechet distance can never be smaller than the larger of those two
    // endpoint distances (it's a MAX-based measure, so it's bounded below by any single
    // pairwise distance actually used, including the forced endpoints).
    const std::vector<Point2D> p{{0, 0}, {1, 5}, {2, 0}};
    const std::vector<Point2D> q{{0, 3}, {1, 3}, {2, 3}};
    const double start_dist = datamunge::geometry::distance(p.front(), q.front());
    const double end_dist = datamunge::geometry::distance(p.back(), q.back());
    EXPECT_GE(discrete_frechet_distance(p, q), std::max(start_dist, end_dist));
}

TEST(DiscreteFrechetDistance, RejectsEmptyCurves) {
    EXPECT_THROW(static_cast<void>(discrete_frechet_distance({}, {{0, 0}})), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(discrete_frechet_distance({{0, 0}}, {})), std::invalid_argument);
}
