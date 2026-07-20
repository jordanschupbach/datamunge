#include <gtest/gtest.h>

#include <datamunge/geometry/geometry.hpp>

#include <algorithm>
#include <stdexcept>

using datamunge::geometry::distance;
using datamunge::geometry::KDTree2D;
using datamunge::geometry::Point2D;

namespace {
std::vector<Point2D> make_points() {
    // A 5x5 integer grid, 25 points total.
    std::vector<Point2D> points;
    for (int x = 0; x < 5; ++x)
        for (int y = 0; y < 5; ++y) points.push_back({static_cast<double>(x), static_cast<double>(y)});
    return points;
}

// Brute-force nearest, for cross-checking the tree against an obviously-correct reference.
Point2D brute_force_nearest(const std::vector<Point2D>& points, const Point2D& query) {
    return *std::min_element(points.begin(), points.end(),
                              [&](const Point2D& a, const Point2D& b) { return distance(query, a) < distance(query, b); });
}
} // namespace

TEST(KDTree2D, NearestMatchesBruteForceAcrossManyQueries) {
    const auto points = make_points();
    const KDTree2D tree(points);

    for (double qx = -1.0; qx <= 5.5; qx += 0.7) {
        for (double qy = -1.0; qy <= 5.5; qy += 0.7) {
            const Point2D query{qx, qy};
            const auto expected = brute_force_nearest(points, query);
            const auto actual = tree.nearest(query);
            EXPECT_DOUBLE_EQ(distance(query, actual), distance(query, expected)) << "query (" << qx << ", " << qy << ")";
        }
    }
}

TEST(KDTree2D, KNearestReturnsNearestFirstAndMatchesBruteForceDistances) {
    const auto points = make_points();
    const KDTree2D tree(points);
    const Point2D query{2.3, 2.4};

    const auto knn = tree.k_nearest(query, 5);
    ASSERT_EQ(knn.size(), 5u);

    // Nearest-first ordering.
    for (std::size_t i = 1; i < knn.size(); ++i) {
        EXPECT_LE(distance(query, knn[i - 1]), distance(query, knn[i]));
    }

    // The k returned distances must match the k smallest brute-force distances.
    std::vector<double> brute_force_distances;
    for (const auto& p : points) brute_force_distances.push_back(distance(query, p));
    std::sort(brute_force_distances.begin(), brute_force_distances.end());

    for (std::size_t i = 0; i < knn.size(); ++i) {
        EXPECT_DOUBLE_EQ(distance(query, knn[i]), brute_force_distances[i]);
    }
}

TEST(KDTree2D, KNearestReturnsFewerThanKWhenTreeIsSmaller) {
    const KDTree2D tree(std::vector<Point2D>{{0, 0}, {1, 1}});
    EXPECT_EQ(tree.k_nearest({0, 0}, 10).size(), 2u);
}

TEST(KDTree2D, PointsInRadiusMatchesBruteForce) {
    const auto points = make_points();
    const KDTree2D tree(points);
    const Point2D query{2.0, 2.0};
    const double radius = 1.5;

    auto actual = tree.points_in_radius(query, radius);
    std::vector<Point2D> expected;
    for (const auto& p : points)
        if (distance(query, p) <= radius) expected.push_back(p);

    ASSERT_EQ(actual.size(), expected.size());
    auto by_coords = [](const Point2D& a, const Point2D& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); };
    std::sort(actual.begin(), actual.end(), by_coords);
    std::sort(expected.begin(), expected.end(), by_coords);
    EXPECT_EQ(actual, expected);
}

TEST(KDTree2D, SizeAndEmptyReflectConstruction) {
    const KDTree2D empty_tree(std::vector<Point2D>{});
    EXPECT_TRUE(empty_tree.empty());
    EXPECT_EQ(empty_tree.size(), 0u);

    const KDTree2D tree(make_points());
    EXPECT_FALSE(tree.empty());
    EXPECT_EQ(tree.size(), 25u);
}

TEST(KDTree2D, NearestOnEmptyTreeThrows) {
    const KDTree2D empty_tree(std::vector<Point2D>{});
    EXPECT_THROW(static_cast<void>(empty_tree.nearest({0, 0})), std::logic_error);
}
