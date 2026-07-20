#include <gtest/gtest.h>

#include <datamunge/cv/hough.hpp>
#include <datamunge/image/draw.hpp>

#include <cmath>

using namespace datamunge::cv;
using namespace datamunge::image;

TEST(HoughLines, RecoversAKnownHorizontalLine) {
    Image img(60, 60, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_line(img, 5, 20, 54, 20, Pixel{255, 255, 255, 255});
    const auto lines = hough_lines(img, 128, 30, 5);
    ASSERT_FALSE(lines.empty());
    EXPECT_NEAR(lines.front().rho, 20.0, 2.0);
    EXPECT_NEAR(lines.front().theta, 1.5708, 0.1); // pi/2: horizontal line's normal is vertical
}

TEST(HoughLines, RecoversADiagonalLineThroughTheOrigin) {
    Image img(60, 60, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_line(img, 5, 5, 54, 54, Pixel{255, 255, 255, 255});
    const auto lines = hough_lines(img, 128, 30, 5);
    ASSERT_FALSE(lines.empty());
    EXPECT_NEAR(lines.front().rho, 0.0, 3.0);
}

TEST(HoughLines, NoEdgesProducesNoLines) {
    const Image blank(40, 40, ImageMode::RGB, Pixel{0, 0, 0, 255});
    EXPECT_TRUE(hough_lines(blank).empty());
}

TEST(LineEndpoints, ClipsToTheImageBoundaryNearTheKnownLine) {
    const HoughLine line{20.0, 1.5708, 100};
    std::pair<int, int> p1, p2;
    ASSERT_TRUE(line_endpoints(line, 60, 60, p1, p2));
    EXPECT_NEAR(p1.second, 20, 1);
    EXPECT_NEAR(p2.second, 20, 1);
}

TEST(HoughCircles, RecoversAKnownCircle) {
    Image img(80, 80, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_circle(img, 40, 40, 20, Pixel{255, 255, 255, 255}, false);
    const auto circles = hough_circles(img, 15, 25, 50.0, 15, 8);
    ASSERT_FALSE(circles.empty());
    bool found = false;
    for (const auto& c : circles)
        if (std::abs(c.cx - 40) <= 3 && std::abs(c.cy - 40) <= 3 && std::abs(c.radius - 20) <= 3) found = true;
    EXPECT_TRUE(found);
}

TEST(HoughCircles, RejectsInvalidRadiusRange) {
    const Image img(40, 40, ImageMode::RGB);
    EXPECT_THROW(hough_circles(img, 30, 10), std::invalid_argument);
    EXPECT_THROW(hough_circles(img, 0, 10), std::invalid_argument);
}

TEST(HoughCircles, ResultsAreSortedByDescendingVotes) {
    Image img(80, 80, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_circle(img, 40, 40, 20, Pixel{255, 255, 255, 255}, false);
    const auto circles = hough_circles(img, 15, 25, 50.0, 5, 8);
    for (std::size_t i = 1; i < circles.size(); ++i) EXPECT_GE(circles[i - 1].votes, circles[i].votes);
}
