#include <gtest/gtest.h>

#include <datamunge/image/draw.hpp>

using namespace datamunge::image;

namespace {
int count_matching(const Image& img, const Pixel& color) {
    int count = 0;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x)
            if (img.get_pixel(x, y) == color) ++count;
    return count;
}
} // namespace

TEST(DrawLine, HorizontalLineSetsExactlyTheExpectedPixels) {
    Image img(10, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_line(img, 2, 5, 7, 5, Pixel{255, 0, 0, 255});
    for (int x = 2; x <= 7; ++x) EXPECT_EQ(img.get_pixel(x, 5), (Pixel{255, 0, 0, 255}));
    EXPECT_EQ(count_matching(img, Pixel{255, 0, 0, 255}), 6);
}

TEST(DrawLine, DiagonalLineConnectsBothEndpoints) {
    Image img(10, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_line(img, 0, 0, 9, 9, Pixel{255, 255, 255, 255});
    EXPECT_EQ(img.get_pixel(0, 0), (Pixel{255, 255, 255, 255}));
    EXPECT_EQ(img.get_pixel(9, 9), (Pixel{255, 255, 255, 255}));
    // A 45-degree Bresenham line touches every row and column exactly once.
    EXPECT_EQ(count_matching(img, Pixel{255, 255, 255, 255}), 10);
}

TEST(DrawLine, OutOfBoundsEndpointsAreClippedNotThrown) {
    Image img(5, 5, ImageMode::RGB, Pixel{0, 0, 0, 255});
    EXPECT_NO_THROW(draw_line(img, -5, -5, 20, 20, Pixel{1, 1, 1, 255}));
    EXPECT_GT(count_matching(img, Pixel{1, 1, 1, 255}), 0); // the in-bounds portion was drawn
}

TEST(DrawRectangle, OutlineOnlyTouchesTheBorder) {
    Image img(10, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_rectangle(img, 2, 2, 7, 7, Pixel{255, 0, 0, 255}, /*filled=*/false);
    EXPECT_EQ(img.get_pixel(2, 2), (Pixel{255, 0, 0, 255}));
    EXPECT_EQ(img.get_pixel(7, 7), (Pixel{255, 0, 0, 255}));
    EXPECT_EQ(img.get_pixel(4, 4), (Pixel{0, 0, 0, 255})); // interior untouched
    // Perimeter of a 6x6 (2..7 inclusive) rectangle: 4*6 - 4 corners double-counted = 20.
    EXPECT_EQ(count_matching(img, Pixel{255, 0, 0, 255}), 20);
}

TEST(DrawRectangle, FilledCoversTheEntireArea) {
    Image img(10, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_rectangle(img, 2, 2, 7, 7, Pixel{255, 0, 0, 255}, /*filled=*/true);
    EXPECT_EQ(count_matching(img, Pixel{255, 0, 0, 255}), 6 * 6);
}

TEST(DrawRectangle, HandlesReversedCorners) {
    Image img(10, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_rectangle(img, 7, 7, 2, 2, Pixel{255, 0, 0, 255}, /*filled=*/true);
    EXPECT_EQ(count_matching(img, Pixel{255, 0, 0, 255}), 6 * 6);
}

TEST(DrawCircle, FilledContainsCenterAndExcludesFarCorners) {
    Image img(21, 21, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_circle(img, 10, 10, 5, Pixel{255, 255, 255, 255}, /*filled=*/true);
    EXPECT_EQ(img.get_pixel(10, 10), (Pixel{255, 255, 255, 255}));
    EXPECT_EQ(img.get_pixel(0, 0), (Pixel{0, 0, 0, 255}));
    // Independent check: filled pixel count must exactly match the brute-force squared-distance test.
    int expected = 0;
    for (int y = -5; y <= 5; ++y)
        for (int x = -5; x <= 5; ++x)
            if (x * x + y * y <= 25) ++expected;
    EXPECT_EQ(count_matching(img, Pixel{255, 255, 255, 255}), expected);
}

TEST(DrawCircle, OutlineDoesNotFillTheInterior) {
    Image img(21, 21, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_circle(img, 10, 10, 5, Pixel{255, 255, 255, 255}, /*filled=*/false);
    EXPECT_EQ(img.get_pixel(10, 10), (Pixel{0, 0, 0, 255})); // center is not part of the outline
    EXPECT_EQ(img.get_pixel(15, 10), (Pixel{255, 255, 255, 255})); // rightmost point of the circle
}

TEST(DrawCircle, NegativeRadiusDrawsNothing) {
    Image img(10, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_circle(img, 5, 5, -3, Pixel{255, 0, 0, 255}, true);
    EXPECT_EQ(count_matching(img, Pixel{255, 0, 0, 255}), 0);
}

TEST(DrawPolygon, FewerThanThreePointsDrawsNothing) {
    Image img(10, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_polygon(img, {{1, 1}, {5, 5}}, Pixel{255, 0, 0, 255}, true);
    EXPECT_EQ(count_matching(img, Pixel{255, 0, 0, 255}), 0);
}

TEST(DrawPolygon, FilledSquareMatchesItsKnownArea) {
    Image img(12, 12, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_polygon(img, {{2, 2}, {8, 2}, {8, 8}, {2, 8}}, Pixel{0, 255, 0, 255}, true);
    EXPECT_EQ(img.get_pixel(5, 5), (Pixel{0, 255, 0, 255}));
    // A 6x6 axis-aligned square (x,y from 2 to 7 inclusive under the half-pixel scanline rule).
    EXPECT_EQ(count_matching(img, Pixel{0, 255, 0, 255}), 36);
}

TEST(DrawPolygon, OutlineDoesNotFillTheInterior) {
    Image img(12, 12, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_polygon(img, {{2, 2}, {8, 2}, {8, 8}, {2, 8}}, Pixel{0, 255, 0, 255}, false);
    EXPECT_EQ(img.get_pixel(5, 5), (Pixel{0, 0, 0, 255}));
    EXPECT_EQ(img.get_pixel(2, 2), (Pixel{0, 255, 0, 255}));
}

TEST(DrawPolygon, TriangleFilledAreaMatchesShoelaceFormula) {
    Image img(20, 20, ImageMode::RGB, Pixel{0, 0, 0, 255});
    // Right triangle with legs 10 and 10: true area = 50.
    draw_polygon(img, {{2, 2}, {12, 2}, {2, 12}}, Pixel{0, 255, 0, 255}, true);
    const int painted = count_matching(img, Pixel{0, 255, 0, 255});
    EXPECT_NEAR(painted, 50, 6); // rasterization rounding, not a strict equality
}
