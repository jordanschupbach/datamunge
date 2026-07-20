#include <gtest/gtest.h>

#include <datamunge/cv/blob.hpp>
#include <datamunge/cv/corners.hpp>
#include <datamunge/image/draw.hpp>

#include <cmath>

using namespace datamunge::cv;
using namespace datamunge::image;

namespace {
Image make_checkerboard(int size, int cell) {
    Image img(size, size, ImageMode::RGB, Pixel{0, 0, 0, 255});
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            if (((x / cell) + (y / cell)) % 2 == 0) img.set_pixel(x, y, Pixel{255, 255, 255, 255});
    return img;
}

bool any_corner_near(const std::vector<Corner>& corners, int gx, int gy, int tolerance) {
    for (const auto& c : corners)
        if (std::abs(c.x - gx) <= tolerance && std::abs(c.y - gy) <= tolerance) return true;
    return false;
}
} // namespace

TEST(HarrisCorners, FindsCheckerboardIntersections) {
    const Image checker = make_checkerboard(40, 10);
    const auto corners = harris_corners(checker, 0.04, 1.0, 0.05, 5);
    ASSERT_FALSE(corners.empty());
    for (int gx : {10, 20, 30})
        for (int gy : {10, 20, 30}) EXPECT_TRUE(any_corner_near(corners, gx, gy, 2)) << "missing corner near (" << gx << "," << gy << ")";
}

TEST(HarrisCorners, FlatImageHasNoCorners) {
    const Image flat(30, 30, ImageMode::RGB, Pixel{100, 100, 100, 255});
    EXPECT_TRUE(harris_corners(flat).empty());
}

TEST(HarrisCorners, ResultsAreSortedByDescendingResponse) {
    const Image checker = make_checkerboard(40, 10);
    const auto corners = harris_corners(checker);
    for (std::size_t i = 1; i < corners.size(); ++i) EXPECT_GE(corners[i - 1].response, corners[i].response);
}

TEST(ShiTomasiCorners, FindsCheckerboardIntersections) {
    const Image checker = make_checkerboard(40, 10);
    const auto corners = shi_tomasi_corners(checker, 1.0, 0.05, 5);
    ASSERT_FALSE(corners.empty());
    EXPECT_TRUE(any_corner_near(corners, 20, 20, 2));
}

TEST(FastCorners, FindsRectangleCorners) {
    Image square(40, 40, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_rectangle(square, 10, 10, 29, 29, Pixel{255, 255, 255, 255}, true);
    const auto corners = fast_corners(square, 20, 9);
    ASSERT_FALSE(corners.empty());
    for (int gx : {10, 29})
        for (int gy : {10, 29}) EXPECT_TRUE(any_corner_near(corners, gx, gy, 2));
}

TEST(FastCorners, FlatImageHasNoCorners) {
    const Image flat(30, 30, ImageMode::RGB, Pixel{50, 50, 50, 255});
    EXPECT_TRUE(fast_corners(flat).empty());
}

TEST(DogBlobs, FindsABlobNearASolidDiskCenter) {
    Image disk(60, 60, ImageMode::RGB, Pixel{0, 0, 0, 255});
    draw_circle(disk, 30, 30, 8, Pixel{255, 255, 255, 255}, true);
    const auto blobs = dog_blobs(disk, 5, 1.0, 2.0);
    ASSERT_FALSE(blobs.empty());
    bool found = false;
    for (const auto& kp : blobs)
        if (std::abs(kp.x - 30) <= 5 && std::abs(kp.y - 30) <= 5) found = true;
    EXPECT_TRUE(found);
}

TEST(DogBlobs, RejectsTooFewScales) {
    const Image img(20, 20, ImageMode::RGB);
    EXPECT_THROW(dog_blobs(img, 1), std::invalid_argument);
}
