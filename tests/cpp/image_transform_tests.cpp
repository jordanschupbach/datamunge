#include <gtest/gtest.h>

#include <datamunge/image/transform.hpp>

#include <stdexcept>

using namespace datamunge::image;

TEST(Resize, SolidColorImageStaysSolidAfterResizing) {
    Image src(20, 20, ImageMode::RGB, Pixel{7, 8, 9, 255});
    const Image small = resize(src, 5, 5, ResampleFilter::Bilinear);
    const Image big = resize(src, 40, 40, ResampleFilter::Nearest);
    EXPECT_EQ(small.width(), 5);
    EXPECT_EQ(big.width(), 40);
    for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 5; ++x) EXPECT_EQ(small.get_pixel(x, y), (Pixel{7, 8, 9, 255}));
    for (int y = 0; y < 40; ++y)
        for (int x = 0; x < 40; ++x) EXPECT_EQ(big.get_pixel(x, y), (Pixel{7, 8, 9, 255}));
}

TEST(Resize, NearestOnIntegerScaleFactorReproducesEachSourcePixelExactly) {
    Image src(2, 2, ImageMode::RGB);
    src.set_pixel(0, 0, Pixel{1, 0, 0, 255});
    src.set_pixel(1, 0, Pixel{2, 0, 0, 255});
    src.set_pixel(0, 1, Pixel{3, 0, 0, 255});
    src.set_pixel(1, 1, Pixel{4, 0, 0, 255});
    const Image up = resize(src, 4, 4, ResampleFilter::Nearest);
    // Each source pixel should map to a clean 2x2 block.
    EXPECT_EQ(up.get_pixel(0, 0).r, 1);
    EXPECT_EQ(up.get_pixel(1, 0).r, 1);
    EXPECT_EQ(up.get_pixel(2, 0).r, 2);
    EXPECT_EQ(up.get_pixel(3, 0).r, 2);
    EXPECT_EQ(up.get_pixel(0, 2).r, 3);
    EXPECT_EQ(up.get_pixel(2, 2).r, 4);
}

TEST(Resize, RejectsNonPositiveDimensions) {
    Image src(4, 4, ImageMode::RGB);
    EXPECT_THROW(resize(src, 0, 4), std::invalid_argument);
    EXPECT_THROW(resize(src, 4, -1), std::invalid_argument);
}

TEST(Crop, ExtractsExactSubregion) {
    Image src(5, 5, ImageMode::Grayscale);
    for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 5; ++x) src.set_pixel(x, y, Pixel{static_cast<std::uint8_t>(x * 5 + y), 0, 0, 255});

    const Image cropped = crop(src, 1, 2, 3, 2);
    EXPECT_EQ(cropped.width(), 3);
    EXPECT_EQ(cropped.height(), 2);
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 3; ++dx) EXPECT_EQ(cropped.get_pixel(dx, dy), src.get_pixel(1 + dx, 2 + dy));
}

TEST(Crop, RejectsRectangleExceedingBounds) {
    Image src(5, 5, ImageMode::RGB);
    EXPECT_THROW(crop(src, 3, 0, 3, 1), std::out_of_range);
    EXPECT_THROW(crop(src, 0, 0, 0, 1), std::invalid_argument);
}

TEST(Flip, HorizontalTwiceIsIdentity) {
    Image src(4, 3, ImageMode::Grayscale);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x) src.set_pixel(x, y, Pixel{static_cast<std::uint8_t>(x + y * 4), 0, 0, 255});
    const Image twice = flip_horizontal(flip_horizontal(src));
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x) EXPECT_EQ(twice.get_pixel(x, y), src.get_pixel(x, y));
}

TEST(Flip, HorizontalReversesColumnOrder) {
    Image src(3, 1, ImageMode::Grayscale);
    src.set_pixel(0, 0, Pixel{1, 0, 0, 255});
    src.set_pixel(1, 0, Pixel{2, 0, 0, 255});
    src.set_pixel(2, 0, Pixel{3, 0, 0, 255});
    const Image flipped = flip_horizontal(src);
    EXPECT_EQ(flipped.get_pixel(0, 0).r, 3);
    EXPECT_EQ(flipped.get_pixel(2, 0).r, 1);
}

TEST(Flip, VerticalTwiceIsIdentity) {
    Image src(3, 4, ImageMode::Grayscale);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 3; ++x) src.set_pixel(x, y, Pixel{static_cast<std::uint8_t>(x + y * 3), 0, 0, 255});
    const Image twice = flip_vertical(flip_vertical(src));
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 3; ++x) EXPECT_EQ(twice.get_pixel(x, y), src.get_pixel(x, y));
}

TEST(Rotate90, ZeroDegreesIsIdentity) {
    Image src(3, 2, ImageMode::Grayscale, Pixel{5, 0, 0, 255});
    const Image out = rotate90(src, 0);
    EXPECT_EQ(out.width(), 3);
    EXPECT_EQ(out.height(), 2);
}

TEST(Rotate90, NinetyDegreesSwapsDimensions) {
    Image src(5, 3, ImageMode::Grayscale);
    const Image out = rotate90(src, 90);
    EXPECT_EQ(out.width(), 3);
    EXPECT_EQ(out.height(), 5);
}

TEST(Rotate90, FourNinetyDegreeRotationsReturnToTheOriginal) {
    Image src(4, 3, ImageMode::Grayscale);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x) src.set_pixel(x, y, Pixel{static_cast<std::uint8_t>(x + y * 4), 0, 0, 255});

    Image out = src;
    for (int i = 0; i < 4; ++i) out = rotate90(out, 90);
    EXPECT_EQ(out.width(), src.width());
    EXPECT_EQ(out.height(), src.height());
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x) EXPECT_EQ(out.get_pixel(x, y), src.get_pixel(x, y));
}

TEST(Rotate90, RejectsNonMultipleOfNinety) {
    Image src(4, 4, ImageMode::RGB);
    EXPECT_THROW(rotate90(src, 45), std::invalid_argument);
}

TEST(Rotate, ThreeSixtyDegreesApproximatelyReproducesTheOriginalCenterColor) {
    // A full rotation should land back on (roughly) the same colors -- some blur/border loss
    // near the edges is expected from resampling, so only check the untouched-by-border center.
    Image src(20, 20, ImageMode::RGB, Pixel{50, 100, 150, 255});
    const Image out = rotate(src, 360.0);
    EXPECT_EQ(out.get_pixel(out.width() / 2, out.height() / 2), (Pixel{50, 100, 150, 255}));
}

TEST(Rotate, NinetyDegreesOnASquareKeepsSolidColorEverywhere) {
    Image src(16, 16, ImageMode::RGB, Pixel{20, 30, 40, 255});
    const Image out = rotate(src, 90.0);
    // A 90-degree rotation of a square, solid-colored image should still be solid-colored
    // everywhere except possibly right at the anti-aliased border -- check the interior.
    for (int y = 2; y < out.height() - 2; ++y)
        for (int x = 2; x < out.width() - 2; ++x) EXPECT_EQ(out.get_pixel(x, y), (Pixel{20, 30, 40, 255}));
}

TEST(Rotate, RejectsEmptySourceImage) {
    Image src;
    EXPECT_THROW(rotate(src, 45.0), std::invalid_argument);
}
