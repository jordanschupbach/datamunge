#include <gtest/gtest.h>

#include <datamunge/image/filters.hpp>

#include <stdexcept>

using namespace datamunge::image;

TEST(Convolve, RejectsNonSquareOrEvenSizedKernel) {
    Image img(4, 4, ImageMode::RGB);
    EXPECT_THROW(convolve(img, Kernel(6, 1.0)), std::invalid_argument); // not a perfect square
    EXPECT_THROW(convolve(img, Kernel(4, 0.25)), std::invalid_argument); // 2x2, even
}

TEST(BoxBlur, SolidColorImageIsUnchanged) {
    Image src(10, 10, ImageMode::RGB, Pixel{40, 80, 120, 255});
    const Image blurred = box_blur(src, 2);
    for (int y = 0; y < 10; ++y)
        for (int x = 0; x < 10; ++x) EXPECT_EQ(blurred.get_pixel(x, y), (Pixel{40, 80, 120, 255}));
}

TEST(BoxBlur, SmoothsASingleBrightPixelIntoItsNeighborhood) {
    Image src(11, 11, ImageMode::Grayscale, Pixel{0, 0, 0, 255});
    src.set_pixel(5, 5, Pixel{255, 0, 0, 255});
    const Image blurred = box_blur(src, 1);
    // The 3x3 box average around the spike center: (255)/9 truncated by clamping/rounding.
    EXPECT_NEAR(blurred.get_pixel(5, 5).r, 255.0 / 9.0, 1.0);
    EXPECT_EQ(blurred.get_pixel(0, 0).r, 0); // far from the spike, untouched
}

TEST(GaussianBlur, SolidColorImageIsUnchanged) {
    Image src(10, 10, ImageMode::RGB, Pixel{10, 20, 30, 255});
    const Image blurred = gaussian_blur(src, 1.5);
    EXPECT_EQ(blurred.get_pixel(5, 5), (Pixel{10, 20, 30, 255}));
}

TEST(GaussianBlur, RejectsNonPositiveSigma) {
    Image img(4, 4, ImageMode::RGB);
    EXPECT_THROW(gaussian_blur(img, 0.0), std::invalid_argument);
    EXPECT_THROW(gaussian_blur(img, -1.0), std::invalid_argument);
}

TEST(Sharpen, ZeroAmountIsANoOp) {
    Image src(6, 6, ImageMode::RGB, Pixel{50, 60, 70, 255});
    const Image out = sharpen(src, 0.0);
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 6; ++x) EXPECT_EQ(out.get_pixel(x, y), (Pixel{50, 60, 70, 255}));
}

TEST(SobelEdges, SolidColorImageHasNoEdgesInTheInterior) {
    Image src(10, 10, ImageMode::RGB, Pixel{128, 128, 128, 255});
    const Image edges = sobel_edges(src);
    EXPECT_EQ(edges.mode(), ImageMode::Grayscale);
    for (int y = 1; y < 9; ++y)
        for (int x = 1; x < 9; ++x) EXPECT_EQ(edges.get_pixel(x, y).r, 0);
}

TEST(SobelEdges, SharpBoundaryProducesANonzeroResponse) {
    Image src(10, 10, ImageMode::RGB, Pixel{0, 0, 0, 255});
    for (int y = 0; y < 10; ++y)
        for (int x = 5; x < 10; ++x) src.set_pixel(x, y, Pixel{255, 255, 255, 255});
    const Image edges = sobel_edges(src);
    EXPECT_GT(edges.get_pixel(5, 5).r, 0);
    EXPECT_EQ(edges.get_pixel(1, 5).r, 0); // far from the boundary, flat
}

TEST(AdjustBrightness, PositiveDeltaBrightensAndClampsAtWhite) {
    Image src(2, 1, ImageMode::RGB);
    src.set_pixel(0, 0, Pixel{100, 100, 100, 255});
    src.set_pixel(1, 0, Pixel{250, 250, 250, 255});
    const Image out = adjust_brightness(src, 50);
    EXPECT_EQ(out.get_pixel(0, 0).r, 150);
    EXPECT_EQ(out.get_pixel(1, 0).r, 255); // clamped
}

TEST(AdjustBrightness, NegativeDeltaDarkensAndClampsAtBlack) {
    Image src(1, 1, ImageMode::RGB, Pixel{20, 20, 20, 255});
    const Image out = adjust_brightness(src, -50);
    EXPECT_EQ(out.get_pixel(0, 0).r, 0);
}

TEST(AdjustBrightness, LeavesAlphaUntouched) {
    Image src(1, 1, ImageMode::RGBA, Pixel{100, 100, 100, 128});
    const Image out = adjust_brightness(src, 50);
    EXPECT_EQ(out.get_pixel(0, 0).a, 128);
}

TEST(AdjustContrast, FactorOneIsANoOp) {
    Image src(1, 1, ImageMode::RGB, Pixel{77, 88, 99, 255});
    const Image out = adjust_contrast(src, 1.0);
    EXPECT_EQ(out.get_pixel(0, 0), (Pixel{77, 88, 99, 255}));
}

TEST(AdjustContrast, FactorZeroCollapsesToMidGray) {
    Image src(1, 1, ImageMode::RGB, Pixel{0, 255, 200, 255});
    const Image out = adjust_contrast(src, 0.0);
    EXPECT_EQ(out.get_pixel(0, 0), (Pixel{128, 128, 128, 255}));
}

TEST(AdjustContrast, RejectsNegativeFactor) {
    Image img(2, 2, ImageMode::RGB);
    EXPECT_THROW(adjust_contrast(img, -0.5), std::invalid_argument);
}

TEST(Threshold, SplitsBelowAndAboveLevelIntoBlackAndWhite) {
    Image src(3, 1, ImageMode::Grayscale);
    src.set_pixel(0, 0, Pixel{50, 0, 0, 255});
    src.set_pixel(1, 0, Pixel{128, 0, 0, 255});
    src.set_pixel(2, 0, Pixel{200, 0, 0, 255});
    const Image out = threshold(src, 128);
    EXPECT_EQ(out.get_pixel(0, 0).r, 0);
    EXPECT_EQ(out.get_pixel(1, 0).r, 255); // exactly at the level counts as "above"
    EXPECT_EQ(out.get_pixel(2, 0).r, 255);
}
