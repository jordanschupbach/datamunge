#include <gtest/gtest.h>

#include <datamunge/image/color.hpp>

#include <stdexcept>

using namespace datamunge::image;

TEST(RgbToGray, WhiteAndBlackAreExact) {
    EXPECT_EQ(rgb_to_gray(255, 255, 255), 255);
    EXPECT_EQ(rgb_to_gray(0, 0, 0), 0);
}

TEST(RgbToGray, WeightsGreenMostAndBlueLeast) {
    // Equal-brightness pure channels: BT.601 luma weights green highest, blue lowest.
    EXPECT_GT(rgb_to_gray(0, 255, 0), rgb_to_gray(255, 0, 0));
    EXPECT_GT(rgb_to_gray(255, 0, 0), rgb_to_gray(0, 0, 255));
}

TEST(RgbHsvRoundTrip, PrimaryAndSecondaryColorsSurviveConversion) {
    const std::vector<Pixel> colors = {{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255},   {255, 255, 0, 255},
                                        {0, 255, 255, 255}, {255, 0, 255, 255}, {123, 200, 50, 255}};
    for (const auto& c : colors) {
        const HSV hsv = rgb_to_hsv(c.r, c.g, c.b);
        const Pixel back = hsv_to_rgb(hsv);
        EXPECT_NEAR(back.r, c.r, 1) << "r mismatch for (" << int(c.r) << "," << int(c.g) << "," << int(c.b) << ")";
        EXPECT_NEAR(back.g, c.g, 1);
        EXPECT_NEAR(back.b, c.b, 1);
    }
}

TEST(RgbHsvRoundTrip, BlackHasZeroSaturationAndValue) {
    const HSV hsv = rgb_to_hsv(0, 0, 0);
    EXPECT_DOUBLE_EQ(hsv.s, 0.0);
    EXPECT_DOUBLE_EQ(hsv.v, 0.0);
}

TEST(RgbHsvRoundTrip, GrayHasZeroSaturation) {
    const HSV hsv = rgb_to_hsv(128, 128, 128);
    EXPECT_DOUBLE_EQ(hsv.s, 0.0);
}

TEST(ToGrayscale, ProducesGrayscaleModeMatchingRgbToGray) {
    Image img(2, 1, ImageMode::RGB);
    img.set_pixel(0, 0, Pixel{255, 0, 0, 255});
    img.set_pixel(1, 0, Pixel{0, 255, 0, 255});
    const Image gray = to_grayscale(img);
    EXPECT_EQ(gray.mode(), ImageMode::Grayscale);
    EXPECT_EQ(gray.get_pixel(0, 0).r, rgb_to_gray(255, 0, 0));
    EXPECT_EQ(gray.get_pixel(1, 0).r, rgb_to_gray(0, 255, 0));
}

TEST(ToRgb, ExpandsGrayscaleToEqualChannels) {
    Image gray(1, 1, ImageMode::Grayscale, Pixel{42, 0, 0, 255});
    const Image rgb = to_rgb(gray);
    EXPECT_EQ(rgb.mode(), ImageMode::RGB);
    EXPECT_EQ(rgb.get_pixel(0, 0), (Pixel{42, 42, 42, 255}));
}

TEST(ToRgba, PreservesColorAndSetsFullAlpha) {
    Image rgb(1, 1, ImageMode::RGB, Pixel{5, 6, 7, 255});
    const Image rgba = to_rgba(rgb);
    EXPECT_EQ(rgba.mode(), ImageMode::RGBA);
    EXPECT_EQ(rgba.get_pixel(0, 0), (Pixel{5, 6, 7, 255}));
}

TEST(SplitAndMergeChannels, RoundTripsAnRgbaImageExactly) {
    Image img(3, 3, ImageMode::RGBA);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x)
            img.set_pixel(x, y,
                          Pixel{static_cast<std::uint8_t>(x * 10), static_cast<std::uint8_t>(y * 10),
                                static_cast<std::uint8_t>(x + y), static_cast<std::uint8_t>(255 - x)});

    const auto planes = split_channels(img);
    ASSERT_EQ(planes.size(), 4u);
    for (const auto& plane : planes) {
        EXPECT_EQ(plane.mode(), ImageMode::Grayscale);
        EXPECT_EQ(plane.width(), 3);
        EXPECT_EQ(plane.height(), 3);
    }

    const Image merged = merge_channels(planes);
    EXPECT_EQ(merged.mode(), ImageMode::RGBA);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x) EXPECT_EQ(merged.get_pixel(x, y), img.get_pixel(x, y));
}

TEST(MergeChannels, RejectsMismatchedPlaneDimensions) {
    std::vector<Image> planes = {Image(2, 2, ImageMode::Grayscale), Image(3, 3, ImageMode::Grayscale)};
    EXPECT_THROW(merge_channels(planes), std::invalid_argument);
}

TEST(MergeChannels, RejectsTooManyPlanes) {
    std::vector<Image> planes(5, Image(2, 2, ImageMode::Grayscale));
    EXPECT_THROW(merge_channels(planes), std::invalid_argument);
}
