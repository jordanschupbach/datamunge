#include <gtest/gtest.h>

#include <datamunge/image/image.hpp>

#include <stdexcept>

using datamunge::image::Image;
using datamunge::image::ImageMode;
using datamunge::image::Pixel;

TEST(Image, ConstructedWithFillColorHasThatColorEverywhere) {
    Image img(5, 4, ImageMode::RGB, Pixel{10, 20, 30, 255});
    EXPECT_EQ(img.width(), 5);
    EXPECT_EQ(img.height(), 4);
    EXPECT_EQ(img.channels(), 3);
    for (int y = 0; y < img.height(); ++y) {
        for (int x = 0; x < img.width(); ++x) {
            EXPECT_EQ(img.get_pixel(x, y), (Pixel{10, 20, 30, 255}));
        }
    }
}

TEST(Image, DefaultFillIsTransparentBlack) {
    Image img(3, 3, ImageMode::RGBA);
    EXPECT_EQ(img.get_pixel(0, 0), (Pixel{0, 0, 0, 255}));
}

TEST(Image, SetPixelThenGetPixelRoundTrips) {
    Image img(4, 4, ImageMode::RGBA);
    img.set_pixel(1, 2, Pixel{200, 150, 50, 128});
    EXPECT_EQ(img.get_pixel(1, 2), (Pixel{200, 150, 50, 128}));
    // Neighboring pixels are untouched.
    EXPECT_EQ(img.get_pixel(0, 2), (Pixel{0, 0, 0, 255}));
}

TEST(Image, GrayscaleModeExpandsToEqualRGBOnRead) {
    Image img(2, 2, ImageMode::Grayscale);
    img.set_pixel(0, 0, Pixel{77, 0, 0, 0}); // g, b, a are ignored for Grayscale writes
    const Pixel p = img.get_pixel(0, 0);
    EXPECT_EQ(p.r, 77);
    EXPECT_EQ(p.g, 77);
    EXPECT_EQ(p.b, 77);
    EXPECT_EQ(p.a, 255); // Grayscale has no alpha channel; always reads back as opaque
}

TEST(Image, GrayscaleAlphaModePreservesAlpha) {
    Image img(2, 2, ImageMode::GrayscaleAlpha);
    img.set_pixel(0, 0, Pixel{100, 0, 0, 42});
    const Pixel p = img.get_pixel(0, 0);
    EXPECT_EQ(p.r, 100);
    EXPECT_EQ(p.g, 100);
    EXPECT_EQ(p.b, 100);
    EXPECT_EQ(p.a, 42);
}

TEST(Image, OutOfRangePixelAccessThrows) {
    Image img(3, 3, ImageMode::RGB);
    EXPECT_THROW(img.get_pixel(3, 0), std::out_of_range);
    EXPECT_THROW(img.get_pixel(0, 3), std::out_of_range);
    EXPECT_THROW(img.get_pixel(-1, 0), std::out_of_range);
    EXPECT_THROW(img.set_pixel(0, -1, Pixel{}), std::out_of_range);
}

TEST(Image, NegativeDimensionsThrow) { EXPECT_THROW(Image(-1, 5, ImageMode::RGB), std::invalid_argument); }

TEST(Image, FillOverwritesEveryPixel) {
    Image img(3, 3, ImageMode::RGB, Pixel{1, 2, 3, 255});
    img.fill(Pixel{9, 9, 9, 255});
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x) EXPECT_EQ(img.get_pixel(x, y), (Pixel{9, 9, 9, 255}));
}

TEST(Image, ChannelCountMatchesMode) {
    EXPECT_EQ(Image::channel_count(ImageMode::Grayscale), 1);
    EXPECT_EQ(Image::channel_count(ImageMode::GrayscaleAlpha), 2);
    EXPECT_EQ(Image::channel_count(ImageMode::RGB), 3);
    EXPECT_EQ(Image::channel_count(ImageMode::RGBA), 4);
}

TEST(Image, DataBufferSizeMatchesWidthHeightChannels) {
    Image img(4, 5, ImageMode::RGBA);
    EXPECT_EQ(img.data().size(), static_cast<std::size_t>(4 * 5 * 4));
}
