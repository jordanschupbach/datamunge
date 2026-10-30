// std::format / std::print support for datamunge::image.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>

namespace img = dm::image;

static_assert(std::formattable<img::ImageMode, char>);
static_assert(std::formattable<img::JpegDctMethod, char>);
static_assert(std::formattable<img::JpegColorSpace, char>);
static_assert(std::formattable<img::ResampleFilter, char>);
static_assert(std::formattable<img::Pixel, char>);
static_assert(std::formattable<img::HSV, char>);
static_assert(std::formattable<img::Image, char>);
static_assert(std::formattable<img::JpegDecodeOptions, char>);
static_assert(std::formattable<img::JpegInfo, char>);

TEST(ImageFormat, Enums) {
    EXPECT_EQ(std::format("{}", img::ImageMode::GrayscaleAlpha), "GrayscaleAlpha");
    EXPECT_EQ(std::format("{}", static_cast<img::ImageMode>(7)), "ImageMode(7)");
    EXPECT_EQ(std::format("{}", img::JpegDctMethod::IntegerAccurate), "IntegerAccurate");
    EXPECT_EQ(std::format("{}", img::JpegColorSpace::YCbCr), "YCbCr");
    EXPECT_EQ(std::format("{}", img::ResampleFilter::Bilinear), "Bilinear");
}

TEST(ImageFormat, ValuesAndImages) {
    EXPECT_EQ(std::format("{}", img::Pixel{255, 0, 0, 128}), "Pixel(255, 0, 0, 128)");
    EXPECT_EQ(std::format("{}", img::HSV{120, 0.5, 1}), "HSV(120, 0.5, 1)");
    EXPECT_EQ(std::format("{}", img::Image(640, 480, img::ImageMode::RGB)), "Image 640x480 RGB");
    EXPECT_EQ(std::format("{}", img::Image(3, 2, img::ImageMode::Grayscale)), "Image 3x2 Grayscale");
    EXPECT_EQ(std::format("{}", img::Image()), "Image 0x0 RGB");
    EXPECT_EQ(std::format("{:>20}", img::Image(4, 4)), "       Image 4x4 RGB");
    EXPECT_EQ(std::format("{}", img::JpegDecodeOptions{}),
              "JpegDecodeOptions(channels=0, dct_method=IntegerFast, fancy_upsampling=true)");
    img::JpegInfo info;
    info.width = 8;
    info.height = 4;
    info.components = 3;
    info.h_sampling = {2, 1, 1};
    info.v_sampling = {2, 1, 1};
    EXPECT_EQ(std::format("{}", info),
              "JpegInfo(8x4, components=3, color_space=YCbCr, progressive=false, h_sampling=[2, 1, 1], "
              "v_sampling=[2, 1, 1], restart_interval=0, jfif=false, adobe=false, adobe_transform=-1, orientation=1)");
}
