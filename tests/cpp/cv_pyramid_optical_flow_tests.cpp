#include <gtest/gtest.h>

#include <datamunge/cv/optical_flow.hpp>
#include <datamunge/cv/pyramid.hpp>

#include <cmath>

using namespace datamunge::cv;
using namespace datamunge::image;

TEST(GaussianPyramid, LevelsHalveInDimensionEachStep) {
    const Image img(64, 48, ImageMode::RGB, Pixel{100, 150, 200, 255});
    const auto pyramid = gaussian_pyramid(img, 4);
    ASSERT_EQ(pyramid.size(), 4u);
    EXPECT_EQ(pyramid[0].width(), 64);
    EXPECT_EQ(pyramid[0].height(), 48);
    EXPECT_EQ(pyramid[1].width(), 32);
    EXPECT_EQ(pyramid[1].height(), 24);
    EXPECT_EQ(pyramid[2].width(), 16);
    EXPECT_EQ(pyramid[2].height(), 12);
    EXPECT_EQ(pyramid[3].width(), 8);
    EXPECT_EQ(pyramid[3].height(), 6);
}

TEST(GaussianPyramid, SolidColorImageStaysSolidAtEveryLevel) {
    const Image img(32, 32, ImageMode::RGB, Pixel{100, 150, 200, 255});
    const auto pyramid = gaussian_pyramid(img, 3);
    for (const auto& level : pyramid) {
        EXPECT_EQ(level.get_pixel(level.width() / 2, level.height() / 2), (Pixel{100, 150, 200, 255}));
    }
}

TEST(GaussianPyramid, RejectsFewerThanOneLevel) {
    const Image img(16, 16, ImageMode::RGB);
    EXPECT_THROW(gaussian_pyramid(img, 0), std::invalid_argument);
}

TEST(LaplacianPyramid, SameLevelCountAsGaussianAndZeroDetailForASolidImage) {
    const Image img(32, 32, ImageMode::RGB, Pixel{100, 150, 200, 255});
    const auto gpyr = gaussian_pyramid(img, 3);
    const auto lpyr = laplacian_pyramid(img, 3);
    ASSERT_EQ(lpyr.size(), gpyr.size());
    for (int y = 0; y < lpyr[0].height(); ++y)
        for (int x = 0; x < lpyr[0].width(); ++x) EXPECT_EQ(lpyr[0].get_pixel(x, y), (Pixel{128, 128, 128, 255}));
    EXPECT_EQ(lpyr.back().get_pixel(0, 0), gpyr.back().get_pixel(0, 0));
}

namespace {
Image make_smooth_pattern(int size) {
    Image img(size, size, ImageMode::RGB);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            constexpr double kTwoPi = 6.283185307179586;
            const double v = 128.0 + 80.0 * std::sin(kTwoPi * x / 40.0) + 40.0 * std::sin(kTwoPi * y / 55.0);
            const auto byte = static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L));
            img.set_pixel(x, y, Pixel{byte, byte, byte, 255});
        }
    }
    return img;
}
} // namespace

TEST(LucasKanadeOpticalFlow, RecoversAKnownTranslationOnASmoothPattern) {
    const Image frame1 = make_smooth_pattern(60);
    Image frame2(60, 60, ImageMode::RGB);
    for (int y = 0; y < 60; ++y) {
        for (int x = 0; x < 60; ++x) {
            constexpr double kTwoPi = 6.283185307179586;
            const double v = 128.0 + 80.0 * std::sin(kTwoPi * (x - 2) / 40.0) + 40.0 * std::sin(kTwoPi * (y - 1) / 55.0);
            const auto byte = static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L));
            frame2.set_pixel(x, y, Pixel{byte, byte, byte, 255});
        }
    }

    const std::vector<std::pair<int, int>> points = {{30, 30}, {20, 40}, {40, 20}};
    const auto flow = lucas_kanade_optical_flow(frame1, frame2, points, 8);
    ASSERT_EQ(flow.size(), 3u);
    for (const auto& f : flow) {
        ASSERT_TRUE(f.valid);
        EXPECT_NEAR(f.dx, 2.0, 1.0);
        EXPECT_NEAR(f.dy, 1.0, 1.0);
    }
}

TEST(LucasKanadeOpticalFlow, MarksBorderPointsInvalid) {
    const Image frame1 = make_smooth_pattern(60);
    const Image frame2 = make_smooth_pattern(60);
    const auto flow = lucas_kanade_optical_flow(frame1, frame2, {{2, 2}}, 8);
    ASSERT_EQ(flow.size(), 1u);
    EXPECT_FALSE(flow[0].valid);
}

TEST(LucasKanadeOpticalFlow, RejectsMismatchedFrameDimensions) {
    const Image frame1(60, 60, ImageMode::RGB);
    const Image frame2(10, 10, ImageMode::RGB);
    EXPECT_THROW(lucas_kanade_optical_flow(frame1, frame2, {{5, 5}}, 3), std::invalid_argument);
}
