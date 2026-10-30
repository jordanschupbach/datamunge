// std::format / std::print support for datamunge::cv.
#include <dm/dm.hpp>

#include <gtest/gtest.h>

#include <format>
#include <string>

namespace cv = dm::cv;

static_assert(std::formattable<cv::KeyPoint, char>);
static_assert(std::formattable<cv::Corner, char>);
static_assert(std::formattable<cv::HoughLine, char>);
static_assert(std::formattable<cv::HoughCircle, char>);
static_assert(std::formattable<cv::FlowVector, char>);
static_assert(std::formattable<cv::LabelMap, char>);
static_assert(std::formattable<cv::StructuringElement, char>);
static_assert(std::formattable<cv::PoolMode, char>);

TEST(CvFormat, ValueTypes) {
    EXPECT_EQ(std::format("{}", cv::KeyPoint{3, 4, 1.6, -2.5}), "KeyPoint(x=3, y=4, scale=1.6, response=-2.5)");
    EXPECT_EQ(std::format("{}", cv::Corner{1, 2, 0.75}), "Corner(x=1, y=2, response=0.75)");
    EXPECT_EQ(std::format("{}", cv::HoughLine{10, 0.5, 7}), "HoughLine(rho=10, theta=0.5, votes=7)");
    EXPECT_EQ(std::format("{}", cv::HoughCircle{10, 12, 5, 42}), "HoughCircle(cx=10, cy=12, radius=5, votes=42)");
    EXPECT_EQ(std::format("{}", cv::FlowVector{0.5, -1, true}), "FlowVector(dx=0.5, dy=-1, valid=true)");
    EXPECT_EQ(std::format("{}", cv::StructuringElement::Cross), "Cross");
    EXPECT_EQ(std::format("{}", cv::PoolMode::Average), "Average");
    EXPECT_EQ(std::format("{}", static_cast<cv::PoolMode>(5)), "PoolMode(5)");
}

TEST(CvFormat, LabelMap) {
    const cv::LabelMap small{{0, 1, 1, 0, 0, 2}, 3, 2, 2};
    EXPECT_EQ(std::format("{}", small), "LabelMap 3x2, count=2\n  0 1 1\n  0 0 2");
    EXPECT_EQ(std::format("{}", cv::LabelMap{}), "LabelMap 0x0, count=0");
    cv::LabelMap big;
    big.width = 20;
    big.height = 15;
    big.labels.assign(300, 0);
    const std::string text = std::format("{}", big);
    EXPECT_EQ(text.rfind("LabelMap 20x15, count=0\n  0 0 0 0 0 0 0 0 ...", 0), 0u);
    EXPECT_TRUE(text.ends_with("\n  ..."));
}
