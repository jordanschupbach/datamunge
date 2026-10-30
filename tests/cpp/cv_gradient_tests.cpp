#include <gtest/gtest.h>

#include <datamunge/cv/gradient.hpp>

#include <stdexcept>
#include <vector>

TEST(CvGradient, RampHasUnitDerivative) {
    const int w = 6, h = 5;
    std::vector<double> img(w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) img[y * w + x] = 2.0 * x - 3.0 * y;
    const auto g = datamunge::cv::sobel_gradient(img, w, h);
    // Interior pixels see the exact slopes; clamped borders see half of them.
    for (int y = 1; y < h - 1; ++y)
        for (int x = 1; x < w - 1; ++x) {
            EXPECT_DOUBLE_EQ(g.gx[y * w + x], 2.0);
            EXPECT_DOUBLE_EQ(g.gy[y * w + x], -3.0);
        }
    EXPECT_DOUBLE_EQ(g.gx[2 * w], 1.0);
    EXPECT_THROW(datamunge::cv::sobel_gradient(img, w, h + 1), std::invalid_argument);
}
