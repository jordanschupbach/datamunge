#include <gtest/gtest.h>

#include <datamunge/algorithms/connected_components.hpp>
#include <datamunge/algorithms/dithering.hpp>
#include <datamunge/algorithms/histogram_equalization.hpp>
#include <datamunge/algorithms/median_filter.hpp>
#include <datamunge/algorithms/sobel.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

using namespace datamunge::algorithms;

TEST(ConnectedComponents, CountsBlobs) {
    std::vector<std::vector<int>> img = {
        {1, 1, 0, 0, 1},
        {1, 1, 0, 0, 1},
        {0, 0, 0, 0, 0},
        {0, 1, 0, 1, 1},
    };
    auto cc = connected_components(img, false);  // 4-connectivity
    EXPECT_EQ(cc.count, 4);                       // TL block, TR column, bottom-left single, bottom-right pair
    // Same label within a blob; different across blobs.
    EXPECT_EQ(cc.labels[0][0], cc.labels[1][1]);
    EXPECT_NE(cc.labels[0][0], cc.labels[0][4]);
    // 8-connectivity does not merge these (no diagonal touch here) -> still 4.
    EXPECT_EQ(connected_components(img, true).count, 4);
}

TEST(FloydSteinberg, PreservesAverageBrightness) {
    // A uniform mid-gray image should dither to ~50% black/white.
    std::vector<std::vector<double>> img(16, std::vector<double>(16, 128.0));
    auto                             d = floyd_steinberg_dither(img);
    std::size_t                      white = 0;
    for (const auto& row : d)
        for (int v : row) { EXPECT_TRUE(v == 0 || v == 255); white += (v == 255); }
    const double frac = static_cast<double>(white) / (16 * 16);
    EXPECT_NEAR(frac, 0.5, 0.15);  // average tone preserved
}

TEST(OrderedDither, ThresholdsByBayerMatrix) {
    std::vector<std::vector<double>> img(8, std::vector<double>(8, 128.0));
    auto                             d = ordered_dither(img);
    std::size_t                      white = 0;
    for (const auto& row : d)
        for (int v : row) { EXPECT_TRUE(v == 0 || v == 255); white += (v == 255); }
    EXPECT_NEAR(static_cast<double>(white) / 64.0, 0.5, 0.1);  // mid-gray -> ~half on
    // A fully bright image is all white, a fully dark image all black.
    std::vector<std::vector<double>> bright(4, std::vector<double>(4, 255.0));
    for (const auto& row : ordered_dither(bright)) for (int v : row) EXPECT_EQ(v, 255);
}

TEST(Sobel, RespondsAtAVerticalEdge) {
    // Left half dark, right half bright: a strong vertical edge in the middle.
    std::vector<std::vector<double>> img(6, std::vector<double>(6, 0.0));
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 3; j < 6; ++j) img[i][j] = 255.0;
    auto r = sobel(img);
    // Gradient magnitude peaks along the edge column (j == 2 or 3), and is ~0 in flat regions.
    EXPECT_GT(r.magnitude[3][2] + r.magnitude[3][3], 0.0);
    EXPECT_NEAR(r.magnitude[3][0], 0.0, 1e-9);  // flat dark region
    // The horizontal gradient dominates for a vertical edge.
    EXPECT_GT(std::fabs(r.gx[3][2]) + std::fabs(r.gx[3][3]), std::fabs(r.gy[3][2]) + std::fabs(r.gy[3][3]));
}

TEST(HistogramEqualization, SpreadsIntensityRange) {
    // A low-contrast image using only values 100..107.
    std::vector<std::vector<int>> img(8, std::vector<int>(8));
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) img[i][j] = 100 + ((i + j) % 8);
    auto eq = histogram_equalize(img);
    int  lo = 255, hi = 0;
    for (const auto& row : eq)
        for (int v : row) { lo = std::min(lo, v); hi = std::max(hi, v); }
    // Equalization stretches the used range far wider than the original 7-level span.
    EXPECT_GT(hi - lo, 100);
    EXPECT_LE(hi, 255);
    EXPECT_GE(lo, 0);
}

TEST(MedianFilter, RemovesSaltAndPepper) {
    // Flat gray 100 with a single salt pixel and a single pepper pixel.
    std::vector<std::vector<int>> img(7, std::vector<int>(7, 100));
    img[3][3] = 255;  // salt
    img[1][5] = 0;    // pepper
    auto out = median_filter(img, 1);
    EXPECT_EQ(out[3][3], 100);  // impulse removed
    EXPECT_EQ(out[1][5], 100);
    // Flat regions untouched.
    EXPECT_EQ(out[5][2], 100);
}
