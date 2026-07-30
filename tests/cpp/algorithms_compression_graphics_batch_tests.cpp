#include <gtest/gtest.h>

#include <datamunge/algorithms/bresenham.hpp>
#include <datamunge/algorithms/flood_fill.hpp>
#include <datamunge/algorithms/huffman_coding.hpp>
#include <datamunge/algorithms/run_length_encoding.hpp>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using datamunge::algorithms::bresenham_line;
using datamunge::algorithms::flood_fill;
using datamunge::algorithms::huffman_average_bits;
using datamunge::algorithms::huffman_build;
using datamunge::algorithms::huffman_decode;
using datamunge::algorithms::huffman_encode;
using datamunge::algorithms::rle_decode;
using datamunge::algorithms::rle_encode;

TEST(Huffman, RoundTripsAndCompresses) {
    std::string text = "abracadabra abracadabra abracadabra!";
    auto        hc   = huffman_build(text);
    auto        enc  = huffman_encode(hc, text);
    auto        dec  = huffman_decode(hc, enc);
    EXPECT_EQ(dec, text);
    // Codes are prefix-free (verified indirectly by exact round-trip) and shorter than 8 bits/char.
    EXPECT_LT(huffman_average_bits(hc, text), 8.0);
    // Frequent symbols get shorter codes than rare ones: 'a' (very common) shorter than '!' (rare).
    EXPECT_LT(hc.codes.at('a').size(), hc.codes.at('!').size());
}

TEST(Huffman, SingleSymbol) {
    std::string text = "aaaaaa";
    auto        hc   = huffman_build(text);
    auto        enc  = huffman_encode(hc, text);
    EXPECT_EQ(huffman_decode(hc, enc), text);
    EXPECT_EQ(hc.codes.at('a'), "0");
}

TEST(RLE, RoundTripsAndCountsRuns) {
    std::string text = "aaabbbcccd";
    auto        runs = rle_encode(text);
    ASSERT_EQ(runs.size(), 4u);  // aaa, bbb, ccc, d
    EXPECT_EQ(runs[0].symbol, 'a');
    EXPECT_EQ(runs[0].count, 3u);
    EXPECT_EQ(runs[3].symbol, 'd');
    EXPECT_EQ(runs[3].count, 1u);
    EXPECT_EQ(rle_decode(runs), text);
    // A run-free string encodes to one cell per symbol (no compression).
    EXPECT_EQ(rle_encode("abcdef").size(), 6u);
}

TEST(Bresenham, RasterizesLines) {
    // A shallow diagonal.
    auto pts = bresenham_line(0, 0, 5, 2);
    ASSERT_FALSE(pts.empty());
    EXPECT_EQ(pts.front(), (std::pair<int, int>{0, 0}));
    EXPECT_EQ(pts.back(), (std::pair<int, int>{5, 2}));
    EXPECT_EQ(pts.size(), 6u);  // one cell per x-column for a shallow line
    // Every consecutive pair is 8-adjacent.
    for (std::size_t i = 1; i < pts.size(); ++i) {
        EXPECT_LE(std::abs(pts[i].first - pts[i - 1].first), 1);
        EXPECT_LE(std::abs(pts[i].second - pts[i - 1].second), 1);
    }
    // A perfectly diagonal line steps one-and-one each time.
    auto diag = bresenham_line(0, 0, 4, 4);
    EXPECT_EQ(diag.size(), 5u);
    EXPECT_EQ(diag[2], (std::pair<int, int>{2, 2}));
    // Horizontal line.
    EXPECT_EQ(bresenham_line(2, 7, 6, 7).size(), 5u);
}

TEST(FloodFill, FillsConnectedRegionOnly) {
    // Grid with two separate regions of 1s in a sea of 0s.
    std::vector<std::vector<int>> grid = {
        {1, 1, 0, 0, 1},
        {1, 1, 0, 0, 1},
        {0, 0, 0, 0, 0},
        {0, 1, 1, 0, 0},
    };
    // Fill the top-left region (4 cells) with 9 using 4-connectivity.
    std::size_t filled = flood_fill(grid, 0, 0, 9, false);
    EXPECT_EQ(filled, 4u);
    EXPECT_EQ(grid[0][0], 9);
    EXPECT_EQ(grid[1][1], 9);
    EXPECT_EQ(grid[0][4], 1);  // the other region is untouched
    EXPECT_EQ(grid[3][1], 1);
    // Filling a region with its own value does nothing.
    EXPECT_EQ(flood_fill(grid, 0, 4, 1, false), 0u);
}
