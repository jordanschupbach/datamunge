#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/floyd_steinberg.hpp>
#include <datamunge/algorithms/ordered_dither.hpp>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(OrderedDitherTrace, EveryTileRankThresholdTieAndCompleteActualPixels) {
  int              matrix[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};
  std::vector<int> ranks;
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t j = 0; j < 4; ++j) {
      auto threshold = (matrix[i][j] + .5) * 255 / 16.;
      EXPECT_EQ(bayer4_rank(i, j), matrix[i][j]);
      EXPECT_EQ(bayer4_threshold(i, j), threshold);
      ranks.push_back(matrix[i][j]);
      for (double value :
           {std::nextafter(threshold, -INFINITY), threshold, std::nextafter(threshold, INFINITY)}) {
        std::vector<std::vector<double>> im(8, std::vector<double>(8, value));
        auto                             t = ordered_dither_trace(im);
        ASSERT_EQ(t.steps.size(), 64);
        std::size_t whites = 0;
        for (std::size_t k = 0; k < 64; ++k) {
          auto& s = t.steps[k];
          EXPECT_EQ(s.row, k / 8);
          EXPECT_EQ(s.col, k % 8);
          EXPECT_EQ(s.input, value);
          EXPECT_EQ(s.rank, matrix[s.row % 4][s.col % 4]);
          EXPECT_EQ(s.threshold, (s.rank + .5) * 255 / 16.);
          EXPECT_EQ(s.output, value > s.threshold ? 255 : 0);
          EXPECT_EQ(t.result.output[s.row][s.col], s.output);
          whites += s.output == 255;
        }
        EXPECT_EQ(t.result.work.white, whites);
        EXPECT_EQ(t.result.work.black, 64 - whites);
        EXPECT_EQ(t.captured_entries, 194);
        EXPECT_EQ(ordered_dither(im), t.result.output);
        EXPECT_NO_THROW(ordered_dither_trace(im, 194));
        EXPECT_THROW(ordered_dither_trace(im, 193), std::length_error);
      }
    }
  std::sort(ranks.begin(), ranks.end());
  for (int k = 0; k < 16; ++k)
    EXPECT_EQ(ranks[k], k);
  EXPECT_EQ(bayer4_rank(SIZE_MAX, SIZE_MAX), 5);
}
TEST(OrderedDitherTrace, OriginalGradientBandsUniformToneCountsAndNativePlotOrientation) {
  auto image  = dither_gradient(40, 64);
  auto r      = ordered_dither_work(image);
  auto left   = dither_region_stats(image, r.output, 0, 64 / 3);
  auto middle = dither_region_stats(image, r.output, 64 / 3, 2 * 64 / 3);
  auto right  = dither_region_stats(image, r.output, 2 * 64 / 3, 64);
  EXPECT_NEAR(double(left.white_fraction), 1. / 6, 1e-15);
  EXPECT_NEAR(double(middle.white_fraction), .4880952380952381, 1e-15);
  EXPECT_NEAR(double(right.white_fraction), .8295454545454546, 1e-15);
  for (int count = 0; count <= 16; ++count) {
    double tone = 255. * count / 16;
    auto t = ordered_dither_work(std::vector<std::vector<double>>(4, std::vector<double>(4, tone)));
    EXPECT_EQ(t.work.white, count);
  }
  auto plot = ordered_dither_image_plot({{0, 255}, {255, 0}});
  ASSERT_EQ(plot.series().size(), 4);
  EXPECT_EQ(plot.series()[0].color.r, 0);
  EXPECT_EQ(plot.series()[1].color.r, 255);
  EXPECT_EQ(plot.series()[2].color.r, 255);
  EXPECT_NO_THROW(ordered_dither_image_plot({{0}}));
  EXPECT_TRUE(ordered_dither({}).empty());
  EXPECT_EQ(ordered_dither({{}, {}}).size(), 2);
  EXPECT_EQ(ordered_dither_trace({}).captured_entries, 2);
  EXPECT_EQ(ordered_dither({{-1., 300.}})[0], (std::vector<int>{0, 255}));
  EXPECT_THROW(ordered_dither({{0.}, {1., 2.}}), std::invalid_argument);
  EXPECT_THROW(ordered_dither({{NAN}}), std::invalid_argument);
  EXPECT_THROW(ordered_dither_image_plot({{12}}), std::invalid_argument);
  EXPECT_THROW(ordered_dither_image_plot({{0, 255}}, 1), std::length_error);
}
