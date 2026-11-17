#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/median_filter.hpp>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
std::vector<std::vector<int>> sorted_reference(
    const std::vector<std::vector<int>>& image,
    int                                  radius) {
  auto out  = image;
  int  rows = image.size(), cols = rows ? image.front().size() : 0;
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) {
      std::vector<int> values;
      for (int r = i - radius; r <= i + radius; ++r)
        for (int c = j - radius; c <= j + radius; ++c)
          values.push_back(image[std::clamp(r, 0, rows - 1)][std::clamp(c, 0, cols - 1)]);
      std::sort(values.begin(), values.end());
      out[i][j] = values[values.size() / 2];
    }
  return out;
}
} // namespace
TEST(MedianFilterTrace, AllBinaryThreeSquareImagesActualClampsAndSortedRankReference) {
  for (int bits = 0; bits < 512; ++bits)
    for (int radius = 0; radius <= 2; ++radius) {
      std::vector<std::vector<int>> image(3, std::vector<int>(3));
      for (int k = 0; k < 9; ++k)
        image[k / 3][k % 3] = (bits >> k) & 1;
      auto t = median_filter_trace(image, radius);
      EXPECT_EQ(t.analysis.image, sorted_reference(image, radius));
      std::size_t       sample = 0, pixel = 0;
      const std::size_t w = (2 * radius + 1) * (2 * radius + 1);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          auto& p = t.pixels[pixel++];
          EXPECT_EQ(p.first, sample);
          EXPECT_EQ(p.count, w);
          EXPECT_EQ(p.row, i);
          EXPECT_EQ(p.col, j);
          EXPECT_EQ(p.median, t.analysis.image[i][j]);
          for (int di = -radius; di <= radius; ++di)
            for (int dj = -radius; dj <= radius; ++dj) {
              auto& s  = t.samples[sample++];
              int   sr = std::clamp(i + di, 0, 2), sc = std::clamp(j + dj, 0, 2);
              EXPECT_EQ(s.output_row, i);
              EXPECT_EQ(s.output_col, j);
              EXPECT_EQ(s.source_row, sr);
              EXPECT_EQ(s.source_col, sc);
              EXPECT_EQ(s.value, image[sr][sc]);
            }
        }
      EXPECT_EQ(t.analysis.work.samples, 9 * w);
      EXPECT_EQ(t.analysis.work.pixels, 9);
      EXPECT_EQ(t.captured_entries, 27 + 9 * w);
      EXPECT_EQ(median_filter(image, radius), t.analysis.image);
    }
}
TEST(MedianFilterTrace, OriginalSeededFixtureAndActualErrorSummaries) {
  auto                                   d     = median_filter_demo();
  auto                                   noisy = d.clean;
  std::mt19937_64                        rng(7);
  std::uniform_real_distribution<double> uniform(0, 1);
  std::size_t                            corrupted = 0;
  for (int i = 0; i < 40; ++i)
    for (int j = 0; j < 40; ++j) {
      EXPECT_EQ(d.clean[i][j], j < 20 ? 60 : 200);
      if (uniform(rng) < .15) {
        noisy[i][j] = uniform(rng) < .5 ? 0 : 255;
        ++corrupted;
      }
    }
  EXPECT_EQ(noisy, d.noisy);
  EXPECT_EQ(corrupted, d.corruptions);
  EXPECT_EQ(d.filtered.image, sorted_reference(noisy, 1));
  EXPECT_EQ(d.noisy_summary.extremes, corrupted);
  EXPECT_EQ(d.noisy_summary.changed, corrupted);
  for (auto candidate : {d.noisy, d.filtered.image}) {
    auto        summary = median_filter_compare(d.clean, candidate);
    std::size_t changes = 0, extremes = 0;
    long long   total   = 0;
    int         maximum = 0;
    for (int i = 0; i < 40; ++i)
      for (int j = 0; j < 40; ++j) {
        auto v = candidate[i][j];
        changes += v != d.clean[i][j];
        extremes += v == 0 || v == 255;
        int error = std::abs(v - d.clean[i][j]);
        total += error;
        maximum = std::max(maximum, error);
      }
    EXPECT_EQ(summary.changed, changes);
    EXPECT_EQ(summary.extremes, extremes);
    EXPECT_EQ(summary.maximum_absolute_error, maximum);
    EXPECT_DOUBLE_EQ(summary.mean_absolute_error, double(total) / 1600);
  }
}
TEST(MedianFilterTrace, ThinFeaturesStepIdentityEmptyGuardsQuotasAndIntegerRange) {
  std::vector<std::vector<int>> step(5, std::vector<int>(5)), thin = step;
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j) {
      step[i][j] = j < 2 ? 60 : 200;
      thin[i][j] = j == 2 ? 255 : 0;
    }
  EXPECT_EQ(median_filter(step), step);
  EXPECT_EQ(median_filter(thin), std::vector<std::vector<int>>(5, std::vector<int>(5)));
  EXPECT_EQ(median_filter(step, 0), step);
  EXPECT_TRUE(median_filter({}).empty());
  EXPECT_EQ(median_filter({{}, {}}, 1).size(), 2);
  EXPECT_EQ(median_filter_trace({{}, {}}, 1, 1).captured_entries, 0);
  EXPECT_THROW(median_filter({{1}, {}}, 1), std::invalid_argument);
  EXPECT_THROW(median_filter({}, -1), std::invalid_argument);
  EXPECT_THROW(median_filter_trace({{1}, {}}, 1, 1), std::invalid_argument);
  EXPECT_THROW(median_filter({}, INT_MAX), std::length_error);
  EXPECT_NO_THROW(median_filter_trace({{7}}, 1, 12));
  EXPECT_THROW(median_filter_trace({{7}}, 1, 11), std::length_error);
  EXPECT_NO_THROW(median_filter_trace({{7}}, 1, 0));
  auto image  = step;
  auto t      = median_filter_trace(image);
  image[0][0] = 999;
  EXPECT_EQ(t.input[0][0], 60);
  auto s = median_filter_compare({{INT_MIN}}, {{INT_MAX}});
  EXPECT_EQ(s.maximum_absolute_error, 4294967295ULL);
  EXPECT_DOUBLE_EQ(s.mean_absolute_error, 4294967295.0);
  EXPECT_THROW(median_filter_compare({{1}}, {{1, 2}}), std::invalid_argument);
  EXPECT_THROW(median_filter_compare({{}}, {}), std::invalid_argument);
  EXPECT_EQ(median_filter_compare({}, {}).pixels, 0);
}
