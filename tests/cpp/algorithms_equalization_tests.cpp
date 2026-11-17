#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/histogram_equalization_analysis.hpp>
#include <numeric>
using namespace datamunge::algorithms;
TEST(Equalization, ExhaustiveClampedInputHistogramsAndExactSmallRationalMap) {
  std::size_t combinations = 1;
  for (int n = 0; n <= 8; ++n) {
    if (n)
      combinations *= 3;
    for (std::size_t mask = 0; mask < combinations; ++mask) {
      auto              v = mask;
      EqualizationImage input(1, std::vector<int>(n));
      for (auto& p : input[0]) {
        p = int(v % 3) + 100;
        v /= 3;
      }
      const auto t = histogram_equalize_trace(input);
      EXPECT_EQ(t.image, histogram_equalize(input));
      EXPECT_EQ(t.pixels, n);
      EXPECT_EQ(t.rows, 1);
      EXPECT_EQ(t.columns, n);
      EXPECT_EQ(t.work.histogram_pixels, n);
      EXPECT_EQ(t.work.output_pixels, n);
      EXPECT_EQ(t.work.cdf_bins, 256);
      EXPECT_EQ(t.work.mapping_bins, 256);
      EXPECT_EQ(std::accumulate(t.histogram.begin(), t.histogram.end(), std::size_t{}), n);
      EXPECT_EQ(
          std::accumulate(t.output_histogram.begin(), t.output_histogram.end(), std::size_t{}), n);
      std::array<std::size_t, 256> counts{}, out{};
      for (int p : input[0])
        ++counts[p];
      std::size_t first = 0;
      for (auto c : counts)
        if (c) {
          first = c;
          break;
        }
      std::size_t sum = 0;
      for (std::size_t j = 0; j < 256; ++j) {
        sum += counts[j];
        EXPECT_EQ(t.histogram[j], counts[j]);
        EXPECT_EQ(t.cdf[j], sum);
        const std::size_t den = n - first;
        int               expected =
            den ? int((2 * 255 * (sum >= first ? sum - first : 0) + den) / (2 * den)) : int(j);
        EXPECT_EQ(t.mapping[j], expected);
        if (j)
          EXPECT_LE(t.mapping[j - 1], t.mapping[j]);
      }
      for (std::size_t j = 0; j < input[0].size(); ++j) {
        EXPECT_EQ(t.image[0][j], t.mapping[input[0][j]]);
        ++out[t.image[0][j]];
      }
      EXPECT_EQ(out, t.output_histogram);
      auto copy       = t;
      copy.mapping[0] = 999;
      copy.image[0].push_back(999);
      EXPECT_NE(t.mapping[0], 999);
      EXPECT_EQ(t.image[0].size(), n);
    }
  }
}
TEST(Equalization, ClampsShapesIdentityAndCompleteCaps) {
  const auto t = histogram_equalize_trace({{-100, 0, 255, std::numeric_limits<int>::max()}});
  EXPECT_EQ(t.work.clamped_pixels, 2);
  EXPECT_EQ(t.image, (EqualizationImage{{0, 0, 255, 255}}));
  for (int v : {-1, 0, 7, 255, 256}) {
    const auto a = histogram_equalize_trace(EqualizationImage(2, std::vector<int>(3, v)));
    EXPECT_TRUE(a.identity_mapping);
    EXPECT_EQ(a.image, EqualizationImage(2, std::vector<int>(3, std::clamp(v, 0, 255))));
  }
  EXPECT_TRUE(histogram_equalize({}).empty());
  EXPECT_EQ(histogram_equalize({{}, {}}), (EqualizationImage{{}, {}}));
  EXPECT_THROW(histogram_equalize({{1, 2}, {3}}), std::invalid_argument);
  EXPECT_THROW(histogram_equalize_trace({{}, {1}}, {1, 1}), std::invalid_argument);
  EXPECT_NO_THROW(histogram_equalize_trace({{1, 2}, {3, 4}}, {2, 4}));
  EXPECT_THROW(histogram_equalize_trace({{1, 2}, {3, 4}}, {1, 4}), std::length_error);
  EXPECT_THROW(histogram_equalize_trace({{1, 2}, {3, 4}}, {2, 3}), std::length_error);
  EXPECT_NO_THROW(histogram_equalize_trace({{1, 2}, {3, 4}}, {0, 0}));
  EXPECT_THROW(
      equalization_detail::dimensions(std::numeric_limits<std::size_t>::max(), 2, {0, 0}),
      std::length_error);
}
TEST(Equalization, RareOccupiedLevelsCanMerge) {
  EqualizationImage input(1, std::vector<int>(1002, 102));
  input[0][0]  = 100;
  input[0][1]  = 101;
  const auto t = histogram_equalize_trace(input);
  EXPECT_EQ(t.mapping[100], 0);
  EXPECT_EQ(t.mapping[101], 0);
  EXPECT_EQ(t.mapping[102], 255);
  EXPECT_EQ(t.output_histogram[0], 2);
  EXPECT_EQ(t.output_histogram[255], 1000);
}

TEST(Equalization, OriginalFixtureCoarseCountsAndNativeDisplayOrientation) {
  const auto input   = equalization_pattern();
  const auto t       = histogram_equalize_trace(input);
  auto       grouped = equalization_hist16(t.output_histogram);
  for (auto n : grouped)
    EXPECT_EQ(n, 256);
  EXPECT_EQ(equalization_pattern(2, 2), (EqualizationImage{{96, 101}, {103, 108}}));
  EXPECT_THROW(equalization_pattern(2, 2, {2, 3}), std::length_error);
  std::array<std::size_t, 256> enormous{};
  enormous[0] = std::numeric_limits<std::size_t>::max();
  enormous[1] = 1;
  EXPECT_THROW(equalization_hist16(enormous), std::length_error);
  const auto p = equalization_image_plot({{0, 85}, {170, 255}}, "orientation");
  ASSERT_EQ(p.series().size(), 4);
  for (const auto& cell : p.series()) {
    const auto x = (*std::min_element(cell.x.begin(), cell.x.end()) +
                    *std::max_element(cell.x.begin(), cell.x.end())) /
                   2;
    const auto y = (*std::min_element(cell.y.begin(), cell.y.end()) +
                    *std::max_element(cell.y.begin(), cell.y.end())) /
                   2;
    const int want = y == 0 ? (x == 0 ? 170 : 255) : (x == 0 ? 0 : 85);
    EXPECT_EQ(int(cell.color.r), want);
    EXPECT_EQ(int(cell.color.g), want);
    EXPECT_EQ(int(cell.color.b), want);
  }
  EXPECT_THROW(equalization_image_plot({{1}}, "small"), std::invalid_argument);
  EXPECT_THROW(equalization_image_plot({{1, 2}, {3, 4}}, "cap", 3), std::length_error);
}
