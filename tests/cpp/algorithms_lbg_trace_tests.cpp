#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/linde_buzo_gray.hpp>
#include <limits>
using namespace datamunge::algorithms;
namespace {
// Independent direct objective and earliest nearest assignment, for produced finite fixtures.
void consistent(const std::vector<std::vector<double>>& data, const LBGResult& r) {
  double total = 0;
  for (std::size_t i = 0; i < data.size(); ++i) {
    std::size_t best     = 0;
    double      distance = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < r.codebook.size(); ++k) {
      double q = 0;
      for (std::size_t j = 0; j < data[i].size(); ++j)
        q += std::pow(data[i][j] - r.codebook[k][j], 2);
      if (q < distance) {
        distance = q;
        best     = k;
      }
    }
    EXPECT_EQ(r.assignment[i], best);
    total += distance;
  }
  EXPECT_NEAR(r.distortion, total / data.size(), 1e-12);
}
} // namespace
TEST(LBGTrace, ActualTinyStagesAndCorrectedCappedLabels) {
  auto t = linde_buzo_gray_trace({{1}, {3}}, 2);
  EXPECT_EQ(t.analysis.result.codebook, (std::vector<std::vector<double>>{{1}, {3}}));
  EXPECT_EQ(t.analysis.result.distortion, 0);
  EXPECT_EQ(t.stages.size(), 2);
  EXPECT_EQ(t.analysis.work.lloyd_iterations, t.iterations.size());
  std::vector<std::vector<double>> data{{0}, {5}, {6}, {7}, {8}};
  auto                             cap = linde_buzo_gray_trace(data, 2, .01, 1, 1e-6);
  EXPECT_FALSE(cap.stages.back().tolerance_stop);
  auto& last = cap.iterations.back();
  EXPECT_EQ(last.assignment[1], 0);
  EXPECT_EQ(cap.analysis.result.assignment[1], 1);
  EXPECT_NEAR(cap.analysis.result.distortion, 2.45, 1e-12);
  EXPECT_NE(last.before_distortion, cap.analysis.result.distortion);
  consistent(data, cap.analysis.result);
}
TEST(LBGTrace, ActualMeanUpdatesAndConsistentStageObjectives) {
  auto data = lbg_demo_mixture();
  ASSERT_EQ(data.size(), 600);
  auto t = linde_buzo_gray_trace(data, 16, .01, 100, 1e-8);
  consistent(data, t.analysis.result);
  for (std::size_t k = 0; k < t.stages.size(); ++k) {
    auto& s = t.stages[k];
    auto  q = lbg_quantize(data, s.final);
    EXPECT_EQ(q.assignment, s.assignment);
    EXPECT_EQ(q.distortion, s.distortion);
    EXPECT_EQ(s.distortion, t.analysis.result.distortion_history[k]);
  }
  std::size_t passes = 0, pairs = 0, squares = 0;
  for (auto& i : t.iterations) {
    ++passes;
    pairs += data.size() * i.before.size();
    squares += data.size() * i.before.size() * 2;
    std::vector<std::vector<double>> sum(i.before.size(), std::vector<double>(2));
    std::vector<std::size_t>         counts(i.before.size());
    for (std::size_t j = 0; j < data.size(); ++j) {
      ++counts[i.assignment[j]];
      for (std::size_t d = 0; d < 2; ++d)
        sum[i.assignment[j]][d] += data[j][d];
    }
    EXPECT_EQ(counts, i.counts);
    for (std::size_t k = 0; k < i.after.size(); ++k)
      for (std::size_t d = 0; d < 2; ++d)
        EXPECT_EQ(i.after[k][d], counts[k] ? sum[k][d] / counts[k] : i.before[k][d]);
    auto q = lbg_quantize(data, i.before);
    EXPECT_EQ(q.assignment, i.assignment);
    EXPECT_EQ(q.distortion, i.before_distortion);
  }
  for (auto& s : t.stages) {
    ++passes;
    pairs += data.size() * s.size;
    squares += data.size() * s.size * 2;
  }
  EXPECT_EQ(t.analysis.work.assignment_passes, passes);
  EXPECT_EQ(t.analysis.work.distance_evaluations, pairs);
  EXPECT_EQ(t.analysis.work.coordinate_squares, squares);
  EXPECT_EQ(linde_buzo_gray(data, 16, .01, 100, 1e-8).codebook, t.analysis.result.codebook);
}
TEST(LBGTrace, MultiplicativeZeroCollapseAndRoundedUpTarget) {
  auto t = linde_buzo_gray_trace({{-1}, {1}}, 3);
  EXPECT_EQ(t.analysis.rounded_size, 4);
  EXPECT_EQ(t.analysis.result.codebook, (std::vector<std::vector<double>>(4, {0})));
  EXPECT_EQ(t.analysis.result.assignment, (std::vector<std::size_t>{0, 0}));
  EXPECT_EQ(t.analysis.result.distortion, 1);
  auto q = lbg_quantize({{1}, {3}}, {{0}, {4}});
  EXPECT_EQ(q.assignment, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(q.distortion, 1);
}
TEST(LBGTrace, CompleteGuardsQuotasEmptyOwnershipAndRange) {
  double q = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(linde_buzo_gray({{1}, {1, 2}}, 2), std::invalid_argument);
  EXPECT_THROW(linde_buzo_gray({{}}, 2), std::invalid_argument);
  EXPECT_THROW(linde_buzo_gray({{q}}, 0), std::invalid_argument);
  EXPECT_THROW(linde_buzo_gray({{1}}, 1, -.1), std::invalid_argument);
  EXPECT_THROW(linde_buzo_gray({{1}}, 1, .01, 0), std::invalid_argument);
  EXPECT_THROW(linde_buzo_gray({{1}}, 1, .01, 10, -1), std::invalid_argument);
  EXPECT_THROW(linde_buzo_gray({{1}}, std::numeric_limits<std::size_t>::max()), std::length_error);
  EXPECT_THROW(lbg_quantize({{1}}, {}), std::invalid_argument);
  EXPECT_THROW(lbg_quantize({{1}}, {{1, 2}}), std::invalid_argument);
  auto t = linde_buzo_gray_trace({{1}, {3}}, 2);
  EXPECT_THROW(
      linde_buzo_gray_trace({{1}, {3}}, 2, .01, 100, 1e-6, t.captured_entries - 1),
      std::length_error);
  EXPECT_EQ(
      linde_buzo_gray_trace({{1}, {3}}, 2, .01, 100, 1e-6, t.captured_entries)
          .analysis.result.codebook,
      t.analysis.result.codebook);
  std::vector<std::vector<double>> data{{1}, {3}};
  auto                             own = linde_buzo_gray_trace(data, 1, .01, 100, 1e-6, 0);
  data[0][0]                           = 9;
  EXPECT_EQ(own.data[0][0], 1);
  EXPECT_TRUE(linde_buzo_gray({}, 2).codebook.empty());
  EXPECT_TRUE(linde_buzo_gray({{1}}, 0).codebook.empty());
  EXPECT_EQ(lbg_quantize({}, {{1}}).distortion, 0);
  const double max = std::numeric_limits<double>::max();
  EXPECT_EQ(linde_buzo_gray({{max}, {max}}, 1).codebook[0][0], max);
  EXPECT_THROW(lbg_quantize({{max}}, {{-max}}), std::overflow_error);
}
