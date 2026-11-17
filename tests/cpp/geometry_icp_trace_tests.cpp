#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/geometry/icp.hpp>
#include <limits>
#include <random>
using namespace datamunge::geometry;
namespace {
double directed_error(const std::vector<Point2D>& source, const std::vector<Point2D>& target) {
  long double sum = 0;
  for (auto p : source) {
    long double best = std::numeric_limits<long double>::infinity();
    for (auto q : target) {
      long double x = (long double)p.x - q.x, y = (long double)p.y - q.y;
      best = std::min(best, x * x + y * y);
    }
    sum += best;
  }
  return std::sqrt(sum / source.size());
}
Point2D apply(Point2D p, double a, Point2D t) {
  return {std::cos(a) * p.x - std::sin(a) * p.y + t.x, std::sin(a) * p.x + std::cos(a) * p.y + t.y};
}
} // namespace
TEST(IcpTrace, FinalErrorDescribesActualUpdateAndZeroBudget) {
  const auto t = icp_trace({{0, 0}}, {{3, 4}}, 1);
  ASSERT_EQ(t.steps.size(), 1);
  EXPECT_EQ(t.steps[0].pre_rmse, 5);
  EXPECT_EQ(t.run.result.rmse, 0);
  EXPECT_EQ(icp({{0, 0}}, {{3, 4}}, 1).rmse, 0);
  const auto z = icp_run({{0, 0}}, {{3, 4}}, 0);
  EXPECT_EQ(z.result.rmse, 5);
  EXPECT_EQ(z.result.iterations, 0);
  EXPECT_EQ(z.work.distance_evaluations, 0);
  EXPECT_EQ(z.work.diagnostic_distance_evaluations, 1);
  EXPECT_EQ(z.stop, IcpStop::IterationLimit);
  EXPECT_EQ(icp_run({}, {{1, 2}}).stop, IcpStop::EmptyCloud);
  EXPECT_EQ(icp_run({{1, 2}}, {}).aligned[0], (Point2D{1, 2}));
}
TEST(IcpTrace, ActualMatchesUpdatesCompositionAndErrors) {
  std::mt19937_64                        rng(1992);
  std::uniform_real_distribution<double> value(-4, 4);
  for (unsigned trial = 0; trial < 500; ++trial) {
    std::vector<Point2D> s, t;
    for (unsigned i = 0; i < 1 + rng() % 12; ++i)
      s.push_back({value(rng), value(rng)});
    for (unsigned j = 0; j < 1 + rng() % 12; ++j)
      t.push_back({value(rng), value(rng)});
    const auto h = icp_trace(s, t, 6, 0);
    ASSERT_EQ(h.steps.size(), 6);
    EXPECT_EQ(h.run.work.distance_evaluations, 6 * s.size() * t.size());
    EXPECT_EQ(h.run.work.point_updates, 6 * s.size());
    EXPECT_EQ(h.run.work.diagnostic_distance_evaluations, 6 * s.size() * (t.size() + 1));
    std::vector<Point2D> current = s;
    for (const auto& step : h.steps) {
      EXPECT_EQ(step.before, current);
      ASSERT_EQ(step.target_indices.size(), s.size());
      long double fixed = 0;
      for (std::size_t i = 0; i < s.size(); ++i) {
        double      best = std::numeric_limits<double>::infinity();
        std::size_t bj   = 0;
        for (std::size_t j = 0; j < t.size(); ++j) {
          double dx = current[i].x - t[j].x, dy = current[i].y - t[j].y, d = dx * dx + dy * dy;
          if (d < best) {
            best = d;
            bj   = j;
          }
        }
        EXPECT_EQ(step.target_indices[i], bj);
        auto q = apply(current[i], step.incremental_angle, step.incremental_translation);
        EXPECT_EQ(q, step.after[i]);
        long double dx = (long double)q.x - t[bj].x, dy = (long double)q.y - t[bj].y;
        fixed += dx * dx + dy * dy;
        auto accumulated = apply(s[i], step.pose.angle, step.pose.translation);
        EXPECT_NEAR(accumulated.x, q.x, 2e-13);
        EXPECT_NEAR(accumulated.y, q.y, 2e-13);
      }
      EXPECT_NEAR(step.pre_rmse, directed_error(current, t), 1e-13);
      EXPECT_NEAR(step.post_fixed_rmse, std::sqrt(fixed / s.size()), 1e-13);
      EXPECT_NEAR(step.post_nearest_rmse, directed_error(step.after, t), 1e-13);
      EXPECT_LE(step.post_fixed_rmse, step.pre_rmse + 1e-12);
      EXPECT_LE(step.post_nearest_rmse, step.post_fixed_rmse + 1e-12);
      current = step.after;
    }
    EXPECT_EQ(current, h.run.aligned);
    EXPECT_EQ(h.run.result.rmse, h.steps.back().post_nearest_rmse);
    const auto plain = icp_run(s, t, 6, 0);
    EXPECT_EQ(plain.aligned, current);
    EXPECT_EQ(plain.result.rmse, h.run.result.rmse);
    EXPECT_EQ(plain.work.diagnostic_distance_evaluations, s.size() * t.size());
  }
}
TEST(IcpTrace, WrongLocalSolutionManyToOneAndStrictPlateau) {
  auto h = icp_trace({{0, -1}, {0, 1}}, {{-1, 0}, {1, 0}}, 10, 1e-14);
  EXPECT_EQ(h.steps.front().target_indices, (std::vector<std::size_t>{0, 0}));
  EXPECT_EQ(h.run.result.angle, 0);
  EXPECT_EQ(h.run.result.rmse, 1);
  EXPECT_EQ(h.run.stop, IcpStop::PreUpdatePlateau);
  EXPECT_EQ(h.run.result.iterations, 3);
  auto full = icp_trace({{0, 0}}, {{0, 0}}, 4, 0);
  EXPECT_EQ(full.steps.size(), 4);
  EXPECT_EQ(full.run.stop, IcpStop::IterationLimit);
  auto strict = icp_trace({{0, 0}}, {{1, 0}}, 10, 1);
  EXPECT_EQ(strict.steps.size(), 3);
}
TEST(IcpTrace, PoliciesOwnershipDomainsAndOverflow) {
  std::vector<Point2D> s{{0, 0}}, t{{1, 0}};
  auto                 h = icp_trace(s, t, 2, 0, {2, 12});
  EXPECT_EQ(h.steps.size(), 2);
  EXPECT_THROW(icp_trace(s, t, 2, 0, {2, 11}), std::length_error);
  EXPECT_THROW(icp_trace(s, t, 2, 0, {1, 0}), std::length_error);
  EXPECT_NO_THROW(icp_run(s, t, 2, 0, {2, 1}));
  EXPECT_NO_THROW(icp_trace(s, t, 2, 0, {0, 0}));
  s[0].x = 5;
  t[0].x = 9;
  EXPECT_EQ(h.source[0].x, 0);
  EXPECT_EQ(h.target[0].x, 1);
  h.steps[0].after[0].x = 77;
  EXPECT_EQ(h.steps[1].before[0].x, 1);
  auto nan = std::numeric_limits<double>::quiet_NaN(),
       inf = std::numeric_limits<double>::infinity(), max = std::numeric_limits<double>::max();
  EXPECT_THROW(icp({}, {}, -1), std::invalid_argument);
  EXPECT_THROW(icp({}, {}, 1, -1), std::invalid_argument);
  EXPECT_THROW(icp({}, {}, 1, nan), std::invalid_argument);
  EXPECT_THROW(icp({{nan, 0}}, {}), std::invalid_argument);
  EXPECT_THROW(icp({}, {{inf, 0}}), std::invalid_argument);
  EXPECT_THROW(icp({{max, 0}}, {{-max, 0}}, 1), std::overflow_error);
  EXPECT_THROW(icp({{1e200, 0}}, {{0, 0}}, 1), std::overflow_error);
}

#include <datamunge/geometry/icp_analysis.hpp>
TEST(IcpTrace, NativeFixturesPoseAndActualViews) {
  const auto s = icp_example_source();
  ASSERT_EQ(s.size(), 25);
  EXPECT_EQ(s[0], (Point2D{0, .2}));
  const auto t = icp_transform_points(s, .12, {.7, -.5});
  const auto r = icp_run(s, t, 100, 1e-15);
  EXPECT_NEAR(r.result.angle, .12, 1e-13);
  EXPECT_NEAR(r.result.translation.x, .7, 1e-13);
  EXPECT_NEAR(r.result.translation.y, -.5, 1e-13);
  EXPECT_FALSE(icp_alignment_plot(s, t, r).to_svg().empty());
  const auto h = icp_trace({{0, 0}}, {{3, 4}}, 1);
  EXPECT_FALSE(icp_step_diagram(h).to_svg().empty());
  EXPECT_THROW(icp_step_diagram(h, 1), std::out_of_range);
  auto bad                       = h;
  bad.steps[0].target_indices[0] = 1;
  EXPECT_THROW(icp_step_diagram(bad), std::invalid_argument);
  bad                   = h;
  bad.steps[0].pre_rmse = -1;
  EXPECT_THROW(icp_step_diagram(bad), std::invalid_argument);
  EXPECT_THROW(icp_step_diagram(icp_trace(s, t, 1)), std::length_error);
  auto bad_run = r;
  bad_run.aligned.pop_back();
  EXPECT_THROW(icp_alignment_plot(s, t, bad_run), std::invalid_argument);
  auto inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW(icp_transform_points({}, inf), std::invalid_argument);
  EXPECT_THROW(icp_transform_points({{inf, 0}}, 0), std::invalid_argument);
  const auto max = std::numeric_limits<double>::max();
  EXPECT_THROW(icp_transform_points({{max, 0}}, 0, {max, 0}), std::overflow_error);
}
