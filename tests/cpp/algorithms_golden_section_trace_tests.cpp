#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/golden_section_search.hpp>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../oracles/golden_section_exact.hpp"
namespace a = datamunge::algorithms;
namespace {
void same(const a::GoldenState& x, const a::GoldenState& y) {
  EXPECT_DOUBLE_EQ(x.a, y.a);
  EXPECT_DOUBLE_EQ(x.b, y.b);
  EXPECT_DOUBLE_EQ(x.c, y.c);
  EXPECT_DOUBLE_EQ(x.d, y.d);
  EXPECT_DOUBLE_EQ(x.fc, y.fc);
  EXPECT_DOUBLE_EQ(x.fd, y.fd);
}
} // namespace
TEST(GoldenSectionTrace, ExactRationalAnalyticMaxima) {
  for (const auto& v : golden_fixture::cases) {
    for (int shape = 0; shape < 2; ++shape) {
      auto f = [&](double x) { return shape ? -std::abs(x - v.p) : -(x - v.p) * (x - v.p); };
      auto r = a::golden_section_run(f, v.a, v.b, 1e-9);
      EXPECT_NEAR(r.result.argmax, v.peak, 1e-7);
      EXPECT_EQ(r.stop, a::GoldenStop::width);
      EXPECT_LE(r.final_state.b - r.final_state.a, 1e-9);
      EXPECT_EQ(r.evaluations, std::size_t(r.result.iterations) + 3);
    }
  }
}
TEST(GoldenSectionTrace, ActualCallsStatesAndCachedReuse) {
  std::mt19937_64 rng(138);
  for (int trial = 0; trial < 200; ++trial) {
    double                                 p = double(rng() % 1001) / 1000;
    std::vector<std::pair<double, double>> calls;
    auto                                   f = [&](double x) {
      double y = -(x - p) * (x - p);
      calls.emplace_back(x, y);
      return y;
    };
    auto t = a::golden_section_trace(f, 0., 1., 1e-8, 100, 0);
    ASSERT_EQ(calls.size(), t.run.evaluations);
    ASSERT_EQ(t.events.size(), std::size_t(t.run.result.iterations) + 2);
    const auto& init = t.events.front();
    EXPECT_EQ(init.kind, a::GoldenEventKind::initial);
    EXPECT_EQ(init.evaluations, 2u);
    same(init.before, init.after);
    EXPECT_EQ(calls[0], std::make_pair(init.after.c, init.after.fc));
    EXPECT_EQ(calls[1], std::make_pair(init.after.d, init.after.fd));
    double width = 1;
    for (std::size_t i = 1; i + 1 < t.events.size(); ++i) {
      const auto& e = t.events[i];
      same(e.before, t.events[i - 1].after);
      EXPECT_EQ(e.kind, a::GoldenEventKind::step);
      EXPECT_EQ(e.iteration, int(i));
      EXPECT_EQ(e.evaluations, i + 2);
      EXPECT_EQ(e.kept_right, e.before.fc < e.before.fd);
      EXPECT_EQ(calls[i + 1], std::make_pair(e.evaluated_x, e.evaluated_value));
      if (e.kept_right) {
        EXPECT_EQ(e.after.a, e.before.c);
        EXPECT_EQ(e.after.b, e.before.b);
        EXPECT_EQ(e.after.c, e.before.d);
        EXPECT_EQ(e.after.fc, e.before.fd);
        EXPECT_EQ(e.after.d, e.evaluated_x);
        EXPECT_EQ(e.after.fd, e.evaluated_value);
      } else {
        EXPECT_EQ(e.after.a, e.before.a);
        EXPECT_EQ(e.after.b, e.before.d);
        EXPECT_EQ(e.after.d, e.before.c);
        EXPECT_EQ(e.after.fd, e.before.fc);
        EXPECT_EQ(e.after.c, e.evaluated_x);
        EXPECT_EQ(e.after.fc, e.evaluated_value);
      }
      width *= (std::sqrt(5.) - 1) / 2;
      EXPECT_NEAR(e.after.b - e.after.a, width, 2e-15);
      EXPECT_LE(e.after.a, p + 2e-15);
      EXPECT_GE(e.after.b, p - 2e-15);
    }
    const auto& stop = t.events.back();
    EXPECT_EQ(stop.kind, a::GoldenEventKind::stop);
    same(stop.before, stop.after);
    same(stop.after, t.run.final_state);
    EXPECT_EQ(calls.back(), std::make_pair(stop.evaluated_x, stop.evaluated_value));
    EXPECT_EQ(stop.evaluated_x, t.run.result.argmax);
    EXPECT_EQ(stop.evaluated_value, t.run.result.value);
  }
}
TEST(GoldenSectionTrace, StopConditionsZeroUpdatesAndTies) {
  auto f         = [](double x) { return x; };
  auto short_run = a::golden_section_run(f, 0, 1, 1, 0);
  EXPECT_EQ(short_run.evaluations, 3u);
  EXPECT_EQ(short_run.result.argmax, .5);
  EXPECT_EQ(short_run.stop, a::GoldenStop::width);
  auto capped = a::golden_section_run(f, 0, 1, 0, 0);
  EXPECT_EQ(capped.evaluations, 3u);
  EXPECT_EQ(capped.stop, a::GoldenStop::iteration_limit);
  auto equal = a::golden_section_trace(f, 2, 2, 0, 10, 2);
  EXPECT_EQ(equal.run.result.argmax, 2);
  EXPECT_EQ(equal.run.evaluations, 3u);
  EXPECT_EQ(equal.events.size(), 2u);
  auto tied = a::golden_section_trace([](double) { return 7.; }, 0, 1, 0, 1, 3);
  EXPECT_FALSE(tied.events[1].kept_right);
  EXPECT_EQ(tied.events[1].after.b, tied.events[0].after.d);
  EXPECT_EQ(tied.run.result.value, 7);
  EXPECT_STREQ(a::golden_stop_name(a::GoldenStop::width), "width");
  EXPECT_STREQ(a::golden_stop_name(a::GoldenStop::iteration_limit), "iteration_limit");
  EXPECT_STREQ(a::golden_stop_name(static_cast<a::GoldenStop>(99)), "unknown");
}
TEST(GoldenSectionTrace, PreservesFiniteFailuresAndStagnation) {
  auto flat = a::golden_section_run(
      [](double x) { return std::max(0., 1 - 20 * std::abs(x - .9)); }, 0, 1, 1e-10);
  EXPECT_EQ(flat.result.value, 0);
  EXPECT_LT(flat.result.argmax, .001);
  EXPECT_EQ(flat.stop, a::GoldenStop::width);
  auto offset =
      a::golden_section_run([](double x) { return 1 - (x - .9) * (x - .9); }, 0, 1, 1e-18);
  EXPECT_GT(std::abs(offset.result.argmax - .9), 1e-9);
  EXPECT_FALSE(offset.final_state.a <= .9 && .9 <= offset.final_state.b);
  auto stalled =
      a::golden_section_run([](double x) { return x; }, 1, std::nextafter(1., 2.), 1e-30);
  EXPECT_EQ(stalled.result.iterations, 500);
  EXPECT_EQ(stalled.evaluations, 503u);
  EXPECT_EQ(stalled.stop, a::GoldenStop::iteration_limit);
  EXPECT_GT(stalled.final_state.b - stalled.final_state.a, 1e-30);
  auto original = a::golden_section_run(
      [](double x) { return std::sin(x) - .1 * (x - 1.5) * (x - 1.5); }, 0, 3, 1e-10);
  double lo = 0, hi = 3;
  for (int i = 0; i < 100; ++i) {
    double m = (lo + hi) / 2;
    if (std::cos(m) - .2 * (m - 1.5) > 0)
      lo = m;
    else
      hi = m;
  }
  double truth = (lo + hi) / 2;
  EXPECT_GT(std::abs(original.result.argmax - truth), 1e-9);
  EXPECT_EQ(original.result.iterations, 51);
  EXPECT_EQ(original.evaluations, 54u);
}
TEST(GoldenSectionTrace, WholeOnlineRecordsAndOwnedCopies) {
  int  calls = 0;
  auto f     = [&](double x) {
    ++calls;
    return x;
  };
  EXPECT_THROW(a::golden_section_trace(f, 0, 1, 1, 0, 1), std::length_error);
  EXPECT_EQ(calls, 2);
  calls  = 0;
  auto t = a::golden_section_trace(f, 0, 1, 1, 0, 2);
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(t.events.size(), 2u);
  calls = 0;
  EXPECT_THROW(a::golden_section_trace(f, 0, 1, 0, 1, 2), std::length_error);
  EXPECT_EQ(calls, 3);
  calls    = 0;
  auto one = a::golden_section_trace(f, 0, 1, 0, 1, 3);
  EXPECT_EQ(calls, 4);
  EXPECT_EQ(one.events.size(), 3u);
  auto copy              = one;
  copy.events[0].after.a = 99;
  EXPECT_EQ(one.events[0].after.a, 0);
  auto unlimited = a::golden_section_trace(f, 0, 1, 0, 3, 0);
  EXPECT_EQ(unlimited.events.size(), 5u);
}
TEST(GoldenSectionTrace, ValidationAndCallbackFailures) {
  int  calls = 0;
  auto f     = [&](double x) {
    ++calls;
    return x;
  };
  double nan = std::numeric_limits<double>::quiet_NaN(),
         inf = std::numeric_limits<double>::infinity(), big = std::numeric_limits<double>::max();
  for (auto endpoints :
       std::vector<std::pair<double, double>>{{nan, 1}, {0, nan}, {-inf, 1}, {0, inf}, {2, 1}})
    EXPECT_THROW(
        a::golden_section_trace(f, endpoints.first, endpoints.second, 0, 0, 1),
        std::invalid_argument);
  for (double tol : {-1., nan, inf})
    EXPECT_THROW(a::golden_section_trace(f, 0, 1, tol, 0, 1), std::invalid_argument);
  EXPECT_THROW(a::golden_section_search(f, 0, 1, 1, -1), std::invalid_argument);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(a::golden_section_search(f, -big, big), std::overflow_error);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(a::golden_section_search(f, big, big), std::overflow_error);
  EXPECT_EQ(calls, 2);
  for (int failed_call : {0, 1, 2, 3}) {
    calls    = 0;
    auto bad = [&](double x) { return calls++ == failed_call ? nan : x; };
    EXPECT_THROW(a::golden_section_search(bad, 0, 1, 0, 1), std::domain_error);
    EXPECT_EQ(calls, failed_call + 1);
  }
  EXPECT_THROW(a::golden_section_run([&](double) { return inf; }, 0, 1), std::domain_error);
  calls         = 0;
  auto throwing = [&](double) {
    ++calls;
    throw std::logic_error("callback");
    return 0.;
  };
  EXPECT_THROW(a::golden_section_search(throwing, 0, 1), std::logic_error);
  EXPECT_EQ(calls, 1);
}
TEST(GoldenSectionTrace, CallableCopyMoveAndBorrowedState) {
  struct Counter {
    int  local = 0;
    int* copies;
    Counter(int* c) : copies(c) {}
    Counter(const Counter& c) : local(c.local), copies(c.copies) { ++*copies; }
    double operator()(double x) {
      ++local;
      return x;
    }
  };
  int     copies = 0;
  Counter c(&copies);
  auto    r = a::golden_section_run(c, 0, 1, 1, 0);
  EXPECT_EQ(copies, 1);
  EXPECT_EQ(c.local, 0);
  EXPECT_EQ(r.evaluations, 3u);
  auto moved = [p = std::make_unique<int>(0)](double x) mutable {
    ++*p;
    return x;
  };
  auto m = a::golden_section_search(std::move(moved), 0, 1, 1, 0);
  EXPECT_EQ(m.argmax, .5);
  int  borrowed = 0;
  auto t        = a::golden_section_trace(
      [&](double x) {
        ++borrowed;
        return x;
      },
      0,
      1,
      1,
      0,
      2);
  EXPECT_EQ(borrowed, 3);
  auto t2 = t;
  EXPECT_EQ(borrowed, 3);
  EXPECT_EQ(t2.run.evaluations, 3u);
}
