#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <string>
#include <thread>

namespace nn = datamunge::nn;

TEST(NnProfiler, InactiveScopesRecordNothing) {
  EXPECT_FALSE(nn::profiling());
  { nn::ProfileScope s("idle"); }
  nn::Profiler p;
  EXPECT_TRUE(p.events().empty());
  EXPECT_FALSE(p.active());
}

TEST(NnProfiler, NestedScopesGetDepthAndSelfTime) {
  nn::Profiler p;
  p.start();
  EXPECT_TRUE(p.active());
  EXPECT_TRUE(nn::profiling());
  {
    nn::ProfileScope outer("outer");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    {
      nn::ProfileScope inner("inner");
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
  }
  p.stop();
  EXPECT_FALSE(nn::profiling());
  const auto ev = p.events();
  ASSERT_EQ(ev.size(), 2u);
  EXPECT_EQ(ev[0].name, "inner");  // ends first
  EXPECT_EQ(ev[0].depth, 1);
  EXPECT_EQ(ev[1].depth, 0);
  EXPECT_GE(ev[0].start_us, ev[1].start_us);
  const auto rows = p.summary();
  ASSERT_EQ(rows.size(), 2u);
  EXPECT_EQ(rows[0].name, "outer");  // most total time
  EXPECT_NEAR(rows[0].self_us, rows[0].total_us - rows[1].total_us, 1e-9);
  EXPECT_GE(rows[1].total_us, 3000.0);
  EXPECT_GE(p.wall_us(), rows[0].total_us);
  { nn::ProfileScope late("after stop"); }
  EXPECT_EQ(p.events().size(), 2u);
}

TEST(NnProfiler, BackwardNodesAreRecorded) {
  nn::Profiler p;
  auto x = nn::tensor({{1.0, 2.0}, {3.0, 4.0}}).requires_grad_(true);
  p.start();
  auto y = nn::sum(nn::mul(nn::matmul(x, x), x));
  auto g = nn::grad({y}, {x});
  p.stop();
  int backward = 0;
  bool matmul  = false;
  for (const auto& e : p.events())
    if (e.category == "backward") ++backward, matmul |= e.name.find("MatMul") != std::string::npos;
  EXPECT_GE(backward, 3);
  EXPECT_TRUE(matmul);
}

TEST(NnProfiler, ThreadsAndExports) {
  nn::Profiler p;
  p.start();
  std::thread t([] { nn::ProfileScope s("worker \"a\""); });
  t.join();
  { nn::ProfileScope s("main", "custom"); }
  p.stop();
  const auto ev = p.events();
  ASSERT_EQ(ev.size(), 2u);
  EXPECT_NE(ev[0].thread, ev[1].thread);
  const std::string json = p.to_chrome_trace();
  EXPECT_NE(json.find("\"ph\":\"X\""), std::string::npos);
  EXPECT_NE(json.find("worker \\\"a\\\""), std::string::npos);
  const std::string md = p.to_markdown(1);
  EXPECT_NE(md.find("(other)"), std::string::npos);
  auto plot = nn::plot_timeline(p, "t");
  EXPECT_FALSE(plot.series().empty());
  EXPECT_THROW((void)nn::plot_timeline(p, "t", 1e12, 2e12), std::invalid_argument);
}

TEST(NnProfiler, StartingAnotherProfilerStopsTheFirst) {
  nn::Profiler a, b;
  a.start();
  b.start();
  EXPECT_FALSE(a.active());
  EXPECT_TRUE(b.active());
  { nn::ProfileScope s("x"); }
  b.stop();
  EXPECT_TRUE(a.events().empty());
  EXPECT_EQ(b.events().size(), 1u);
}
