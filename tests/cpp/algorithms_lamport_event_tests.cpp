#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <datamunge/algorithms/lamport_clock.hpp>
#include <limits>
#include <random>
#include <vector>
using namespace datamunge::algorithms;
TEST(LamportEvent, OriginalSevenEventsAndActualStateBoundaries) {
  LamportClock a, b;
  const auto   l = a.local_step();
  EXPECT_EQ(l.before, 0);
  EXPECT_EQ(l.after, 1);
  const auto s = a.send_step();
  EXPECT_EQ(s.before, 1);
  EXPECT_EQ(s.after, 2);
  EXPECT_EQ(s.kind, LamportEventKind::Send);
  b.local_event();
  const auto r = b.receive_step(s.after);
  EXPECT_EQ(r.before, 1);
  EXPECT_EQ(r.message_time, 2);
  EXPECT_EQ(r.after, 3);
  const auto s2 = b.send_step();
  EXPECT_EQ(s2.after, 4);
  EXPECT_EQ(a.receive(s2.after), 5);
  EXPECT_EQ(a.local_event(), 6);
  EXPECT_EQ(a.time(), 6);
  EXPECT_EQ(b.time(), 4);
}
TEST(LamportEvent, ArbitraryNonnegativeStampsAgainstIndependentIntegerMaxRecurrence) {
  LamportClock c;
  std::int64_t time = 0;
  std::mt19937 rng(192);
  for (int i = 0; i < 10000; ++i) {
    const int          mode  = rng() % 3;
    const std::int64_t stamp = rng() % 100000;
    LamportEvent       e;
    if (mode == 0)
      e = c.local_step();
    else if (mode == 1)
      e = c.send_step();
    else
      e = c.receive_step(stamp);
    EXPECT_EQ(e.before, time);
    time = (mode == 2 ? std::max(time, stamp) : time) + 1;
    EXPECT_EQ(e.after, time);
    EXPECT_EQ(c.time(), time);
    EXPECT_EQ(e.message_time, mode == 2 ? stamp : 0);
    EXPECT_EQ(int(e.kind), mode);
  }
}
TEST(LamportEvent, ActualSendReceiveAndTransitiveHappenedBeforeGraph) {
  for (int fixture = 0; fixture < 10; ++fixture) {
    std::mt19937                rng(192 + fixture);
    std::array<LamportClock, 3> clocks;
    std::array<int, 3>          last{-1, -1, -1};
    struct Message {
      int          sender;
      std::int64_t stamp;
    };
    std::vector<Message>             pending;
    std::vector<std::int64_t>        stamps;
    std::vector<std::pair<int, int>> edges;
    for (int id = 0; id < 100; ++id) {
      const int    p = rng() % 3;
      LamportEvent e;
      if (!pending.empty() && rng() % 3 == 0) {
        const auto index = rng() % pending.size();
        const auto m     = pending[index];
        e                = clocks[p].receive_step(m.stamp);
        edges.emplace_back(m.sender, id);
        pending.erase(pending.begin() + index);
      } else if (rng() % 2) {
        e = clocks[p].send_step();
        pending.push_back({id, e.after});
      } else
        e = clocks[p].local_step();
      if (last[p] >= 0)
        edges.emplace_back(last[p], id);
      last[p] = id;
      stamps.push_back(e.after);
    }
    bool reach[100][100]{};
    for (auto [u, v] : edges)
      reach[u][v] = true;
    for (int k = 0; k < 100; ++k)
      for (int i = 0; i < 100; ++i)
        for (int j = 0; j < 100; ++j)
          reach[i][j] |= reach[i][k] && reach[k][j];
    for (int i = 0; i < 100; ++i)
      for (int j = 0; j < 100; ++j)
        if (reach[i][j])
          EXPECT_LT(stamps[i], stamps[j]);
  }
}
TEST(LamportEvent, FullSignedBoundaryRejectsAtomicallyAndCopiesOwnState) {
  LamportClock c;
  EXPECT_THROW((void)c.receive(-1), std::invalid_argument);
  EXPECT_EQ(c.time(), 0);
  EXPECT_THROW((void)c.receive(std::numeric_limits<std::int64_t>::max()), std::overflow_error);
  EXPECT_EQ(c.time(), 0);
  EXPECT_EQ(
      c.receive(std::numeric_limits<std::int64_t>::max() - 1),
      std::numeric_limits<std::int64_t>::max());
  const auto copy = c;
  EXPECT_THROW((void)c.local_step(), std::overflow_error);
  EXPECT_THROW((void)c.send_step(), std::overflow_error);
  EXPECT_THROW((void)c.receive_step(0), std::overflow_error);
  EXPECT_EQ(c.time(), copy.time());
  LamportClock a;
  auto         b = a;
  b.local_event();
  EXPECT_EQ(a.time(), 0);
  EXPECT_EQ(b.time(), 1);
}
TEST(LamportEvent, CompleteAlternatingMessageChainsHaveTwoTicksPerMessage) {
  for (int n : {1, 2, 4, 8, 16, 64, 1024}) {
    std::array<LamportClock, 2> c;
    for (int i = 0; i < n; ++i) {
      const auto s = c[i % 2].send_step();
      const auto r = c[1 - i % 2].receive_step(s.after);
      EXPECT_EQ(s.after, 2 * i + 1);
      EXPECT_EQ(r.after, 2 * i + 2);
    }
    EXPECT_EQ(std::max(c[0].time(), c[1].time()), 2 * n);
  }
}
