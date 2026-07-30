#include <gtest/gtest.h>

#include <datamunge/algorithms/banker.hpp>
#include <datamunge/algorithms/cpu_scheduling.hpp>
#include <datamunge/algorithms/realtime_scheduling.hpp>

#include <cmath>
#include <vector>

using namespace datamunge::algorithms;

// ---- Banker's algorithm ----------------------------------------------------

// The canonical Silberschatz 5-process / 3-resource example.
namespace {
std::vector<std::vector<int>> banker_alloc() {
    return {{0, 1, 0}, {2, 0, 0}, {3, 0, 2}, {2, 1, 1}, {0, 0, 2}};
}
std::vector<std::vector<int>> banker_max() {
    return {{7, 5, 3}, {3, 2, 2}, {9, 0, 2}, {2, 2, 2}, {4, 3, 3}};
}
}  // namespace

TEST(Banker, SafeStateHasSafeSequence) {
    auto r = banker_safety(banker_alloc(), banker_max(), {3, 3, 2});
    ASSERT_TRUE(r.safe);
    // Greedy-by-index safe sequence for this classic instance.
    std::vector<std::size_t> expected = {1, 3, 4, 0, 2};
    EXPECT_EQ(r.sequence, expected);
}

TEST(Banker, GrantableRequestKeepsStateSafe) {
    // P1 requests (1,0,2): within need and available, resulting state safe.
    EXPECT_TRUE(banker_request_grantable(1, {1, 0, 2}, banker_alloc(), banker_max(), {3, 3, 2}));
}

TEST(Banker, RequestLeadingToUnsafeStateDenied) {
    // P4 requests (3,3,0): within need/available but leaves no safe sequence.
    EXPECT_FALSE(banker_request_grantable(4, {3, 3, 0}, banker_alloc(), banker_max(), {3, 3, 2}));
}

TEST(Banker, RequestExceedingNeedDenied) {
    // P0 need is (7,4,3); requesting (8,0,0) exceeds it.
    EXPECT_FALSE(banker_request_grantable(0, {8, 0, 0}, banker_alloc(), banker_max(), {3, 3, 2}));
}

// ---- Round robin -----------------------------------------------------------

TEST(RoundRobin, TwoProcessesQuantumTwo) {
    std::vector<SchedProcess> procs = {{0, 0, 4}, {1, 0, 2}};
    auto                      r     = round_robin(procs, 2);
    // P0: 0-2, P1: 2-4, P0: 4-6.
    EXPECT_EQ(r.metrics[0].completion, 6);
    EXPECT_EQ(r.metrics[1].completion, 4);
    EXPECT_EQ(r.metrics[0].waiting, 2);
    EXPECT_EQ(r.metrics[1].waiting, 2);
    EXPECT_DOUBLE_EQ(r.avg_waiting, 2.0);
    ASSERT_EQ(r.gantt.size(), 3u);
    EXPECT_EQ(r.gantt[0].id, 0u);
    EXPECT_EQ(r.gantt[1].id, 1u);
    EXPECT_EQ(r.gantt[2].id, 0u);
}

// ---- Shortest job next -----------------------------------------------------

TEST(ShortestJobNext, PicksShortestBurstFirst) {
    std::vector<SchedProcess> procs = {{0, 0, 3}, {1, 0, 1}, {2, 0, 2}};
    auto                      r     = shortest_job_next(procs);
    // Order P1(1) -> P2(2) -> P0(3): completions 1, 3, 6.
    EXPECT_EQ(r.metrics[0].completion, 6);
    EXPECT_EQ(r.metrics[1].completion, 1);
    EXPECT_EQ(r.metrics[2].completion, 3);
    EXPECT_NEAR(r.avg_waiting, 4.0 / 3.0, 1e-9);
}

// ---- Shortest remaining time (preemptive) ----------------------------------

TEST(ShortestRemainingTime, ClassicFourProcessExample) {
    std::vector<SchedProcess> procs = {{0, 0, 8}, {1, 1, 4}, {2, 2, 9}, {3, 3, 5}};
    auto                      r     = shortest_remaining_time(procs);
    EXPECT_EQ(r.metrics[0].completion, 17);
    EXPECT_EQ(r.metrics[1].completion, 5);
    EXPECT_EQ(r.metrics[2].completion, 26);
    EXPECT_EQ(r.metrics[3].completion, 10);
    EXPECT_DOUBLE_EQ(r.avg_waiting, 6.5);
    EXPECT_DOUBLE_EQ(r.avg_turnaround, 13.0);
}

// ---- Real-time scheduling --------------------------------------------------

TEST(RealTime, Utilization) {
    std::vector<RTTask> tasks = {{0, 4, 1}, {1, 5, 2}};
    EXPECT_NEAR(rt_utilization(tasks), 0.25 + 0.4, 1e-9);
}

TEST(RealTime, RateMonotonicBoundTwoTasks) {
    EXPECT_NEAR(rate_monotonic_bound(2), 2.0 * (std::pow(2.0, 0.5) - 1.0), 1e-9);
}

TEST(RealTime, SchedulabilityTests) {
    std::vector<RTTask> ok = {{0, 4, 1}, {1, 5, 2}};  // U=0.65 < 0.828
    EXPECT_TRUE(rate_monotonic_schedulable(ok));
    EXPECT_TRUE(edf_schedulable(ok));

    // U=0.833: fails the RMS sufficient bound (0.828) but EDF (U<=1) accepts.
    std::vector<RTTask> edf_only = {{0, 4, 2}, {1, 6, 2}};
    EXPECT_FALSE(rate_monotonic_schedulable(edf_only));
    EXPECT_TRUE(edf_schedulable(edf_only));
}

TEST(RealTime, PriorityOrderByPeriod) {
    std::vector<RTTask>      tasks = {{0, 5, 2}, {1, 4, 1}, {2, 7, 2}};
    std::vector<std::size_t> pri   = rate_monotonic_priorities(tasks);
    std::vector<std::size_t> expected = {1, 0, 2};  // periods 4 < 5 < 7
    EXPECT_EQ(pri, expected);
}

TEST(RealTime, EdfFeasibleSetMeetsDeadlines) {
    std::vector<RTTask> tasks = {{0, 4, 1}, {1, 5, 2}};
    auto                r     = edf_simulate(tasks, 40);
    EXPECT_FALSE(r.deadline_missed);
}

TEST(RealTime, OverloadedSetMissesDeadline) {
    std::vector<RTTask> tasks = {{0, 2, 2}, {1, 3, 2}};  // U = 1.667 > 1
    auto                r     = edf_simulate(tasks, 30);
    EXPECT_TRUE(r.deadline_missed);
}
