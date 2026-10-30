#include <gtest/gtest.h>

#include <datamunge/algorithms/bakery.hpp>
#include <datamunge/algorithms/bully_election.hpp>
#include <datamunge/algorithms/clock_synchronization.hpp>
#include <datamunge/algorithms/lamport_clock.hpp>
#include <datamunge/algorithms/vector_clock.hpp>

#include <cstddef>
#include <vector>

using namespace datamunge::algorithms;

TEST(LamportClock, RespectsHappenedBefore) {
    LamportClock p0, p1;
    p0.local_event();               // p0 = 1
    const auto send_ts = p0.send(); // p0 = 2, message stamped 2
    p1.local_event();               // p1 = 1
    const auto recv_ts = p1.receive(send_ts);  // p1 = max(1,2)+1 = 3
    EXPECT_EQ(send_ts, 2);
    EXPECT_EQ(recv_ts, 3);
    EXPECT_LT(send_ts, recv_ts);    // send happened-before receive => smaller timestamp
    // A later local event keeps advancing.
    EXPECT_EQ(p1.local_event(), 4);
}

TEST(VectorClock, DetectsCausalityAndConcurrency) {
    // Three processes. p0 sends to p1; p2 acts independently (concurrent with both).
    VectorClock c0(3), c1(3), c2(3);
    vc_event(c0, 0);                       // c0 = [1,0,0]  event A
    VectorClock msg = vc_send(c0, 0);      // c0 = [2,0,0], msg = [2,0,0]  (send)
    VectorClock a_send = msg;
    vc_receive(c1, 1, msg);                // c1 = [2,1,0]  event B (receive)
    VectorClock b_recv = c1;
    vc_event(c2, 2);                       // c2 = [0,0,1]  event C (independent)
    VectorClock c_indep = c2;

    EXPECT_TRUE(vc_happens_before(a_send, b_recv));   // send -> receive
    EXPECT_FALSE(vc_happens_before(b_recv, a_send));
    EXPECT_TRUE(vc_concurrent(a_send, c_indep));      // C is concurrent with the send
    EXPECT_TRUE(vc_concurrent(b_recv, c_indep));
}

TEST(BullyElection, HighestAliveWins) {
    // 6 processes; the top two (4,5) are down.
    std::vector<char> alive = {1, 1, 1, 1, 0, 0};
    auto              r     = bully_elect(6, alive, /*initiator=*/1);
    EXPECT_EQ(r.leader, 3);  // highest alive id
    EXPECT_GT(r.election_messages, 0u);

    // If the very top is alive, it wins.
    std::vector<char> alive2 = {1, 1, 1, 1, 1, 1};
    EXPECT_EQ(bully_elect(6, alive2, 0).leader, 5);
}

TEST(Bakery, ServesInTicketOrderWithMutualExclusion) {
    std::vector<BakeryTicket> tickets = {{3, 0}, {1, 1}, {2, 2}, {1, 3}, {0, 4}};  // thread 4 not requesting
    auto                      order = bakery_service_order(tickets);
    // Order by (number, id): (1,1), (1,3), (2,2), (3,0). Thread 4 excluded.
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], 1u);
    EXPECT_EQ(order[1], 3u);
    EXPECT_EQ(order[2], 2u);
    EXPECT_EQ(order[3], 0u);
    EXPECT_EQ(bakery_next(tickets), 1u);
    EXPECT_TRUE(bakery_mutual_exclusion_holds(tickets));
    // Duplicate (number,id) would violate exclusion.
    std::vector<BakeryTicket> bad = {{2, 5}, {2, 5}};
    EXPECT_FALSE(bakery_mutual_exclusion_holds(bad));
}

TEST(ClockSync, CristianAndBerkeley) {
    // Cristian: server said 1000.0, round trip 40 ms -> estimate 1020.0, bound 20 ms (or less).
    EXPECT_DOUBLE_EQ(cristian_estimate(1000.0, 40.0), 1020.0);
    EXPECT_DOUBLE_EQ(cristian_error_bound(40.0, 5.0), 15.0);

    // Berkeley: average the clocks; adjustments bring each to the mean.
    std::vector<double> clocks = {100.0, 104.0, 97.0, 99.0};  // mean = 100
    auto                r      = berkeley_sync(clocks, 0.0, 0);
    EXPECT_DOUBLE_EQ(r.synchronized_time, 100.0);
    EXPECT_DOUBLE_EQ(r.adjustments[1], -4.0);  // node 1 must go back 4
    EXPECT_DOUBLE_EQ(r.adjustments[2], 3.0);   // node 2 must go forward 3
    // Outlier rejection: a wildly-off clock is excluded from the average.
    std::vector<double> withOutlier = {100.0, 101.0, 99.0, 500.0};
    auto                r2          = berkeley_sync(withOutlier, 50.0, 0);
    EXPECT_DOUBLE_EQ(r2.synchronized_time, 100.0);  // 500 excluded -> mean of {100,101,99}
}

TEST(ClockSync, FaultTolerantWindowAndDriftLaw) {
    // Largest group within 2 s: {10, 11, 11.5}; the two outliers are rejected but still adjusted.
    const auto r = berkeley_sync_window({10.0, 30.0, 11.0, 11.5, -40.0}, 2.0);
    EXPECT_NEAR(r.synchronized_time, 32.5 / 3.0, 1e-12);
    EXPECT_EQ(r.included, (std::vector<int>{1, 0, 1, 1, 0}));
    EXPECT_NEAR(r.adjustments[1], 32.5 / 3.0 - 30.0, 1e-12);
    EXPECT_NEAR(berkeley_sync_window({1.0, 3.0}, 0.0).synchronized_time, 2.0, 1e-12);
    EXPECT_DOUBLE_EQ(crystal_drift_ppm(25.0), 0.0);
    EXPECT_NEAR(crystal_drift_ppm(15.0), -3.4, 1e-12);
    EXPECT_NEAR(crystal_drift_ppm(35.0), -3.4, 1e-12);
}

TEST(ClockSync, NetworkSkewStaysWithinTheBound) {
    // Four good clocks with different steady drifts and one broken clock 200 ppm fast.
    const std::vector<std::vector<double>> drift = {
        std::vector<double>(10, -1.0), std::vector<double>(10, 3.0), std::vector<double>(10, -4.0),
        std::vector<double>(10, 0.5), std::vector<double>(10, 200.0)};
    BerkeleyNetworkOptions o;
    o.round_interval = 3600.0;
    o.window         = 0.2;
    const auto t = simulate_berkeley_network(drift, o);
    ASSERT_EQ(t.time_days.size(), 240u);
    const double eps2 = o.max_delay - o.min_delay;  // spread right after a round
    const double grow = (3.0 - (-4.0)) * 1e-6 * o.round_interval;
    for (std::size_t k = 0; k < t.time_days.size(); ++k) {
        EXPECT_EQ(t.rejected[k], 1u) << k;  // the broken clock
        EXPECT_LE(t.skew_after[k], eps2 + 1e-12);
        if (k > 0) EXPECT_LE(t.skew_before[k], eps2 + grow + 1e-12);
    }
    // Without synchronization the clocks drift apart linearly.
    o.synchronize = false;
    const auto free = simulate_berkeley_network(drift, o);
    EXPECT_NEAR(free.before[1].back() - free.before[2].back(), 7e-6 * 10 * 86400.0, 1e-9);
}
