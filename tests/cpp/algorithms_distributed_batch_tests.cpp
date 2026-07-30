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
