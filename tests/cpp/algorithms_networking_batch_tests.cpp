#include <gtest/gtest.h>

#include <datamunge/algorithms/exponential_backoff.hpp>
#include <datamunge/algorithms/karn.hpp>
#include <datamunge/algorithms/leaky_bucket.hpp>
#include <datamunge/algorithms/nagle.hpp>
#include <datamunge/algorithms/token_bucket.hpp>
#include <datamunge/algorithms/truncated_binary_exponential_backoff.hpp>

#include <vector>

using namespace datamunge::algorithms;

// ---- Exponential backoff ----------------------------------------------------

TEST(ExponentialBackoff, DoublesThenCaps) {
    auto s = exponential_backoff_sequence(8, 1.0, 30.0);
    EXPECT_EQ(s, (std::vector<double>{1, 2, 4, 8, 16, 30, 30, 30}));
}

TEST(ExponentialBackoff, JitterWithinWindow) {
    EXPECT_DOUBLE_EQ(exponential_backoff_jittered(3, 1.0, 100.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(exponential_backoff_jittered(3, 1.0, 100.0, 0.5), 4.0);  // 0.5 * 8
    EXPECT_DOUBLE_EQ(exponential_backoff_jittered(3, 1.0, 100.0, 1.0), 8.0);
}

// ---- Truncated binary exponential backoff -----------------------------------

TEST(TruncatedBackoff, WindowDoublesAndTruncates) {
    EXPECT_EQ(contention_window(0), 1);
    EXPECT_EQ(contention_window(3), 8);
    EXPECT_EQ(contention_window(10), 1024);
    EXPECT_EQ(contention_window(15), 1024);   // truncated at 10
    EXPECT_EQ(max_backoff_slots(3), 7);
    EXPECT_FALSE(backoff_give_up(15));
    EXPECT_TRUE(backoff_give_up(16));          // Ethernet abandons after 16
}

// ---- Nagle's algorithm ------------------------------------------------------

TEST(Nagle, CoalescesSmallWrites) {
    std::vector<NagleEvent> ev = {{NagleEvent::Write, 100}, {NagleEvent::Write, 100},
                                  {NagleEvent::Write, 100}, {NagleEvent::Write, 100},
                                  {NagleEvent::Ack, 100}};
    auto with    = nagle_simulate(ev, 500);
    auto without = no_nagle_simulate(ev, 500);
    EXPECT_EQ(with, (std::vector<int>{100, 300}));               // first sent, rest coalesced
    EXPECT_EQ(without.size(), 4u);                               // one packet per write
}

TEST(Nagle, FullSegmentsSentRemainderHeld) {
    // 1200 bytes, MSS 500: the two full segments go immediately; the 200-byte tail is held
    // (Nagle) because there is now unacknowledged data outstanding.
    std::vector<NagleEvent> ev = {{NagleEvent::Write, 1200}};
    auto                    p  = nagle_simulate(ev, 500);
    EXPECT_EQ(p, (std::vector<int>{500, 500}));
    // An acknowledgement releases the held remainder.
    ev.push_back({NagleEvent::Ack, 1000});
    EXPECT_EQ(nagle_simulate(ev, 500), (std::vector<int>{500, 500, 200}));
}

// ---- Karn's algorithm -------------------------------------------------------

TEST(Karn, IgnoresRetransmittedSamples) {
    KarnEstimator k(1.0);
    k.on_ack(0.2, false);
    double srtt_before = k.srtt();
    k.on_ack(9.0, true);                 // ambiguous retransmit sample -> ignored
    EXPECT_DOUBLE_EQ(k.srtt(), srtt_before);
}

TEST(Karn, TimeoutBacksOffRto) {
    KarnEstimator k(1.0);
    k.on_ack(0.2, false);
    double rto = k.rto();
    k.on_timeout();
    EXPECT_DOUBLE_EQ(k.rto(), 2 * rto);
    k.on_timeout();
    EXPECT_DOUBLE_EQ(k.rto(), 4 * rto);
}

TEST(Karn, SrttTracksSamples) {
    KarnEstimator k(1.0);
    for (int i = 0; i < 50; ++i) k.on_ack(0.3, false);  // steady RTT
    EXPECT_NEAR(k.srtt(), 0.3, 1e-6);
    EXPECT_NEAR(k.rttvar(), 0.0, 1e-3);
}

// ---- Token bucket -----------------------------------------------------------

TEST(TokenBucket, AllowsBurstThenLimits) {
    TokenBucket tb(5, 1);              // capacity 5, 1 token/sec
    int         allowed = 0;
    for (int i = 0; i < 7; ++i)
        if (tb.allow(0.0)) ++allowed;
    EXPECT_EQ(allowed, 5);            // burst up to capacity
    EXPECT_FALSE(tb.allow(0.0));      // empty
    EXPECT_NEAR(tb.available(3.0), 3.0, 1e-9);  // refilled 3 tokens by t=3
    EXPECT_TRUE(tb.allow(3.0));
}

// ---- Leaky bucket -----------------------------------------------------------

TEST(LeakyBucket, AcceptsToCapacityThenDrains) {
    LeakyBucket lb(5, 1);            // capacity 5, leak 1/sec
    int         accepted = 0;
    for (int i = 0; i < 7; ++i)
        if (lb.add(0.0)) ++accepted;
    EXPECT_EQ(accepted, 5);         // fills to capacity, rest dropped
    EXPECT_NEAR(lb.level(2.0), 3.0, 1e-9);  // leaked 2 units by t=2
    EXPECT_TRUE(lb.add(2.0));       // room again
}
