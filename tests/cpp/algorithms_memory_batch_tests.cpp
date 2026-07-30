#include <gtest/gtest.h>

#include <datamunge/algorithms/buddy_allocator.hpp>
#include <datamunge/algorithms/cheney.hpp>
#include <datamunge/algorithms/generational.hpp>
#include <datamunge/algorithms/mark_compact.hpp>
#include <datamunge/algorithms/mark_sweep.hpp>
#include <datamunge/algorithms/object_graph.hpp>
#include <datamunge/algorithms/reference_counting.hpp>

#include <algorithm>
#include <vector>

using namespace datamunge::algorithms;

namespace {
// root 0 -> 1 -> 2 (live); cycle 3<->4, node 5, chain 6->7 all unreachable.
ObjectGraph sample() {
    ObjectGraph g;
    g.num_objects = 8;
    g.refs        = {{1}, {2}, {}, {4}, {3}, {}, {7}, {}};
    g.roots       = {0};
    return g;
}
}  // namespace

TEST(MarkSweep, CollectsAllUnreachableIncludingCycles) {
    auto r = mark_sweep(sample());
    EXPECT_EQ(r.live, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(r.freed, (std::vector<int>{3, 4, 5, 6, 7}));  // cycle 3,4 is collected
}

TEST(MarkCompact, SlidesSurvivorsAndUpdatesRefs) {
    auto r = mark_compact(sample());
    EXPECT_EQ(r.order, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(r.new_address[0], 0);
    EXPECT_EQ(r.new_address[1], 1);
    EXPECT_EQ(r.new_address[2], 2);
    EXPECT_EQ(r.new_address[3], -1);       // dead
    // object 0's ref to 1 becomes new address 1.
    EXPECT_EQ(r.updated_refs[0], (std::vector<int>{1}));
}

TEST(Cheney, CopiesLiveBreadthFirst) {
    auto r = cheney_copy(sample());
    EXPECT_EQ(r.to_space_order, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(r.forwarding[0], 0);
    EXPECT_EQ(r.forwarding[5], -1);        // garbage, not copied
}

TEST(ReferenceCounting, FreesAcyclicButLeaksCycle) {
    auto r = reference_counting_collect(sample());
    // 5, 6, 7 are acyclic garbage -> reclaimed by cascading counts.
    EXPECT_TRUE(r.freed[5]);
    EXPECT_TRUE(r.freed[6]);
    EXPECT_TRUE(r.freed[7]);
    // 3 and 4 form a cycle -> leaked.
    EXPECT_EQ(r.leaked, (std::vector<int>{3, 4}));
    EXPECT_FALSE(r.freed[3]);
    // live objects are never freed.
    EXPECT_FALSE(r.freed[0]);
    EXPECT_FALSE(r.freed[1]);
}

TEST(Generational, MinorScansLessThanFull) {
    // A long-lived old chain plus a small young nursery.
    ObjectGraph g;
    g.num_objects = 7;
    // 0(old)->1(old)->2(old)->3(old); root also holds young 4->5; young 6 is garbage.
    g.refs  = {{1}, {2}, {3}, {}, {5}, {}, {}};
    g.roots = {0, 4};
    std::vector<int> gen = {1, 1, 1, 1, 0, 0, 0};  // 0..3 old, 4..6 young

    auto r = minor_gc(g, gen);
    EXPECT_LT(r.minor_scanned, r.full_scanned);    // minor collection scans fewer objects
    EXPECT_EQ(r.minor_scanned, 2);                 // young 4 and 5
    EXPECT_EQ(r.full_scanned, 6);                  // 0,1,2,3,4,5
    EXPECT_TRUE(r.promoted[4]);
    EXPECT_TRUE(r.promoted[5]);
    EXPECT_EQ(r.young_freed, (std::vector<int>{6}));
}

TEST(Generational, RememberedSetCapturesOldToYoung) {
    // Old object 0 references young object 1; no direct root to 1.
    ObjectGraph g;
    g.num_objects = 2;
    g.refs        = {{1}, {}};
    g.roots       = {0};
    std::vector<int> gen = {1, 0};
    auto             r   = minor_gc(g, gen);
    EXPECT_EQ(r.remembered_set, (std::vector<int>{0}));
    EXPECT_TRUE(r.promoted[1]);  // kept alive via the remembered set
}

// ---- Buddy allocator --------------------------------------------------------

TEST(BuddyAllocator, RoundsUpAndTracksFragmentation) {
    BuddyAllocator ba(1024, 32);
    int a = ba.allocate(30);   // -> 32-byte block
    int b = ba.allocate(100);  // -> 128-byte block
    int c = ba.allocate(50);   // -> 64-byte block
    EXPECT_GE(a, 0);
    EXPECT_GE(b, 0);
    EXPECT_GE(c, 0);
    EXPECT_EQ(ba.allocated_bytes(), 32 + 128 + 64);
    EXPECT_EQ(ba.requested_bytes(), 30 + 100 + 50);
    EXPECT_EQ(ba.internal_fragmentation(), (32 - 30) + (128 - 100) + (64 - 50));
}

TEST(BuddyAllocator, FreeCoalescesBuddies) {
    BuddyAllocator ba(1024, 32);
    int a = ba.allocate(32);
    int b = ba.allocate(32);
    (void)a;
    (void)b;
    ba.free(a);
    ba.free(b);
    // Everything free again -> one block at the top order (1024 / 32 = 2^5).
    auto counts = ba.free_blocks_per_order();
    EXPECT_EQ(counts[ba.max_order()], 1);
    for (int o = 0; o < ba.max_order(); ++o) EXPECT_EQ(counts[o], 0);
}

TEST(BuddyAllocator, ExhaustionReturnsMinusOne) {
    BuddyAllocator ba(64, 32);
    int a = ba.allocate(32);
    int b = ba.allocate(32);
    int c = ba.allocate(32);  // arena full (only two 32-byte blocks)
    EXPECT_GE(a, 0);
    EXPECT_GE(b, 0);
    EXPECT_EQ(c, -1);
}
