#include <datamunge/algorithms/bakery.hpp>

#include <gtest/gtest.h>

#include <thread>

namespace alg = datamunge::algorithms;

TEST(BakeryModel, MutualExclusionHoldsWithChoosing) {
    for (std::size_t n : {2u, 3u}) {
        const auto r = alg::bakery_model_check({n, true, 4});
        EXPECT_TRUE(r.mutual_exclusion) << n;
        EXPECT_EQ(r.max_critical, 1u);
        EXPECT_TRUE(r.counterexample.empty());
        EXPECT_GT(r.states, 100u);
        EXPECT_GT(r.cut_by_bound, 0u);
    }
}

TEST(BakeryModel, ChoosingFlagIsNecessary) {
    const auto r = alg::bakery_model_check({2, false, 4});
    EXPECT_FALSE(r.mutual_exclusion);
    EXPECT_EQ(r.max_critical, 2u);
    ASSERT_FALSE(r.counterexample.empty());
    // Replaying the counterexample in the simulator's terms: the last step enters a critical section.
    EXPECT_NE(r.counterexample.back().action.find("enter critical section"), std::string::npos);
    EXPECT_THROW(alg::bakery_model_check({0, true, 4}), std::invalid_argument);
}

TEST(BakeryModel, SimulationServesEveryone) {
    const auto s = alg::bakery_simulate({3, true, 0}, 20000, 7);
    EXPECT_EQ(s.violations, 0u);
    EXPECT_EQ(s.phase.size(), 20000u);
    for (auto e : s.entries) EXPECT_GT(e, 50u);
    const auto bad = alg::bakery_simulate({3, false, 0}, 200000, 7);
    EXPECT_GT(bad.violations, 0u);
}

TEST(BakeryLock, ProtectsACounter) {
    const std::size_t threads = 4, per = 20000;
    alg::BakeryLock   lock(threads);
    long              counter = 0;
    std::vector<std::thread> pool;
    for (std::size_t t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            for (std::size_t k = 0; k < per; ++k) {
                lock.lock(t);
                const long v = counter;  // non-atomic read-modify-write inside the lock
                counter      = v + 1;
                lock.unlock(t);
            }
        });
    for (auto& th : pool) th.join();
    EXPECT_EQ(counter, static_cast<long>(threads * per));
    EXPECT_GE(lock.max_ticket(), 1u);
    EXPECT_THROW(alg::BakeryLock(0), std::invalid_argument);
}
