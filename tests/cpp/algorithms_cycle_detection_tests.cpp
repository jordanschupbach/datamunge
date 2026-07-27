#include <gtest/gtest.h>

#include <datamunge/algorithms/cycle_detection.hpp>

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

using datamunge::algorithms::brent_cycle_detection;
using datamunge::algorithms::CycleResult;
using datamunge::algorithms::floyd_cycle_detection;

namespace {

// Brute-force reference: iterate f from x0, recording the first index at which each value is seen;
// the first repeat gives mu (start of the cycle) and lambda (period).
CycleResult brute_force(std::uint64_t x0, const std::function<std::uint64_t(std::uint64_t)>& f) {
    std::unordered_map<std::uint64_t, std::size_t> seen;
    std::uint64_t x = x0;
    std::size_t i = 0;
    while (true) {
        auto it = seen.find(x);
        if (it != seen.end()) return CycleResult{i - it->second, it->second};
        seen[x] = i;
        x = f(x);
        ++i;
    }
}

} // namespace

TEST(BrentCycleDetection, ExplicitRhoGraph) {
    // 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 3 (tail 0,1,2; cycle 3,4,5,6).
    const std::vector<std::uint64_t> next = {1, 2, 3, 4, 5, 6, 3};
    auto f = [&](std::uint64_t x) { return next[x]; };
    const auto r = brent_cycle_detection(0, f);
    EXPECT_EQ(r.mu, 3u);
    EXPECT_EQ(r.lambda, 4u);
}

TEST(BrentCycleDetection, FixedPoint) {
    auto f = [](std::uint64_t x) { return x; };
    const auto r = brent_cycle_detection(42, f);
    EXPECT_EQ(r.lambda, 1u);
    EXPECT_EQ(r.mu, 0u);
}

TEST(BrentCycleDetection, PureCycleFromStart) {
    // 0 -> 1 -> 2 -> 0: no tail, period 3.
    auto f = [](std::uint64_t x) { return (x + 1) % 3; };
    const auto r = brent_cycle_detection(0, f);
    EXPECT_EQ(r.lambda, 3u);
    EXPECT_EQ(r.mu, 0u);
}

TEST(BrentCycleDetection, MatchesBruteForceOnRandomMaps) {
    // Random maps on {0..n-1} have both a tail and a cycle; check Brent against the brute-force truth.
    std::mt19937_64 rng(12345);
    for (int trial = 0; trial < 200; ++trial) {
        const std::uint64_t n = 2 + rng() % 60;
        std::vector<std::uint64_t> next(n);
        std::uniform_int_distribution<std::uint64_t> pick(0, n - 1);
        for (auto& v : next) v = pick(rng);
        auto f = [&](std::uint64_t x) { return next[x]; };
        const std::uint64_t x0 = pick(rng);
        const auto got = brent_cycle_detection(x0, f);
        const auto want = brute_force(x0, f);
        EXPECT_EQ(got.lambda, want.lambda) << "trial " << trial;
        EXPECT_EQ(got.mu, want.mu) << "trial " << trial;
    }
}

TEST(FloydCycleDetection, ExplicitRhoGraphAndSpecialCases) {
    const std::vector<std::uint64_t> next = {1, 2, 3, 4, 5, 6, 3};
    auto f = [&](std::uint64_t x) { return next[x]; };
    const auto r = floyd_cycle_detection(0, f);
    EXPECT_EQ(r.mu, 3u);
    EXPECT_EQ(r.lambda, 4u);

    auto fixed = [](std::uint64_t x) { return x; };
    EXPECT_EQ(floyd_cycle_detection(42, fixed).lambda, 1u);
    EXPECT_EQ(floyd_cycle_detection(42, fixed).mu, 0u);

    auto cyc = [](std::uint64_t x) { return (x + 1) % 3; };
    EXPECT_EQ(floyd_cycle_detection(0, cyc).lambda, 3u);
    EXPECT_EQ(floyd_cycle_detection(0, cyc).mu, 0u);
}

TEST(FloydCycleDetection, AgreesWithBrentAndBruteForce) {
    std::mt19937_64 rng(999);
    for (int trial = 0; trial < 200; ++trial) {
        const std::uint64_t n = 2 + rng() % 60;
        std::vector<std::uint64_t> next(n);
        std::uniform_int_distribution<std::uint64_t> pick(0, n - 1);
        for (auto& v : next) v = pick(rng);
        auto f = [&](std::uint64_t x) { return next[x]; };
        const std::uint64_t x0 = pick(rng);
        const auto floyd = floyd_cycle_detection(x0, f);
        const auto brent = brent_cycle_detection(x0, f);
        const auto truth = brute_force(x0, f);
        EXPECT_EQ(floyd.lambda, truth.lambda) << "trial " << trial;
        EXPECT_EQ(floyd.mu, truth.mu) << "trial " << trial;
        EXPECT_EQ(floyd.lambda, brent.lambda);
        EXPECT_EQ(floyd.mu, brent.mu);
    }
}

TEST(BrentCycleDetection, PollardRhoStyleIteration) {
    // g(x) = (x^2 + 1) mod m -- the Pollard-rho iteration; just confirm a valid cycle is found.
    const std::uint64_t m = 8051;
    auto g = [&](std::uint64_t x) { return (x * x + 1) % m; };
    const auto r = brent_cycle_detection(2, g);
    EXPECT_GE(r.lambda, 1u);
    // Verify the reported cycle actually closes: x_{mu} == x_{mu + lambda}.
    std::uint64_t a = 2;
    for (std::size_t i = 0; i < r.mu; ++i) a = g(a);
    std::uint64_t b = a;
    for (std::size_t i = 0; i < r.lambda; ++i) b = g(b);
    EXPECT_EQ(a, b);
}
